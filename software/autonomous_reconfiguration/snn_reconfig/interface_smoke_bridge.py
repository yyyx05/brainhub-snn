"""Frontend-to-Verilator command replay for the bounded shared-array fixture.

This is not a live MMIO transport. Wait commands are checked by the simulator,
not completed while this object records them. Catalog assets are identified but
replaced by the existing small roundtrip's synthetic worksets in the testbench.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_bridge import execute_selected_profile


@dataclass(frozen=True)
class FixtureBinding:
    opcode: int
    contract: str
    learning_scope: str
    descriptor: int


BINDINGS = {
    "fc_nmnist_h160": FixtureBinding(0x01, "fc_native_stream_v1", "W2 only", 80),
    "srnn_shd_h450_m2": FixtureBinding(
        0x10, "srnn_native_stream_v1", "M2 Wout only", 80
    ),
    "scnn_dvsgesture_vgg9": FixtureBinding(
        0x20, "scnn_graph_frame_service_v1", "classifier only", 0xC0801
    ),
}

# Each command executes a bounded ready/valid or completion wait in the TB.
STOP, DRAIN, LOAD_FIXTURE, PROFILE, ALIGN, DISPATCH, COMPLETE, RESUME = range(1, 9)


class FixtureReplayTransport:
    """Compile NativeControlTransport calls into an ordered simulation program."""

    def __init__(self) -> None:
        self.commands: list[tuple[int, int, int]] = []
        self.selections: list[dict[str, Any]] = []
        self._binding: FixtureBinding | None = None
        self._assets: list[dict[str, Any]] = []
        self._stage = 0

    def plan(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Validate the whole binding before allowing any transport side effect."""
        if decision.get("status") != "hardware_profile_selected":
            raise ValueError("rejected decisions must emit no simulator commands")
        if self._stage:
            raise ValueError("previous replay transaction is incomplete")
        selected = decision["selected_profile"]
        binding = BINDINGS.get(selected.get("profile_id"))
        if binding is None:
            raise ValueError("profile has no small-fixture binding")
        if selected.get("profile_code") != f"0x{binding.opcode:02x}":
            raise ValueError("opcode does not match the fixed profile")
        if selected.get("dispatch_contract_id") != binding.contract:
            raise ValueError("dispatch contract does not match the fixed profile")
        if selected.get("learning_scope", {}).get("trainable_parameters") != binding.learning_scope:
            raise ValueError("fixed writable mask/scope changed")
        if decision["learning_decision"].get("enabled") or decision["hardware_plan"].get("learning_enable"):
            raise ValueError("learning-on is blocked: this fixture has no learner/publication DUT")
        self._binding = binding
        self._assets = selected["deployment_assets"]
        result = execute_selected_profile(decision, self)
        result["status"] = "simulation_replay_pending"
        result["boundary"] = (
            "Commands recorded only; completion requires a checked DUT transcript. "
            "Small synthetic worksets replace catalog assets; no final-top/DMA/M2/learning claim."
        )
        self.selections.append({
            "profile_id": selected["profile_id"],
            "opcode": binding.opcode,
            "dispatch_contract": binding.contract,
            "learning_scope": binding.learning_scope,
            "catalog_assets_not_loaded": self._assets,
            "execution": result,
        })
        return result

    def _append(self, operation: int, payload: int = 0) -> None:
        if self._binding is None or operation != self._stage + 1:
            raise ValueError("out-of-order simulation operation")
        self.commands.append((operation, self._binding.opcode, payload))
        self._stage = operation if operation != RESUME else 0

    def stop_accepting_new_work(self) -> None:
        self._append(STOP)

    def wait_for_quiescent(self) -> None:
        self._append(DRAIN)

    def load_deployment_assets(self, assets: list[dict[str, Any]]) -> None:
        if assets != self._assets:
            raise ValueError("deployment assets differ from selected metadata")
        self._append(LOAD_FIXTURE)

    def request_profile(self, opcode: str, learning_enable: bool) -> None:
        if self._binding is None or int(opcode, 16) != self._binding.opcode or learning_enable:
            raise ValueError("unsupported opcode or learning request")
        self._append(PROFILE)

    def wait_for_profile_alignment(self) -> None:
        self._append(ALIGN)

    def submit_runtime_descriptors(self, profile_id: str, dispatch_contract_id: str) -> None:
        if BINDINGS.get(profile_id) != self._binding or dispatch_contract_id != self._binding.contract:
            raise ValueError("descriptor family does not match active profile")
        self._append(DISPATCH, self._binding.descriptor)

    def wait_for_learning_commit(self) -> None:
        raise ValueError("no learning commit DUT in this fixture")

    def continue_without_learning_commit(self) -> None:
        self._append(COMPLETE)

    def resume_accepting_work(self) -> None:
        self._append(RESUME)

    def write_program(self, path: Path) -> None:
        if self._stage or not self.commands:
            raise ValueError("cannot write empty or incomplete replay")
        path.write_text("".join(f"{op} {code} {data}\n" for op, code, data in self.commands), encoding="ascii")


def check_transcript(log: str, commands: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Check actual TB acknowledgements, not the host's planned wait records."""
    acknowledgements = []
    profiles = []
    observations = []
    drains = []
    counters = None
    for line in log.splitlines():
        if line.startswith("IF_ACK,"):
            acknowledgements.append(tuple(map(int, line.split(",")[1:])))
        elif line.startswith("IF_PROFILE,"):
            profiles.append(tuple(map(int, line.split(",")[1:])))
        elif line.startswith("IF_OBSERVATIONS,"):
            observations.append(tuple(map(int, line.split(",")[1:])))
        elif line.startswith("IF_DRAIN,"):
            drains.append(tuple(map(int, line.split(",")[1:])))
        elif line.startswith("IF_COUNTS,"):
            if counters is not None:
                raise ValueError("duplicate DUT counters")
            counters = tuple(map(int, line.split(",")[1:]))
    expected_acks = [(index, *command) for index, command in enumerate(commands)]
    codes = [code for op, code, _ in commands if op == PROFILE]
    checks = {
        "all_commands_acknowledged_in_order": acknowledgements == expected_acks,
        "fixed_profile_sequence": codes == [1, 16, 32, 1],
        "profile_version_and_ready_valid_backpressure": profiles == [
            (code, index + 1, 5) for index, code in enumerate(codes)
        ],
        "fixture_numeric_observations": observations == [
            (1, 80, 80, 0, 0), (16, 0, 0, 80, 0),
            (32, 0, 0, 0, 1), (1, 80, 80, 0, 0),
        ],
        "owner_retained_until_sink_drain": drains == [(code, 5) for code in codes],
        # Reset releases, profiles, dense layers, graph descriptors, groups,
        # recurrent groups, maintenance leases, Conv contexts.
        "dut_handshake_counts_and_no_midsequence_reset": counters == (1, 4, 3, 1, 4, 1, 4, 2),
        "dut_terminal_pass": "IF_PASS" in log.splitlines(),
        "no_dut_failure": "%Error" not in log and "IF_FATAL" not in log,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "acknowledged_commands": len(acknowledgements),
        "profiles": profiles,
        "observations": observations,
        "owner_drains": drains,
        "dut_counters": counters,
    }
