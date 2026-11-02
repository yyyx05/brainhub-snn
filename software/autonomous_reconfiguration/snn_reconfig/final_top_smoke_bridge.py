"""DecisionEngine replay into the real final top's native interfaces.

This records a simulation program, not completed handshakes or a DMA load.
Synthetic assets deliberately replace the selected catalog assets. The final
top has no opcode-ready port: opcodes select native config/layer/graph requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_bridge import execute_selected_profile


STOP, DRAIN, LOAD_FIXTURE, CONFIG, ALIGN, DISPATCH, COMPLETE, RESUME = range(1, 9)


@dataclass(frozen=True)
class Binding:
    opcode: int
    contract: str
    scope: str
    parameter_mask: int
    parameter_owner: int


BINDINGS = {
    "fc_nmnist_h160": Binding(1, "fc_native_stream_v1", "W2 only", 1, 1),
    "srnn_shd_h450_m2": Binding(16, "srnn_native_stream_v1", "M2 Wout only", 2, 2),
    "scnn_dvsgesture_vgg9": Binding(32, "scnn_graph_frame_service_v1", "classifier only", 4, 3),
}

# Order matches frame_descriptor_t in the production system, without padding.
DESCRIPTOR_FIELDS = (
    "parameter_base", "in_width", "in_height", "in_channels", "out_width",
    "out_height", "out_channels", "stride", "padding", "pool_mode", "work_base",
    "leak", "threshold", "reset_to_zero", "trace_decay", "layer_tag",
    "feature_write_slot", "feature_write_base", "feature_write_row_stride",
    "feature_write_channel_stride",
)
DENSE_FIELDS = (
    "mode", "neuron_count", "feedforward_group_count", "recurrent_group_count",
    "recurrent_enable", "weight_int8", "current_shift6", "leak", "threshold",
    "reset_to_zero", "trace_decay", "transaction_tag", "work_base", "tile_base",
    "source_mask", "weight_addr0", "weight_addr1", "weight_addr2", "weight_addr3",
    "parameter_base", "hidden_stride", "source_count", "tile_count", "version",
    "expected_pre_state", "expected_spike",
)


def scnn_descriptors() -> list[dict[str, int]]:
    """Existing B2T2 small frame-service geometry, not VGG9 model descriptors."""
    return [dict(zip(DESCRIPTOR_FIELDS, (
        0, 2, 2, 1, 2, 2, 1, 1, 1, 0, 0, 0x7FFF, 256, 0, 0x7FFF,
        layer + 1, 0, 0, 1, 2,
    )), sample=frame // 2, timestep=frame % 2, last=int(layer == 7))
        for frame in range(4) for layer in range(8)]


class FinalTopReplayTransport:
    def __init__(self) -> None:
        self.commands: list[tuple[int, int, tuple[int, ...]]] = []
        self.selections: list[dict[str, Any]] = []
        self._binding: Binding | None = None
        self._assets: list[dict[str, Any]] = []
        self._stage = 0

    def plan(self, decision: dict[str, Any]) -> dict[str, Any]:
        if decision.get("status") != "hardware_profile_selected":
            raise ValueError("rejected selection must emit no DUT commands")
        if self._stage:
            raise ValueError("previous transaction is incomplete")
        selected = decision.get("selected_profile", {})
        binding = BINDINGS.get(selected.get("profile_id"))
        if binding is None:
            raise ValueError("no bounded final-top binding")
        if selected.get("profile_code") != f"0x{binding.opcode:02x}":
            raise ValueError("fixed opcode changed")
        if selected.get("dispatch_contract_id") != binding.contract:
            raise ValueError("fixed dispatch contract changed")
        if selected.get("learning_scope", {}).get("trainable_parameters") != binding.scope:
            raise ValueError("fixed writable scope changed")
        if decision.get("learning_decision", {}).get("enabled") or decision.get("hardware_plan", {}).get("learning_enable"):
            raise ValueError("this bounded replay is learning-off only")
        self._binding = binding
        self._assets = selected["deployment_assets"]
        # execute_selected_profile validates all plan consistency before STOP.
        result = execute_selected_profile(decision, self)
        result.update(status="simulation_replay_pending", boundary=(
            "Real final-top completion requires checked simulator acknowledgements; "
            "synthetic fixtures are not catalog DMA, M2 inference or learning publication."
        ))
        self.selections.append({
            "profile_id": selected["profile_id"], "opcode": binding.opcode,
            "dispatch_contract": binding.contract, "writable_scope": binding.scope,
            "catalog_assets_not_loaded": self._assets, "execution": result,
        })
        return result

    def _append(self, operation: int, *payload: int) -> None:
        if self._binding is None or operation != self._stage + 1:
            raise ValueError("out-of-order native replay operation")
        self.commands.append((operation, self._binding.opcode, tuple(payload)))
        self._stage = operation if operation != RESUME else 0

    def stop_accepting_new_work(self) -> None:
        self._append(STOP)

    def wait_for_quiescent(self) -> None:
        self._append(DRAIN)

    def load_deployment_assets(self, assets: list[dict[str, Any]]) -> None:
        if assets != self._assets:
            raise ValueError("selected asset metadata changed")
        # base, count, code, residual: native FC load or external SRNN responder.
        code = self._binding.opcode
        self._append(LOAD_FIXTURE, 64 if code == 1 else 0, 40 if code == 32 else 4,
                     0 if code == 32 else 10, 0)

    def request_profile(self, opcode: str, learning_enable: bool) -> None:
        if int(opcode, 16) != self._binding.opcode or learning_enable:
            raise ValueError("unsupported native profile/learning request")
        b = self._binding
        self._append(CONFIG, b.parameter_mask, int(b.opcode == 32), b.opcode,
                     b.opcode, 0, 5, 7)

    def wait_for_profile_alignment(self) -> None:
        self._append(ALIGN)

    def submit_runtime_descriptors(self, profile_id: str, dispatch_contract_id: str) -> None:
        if BINDINGS.get(profile_id) != self._binding or dispatch_contract_id != self._binding.contract:
            raise ValueError("descriptor family does not match selection")
        code = self._binding.opcode
        if code == 32:
            records = scnn_descriptors()
            payload = [2, 2, 0, len(records)]
            for record in records:
                payload.extend(record[name] for name in ("sample", "timestep", "last", *DESCRIPTOR_FIELDS))
        else:
            # Two valid source lanes with code 10, one group, two neurons.
            payload = [int(code == 16), 2, 1, 0, 0, 0, 0, 0, 8, 1, 0,
                       code * 100 + 1, 0, 0, 3, 0, 0, 0, 0,
                       64 if code == 1 else 0, 2, 2, 1, 5, 20, 1]
        self._append(DISPATCH, *payload)

    def wait_for_learning_commit(self) -> None:
        raise ValueError("learning publication is not exercised")

    def continue_without_learning_commit(self) -> None:
        self._append(COMPLETE)

    def resume_accepting_work(self) -> None:
        self._append(RESUME)

    def write_program(self, path: Path) -> None:
        if self._stage or not self.commands:
            raise ValueError("empty or incomplete replay")
        path.write_text("".join(
            " ".join(map(str, (op, code, len(payload), *payload))) + "\n"
            for op, code, payload in self.commands), encoding="ascii")


def check_transcript(log: str, commands: list[tuple[int, int, tuple[int, ...]]]) -> dict[str, Any]:
    rows: dict[str, list[tuple[int, ...]]] = {}
    for line in log.splitlines():
        if line.startswith("FT_") and "," in line:
            name, *values = line.split(",")
            rows.setdefault(name, []).append(tuple(map(int, values)))
    acks = [(n, op, code, len(payload)) for n, (op, code, payload) in enumerate(commands)]
    counts = rows.get("FT_COUNTS", [])
    scnn = rows.get("FT_SCNN", [])
    checks = {
        "ordered_program_acknowledgements": rows.get("FT_ACK") == acks,
        "selected_native_configuration": rows.get("FT_CONFIG") == [(1, 1, 1, 0), (16, 2, 2, 0), (32, 4, 3, 0), (1, 1, 1, 0)],
        "owner_and_sink_quiescent_drains": rows.get("FT_DRAIN") == [(1, 0), (16, 0), (32, 0), (1, 0)],
        "dense_numeric_inference_and_capture": rows.get("FT_DENSE") == [(1, 2, 1, 1), (16, 2, 1, 1), (1, 2, 1, 1)],
        "full_graph_and_state_trace_service": len(scnn) == 1 and scnn[0][:10] == (32, 4, 68, 68, 4, 4, 144, 144, 144, 144),
        "descriptor_and_feature_request_backpressure": len(scnn) == 1 and len(scnn[0]) == 13 and scnn[0][10] > 0 and scnn[0][11] > 0,
        "one_reset_no_learning_writes_or_commits": counts == [(1, 3, 3, 6, 32, 0, 0, 0, 0)],
        "actual_terminal_pass": log.splitlines().count("FT_PASS") == 1,
        "no_failure": "FT_FATAL" not in log and "%Error" not in log,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "acknowledged_commands": len(rows.get("FT_ACK", [])), "rows": rows}
