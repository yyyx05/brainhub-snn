"""Platform-neutral execution order for a selected hardware profile."""

from __future__ import annotations

from typing import Any, Protocol


class NativeControlTransport(Protocol):
    """Operations that a future MMIO, simulation, or board driver must provide."""

    def stop_accepting_new_work(self) -> None: ...

    def wait_for_quiescent(self) -> None: ...

    def load_deployment_assets(self, assets: list[dict[str, Any]]) -> None: ...

    def request_profile(self, opcode: str, learning_enable: bool) -> None: ...

    def wait_for_profile_alignment(self) -> None: ...

    def submit_runtime_descriptors(
        self, profile_id: str, dispatch_contract_id: str
    ) -> None: ...

    def wait_for_learning_commit(self) -> None: ...

    def continue_without_learning_commit(self) -> None: ...

    def resume_accepting_work(self) -> None: ...


class TraceTransport:
    """Record a host sequence without driving RTL, MMIO, or a board."""

    def __init__(self, contracts: dict[str, dict[str, Any]] | None = None) -> None:
        self.contracts = contracts or {}
        self.events: list[dict[str, Any]] = []

    def _record(self, operation: str, **details: Any) -> None:
        self.events.append({"operation": operation, **details})

    def stop_accepting_new_work(self) -> None:
        self._record("stop_accepting_new_work")

    def wait_for_quiescent(self) -> None:
        self._record("wait_for_owner_drain_and_quiescent")

    def load_deployment_assets(self, assets: list[dict[str, Any]]) -> None:
        self._record("load_deployment_assets", assets=assets)

    def request_profile(self, opcode: str, learning_enable: bool) -> None:
        self._record(
            "request_profile",
            profile_opcode=opcode,
            learning_enable=learning_enable,
        )

    def wait_for_profile_alignment(self) -> None:
        self._record("wait_for_profile_alignment_and_version_visibility")

    def submit_runtime_descriptors(
        self, profile_id: str, dispatch_contract_id: str
    ) -> None:
        details: dict[str, Any] = {
            "profile_id": profile_id,
            "dispatch_contract_id": dispatch_contract_id,
        }
        if self.contracts:
            contract = self.contracts[dispatch_contract_id]
            if contract["profile_id"] != profile_id:
                raise ValueError("dispatch contract does not belong to the selected profile")
            details["request_family"] = contract["request_family"]
            details["interface_groups"] = contract["interface_groups"]
        self._record("submit_runtime_descriptors", **details)

    def wait_for_learning_commit(self) -> None:
        self._record("wait_for_learning_commit_and_published_version")

    def continue_without_learning_commit(self) -> None:
        self._record("continue_without_learning_commit")

    def resume_accepting_work(self) -> None:
        self._record("resume_accepting_work")


def execute_selected_profile(
    decision: dict[str, Any], transport: NativeControlTransport
) -> dict[str, Any]:
    """Execute one accepted decision using the required ownership sequence."""

    if decision.get("status") != "hardware_profile_selected":
        raise ValueError("only an accepted hardware profile can be executed")
    selected = decision.get("selected_profile")
    plan = decision.get("hardware_plan")
    learning = decision.get("learning_decision")
    if not isinstance(selected, dict) or not isinstance(plan, dict) or not isinstance(learning, dict):
        raise ValueError("selected decision is missing profile, plan, or learning metadata")
    if plan.get("control_transport") != "native_profile_descriptor_handshake":
        raise ValueError("unsupported control transport")
    opcode = selected.get("profile_code")
    if opcode != plan.get("profile_opcode"):
        raise ValueError("selected profile opcode and hardware plan differ")
    learning_enable = bool(learning.get("enabled", False))
    if learning_enable != bool(plan.get("learning_enable", False)):
        raise ValueError("learning decision and hardware plan differ")
    dispatch_contract_id = selected.get("dispatch_contract_id")
    if not isinstance(dispatch_contract_id, str) or not dispatch_contract_id:
        raise ValueError("selected profile is missing its dispatch contract")
    if dispatch_contract_id != plan.get("profile_dispatch_contract"):
        raise ValueError("selected profile and hardware plan dispatch contracts differ")

    trace: list[str] = []

    def run(step: str, operation: Any, *args: Any) -> None:
        operation(*args)
        trace.append(step)

    run("stop_accepting_new_work", transport.stop_accepting_new_work)
    run("wait_for_owner_drain_and_quiescent", transport.wait_for_quiescent)
    run(
        "load_qualified_weights_state_and_descriptors_if_needed",
        transport.load_deployment_assets,
        selected["deployment_assets"],
    )
    run(
        "request_profile_opcode_and_learning_enable",
        transport.request_profile,
        opcode,
        learning_enable,
    )
    run("wait_for_profile_alignment_and_version_visibility", transport.wait_for_profile_alignment)
    run(
        "submit_runtime_descriptors",
        transport.submit_runtime_descriptors,
        selected["profile_id"],
        dispatch_contract_id,
    )
    if learning_enable:
        run(
            "wait_for_learning_commit_and_published_version_before_next_inference",
            transport.wait_for_learning_commit,
        )
    else:
        run(
            "continue_inference_without_learning_commit",
            transport.continue_without_learning_commit,
        )
    run("resume_accepting_work", transport.resume_accepting_work)
    return {
        "status": "execution_sequence_completed",
        "profile_id": selected["profile_id"],
        "profile_opcode": opcode,
        "dispatch_contract_id": dispatch_contract_id,
        "learning_enable": learning_enable,
        "trace": trace,
        "boundary": (
            "Host sequence only. A concrete transport must still prove signal/register "
            "handshakes, timeout behavior, and error propagation."
        ),
    }
