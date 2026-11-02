"""Run the fixed FC/SRNN/SCNN selector sequence as a software dry-run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ROOT = PROJECT_ROOT / "software" / "autonomous_reconfiguration"
sys.path.insert(0, str(COMPONENT_ROOT))

from snn_reconfig.dataset_adapters import (  # noqa: E402
    fingerprint_dvsgesture_q8_8_hex,
    fingerprint_shd_h5,
)
from snn_reconfig.decision import DecisionEngine, load_json  # noqa: E402
from snn_reconfig.runtime_bridge import TraceTransport, execute_selected_profile  # noqa: E402


def _request_sequence() -> list[tuple[str, dict[str, Any]]]:
    examples = COMPONENT_ROOT / "examples"
    return [
        ("nmnist_initial", load_json(examples / "nmnist_query.json")),
        (
            "shd_real_project_data",
            {
                "fingerprint": fingerprint_shd_h5(
                    PROJECT_ROOT / "imports" / "srnn" / "dataset_cache" / "SHD" / "shd_train.h5",
                    range(32),
                )
            },
        ),
        ("unsupported_smnist", load_json(examples / "smnist_unsupported_query.json")),
        (
            "dvsgesture_real_project_data",
            {
                "fingerprint": fingerprint_dvsgesture_q8_8_hex(
                    PROJECT_ROOT
                    / "imports"
                    / "scnn"
                    / "datasets"
                    / "DVSGesture"
                    / "hardware"
                    / "test_10x20_inputs_q8_8.hex",
                    10,
                )
            },
        ),
        ("nmnist_return", load_json(examples / "nmnist_query.json")),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "paper_evidence"
            / "supplement_20261004"
            / "software"
            / "autonomous_reconfiguration_demo"
            / "report.json"
        ),
    )
    args = parser.parse_args()

    engine = DecisionEngine.from_files(
        COMPONENT_ROOT / "catalog" / "hardware_profiles.json",
        COMPONENT_ROOT / "catalog" / "fingerprint_prototypes.json",
    )
    dispatch_catalog = load_json(
        COMPONENT_ROOT / "catalog" / "profile_dispatch_contracts.json"
    )
    contracts = {
        item["contract_id"]: item for item in dispatch_catalog["contracts"]
    }
    transport = TraceTransport(contracts)
    steps: list[dict[str, Any]] = []

    for name, request in _request_sequence():
        before = len(transport.events)
        decision = engine.decide(request, "off")
        execution = None
        if decision["status"] == "hardware_profile_selected":
            execution = execute_selected_profile(decision, transport)
        after = len(transport.events)
        steps.append(
            {
                "name": name,
                "decision_status": decision["status"],
                "profile_id": (
                    decision["selected_profile"]["profile_id"]
                    if decision["selected_profile"]
                    else None
                ),
                "profile_opcode": (
                    decision["selected_profile"]["profile_code"]
                    if decision["selected_profile"]
                    else None
                ),
                "dispatch_contract_id": (
                    decision["selected_profile"]["dispatch_contract_id"]
                    if decision["selected_profile"]
                    else None
                ),
                "transport_event_range": [before, after],
                "transport_events_added": after - before,
                "execution": execution,
            }
        )

    selected_opcodes = [
        step["profile_opcode"] for step in steps if step["profile_opcode"] is not None
    ]
    rejected = [step for step in steps if step["decision_status"] != "hardware_profile_selected"]
    checks = {
        "selected_opcode_sequence": selected_opcodes == ["0x01", "0x10", "0x20", "0x01"],
        "unsupported_task_rejected": len(rejected) == 1,
        "rejected_task_emitted_zero_transport_events": (
            len(rejected) == 1 and rejected[0]["transport_events_added"] == 0
        ),
        "each_selected_task_has_complete_sequence": all(
            step["transport_events_added"] == 8
            for step in steps
            if step["profile_opcode"] is not None
        ),
        "no_global_reset_operation": all(
            event["operation"] != "global_reset" for event in transport.events
        ),
        "all_selected_tasks_resume_accepting_work": all(
            transport.events[step["transport_event_range"][1] - 1]["operation"]
            == "resume_accepting_work"
            for step in steps
            if step["profile_opcode"] is not None
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "1.0",
        "status": status,
        "mode": "software_dry_run_only",
        "sequence": steps,
        "checks": checks,
        "transport_events": transport.events,
        "boundary": (
            "This report proves fixed-profile host selection, rejection, and ordered "
            "dispatch planning only. It is not an RTL, MMIO, board, latency, or power result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output": str(args.output)}, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
