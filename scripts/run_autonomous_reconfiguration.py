"""Project entry point for fixed-profile autonomous SNN reconfiguration.

The command selects only one of the three qualified hardware profiles and emits
a dry-run host sequence. It does not drive RTL, MMIO, a board, or an EDA tool.
"""

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


def _parse_indices(specification: str) -> list[int]:
    result: list[int] = []
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, stop_text = item.split(":", 1)
            result.extend(range(int(start_text), int(stop_text)))
        else:
            result.append(int(item))
    if not result:
        raise ValueError("at least one SHD sample index is required")
    return result


def _load_request(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.fingerprint_json:
        return load_json(args.fingerprint_json), {
            "adapter": "fingerprint-json",
            "path": str(args.fingerprint_json),
        }
    if args.shd_h5:
        indices = _parse_indices(args.shd_indices)
        return {
            "dataset_id": "SHD-project-input",
            "fingerprint": fingerprint_shd_h5(args.shd_h5, indices),
        }, {
            "adapter": "shd-h5",
            "path": str(args.shd_h5),
            "sample_indices": indices,
        }
    return {
        "dataset_id": "DVS-Gesture-project-input",
        "fingerprint": fingerprint_dvsgesture_q8_8_hex(
            args.dvsgesture_q8_8, args.dvs_sample_count
        ),
    }, {
        "adapter": "dvsgesture-q8-8",
        "path": str(args.dvsgesture_q8_8),
        "sample_count": args.dvs_sample_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fingerprint-json", type=Path)
    source.add_argument("--shd-h5", type=Path)
    source.add_argument("--dvsgesture-q8-8", type=Path)
    parser.add_argument("--shd-indices", default="0:32")
    parser.add_argument("--dvs-sample-count", type=int, default=10)
    parser.add_argument("--learning", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profiles_path = COMPONENT_ROOT / "catalog" / "hardware_profiles.json"
    prototypes_path = COMPONENT_ROOT / "catalog" / "fingerprint_prototypes.json"
    dispatch_path = COMPONENT_ROOT / "catalog" / "profile_dispatch_contracts.json"
    request, source_metadata = _load_request(args)
    decision = DecisionEngine.from_files(profiles_path, prototypes_path).decide(
        request, args.learning
    )
    dispatch_catalog = load_json(dispatch_path)
    contracts = {
        item["contract_id"]: item for item in dispatch_catalog["contracts"]
    }

    execution = None
    dry_run_events: list[dict[str, Any]] = []
    if decision["status"] == "hardware_profile_selected":
        transport = TraceTransport(contracts)
        execution = execute_selected_profile(decision, transport)
        dry_run_events = transport.events

    result = {
        "schema_version": "1.0",
        "mode": "dry_run_contract_trace",
        "input_source": source_metadata,
        "decision": decision,
        "execution": execution,
        "dry_run_events": dry_run_events,
        "boundary": (
            "Project-level fixed-profile integration only. No MMIO, RTL simulation, "
            "board action, or arbitrary depth/scale/timestep/precision/activation-unit "
            "selection is performed."
        ),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
