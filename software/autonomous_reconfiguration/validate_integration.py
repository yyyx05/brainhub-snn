"""Cross-check the host decision catalog against current project assets and RTL ports."""

from __future__ import annotations

import json
import re
from pathlib import Path

from snn_reconfig.decision import DecisionEngine, load_json


COMPONENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_ROOT.parents[1]
CATALOG_ROOT = COMPONENT_ROOT / "catalog"
EXPECTED_EXAMPLES = {
    "nmnist_query.json": ("hardware_profile_selected", "0x01"),
    "shd_query.json": ("hardware_profile_selected", "0x10"),
    "dvsgesture_query.json": ("hardware_profile_selected", "0x20"),
    "smnist_unsupported_query.json": ("requires_accelerator_extension", None),
    "cifar10dvs_unsupported_query.json": ("requires_accelerator_extension", None),
}
RTL_PROFILE_CONSTANTS = {
    "fc_nmnist_h160": ("PROFILE_FC_H160", "01"),
    "srnn_shd_h450_m2": ("PROFILE_SRNN_M2", "10"),
    "scnn_dvsgesture_vgg9": ("PROFILE_SCNN_VGG9", "20"),
}
EXPECTED_LEARNING_SCOPES = {
    "fc_nmnist_h160": "W2 only",
    "srnn_shd_h450_m2": "M2 Wout only",
    "scnn_dvsgesture_vgg9": "classifier only",
}


def main() -> int:
    profiles = load_json(CATALOG_ROOT / "hardware_profiles.json")
    interface = load_json(CATALOG_ROOT / "native_control_interface.json")
    dispatch_catalog = load_json(CATALOG_ROOT / "profile_dispatch_contracts.json")
    engine = DecisionEngine(profiles, load_json(CATALOG_ROOT / "fingerprint_prototypes.json"))

    errors: list[str] = []
    asset_count = 0
    opcodes: set[str] = set()
    profile_ids = {profile["profile_id"] for profile in profiles["profiles"]}
    dispatch_by_profile = {
        contract["profile_id"]: contract for contract in dispatch_catalog["contracts"]
    }
    profile_rtl = (PROJECT_ROOT / "rtl" / "top" / "unified_tp_online_system_top.sv").read_text(
        encoding="utf-8", errors="replace"
    )
    final_top = (PROJECT_ROOT / "rtl" / "top" / "unified_h40_final_composition_top_v1.sv").read_text(
        encoding="utf-8", errors="replace"
    )
    port_header = final_top.split(");", 1)[0]

    for profile in profiles["profiles"]:
        profile_id = profile["profile_id"]
        opcode = profile["profile_code"]
        if opcode in opcodes:
            errors.append(f"duplicate profile opcode: {opcode}")
        opcodes.add(opcode)
        constant, expected_hex = RTL_PROFILE_CONSTANTS[profile_id]
        if not re.search(rf"\b{constant}\s*=\s*7'h{expected_hex}\b", profile_rtl):
            errors.append(f"RTL profile constant mismatch for {profile_id}")
        for asset in profile["deployment_assets"]:
            asset_count += 1
            path = PROJECT_ROOT / asset["path"]
            if not path.is_file():
                errors.append(f"missing deployment asset: {asset['path']}")
        dispatch = dispatch_by_profile.get(profile_id)
        if dispatch is None:
            errors.append(f"missing dispatch contract for {profile_id}")
        elif dispatch["contract_id"] != profile.get("dispatch_contract_id"):
            errors.append(f"dispatch contract ID mismatch for {profile_id}")
        elif dispatch["profile_code"] != opcode:
            errors.append(f"dispatch contract opcode mismatch for {profile_id}")
        actual_learning_scope = profile.get("learning_scope", {}).get(
            "trainable_parameters"
        )
        expected_learning_scope = EXPECTED_LEARNING_SCOPES.get(profile_id)
        if actual_learning_scope != expected_learning_scope:
            errors.append(
                f"learning scope mismatch for {profile_id}: "
                f"expected {expected_learning_scope!r}, got {actual_learning_scope!r}"
            )

    unknown_dispatch_profiles = set(dispatch_by_profile) - profile_ids
    if unknown_dispatch_profiles:
        errors.append(
            "dispatch catalog references unknown profiles: "
            + ", ".join(sorted(unknown_dispatch_profiles))
        )

    port_count = 0
    for operation in interface["operations"]:
        for signal in operation["drive"] + operation["observe"]:
            port_count += 1
            if not re.search(rf"\b(?:input|output)\b[^\n]*\b{re.escape(signal)}\b", port_header):
                errors.append(f"native interface signal is not a final-top port: {signal}")

    dispatch_ports: set[str] = set()
    for contract in dispatch_catalog["contracts"]:
        for group in contract["interface_groups"]:
            for signal in group["drive"] + group["observe"]:
                dispatch_ports.add(signal)
                if not re.search(
                    rf"\b(?:input|output)\b[^\n]*\b{re.escape(signal)}\b", port_header
                ):
                    errors.append(
                        f"dispatch anchor is not a final-top port: "
                        f"{contract['contract_id']}:{signal}"
                    )

    example_results: dict[str, dict[str, str | None]] = {}
    for name, (expected_status, expected_opcode) in EXPECTED_EXAMPLES.items():
        result = engine.decide(load_json(COMPONENT_ROOT / "examples" / name), "off")
        selected_opcode = (
            result["selected_profile"]["profile_code"] if result["selected_profile"] else None
        )
        example_results[name] = {
            "status": result["status"],
            "profile_code": selected_opcode,
        }
        if result["status"] != expected_status or selected_opcode != expected_opcode:
            errors.append(f"unexpected decision for {name}: {example_results[name]}")
        if expected_opcode is None and result["hardware_plan"] is not None:
            errors.append(f"unsupported example emitted a hardware plan: {name}")

    report = {
        "status": "pass" if not errors else "fail",
        "profiles_checked": len(profiles["profiles"]),
        "unique_opcodes_checked": len(opcodes),
        "deployment_assets_checked": asset_count,
        "native_ports_checked": port_count,
        "dispatch_contracts_checked": len(dispatch_catalog["contracts"]),
        "unique_dispatch_anchor_ports_checked": len(dispatch_ports),
        "fixed_learning_scopes_checked": len(EXPECTED_LEARNING_SCOPES),
        "example_results": example_results,
        "errors": errors,
        "boundary": (
            "Static host-to-native-interface contract check only; no MMIO driver, board, "
            "RTL simulation, or physical-result claim."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
