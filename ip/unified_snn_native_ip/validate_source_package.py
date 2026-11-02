"""Validate the generated native-interface source IP package without Vivado."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def validate(package_root: Path) -> dict[str, object]:
    package_root = package_root.resolve()
    manifest_path = package_root / "ip_manifest.json"
    filelist_path = package_root / "src" / "filelist.f"
    xdc_path = package_root / "constraints" / "unified_h40_final_composition_v1_ooc.xdc"
    package_tcl = package_root / "package_ip.tcl"
    closure_report_path = package_root / "source_closure_report.json"

    required = (manifest_path, filelist_path, xdc_path, package_tcl, closure_report_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"package is missing required files: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    closure_report = json.loads(closure_report_path.read_text(encoding="utf-8-sig"))
    entries = [
        line.strip()
        for line in filelist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != len(set(entries)):
        raise RuntimeError("filelist contains duplicate source paths")
    missing_sources = [entry for entry in entries if not (package_root / entry).is_file()]
    if missing_sources:
        raise RuntimeError(f"filelist references missing sources: {missing_sources[:5]}")
    if manifest["source_count"] != len(entries):
        raise RuntimeError("manifest source_count does not match filelist")
    if closure_report.get("status") != "pass" or closure_report.get("closure_lint") != "pass":
        raise RuntimeError("source closure report did not pass")
    if closure_report.get("reachable_source_count") != len(entries):
        raise RuntimeError("source closure count does not match filelist")
    packaged_sources = sorted(
        path.relative_to(package_root).as_posix()
        for path in (package_root / "src" / "rtl").rglob("*.sv")
    )
    if packaged_sources != sorted(entries):
        raise RuntimeError("package contains unlisted or missing SystemVerilog sources")
    memory_relative = "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
    if manifest.get("auxiliary_source_files") != [memory_relative]:
        raise RuntimeError("manifest must include the original surrogate ROM dependency")
    words = (package_root / memory_relative).read_text(encoding="ascii").split()
    if len(words) != 1302 or any(not re.fullmatch(r"[0-9a-fA-F]{3}", word)
                               or int(word, 16) > 0x1FF for word in words):
        raise RuntimeError("surrogate ROM must contain 1302 valid 9-bit words")

    top_rel = manifest["top_source"]
    top_path = package_root / top_rel
    if not top_path.is_file():
        raise RuntimeError(f"top source is missing: {top_rel}")
    top_text = top_path.read_text(encoding="utf-8", errors="replace")
    module_pattern = re.compile(
        rf"\bmodule\s+{re.escape(manifest['top_module'])}\b", re.MULTILINE
    )
    if len(module_pattern.findall(top_text)) != 1:
        raise RuntimeError("top module declaration is missing or ambiguous")
    for port in ("i_clk", "i_rst_n"):
        if not re.search(rf"\b{port}\b", top_text):
            raise RuntimeError(f"top source is missing required port {port}")

    if "packaging_adapter_source" in manifest:
        from generate_native_wrapper import render_wrapper
        adapter = package_root / manifest["packaging_adapter_source"]
        if adapter.read_text(encoding="ascii") != render_wrapper(top_path):
            raise RuntimeError("packaging adapter differs from direct core port/parameter forwarding")

    return {
        "status": "pass",
        "package_root": str(package_root),
        "top_module": manifest["top_module"],
        "source_count": len(entries),
        "input_source_count": manifest["input_source_count"],
        "excluded_unreachable_source_count": manifest[
            "excluded_unreachable_source_count"
        ],
        "source_selection": manifest["source_selection"],
        "part": manifest["part"],
        "clock_period_ns": manifest["clock_period_ns"],
        "interface_kind": manifest["interface_kind"],
        "auxiliary_source_files": [memory_relative],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.package_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
