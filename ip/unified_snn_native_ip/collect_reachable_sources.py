"""Collect the SystemVerilog files reachable from the final IP top module."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


TOP_MODULE = "unified_h40_final_composition_top_v1"
VERILATOR_OPTIONS = (
    "--timing --assert -DUNIFIED_USE_XPM_MEMORY -Wno-MODMISSING "
    "-Wno-fatal -Wno-DECLFILENAME -Wno-PROCASSINIT "
    "-Wno-PINCONNECTEMPTY -Wno-PINMISSING -Wno-WIDTHTRUNC "
    "-Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM "
    "-Wno-BLKSEQ -Wno-CMPCONST -Wno-SYNCASYNCNET "
    "-Wno-TIMESCALEMOD -Wno-WIDTHCONCAT"
)


def _msys_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/{drive}/{remainder}"


def _bash_executable() -> str:
    configured = os.environ.get("SNN_BASH_EXE")
    if configured:
        return configured
    if os.name == "nt":
        candidate = Path(r"C:\msys64\usr\bin\bash.exe")
        if candidate.is_file():
            return str(candidate)
    return "bash"


def _run_bash(project_root: Path, command: str, log_path: Path) -> None:
    full_command = (
        "set -euo pipefail; "
        "export PATH=/mingw64/bin:/usr/bin:$PATH; "
        f"cd {shlex.quote(_msys_path(project_root))}; "
        f"{command} > {shlex.quote(_msys_path(log_path))} 2>&1"
    )
    result = subprocess.run(
        [_bash_executable(), "-lc", full_command],
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"Verilator command failed with exit code {result.returncode}:\n{tail}"
        )


def _parse_reachable_files(meta_path: Path, tree_path: Path) -> tuple[list[str], int]:
    file_pattern = re.compile(
        r'^\s*"([^"]+)": \{"filename":"([^"]+)".*"realpath":"([^"]+)"'
    )
    module_pattern = re.compile(r'"type":"MODULE".*?"loc":"([^,]+),')

    file_map: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = file_pattern.search(line)
        if match:
            file_map[match.group(1)] = match.group(2).replace("\\", "/")

    module_file_ids: set[str] = set()
    module_count = 0
    for line in tree_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = module_pattern.search(line)
        if match:
            module_file_ids.add(match.group(1))
            module_count += 1

    reachable = sorted(
        {
            file_map[file_id]
            for file_id in module_file_ids
            if file_id in file_map and file_map[file_id].startswith("rtl/")
        }
    )
    if not reachable:
        raise RuntimeError("Verilator hierarchy did not expose any reachable RTL files")
    return reachable, module_count


def collect(project_root: Path, output_list: Path, report_path: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    rtl_root = project_root / "rtl"
    all_sources = sorted(
        path.relative_to(project_root).as_posix() for path in rtl_root.rglob("*.sv")
    )
    if not all_sources:
        raise RuntimeError(f"no SystemVerilog sources found under {rtl_root}")

    work_root = project_root / "work" / "ip_source_closure_collect"
    work_root.mkdir(parents=True, exist_ok=True)
    all_list = work_root / "all_sources.f"
    tree_path = work_root / "hierarchy.tree.json"
    meta_path = work_root / "hierarchy.meta.json"
    hierarchy_log = work_root / "hierarchy.log"
    closure_list = work_root / "reachable_sources.f"
    closure_lint_log = work_root / "closure_lint.log"
    all_list.write_text("\n".join(all_sources) + "\n", encoding="utf-8")

    _run_bash(
        project_root,
        " ".join(
            (
                "verilator --json-only",
                VERILATOR_OPTIONS,
                f"--top-module {TOP_MODULE}",
                f"--json-only-output {shlex.quote(_msys_path(tree_path))}",
                f"--json-only-meta-output {shlex.quote(_msys_path(meta_path))}",
                f"-f {shlex.quote(_msys_path(all_list))}",
            )
        ),
        hierarchy_log,
    )
    reachable, elaborated_module_count = _parse_reachable_files(meta_path, tree_path)
    top_source = f"rtl/top/{TOP_MODULE}.sv"
    if top_source not in reachable:
        raise RuntimeError(f"reachable source set is missing the top source: {top_source}")
    for relative in reachable:
        if not (project_root / relative).is_file():
            raise RuntimeError(f"reachable source does not exist: {relative}")

    closure_list.write_text("\n".join(reachable) + "\n", encoding="utf-8")
    _run_bash(
        project_root,
        " ".join(
            (
                "verilator --lint-only",
                VERILATOR_OPTIONS,
                f"--top-module {TOP_MODULE}",
                f"-f {shlex.quote(_msys_path(closure_list))}",
            )
        ),
        closure_lint_log,
    )

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(reachable) + "\n", encoding="utf-8")
    report = {
        "status": "pass",
        "top_module": TOP_MODULE,
        "selection_method": "verilator_elaborated_top_reachability",
        "xpm_define_enabled": True,
        "input_source_count": len(all_sources),
        "reachable_source_count": len(reachable),
        "excluded_unreachable_source_count": len(all_sources) - len(reachable),
        "elaborated_module_variant_count": elaborated_module_count,
        "closure_lint": "pass",
        "source_list": reachable,
        "boundary": (
            "Source reachability and Verilator lint only; Vivado synthesis and IP "
            "integrity remain separate release checks."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.project_root, args.output_list, args.report),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
