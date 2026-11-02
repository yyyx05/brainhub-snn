#!/usr/bin/env python3
"""Isolated P4 simulation and post-synthesis report parsing; never launches EDA."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "verification/experiments/p4_factor"
PRODUCTION_REL = "rtl/tiles/unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv"
PRODUCTION_MODULE = "unified_dual_role_dsp_lane_p4_cascade_candidate_v1"
PART = "xc7vx690tffg1761-3"
RESOURCE_KEYS = ("lut", "ff", "ramb36", "ramb18", "bram36eq", "dsp")


def check_reference(root: Path = ROOT) -> Path:
    sources = (root / "rtl/final_sources.f").read_text(encoding="utf-8").splitlines()
    if PRODUCTION_REL not in [line.strip() for line in sources]:
        raise ValueError("P4 primitive is not in the current production source list")
    source = root / PRODUCTION_REL
    if not re.search(r"\bmodule\s+" + PRODUCTION_MODULE + r"\s*\(",
                     source.read_text(encoding="utf-8")):
        raise ValueError("Production P4 module declaration is missing")
    return source


def rtl_snapshot(root: Path = ROOT) -> dict[str, bytes]:
    # Ephemeral before/after comparison only. No stored baseline or content hash.
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in (root / "rtl").rglob("*") if p.is_file()}


def run_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use an alphanumeric run name, not a path")
    return EXPERIMENT / "runs" / name


def simulate(out: Path, simulator: str) -> int:
    source = check_reference()
    before = rtl_snapshot()
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "not_run", "scope": "behavioral_RTL_unit_only",
              "production_source": PRODUCTION_REL, "simulator": simulator,
              "production_unchanged": None}
    code = 2
    try:
        compiler = shutil.which("iverilog" if simulator == "iverilog" else "verilator")
        runtime = shutil.which("vvp") if simulator == "iverilog" else None
        if compiler is None or (simulator == "iverilog" and runtime is None):
            result.update(status="missing_tool", error=f"{simulator} toolchain unavailable")
        else:
            inputs = [str(source), str(EXPERIMENT / "p4_sum_products.sv"),
                      str(EXPERIMENT / "p4_factor_top.sv"), str(EXPERIMENT / "tb_p4_factor.sv")]
            if simulator == "iverilog":
                build = [compiler, "-g2012", "-s", "tb_p4_factor", "-o", str(out / "sim.vvp"), *inputs]
                execute = [runtime, str(out / "sim.vvp")]
            else:
                build = [compiler, "--binary", "--timing", "--assert", "-Wno-fatal",
                         "--top-module", "tb_p4_factor", "--Mdir", str(out / "obj"),
                         "-o", "p4_sim", *inputs]
                execute = [str(out / "obj/p4_sim")]
            for phase, command in (("build", build), ("simulation", execute)):
                with (out / f"{phase}.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=out, stdout=log,
                                               stderr=subprocess.STDOUT, check=False)
                result[f"{phase}_exit"] = completed.returncode
                if completed.returncode:
                    result["status"] = f"{phase}_failed"
                    code = 1
                    break
            else:
                log = (out / "simulation.log").read_text(encoding="utf-8", errors="replace")
                marker = re.search(r"P4_FACTOR_RTL_PASS checks=(\d+) accepted=(\d+) "
                                   r"bubbles=(\d+) restarts=(\d+) wraps=(\d+)", log)
                if marker and all(int(x) > 0 for x in marker.groups()) and not re.search(
                        r"(?im)^\s*(?:ERROR:|FATAL:|%Error)", log):
                    result["status"] = "behavioral_rtl_pass"
                    result["coverage"] = dict(zip(
                        ("checks", "accepted", "bubbles", "restarts", "wraps"),
                        map(int, marker.groups())))
                    code = 0
                else:
                    result.update(status="simulation_failed", error="Missing scoreboard completion or RTL error")
                    code = 1
    except OSError as exc:
        result.update(status="tool_error", error=str(exc))
        code = 2
    finally:
        result["production_unchanged"] = before == rtl_snapshot()
        if not result["production_unchanged"]:
            result.update(status="source_changed", error="Production RTL changed during simulation")
            code = 1
        (out / "simulation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return code


def key_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def number(value: str) -> float | None:
    try:
        result = float(value.replace(",", ""))
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def utilization(text: str) -> dict[str, float | None]:
    aliases = {"Slice LUTs": "lut", "Slice LUTs*": "lut", "Slice Registers": "ff", "RAMB36/FIFO*": "ramb36",
               "RAMB36/FIFO": "ramb36", "RAMB36": "ramb36", "RAMB18": "ramb18",
               "DSPs": "dsp", "DSP48E1": "dsp"}
    result = dict.fromkeys(RESOURCE_KEYS)
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 2 and cells[1] in aliases:
            value = number(cells[2])
            result[aliases[cells[1]]] = value if value is not None and value >= 0 else None
    if result["ramb36"] is not None and result["ramb18"] is not None:
        result["bram36eq"] = result["ramb36"] + result["ramb18"] / 2
    return result


def timing(text: str) -> dict[str, float | None]:
    keys = ("wns_ns", "tns_ns", "setup_failing", "setup_endpoints",
            "whs_ns", "ths_ns", "hold_failing", "hold_endpoints")
    result = dict.fromkeys(keys)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line and "WHS(ns)" in line and "THS(ns)" in line:
            for row in lines[index + 1:index + 6]:
                fields = row.split()
                if len(fields) >= 8 and any(char.isdigit() for char in row):
                    return dict(zip(keys, (number(field) for field in fields[:8])))
    return result


def summarize(out: Path) -> dict:
    arms = {}
    for arm, control in (("factor", "0"), ("sop", "1")):
        folder = out / arm
        info = key_values(folder / "run_info.txt")
        expected = {"arm": arm, "sop_control": control, "stage": "post_synthesis",
                    "part": PART, "top": "p4_factor_top", "mode": "out_of_context",
                    "flatten_hierarchy": "rebuilt", "directive": "Default",
                    "production_source": PRODUCTION_REL}
        missing = [f"metadata:{key}" for key, value in expected.items() if info.get(key) != value]
        if number(info.get("period_ns", "")) != 15:
            missing.append("metadata:period_ns")
        if not info.get("tool_version"):
            missing.append("metadata:tool_version")
        if info.get("blackboxes") != "0":
            missing.append("blackboxes_zero_unconfirmed")
        if key_values(folder / "status.txt").get("status") != "complete":
            missing.append("flow_incomplete")
        reports = {}
        for filename, parser in (("utilization.rpt", utilization), ("timing_summary.rpt", timing)):
            path = folder / filename
            parsed = parser(path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "")
            reports.update(parsed)
            missing.extend(key for key, value in parsed.items() if value is None)
        # The registered timing shell must have actual setup and hold endpoints.
        for key in ("setup_endpoints", "hold_endpoints"):
            if reports[key] is not None and reports[key] <= 0:
                missing.append(key + ":no_paths")
        if missing:
            status = "incomplete"
        else:
            met = (reports["wns_ns"] >= 0 and reports["whs_ns"] >= 0
                   and all(reports[key] == 0 for key in
                           ("tns_ns", "ths_ns", "setup_failing", "hold_failing")))
            status = "post_synth_estimate_met" if met else "post_synth_estimate_violated"
        arms[arm] = {"status": status, "metadata": info, "metrics": reports, "missing": missing}
    matched = (all(arms[arm]["status"] != "incomplete" for arm in arms)
               and arms["factor"]["metadata"].get("tool_version") == arms["sop"]["metadata"].get("tool_version")
               and key_values(out / "flow.txt").get("status") == "complete")
    delta = {key: arms["sop"]["metrics"][key] - arms["factor"]["metrics"][key]
             if matched else None for key in RESOURCE_KEYS}
    return {"scope": "one_P4_lane_plus_identical_IO_registers_post_synthesis_only",
            "comparison_status": "matched_reports" if matched else "incomplete",
            "arms": arms, "sop_minus_factor": delta,
            "physical_equivalence": "not_checked",
            "mapping_note": "SOP may be refactored by synthesis; inspect post_synth.v, hierarchy.rpt and DCP. "
                            "Equal resource counts alone do not prove identical mapping.",
            "whole_core_cost": None, "routed_sta": "not_run", "power": "not_measured"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-check")
    sim = commands.add_parser("simulate")
    sim.add_argument("--run", required=True)
    sim.add_argument("--simulator", choices=("iverilog", "verilator"), default="iverilog")
    summary = commands.add_parser("summarize")
    summary.add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        if args.command == "source-check":
            check_reference()
            print("SOURCE_REFERENCE_OK: current production primitive, read-only; not RTL simulation PASS")
            return 0
        out = run_path(args.run)
        if args.command == "simulate":
            return simulate(out, args.simulator)
        if not out.is_dir():
            raise FileNotFoundError(f"No experiment results: {out}")
        result = summarize(out)
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        if result["comparison_status"] == "incomplete":
            return 2
        return 0 if all(arm["status"] == "post_synth_estimate_met" for arm in result["arms"].values()) else 1
    except (OSError, ValueError) as exc:
        print(f"NOT_PASS: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
