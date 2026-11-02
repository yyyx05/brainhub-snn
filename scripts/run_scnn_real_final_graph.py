#!/usr/bin/env python3
"""Prepare/run the raw primary SCNN on the production final-composition top.

Default: list dependencies. No SSH, synthesis, automatic retry or local large
compilation. Run prepare on the build host to make its paths native.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from generate_scnn_real_final_graph import (
    ROOT, TOP, SOURCE, FINAL, BODY, CHECKS, CONV_WORDS, CONV_BASES,
    CLASSIFIER_BASE, PARAM_DEPTH, testbench_source,
)
from supplement_scnn_same_dut_reference import write_fixture

BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_real_final_graph"
PASS = "SCNN_REAL_FINAL_GRAPH_PASS"


def dependencies():
    paths = set(ROOT.glob("rtl/**/*.sv"))
    paths.update((SOURCE, BODY, CHECKS, Path(__file__).resolve()))
    for name in ("generate_scnn_real_final_graph.py", "run_fc_final_counts.py",
                 "supplement_scnn_same_dut_reference.py", "supplement_scnn_graph_reference.py",
                 "analyze_scnn_b4t4_rtl_update_contract.py", "supplement_scnn_graph_host.mk"):
        paths.add(ROOT / "scripts" / name)
    paths.add(ROOT / "tests/test_scnn_real_final_graph.py")
    for bank in (*range(1, 9), 12):
        paths.add(ROOT / f"models/scnn_dvsgesture_vgg9_q0_15/bank_{bank:02d}.hex")
    for directory in (
        "results/paper_evidence/rtl/scnn_q8_8_value_input_20260926/full_frame_t1_reference",
        "results/paper_evidence/rtl/scnn_full_graph_same_array_20260927/t2_reference",
    ):
        paths.update((ROOT / directory).glob("*.hex"))
    paths.add(ROOT / "imports/scnn/datasets/DVSGesture/hardware/test_10_labels.hex")
    paths.add(ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem")
    # Include any other literal ROM assets used by the current source tree.
    for rtl in ROOT.glob("rtl/**/*.sv"):
        for literal in re.findall(r'"([^"\n]+\.(?:hex|mem))"', rtl.read_text(encoding="utf-8-sig")):
            if (ROOT / literal).is_file():
                paths.add(ROOT / literal)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing dependencies: " + ", ".join(missing))
    return sorted(paths)


def build_command(output, verilator="verilator", jobs=2):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    warnings = ["-Wall", "-Wno-fatal"] + ["-Wno-" + name for name in (
        "DECLFILENAME", "PROCASSINIT", "PINCONNECTEMPTY", "PINMISSING", "WIDTHTRUNC", "WIDTHEXPAND",
        "UNUSEDSIGNAL", "UNUSEDPARAM", "BLKSEQ", "CMPCONST", "SYNCASYNCNET", "TIMESCALEMOD", "WIDTHCONCAT")]
    return [str(verilator), "--binary", "-j", str(jobs), "--timing", "--assert", "--prefix", "Vscgraph",
            "--top-module", TOP, "--Mdir", str(output / "obj"), *warnings,
            "--output-groups", "4", "--output-split", "5000", "--output-split-cfuncs", "2000",
            "-MAKEFLAGS", "OPT_FAST=-O1 OPT_GLOBAL=-O1 -f " + str(ROOT / "scripts/supplement_scnn_graph_host.mk"),
            "-f", str(output / "sources.f")]


def prepare(output):
    paths = dependencies()
    source = testbench_source()
    output.mkdir(parents=True, exist_ok=False)
    tb = output / (TOP + ".sv")
    tb.write_text(source, encoding="ascii")
    fixtures = {mode: write_fixture(output / ("fixture_" + mode), mode == "on") for mode in ("off", "on")}
    rtl = sorted(ROOT.glob("rtl/**/*.sv"))
    (output / "sources.f").write_text("".join('"' + p.as_posix() + '"\n' for p in [*rtl, tb]), encoding="ascii")
    (output / "dependencies.txt").write_text("".join(p.relative_to(ROOT).as_posix() + "\n" for p in paths), encoding="ascii")
    record = {
        "status": "prepared_not_compiled_not_simulated", "top": "unified_h40_final_composition_top_v1",
        "physical_arrays": 1, "batch": 1, "timesteps": 2, "fresh_raw_t0": True, "descriptors": 24,
        "window_requests": [{"base": 0, "count": 2}, {"base": 0, "count": 1, "learning": False}],
        "param_depth": PARAM_DEPTH, "param_addr_width": 24, "conv_bases": CONV_BASES,
        "conv_words": CONV_WORDS, "classifier_base": CLASSIFIER_BASE, "classifier_words": 22528,
        "parameter_window": "external DDR address window, not an on-chip 16M store; final-source PPA must match",
        "stimulus": "raw sample0 T0/T1 plus original banks; TB external DDR only; state from DUT public spill",
        "oracle": "existing independent integer full-graph references plus cold B1T2 magnitude update/640",
        "protocol_differences_from_old_subtop": [
            "one request per window, one descriptor per Conv, last on Conv8; global class_count=11",
            "all Conv weights fetched through production parameter/cache refill, never direct cachefill",
            "Conv and classifier physical state/trace spill and restore through public held service",
            "classifier-only selected path has no Conv8 prescan; explicit zero-prescan assertion",
            "protected classifier publication stays in pair1, unlike old graph-online pair flip",
            "shared learner terminal window advances protocol version even when actual code changes are zero",
        ],
        "fixtures": fixtures,
        "dependency_files": len(paths), "dependency_bytes": sum(p.stat().st_size for p in paths),
        "memory_estimate_not_measurement": {
            "external_code_payload_bytes": (CLASSIFIER_BASE + 22528)*2,
            "server_recommended_available_gib": 16, "compiler_jobs": 2,
            "note": "whole final top generated C++ may require substantially more than payload; do not build on low-RAM laptop",
        },
        "not_claimed": ["simulation PASS before server execution", "full Test", "B4T12 natural code crossing",
                        "accuracy/adaptation benefit", "DDR timing", "board results", "new-source PPA"],
    }
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    return record


def run_logged(command, log):
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"command": command, "exit_code": result.returncode, "seconds": round(time.monotonic()-started, 3)}
    log.with_suffix(".status.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    if result.returncode:
        raise RuntimeError(f"preserved failure: {log}")
    return record


def parse_result(log, expected, learning):
    rows = [dict((key, int(value)) for key, value in re.findall(r"(\w+)=(-?\d+)", line))
            for line in log.splitlines() if line.startswith(PASS + " ")]
    if len(rows) != 1:
        raise ValueError("missing/duplicate final completion marker")
    row = rows[0]
    exact = dict(learning=int(learning), frames=3, descriptors=24, requests=2, features=6144,
                 currents=33, logits=33, traces=4096*int(learning),
                 rmw=expected["expected_rmw_transactions"], code_changes=expected["expected_code_changes"],
                 residual_changes=expected["expected_residual_changes"], publications=1892*(1+int(learning)),
                 commits=int(learning))
    if any(row.get(key) != value for key, value in exact.items()) or row.get("fresh_reads", 0) <= 0:
        raise ValueError(f"final measured counts differ from reference: {row}")
    if len(re.findall(r"^SCNN_REAL_FINAL_FRAME_PASS ", log, flags=re.M)) != 3:
        raise ValueError("missing per-frame numeric completion")
    banks = [tuple(map(int, m)) for m in re.findall(r"^SCNN_REAL_FINAL_DDR bank=(\d+) reads=(\d+)$", log, flags=re.M)]
    if sorted(b for b, _ in banks) != list(range(9)) or any(n <= 0 for _, n in banks):
        raise ValueError("missing production DDR bank reads")
    return row


def summarize(output):
    rows = {}
    for mode in ("off", "on"):
        log = output / ("run_" + mode + ".log")
        if json.loads(log.with_suffix(".status.json").read_text())["exit_code"] != 0:
            raise ValueError(f"nonzero simulator exit: {mode}")
        expected = json.loads((output / ("fixture_" + mode) / "expectations.json").read_text())
        rows[mode] = parse_result(log.read_text(encoding="utf-8", errors="replace"), expected, mode == "on")
    summary = dict(status="PASS", final_composition_top_simulated=True, rows=rows,
                   raw_graph_frames=3, full_test=False, b4t12=False, board_measured=False,
                   changed_code_visibility_demonstrated=rows["on"]["code_changes"] > 0,
                   note="code0 only proves unchanged-code/residual and fresh-forward consistency, not nonzero code-crossing visibility")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("files", "prepare", "build", "run", "summarize", "all"), default="files")
    parser.add_argument("--run-name", default="prepared_b1t2")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--learning", choices=("both", "off", "on"), default="both")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must be a simple directory name")
    output = BASE / args.run_name
    if args.stage == "files":
        print("\n".join(p.relative_to(ROOT).as_posix() for p in dependencies()))
        return
    if args.stage in ("prepare", "all"):
        print(json.dumps(prepare(output), indent=2))
    if args.stage in ("build", "all"):
        if sys.platform == "win32":
            parser.error("large final-top build is server-only; use prepare locally")
        run_logged(build_command(output, args.verilator, args.jobs), output / "build.log")
        log = (output / "build.log").read_text(encoding="utf-8", errors="replace")
        if any("%Warning-" + name in log for name in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN")):
            raise RuntimeError("build has structural warnings; inspect preserved log before simulation")
    if args.stage in ("run", "all"):
        modes = ("off", "on") if args.learning == "both" else (args.learning,)
        for mode in modes:
            run_logged([str(output / "obj/Vscgraph"),
                        "+SCNN_FINAL_FIXTURE=" + (output / ("fixture_" + mode)).relative_to(ROOT).as_posix(),
                        "+SCNN_LEARNING=" + str(int(mode == "on"))], output / ("run_" + mode + ".log"))
    if args.stage == "summarize" or (args.stage == "all" and args.learning == "both"):
        print(json.dumps(summarize(output), indent=2))


if __name__ == "__main__":
    main()
