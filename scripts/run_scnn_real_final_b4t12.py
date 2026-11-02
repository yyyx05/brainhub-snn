#!/usr/bin/env python3
"""Prepare and execute one continuous final-top B4T12 plus raw query4 test."""
import argparse
import json
from pathlib import Path
import re
import sys

import generate_scnn_real_final_b4t12 as generator
import run_scnn_real_final_graph as base

ROOT = generator.ROOT
FIXTURE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_same_dut/natural_visibility_prepare/seed38472_b4t12_q22_t4_finaltop_v2"
BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_real_final_graph"
PASS = "SCNN_TEMPORAL_FINAL_PASS"


def dependencies():
    paths = set(base.dependencies())
    paths.update([Path(__file__).resolve(), ROOT / "scripts/generate_scnn_real_final_b4t12.py",
                  generator.BODY, generator.CHECKS, ROOT / "tests/test_scnn_real_final_b4t12.py",
                  ROOT / "tests/tb_scnn_long_watchdog.sv"])
    paths.update(p for p in FIXTURE.iterdir() if p.is_file())
    return sorted(paths)


def prepare(output):
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["seed"], manifest["support_ids"], manifest["query_id"], manifest["total_graph_frames"]) != (38472, [0, 1, 2, 3], 22, 52):
        raise ValueError("unexpected fixed experiment")
    source = generator.testbench_source()
    output.mkdir(parents=True, exist_ok=False)
    tb = output / (generator.TOP + ".sv")
    tb.write_text(source, encoding="ascii")
    (output / "sources.f").write_text("".join('"' + p.as_posix() + '"\n' for p in [*sorted(ROOT.glob("rtl/**/*.sv")), tb]), encoding="ascii")
    (output / "dependencies.txt").write_text("".join(p.relative_to(ROOT).as_posix()+"\n" for p in dependencies()), encoding="ascii")
    control = output / "control"
    control.mkdir()
    values = []
    for window in manifest["commits"]:
        values.extend([window["code_changes"], window["residual_changes"], window["nonzero_delta_rmw"], window["max_abs_delta"]])
    (control / "window_counts.hex").write_text("".join(f"{n:04x}\n" for n in values), encoding="ascii")
    record = dict(status="prepared_not_simulated", top="unified_h40_final_composition_top_v1",
                  stimulus="original parameter banks, raw first four fixed shifted support samples, raw query22",
                  state="same DUT, no reset/reload across commits, observed state spilled/restored per sample",
                  schedule=dict(batch=4, count=4, time_bases=[0, 4, 8], query_frames=4),
                  scoreboards="all pooled features, currents, centered logits, traces, gradients, updates, publication; bounded Conv membrane probes and full spill consistency",
                  not_claimed=["full Test accuracy", "full per-layer numerical oracle on every scalar", "RTL PASS before completion", "DDR/board timing", "new PPA"])
    (output / "preparation.json").write_text(json.dumps(record, indent=2)+"\n", encoding="ascii")
    return record


def parse_result(log):
    matches = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(-?\d+)", line))
               for line in log.splitlines() if line.startswith(PASS+" ")]
    exact = dict(frames=52, descriptors=416, features=106496, currents=572, logits=572,
                 traces=98304, probes=1248, rmw=60273, commits=3, publications=7568,
                 code_changes=662, query_currents=4, query_logits=4)
    if len(matches) != 1 or any(matches[0].get(k) != v for k, v in exact.items()) or matches[0].get("fresh_reads", 0) <= 0:
        raise ValueError("missing/incomplete temporal final completion")
    frames = re.findall(r"^SCNN_TEMPORAL_FRAME_STATUS frame=(\d+) ", log, re.M)
    if list(map(int, frames)) != list(range(52)):
        raise ValueError("missing or reordered raw frames")
    windows = re.findall(r"^SCNN_TEMPORAL_COMMIT window=(\d+) code_changes=(\d+) residual_changes=(\d+) rmw=(\d+) ", log, re.M)
    if [tuple(map(int, w)) for w in windows] != [(1, 0, 20041, 20041), (2, 0, 20011, 20011), (3, 662, 20221, 20221)]:
        raise ValueError("wrong update windows")
    if len(re.findall(r"^SCNN_TEMPORAL_CROSSING window=3 index=18433 ", log, re.M)) != 1:
        raise ValueError("missing natural threshold crossing")
    if re.search(r"%Error|%Fatal|Assertion failed", log):
        raise ValueError("error in temporal log")
    return matches[0]


def summarize(output):
    status = json.loads((output / "run.status.json").read_text())
    if status["exit_code"] != 0:
        raise ValueError("nonzero simulation exit")
    row = parse_result((output / "run.log").read_text(encoding="utf-8", errors="replace"))
    record = dict(status="PASS", measured=row, seconds=status["seconds"], full_test=False,
                  scope="fixed B4T12 natural code crossing and raw query4 visibility in one final-composition DUT",
                  board=False, post_route=False)
    (output / "summary.json").write_text(json.dumps(record, indent=2)+"\n", encoding="ascii")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("files", "prepare", "lint", "build", "run", "summarize", "all"), default="files")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verilator", default="verilator")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("simple run-name required")
    output = BASE / args.run_name
    if args.stage == "files":
        print("\n".join(p.relative_to(ROOT).as_posix() for p in dependencies()))
        return
    if args.stage in ("prepare", "all"):
        print(json.dumps(prepare(output)), flush=True)
    if args.stage in ("lint", "build", "all"):
        if sys.platform == "win32" and args.stage != "lint":
            parser.error("large compilation is server-only")
        command = base.build_command(output, args.verilator, args.jobs)
        command[command.index(base.TOP)] = generator.TOP
        stage = "lint" if args.stage == "lint" else "build"
        if stage == "lint":
            command[command.index("--binary")] = "--lint-only"
        base.run_logged(command, output / (stage+".log"))
        log = (output / (stage+".log")).read_text(encoding="utf-8", errors="replace")
        if any("%Warning-"+n in log for n in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN")):
            raise RuntimeError("structural warning; inspect before running")
    if args.stage in ("run", "all"):
        base.run_logged([str(output / "obj/Vscgraph"),
                         "+SCNN_TEMPORAL_FIXTURE="+FIXTURE.relative_to(ROOT).as_posix(),
                         "+SCNN_TEMPORAL_CONTROL="+(output / "control").relative_to(ROOT).as_posix()], output / "run.log")
    if args.stage in ("summarize", "all"):
        print(json.dumps(summarize(output), indent=2), flush=True)


if __name__ == "__main__":
    main()
