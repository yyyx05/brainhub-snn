#!/usr/bin/env python3
"""Bounded raw700 -> M2 execution inside the unmodified production final top."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_fc_final_counts import ROOT, build_settings, generate_tb

BASE = ROOT / "results/paper_evidence/supplement_20261004"
RAW = BASE / "algorithm/srnn_backbone_numeric/official01_b2t8_r2_20261007"


def prepare_reference(output):
    import numpy as np
    import build_srnn_formal_m2_fixture as m2
    from supplement_srnn_backbone_numeric import read_signed, write_hex

    output.mkdir(parents=True, exist_ok=False)
    with np.load(RAW / "backbone_reference.npz") as source:
        spikes = source["spike"][:, :1, :450].copy()
        labels = source["labels"][:1].copy()
    wh1 = read_signed(m2.WH1, 16, (256, 450))
    wout = read_signed(m2.WOUT, 16, (20, 256))
    write_hex(output / "labels.mem", labels, 8)
    counts = {}
    for learning in (False, True):
        prefix = "on" if learning else "off"
        arrays, traffic = m2.reference(spikes * 16384, labels, wh1, wout, learning=learning)
        for key in ("pre", "membrane", "spike", "trace", "feature", "logit",
                    "wout_before", "wout_after", "residual", "gradient"):
            bits = (64 if key in ("logit", "gradient") else
                    16 if key in ("wout_before", "wout_after", "residual") else 32)
            write_hex(output / f"{prefix}_{key}.mem", arrays[key], bits)
        counts[prefix] = traffic
    record = dict(batch=1, steps=8, sample_index=0, labels=labels.tolist(),
                  raw_reference=str(RAW.relative_to(ROOT)), counters=counts,
                  boundary="Python expectations only; the DUT must generate every backbone spike. "
                           "B1 is bounded by the existing 480-record phase capture; not B64 accuracy. "
                           "The production reducer retains its fixed /64 scaling.")
    (output / "reference.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-reference", action="store_true")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--verilator")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--learning", choices=("off", "on"), default="off")
    parser.add_argument("--reuse-executable", type=Path)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    if not fixture.is_relative_to(ROOT):
        parser.error("fixture must be inside the project")
    if args.prepare_reference:
        print(json.dumps(prepare_reference(fixture), indent=2))
        return
    if not args.run_name or not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("a simple new --run-name is required")
    if not args.verilator and not args.reuse_executable:
        parser.error("--verilator required for a new build")
    metadata = json.loads((fixture / "reference.json").read_text(encoding="utf-8"))
    if (metadata["batch"], metadata["steps"], metadata["sample_index"]) != (1, 8, 0):
        parser.error("expected the predetermined sample0 B1T8 reference")
    output = BASE / "rtl/srnn_same_dut" / args.run_name
    work = ROOT / "work/supplement_20261004/srnn_same_dut" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    top = "tb_srnn_same_dut"
    tb = work / (top + ".sv")
    generate_tb(tb, top=top, body=ROOT / "tb/include/srnn_same_dut_body.svh")
    sources = work / "sources.f"
    sources.write_text("".join('"' + path.as_posix() + '"\n'
                              for path in sorted(ROOT.glob("rtl/**/*.sv")) + [tb]), encoding="ascii")
    build, exe, env = build_settings(work / "obj", sources, verilator=args.verilator,
                                     jobs=args.jobs, top=top)
    if args.reuse_executable:
        exe = args.reuse_executable.resolve()
        if not exe.is_relative_to(ROOT):
            parser.error("reused executable must be inside this project")
    run = [str(exe), "+RAW=" + RAW.relative_to(ROOT).as_posix(),
           "+FIXTURE=" + fixture.relative_to(ROOT).as_posix(),
           "+LEARNING=" + str(int(args.learning == "on"))]
    record = dict(build=build, run=run, cwd=str(ROOT), status="prepared",
                  runner_modifies_rtl=False, learning=args.learning)
    path = output / "commands.json"
    for stage, command in ([] if args.reuse_executable else [("build", build)]) + [("run", run)]:
        record["status"] = stage + "_running"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        start = time.monotonic()
        with (output / (stage + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage + "_exit"] = result.returncode
        record[stage + "_seconds"] = time.monotonic() - start
        record["status"] = stage + ("_completed" if result.returncode == 0 else "_failed")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"{stage} failed; preserved at {output}")
    if "SRNN_SAME_DUT_PASS" not in (output / "run.log").read_text(encoding="utf-8"):
        raise RuntimeError("missing numerical completion marker")
    record["status"] = "PASS_bounded_same_DUT_B1T8_not_dataset_accuracy"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
