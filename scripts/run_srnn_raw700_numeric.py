#!/usr/bin/env python3
"""Check actual raw700 backbone observations at the production final top."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_fc_final_counts import ROOT, build_settings, generate_tb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-name", default="server_run1")
    parser.add_argument("--verilator", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("run-name must be a simple directory name")
    fixture = args.fixture.resolve()
    if not fixture.is_relative_to(ROOT):
        parser.error("fixture must stay in the project")
    metadata = json.loads((fixture / "summary.json").read_text(encoding="utf-8"))
    if metadata["shape"] != [8, 2, 480]:
        parser.error("this raw700 experiment requires the fixed B2T8 reference")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/srnn_raw700" / args.run_name
    work = ROOT / "work/supplement_20261004/srnn_raw700" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    top = "tb_srnn_raw700_numeric"
    tb = work / (top + ".sv")
    generate_tb(tb, top=top, body=ROOT / "tb/include/supplement_srnn_raw700_body.svh")
    sources = work / "sources.f"
    paths = sorted(ROOT.glob("rtl/**/*.sv")) + [ROOT / "tb/monitors/supplement_h40_activity_monitor.sv", tb]
    sources.write_text("".join('"' + path.as_posix() + '"\n' for path in paths), encoding="utf-8")
    build, exe, env = build_settings(work / "obj", sources, verilator=args.verilator, jobs=args.jobs, top=top)
    run = [str(exe), "+FIXTURE=" + fixture.relative_to(ROOT).as_posix(), "+DUMP=" + str(out / "observations.csv")]
    record = {"cwd": str(ROOT), "build": build, "run": run, "status": "prepared"}
    record_path = out / "commands.json"
    record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(out, flush=True)
    if args.prepare_only:
        return
    for stage, command in (("build", build), ("run", run)):
        record["status"] = stage + "_running"
        record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        start = time.monotonic()
        with (out / (stage + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage + "_exit"] = result.returncode
        record[stage + "_seconds"] = time.monotonic() - start
        record["status"] = stage + ("_completed" if result.returncode == 0 else "_failed")
        record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"{stage} failed; evidence preserved in {out}")
    log = (out / "run.log").read_text(encoding="utf-8")
    if "SRNN_RAW700_BACKBONE_PASS" not in log:
        raise RuntimeError("missing raw700 numeric completion")
    record["status"] = "PASS_backbone_only_not_M2_or_full_dataset"
    record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
