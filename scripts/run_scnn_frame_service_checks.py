#!/usr/bin/env python3
"""Run focused current-source frame-service checks; no SSH or Vivado."""
import argparse
import json
import subprocess
from pathlib import Path

from run_fc_final_counts import ROOT, build_settings, generate_tb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("descriptor", "integration", "lint"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--verilator")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("run-name must be a simple fresh name")
    output = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_frame_service" / args.run_name
    work = ROOT / "work/scnn_frame_service" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    if args.case == "descriptor":
        top = "tb_unified_scnn_frame_descriptor_sequencer_v1"
        sources = [ROOT / ("rtl/control/" + name + ".sv") for name in (
            "unified_scnn_frame_descriptor_sequencer_v1",
            "unified_scnn_selected_layers_graph_batch_coordinator_v1")]
        sources.append(ROOT / ("tb/" + top + ".sv"))
    else:
        top = "unified_h40_final_composition_top_v1" if args.case == "lint" else "tb_scnn_final_frame_service"
        sources = sorted(ROOT.glob("rtl/**/*.sv"))
        if args.case == "integration":
            tb = output / (top + ".sv")
            generate_tb(tb, top=top, body=ROOT / "tb/include/scnn_final_frame_service_body.svh",
                        parameter_overrides={"PARAM_DEPTH": "128", "MAX_FEATURES": "8", "MAX_CLASSES": "2",
                            "MAX_BATCH": "4", "MAX_TIMESTEPS": "20", "SPATIAL": "4", "POOLED_SPATIAL": "2",
                            "MAX_INPUT_CHANNELS": "1", "MAX_OUTPUT_CHANNELS": "1"})
            sources.append(tb)
    source_file = work / "sources.f"
    source_file.write_text("".join('"' + p.as_posix() + '"\n' for p in sources), encoding="ascii")
    command, exe, env = build_settings(work / "obj", source_file, verilator=args.verilator,
                                       jobs=args.jobs, top=top)
    if args.case == "lint":
        command[command.index("--binary")] = "--lint-only"
    commands = [("lint" if args.case == "lint" else "build", command)]
    if args.case == "integration":
        commands.extend(("run_" + arm, [str(exe), "+LEARNING=" + value, "+WINDOWS=3"])
                        for arm, value in (("off", "0"), ("on", "1")))
    elif args.case != "lint":
        commands.append(("run", [str(exe)]))
    record = dict(case=args.case, status="prepared", commands=commands, cwd=str(ROOT))
    for stage, cmd in commands:
        record["status"] = stage + "_running"
        (output / "status.json").write_text(json.dumps(record, indent=2), encoding="ascii")
        with (output / (stage + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage + "_exit_code"] = result.returncode
        log_text = (output / (stage + ".log")).read_text(encoding="utf-8", errors="replace")
        accepted = result.returncode == 0
        if stage.startswith("run"):
            accepted = accepted and "PASS: SCNN_" in log_text
        else:
            accepted = accepted and not any("%Warning-" + name in log_text
                for name in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN"))
        record["status"] = stage + ("_pass" if accepted else "_failed")
        (output / "status.json").write_text(json.dumps(record, indent=2), encoding="ascii")
        if not accepted:
            raise RuntimeError(f"{stage} failed: {output}")
    record["status"] = "pass"
    (output / "status.json").write_text(json.dumps(record, indent=2), encoding="ascii")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
