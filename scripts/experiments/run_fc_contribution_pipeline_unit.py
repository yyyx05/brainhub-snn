#!/usr/bin/env python3
"""Run a bounded old/new scheduler unit comparison using the existing SSH helpers."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def worker(verilator):
    status = {"status": "running"}

    def save():
        Path("status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def execute(stage, command, timeout):
        started = time.monotonic()
        status["status"] = stage + "_running"
        save()
        with Path(stage + ".log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                status[stage] = {"command": command, "exit_code": None, "timeout_seconds": timeout,
                                 "elapsed_seconds": time.monotonic() - started}
                raise
        status[stage] = {"command": command, "exit_code": result.returncode,
                         "elapsed_seconds": time.monotonic() - started}
        save()
        if result.returncode:
            raise RuntimeError(stage + " failed")

    try:
        execute("build", [verilator, "--binary", "--timing", "--assert", "-Wno-fatal", "-j", "2",
            "--top-module", "tb_fc_contribution_pipeline", "--Mdir", "obj", "-CFLAGS", "-O1",
            "reference_fc_contribution_scheduler.sv", "unified_fc_tp_w2_contribution_scheduler.sv",
            "tb_fc_contribution_pipeline.sv"], 300)
        build_text = Path("build.log").read_text(encoding="utf-8", errors="replace")
        status["structural_warnings"] = re.findall(r"%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*", build_text)
        if status["structural_warnings"]:
            raise RuntimeError("structural warning")
        execute("run", ["obj/Vtb_fc_contribution_pipeline"], 60)
        text = Path("run.log").read_text(encoding="utf-8", errors="replace")
        cases = [dict((key, int(value)) for key, value in re.findall(r"(\w+)=(\d+)", line))
                 for line in text.splitlines() if line.startswith("FC_SCANNER_CASE ")]
        summary = re.findall(r"FC_SCANNER_UNIT_PASS cases=(\d+) checked_tokens=(\d+) reset_during_stall=1", text)
        if len(summary) != 1 or len(cases) != 16 or int(summary[0][0]) != len(cases):
            raise RuntimeError("missing unit completion/cases")
        status.update(status="PASS_unit_stream_equivalence_not_full_core_or_physical_QoR",
                      cases=cases, checked_tokens=int(summary[0][1]))
    except Exception as error:
        status.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        save()
    print(json.dumps(status, indent=2))
    return 0 if status["status"].startswith("PASS_") else 1


def launch(name, candidate_name, candidate_testbench=False):
    from prepare_fc_contribution_pipeline import ROOT, SOURCE, EXPERIMENT
    from run_srnn_resident_remote import connect, run, upload, fetch_tree, REMOTE, RESULT, VERILATOR
    client = connect()
    destination = REMOTE + "/" + name
    try:
        print(run(client, "df -h /root/autodl-tmp"))
        with client.open_sftp() as sftp:
            sftp.mkdir(destination)
            candidate_root = ROOT / EXPERIMENT / candidate_name
            for local, remote in (
                (candidate_root / SOURCE, SOURCE.name),
                (candidate_root / "reference_fc_contribution_scheduler.sv", "reference_fc_contribution_scheduler.sv"),
                (candidate_root / "source_changes.diff", "source_changes.diff"),
                ((candidate_root if candidate_testbench else ROOT / EXPERIMENT) /
                 "tb_fc_contribution_pipeline.sv", "tb_fc_contribution_pipeline.sv"),
                (Path(__file__), "worker.py"),
            ):
                upload(sftp, local, destination + "/" + remote)
        command = "cd " + shlex.quote(destination) + " && python3 worker.py --worker --verilator " + shlex.quote(VERILATOR)
        try:
            print(run(client, command))
        finally:
            with client.open_sftp() as sftp:
                fetch_tree(sftp, destination, ROOT / RESULT / name)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--name", default="fc_contribution_pipeline_unit_r1")
    parser.add_argument("--candidate", default="candidate_r1")
    parser.add_argument("--candidate-testbench", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.verilator))
    launch(args.name, args.candidate, args.candidate_testbench)
