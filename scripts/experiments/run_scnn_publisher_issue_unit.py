#!/usr/bin/env python3
"""Bounded original/candidate publisher comparison, not full-core performance."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def parse_log(text):
    if re.search(r"%Error|%Fatal|Assertion failed", text):
        raise ValueError("simulation diagnostic in log")
    def records(prefix):
        return [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", line))
                for line in text.splitlines() if line.startswith(prefix + " ")]
    cases = records("SCNN_PUBLISHER_CASE")
    summaries = records("SCNN_PUBLISHER_CHECKER_PASS")
    if records("SCNN_PUBLISHER_UNIT_PASS") != [dict(variants=2, cases_per_variant=14)]:
        raise ValueError("missing/duplicate final marker")
    if len(cases) != 28 or len(summaries) != 2 or {r["opt"] for r in summaries} != {0, 1}:
        raise ValueError("incomplete comparison")
    for s in summaries:
        if (s["cases"], s["aborts"], s["bad_responses"], s["invalid_starts"]) != (14, 4, 2, 1):
            raise ValueError("missing recovery checks")
    groups = [[r for r in cases if r["opt"] == opt] for opt in (0, 1)]
    comparisons = []
    for group in groups:
        if [r["case_id"] for r in group] != list(range(14)):
            raise ValueError("case order or duplicate")
        for r in group:
            if r["reads"] != r["features"] * r["classes"] or r["rows"] != 2 * r["classes"] * ((r["features"]+23)//24):
                raise ValueError("incomplete matrix coverage")
    for a, b in zip(*groups):
        if {k:v for k,v in a.items() if k not in ("opt", "cycles")} != {k:v for k,v in b.items() if k not in ("opt", "cycles")}:
            raise ValueError("unmatched case")
        if not a["stalls"] and a["cycles"] - b["cycles"] != a["reads"]:
            raise ValueError("no-stall cycle saving differs from one per scalar")
        comparisons.append(dict(case_id=a["case_id"], features=a["features"], classes=a["classes"],
            stalls=a["stalls"], original_cycles=a["cycles"], candidate_cycles=b["cycles"],
            reads=a["reads"], rows=a["rows"], fair_no_stall_speedup=a["cycles"]/b["cycles"] if not a["stalls"] else None))
    return dict(cases=comparisons, summaries=summaries,
                scope="Single-outstanding scalar service, one-cycle minimum response. Stalled cases are functional, not a fair performance ratio. Unit abort is not full-core cancellation; no mapped resources or full-core throughput.")


def worker(verilator):
    status = dict(status="running")
    def save():
        Path("status.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    def execute(stage, command, timeout):
        status["status"] = stage+"_running"
        save()
        started = time.monotonic()
        with Path(stage+".log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                status[stage] = dict(command=command, exit_code=None, timeout_seconds=timeout,
                                     elapsed_seconds=time.monotonic()-started)
                raise
        status[stage] = dict(command=command, exit_code=result.returncode,
                             elapsed_seconds=time.monotonic()-started)
        save()
        if result.returncode:
            raise RuntimeError(stage+" failed")
    try:
        execute("build", [verilator, "--binary", "--timing", "--assert", "-Wno-fatal", "-j", "2",
            "--top-module", "tb_scnn_publisher_issue", "--Mdir", "obj", "-CFLAGS", "-O1",
            "unified_array16_weight_refill_dispatcher_v1.sv", "reference_scnn_publisher.sv",
            "unified_scnn_classifier_cache_refill_publisher_v1.sv", "tb_scnn_publisher_issue.sv"], 300)
        build = Path("build.log").read_text(encoding="utf-8", errors="replace")
        status["structural_warnings"] = re.findall(r"%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*", build)
        if status["structural_warnings"] or "%Error" in build:
            raise RuntimeError("build diagnostic")
        execute("run", ["obj/Vtb_scnn_publisher_issue"], 60)
        status["checks"] = parse_log(Path("run.log").read_text(encoding="utf-8", errors="replace"))
        status["status"] = "PASS_publisher_unit_not_full_core_or_physical_QoR"
    except Exception as error:
        status.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        save()
    print(json.dumps(status, indent=2))
    return 0 if status["status"].startswith("PASS_") else 1


def launch(name, candidate_name):
    from prepare_scnn_publisher_issue import ROOT, SOURCE, EXPERIMENT
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
                (candidate_root / "reference_scnn_publisher.sv", "reference_scnn_publisher.sv"),
                (candidate_root / "source_changes.diff", "source_changes.diff"),
                (ROOT / "rtl/memory/unified_array16_weight_refill_dispatcher_v1.sv", "unified_array16_weight_refill_dispatcher_v1.sv"),
                (ROOT / EXPERIMENT / "tb_scnn_publisher_issue.sv", "tb_scnn_publisher_issue.sv"),
                (Path(__file__), "worker.py"),
            ):
                upload(sftp, local, destination + "/" + remote)
        try:
            print(run(client, "cd " + shlex.quote(destination) + " && python3 worker.py --worker --verilator " + shlex.quote(VERILATOR)))
        finally:
            with client.open_sftp() as sftp:
                fetch_tree(sftp, destination, ROOT / RESULT / name)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--name", default="scnn_publisher_issue_unit_r1")
    parser.add_argument("--candidate", default="candidate_r1")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.verilator))
    launch(args.name, args.candidate)
