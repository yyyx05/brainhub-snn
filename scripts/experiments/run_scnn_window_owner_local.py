"""Exercise the isolated two-slot SCNN lease/version controller."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

NAME = "scnn_window_owner_r1"
TOP = "tb_window_owner"
BINS = Path("D:/vivado2022.2/Xilinx/Vivado/2022.2/bin")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def launch(attempt_name):
    root = Path(__file__).resolve().parents[2]
    experiment = root / "verification/experiments" / NAME
    sources = {
        "candidate.sv": experiment / "rtl/scnn_window_owner_v1.sv",
        TOP + ".sv": experiment / (TOP + ".sv"),
        "native_helper.py": root / "scripts/experiments/run_scnn_xpm_hold_local.py",
        "runner.py": Path(__file__).resolve(),
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    attempt = experiment / "runs" / attempt_name
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    records = []
    for name, source in sources.items():
        target = inputs / name
        shutil.copyfile(source, target)
        if source.read_bytes() != target.read_bytes():
            raise ValueError("input copy mismatch: " + name)
        records.append(dict(source=str(source), archived=name, bytes=target.stat().st_size))
    save(attempt / "inputs.json", records)
    return subprocess.call([sys.executable, "-B", str(inputs / "runner.py"), "--worker"],
                           stdin=subprocess.DEVNULL)


def worker():
    from native_helper import native_stage

    inputs = Path(__file__).resolve().parent
    attempt = inputs.parent
    work = attempt / "xsim"
    work.mkdir(exist_ok=False)
    status = dict(status="STARTING", engine="xsim", jobs=1,
                  scope="Isolated metadata/lease control; drain/final-release inputs are driven by TB. No RAM, publisher or full-core integration.",
                  job_and_process_commit_limit_bytes=3 * 1024**3, stages={})

    def stage(name, argv, limit):
        record = dict(argv=argv, exit=None)
        status["stages"][name] = record
        status["status"] = name.upper() + "_RUNNING"
        save(attempt / "status.json", status)
        started = time.monotonic()
        with (work / (name + ".log")).open("xb") as out:
            record.update(native_stage(["cmd.exe", "/d", "/c"] + argv, work, out, limit))
        record["wall_seconds"] = time.monotonic() - started
        log = (work / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        record["warnings"] = re.findall(r"^WARNING:[^\n]*", log, re.M)
        record["errors"] = re.findall(r"^(?:ERROR:|FATAL_ERROR:|FATAL:|Fatal:)[^\n]*", log, re.M)
        save(attempt / "status.json", status)
        if record["exit"] != 0 or record["timed_out"] or record["errors"]:
            raise RuntimeError(name + " failed; originals retained")
        print(name + ": exit=0", flush=True)
        return log

    try:
        stage("compile", [str(BINS / "xvlog.bat"), "-sv", str(inputs / "candidate.sv"),
                          str(inputs / (TOP + ".sv"))], 180)
        stage("elaborate", [str(BINS / "xelab.bat"), TOP, "-s", "owner_test",
                           "--mt", "off", "--timescale", "1ns/1ps"], 300)
        log = stage("run", [str(BINS / "xsim.bat"), "owner_test", "-runall"], 60)
        rows = re.findall(r"^PASS_SCNN_WINDOW_OWNER (.*)$", log, re.M)
        if len(rows) != 1 or re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
            raise ValueError("missing unique terminal or runtime assertion")
        summary = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
        for key in ("cases", "checks", "rejected", "promotions", "retirements",
                    "drains", "stale_completions"):
            if summary.get(key, 0) <= 0:
                raise ValueError("missing directed coverage: " + key)
        cases = re.findall(r"^CASE[^\n]*", log, re.M)
        if len(cases) != summary["cases"]:
            raise ValueError("case record count mismatch")
        status.update(status="PASS_LOCAL_SCNN_WINDOW_OWNER",
                      local_directed_accepted=True, summary=summary, case_records=cases)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        status.update(status="FAIL", local_directed_accepted=False, error=repr(error))
        raise
    finally:
        save(attempt / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"r[1-9][0-9]*", args.attempt):
        parser.error("attempt must be r followed by a positive integer")
    if args.worker:
        worker()
    else:
        sys.exit(launch(args.attempt))
