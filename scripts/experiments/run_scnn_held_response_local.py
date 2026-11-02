"""Check the isolated held-response repair with behavioral and vendor RAM."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

NAME = "scnn_held_response_r1"
TOP = "tb_held_response"
VENDOR = Path("D:/vivado2022.2/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def launch(attempt_name):
    root = Path(__file__).resolve().parents[2]
    experiment = root / "verification/experiments" / NAME
    attempt = experiment / "runs" / attempt_name
    sources = {
        "candidate.sv": experiment / "rtl/scnn_held_response_weight_store_v1.sv",
        TOP + ".sv": experiment / (TOP + ".sv"),
        "runner.py": Path(__file__).resolve(),
        "native_helper.py": root / "scripts/experiments/run_scnn_xpm_hold_local.py",
    }
    for source in [*sources.values(), VENDOR]:
        if not source.is_file():
            raise FileNotFoundError(source)
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
    info = VENDOR.stat()
    save(attempt / "vendor_dependency.json", dict(path=str(VENDOR), bytes=info.st_size,
         mtime_ns=info.st_mtime_ns, copied=False, modified=False))
    return subprocess.call([sys.executable, "-B", str(inputs / "runner.py"), "--worker"],
                           stdin=subprocess.DEVNULL)


def worker():
    from native_helper import native_stage

    inputs = Path(__file__).resolve().parent
    attempt = inputs.parent
    dependency = json.loads((attempt / "vendor_dependency.json").read_text())
    bins = VENDOR.parents[5] / "bin"
    status = dict(status="STARTING", engine="xsim", jobs=1,
                  scope="Directed unit repair, not full-system reachability, PPA or throughput.",
                  job_and_process_commit_limit_bytes=3 * 1024**3, arms={})
    terminals, cases = {}, {}

    def stage(arm, name, argv, limit):
        work = attempt / arm
        record = dict(argv=argv, exit=None)
        status["arms"][arm][name] = record
        status["status"] = arm.upper() + "_" + name.upper() + "_RUNNING"
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
            raise RuntimeError(arm + "/" + name + " failed; originals retained")
        print(arm + "/" + name + ": exit=0", flush=True)
        return log

    try:
        for arm in ("behavioral", "xpm"):
            (attempt / arm).mkdir(exist_ok=False)
            status["arms"][arm] = {}
            argv = [str(bins / "xvlog.bat"), "-sv"]
            if arm == "xpm":
                argv += ["-d", "UNIFIED_USE_XPM_MEMORY", str(VENDOR)]
            argv += [str(inputs / "candidate.sv"), str(inputs / (TOP + ".sv"))]
            stage(arm, "compile", argv, 180)
            stage(arm, "elaborate", [str(bins / "xelab.bat"), TOP, "-s", "repair_test",
                                  "--mt", "off", "--timescale", "1ns/1ps"], 300)
            log = stage(arm, "run", [str(bins / "xsim.bat"), "repair_test", "-runall"], 60)
            rows = re.findall(r"^HELD_RESPONSE_PASS (.*)$", log, re.M)
            if len(rows) != 1 or re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
                raise ValueError(arm + ": missing unique terminal or runtime assertion")
            summary = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
            keys = ("checks", "reads", "writes", "blocked", "held_cycles", "simultaneous", "resets")
            if set(summary) != set(keys) or any(summary[key] <= 0 for key in keys):
                raise ValueError(arm + ": missing directed coverage")
            terminals[arm] = summary
            cases[arm] = re.findall(r"^CASE[^\n]*", log, re.M)
            status["arms"][arm]["summary"] = summary
            status["arms"][arm]["case_records"] = cases[arm]
            save(attempt / "status.json", status)
        if terminals["behavioral"] != terminals["xpm"] or not cases["behavioral"] or cases["behavioral"] != cases["xpm"]:
            raise ValueError("behavioral and XPM directed records differ")
        status.update(status="PASS_LOCAL_HELD_RESPONSE_REPAIR", local_directed_accepted=True,
                      directed_case_records_equal=True, summary=terminals["xpm"])
        print(json.dumps(status["summary"], indent=2), flush=True)
    except Exception as error:
        status.update(status="FAIL", local_directed_accepted=False, error=repr(error))
        raise
    finally:
        info = VENDOR.stat()
        status["vendor_metadata_unchanged"] = (info.st_size == dependency["bytes"] and
                                               info.st_mtime_ns == dependency["mtime_ns"])
        if not status["vendor_metadata_unchanged"]:
            status.update(status="FAIL", local_directed_accepted=False)
        save(attempt / "status.json", status)
        if not status["vendor_metadata_unchanged"]:
            raise ValueError("vendor dependency metadata changed")


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
