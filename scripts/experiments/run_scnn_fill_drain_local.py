"""Exercise real SCNN publisher cancellation/drain and window RAM contents."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

NAME = "scnn_fill_drain_r1"
TOP = "tb_fill_drain"
VENDOR = Path("D:/vivado2022.2/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def readable_path(path):
    return Path("\\\\?\\" + str(path.absolute())) if sys.platform == "win32" else path


def launch(attempt_name):
    root = Path(__file__).resolve().parents[2]
    experiment = root / "verification/experiments" / NAME
    archive = root / "results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_reuse_core_r1/rtl"
    sources = {
        "wrapper.sv": experiment / "rtl/scnn_owned_publisher_v1.sv",
        "owner.sv": root / "verification/experiments/scnn_window_owner_r1/rtl/scnn_window_owner_v1.sv",
        "publisher.sv": root / "verification/experiments/scnn_row_window_r1/rtl/scnn_row_window_publisher_v1.sv",
        "dispatcher.sv": archive / "memory/unified_array16_weight_refill_dispatcher_v1.sv",
        "store.sv": root / "verification/experiments/scnn_held_response_r1/rtl/scnn_held_response_weight_store_v1.sv",
        TOP + ".sv": experiment / (TOP + ".sv"),
        "native_helper.py": root / "scripts/experiments/run_scnn_xpm_hold_local.py",
        "runner.py": Path(__file__).resolve(),
    }
    for source in [*sources.values(), VENDOR]:
        if not readable_path(source).is_file():
            raise FileNotFoundError(source)
    attempt = experiment / "runs" / attempt_name
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    records = []
    for name, source in sources.items():
        target = inputs / name
        shutil.copyfile(readable_path(source), target)
        if readable_path(source).read_bytes() != target.read_bytes():
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
    work = attempt / "xpm"
    work.mkdir(exist_ok=False)
    dependency = json.loads((attempt / "vendor_dependency.json").read_text())
    bins = VENDOR.parents[5] / "bin"
    status = dict(status="STARTING", engine="xsim", jobs=1,
                  scope="Local real publisher/dispatcher/owner/RAM fill drain. No scheduler retirement, parameter plane, whole-core, PPA or throughput proof.",
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
        argv = [str(bins / "xvlog.bat"), "-sv", "-d", "UNIFIED_USE_XPM_MEMORY", str(VENDOR)]
        argv += [str(inputs / name) for name in
                 ("dispatcher.sv", "publisher.sv", "owner.sv", "wrapper.sv", "store.sv", TOP + ".sv")]
        stage("compile", argv, 180)
        stage("elaborate", [str(bins / "xelab.bat"), TOP, "-s", "fill_drain_test",
                           "--mt", "off", "--timescale", "1ns/1ps"], 300)
        log = stage("run", [str(bins / "xsim.bat"), "fill_drain_test", "-runall"], 120)
        rows = re.findall(r"^PASS_SCNN_FILL_DRAIN (.*)$", log, re.M)
        if len(rows) != 1 or re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
            raise ValueError("missing unique terminal or runtime assertion")
        summary = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
        for key in ("cases", "checks", "scalar_requests", "scalar_responses",
                    "cancelled_responses", "row_writes", "ram_reads"):
            if summary.get(key, 0) <= 0:
                raise ValueError("missing directed coverage: " + key)
        cases = re.findall(r"^CASE[^\n]*", log, re.M)
        if len(cases) != summary["cases"]:
            raise ValueError("case record count mismatch")
        status.update(status="PASS_LOCAL_SCNN_FILL_DRAIN",
                      local_directed_accepted=True, summary=summary, case_records=cases)
        print(json.dumps(summary, indent=2), flush=True)
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
