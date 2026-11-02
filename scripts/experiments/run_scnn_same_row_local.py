"""Bounded paired unit check for SCNN same-row read reuse; no whole-core QoR."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NAME = "scnn_same_row_r1"
TOP = "tb_scnn_row_ports"


def linux(path):
    value = str(path.resolve())
    return "/mnt/" + value[0].lower() + value[2:].replace("\\", "/") if os.name == "nt" else value


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def launch(attempt_name):
    root = Path(__file__).resolve().parents[2]
    exp = root / "verification/experiments" / NAME
    attempt = exp / "runs" / attempt_name
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    sources = {
        "reference.sv": root / "verification/experiments/scnn_row_reuse_core_r1/rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv",
        "candidate.sv": exp / "rtl/scnn_same_row_weight_store_v1.sv",
        TOP + ".sv": exp / (TOP + ".sv"),
        "runner.py": Path(__file__).resolve(),
    }
    records = []
    for name, source in sources.items():
        target = inputs / name
        shutil.copyfile(source, target)
        if source.read_bytes() != target.read_bytes():
            raise ValueError("input copy mismatch: " + name)
        records.append(dict(source=str(source), archived=name, bytes=target.stat().st_size))
    save(attempt / "inputs.json", records)
    return subprocess.call([sys.executable, "-B", str(inputs / "runner.py"),
                            "--worker-root", str(root)], stdin=subprocess.DEVNULL)


def worker(root):
    inputs = Path(__file__).resolve().parent
    attempt = inputs.parent
    runtime = root / "tools/verilator_5_048_local"
    executable = linux(runtime / "bin/verilator")
    status = dict(status="STARTING", scope="Behavioral RAM paired unit only; no XPM/netlist, full scheduler, PPA or system throughput proof.",
                  jobs=1, virtual_memory_kib_per_process=3145728, stages={})

    def stage(name, argv, limit):
        record = dict(argv=argv, exit=None)
        status["stages"][name] = record
        status["status"] = name.upper() + "_RUNNING"
        save(attempt / "status.json", status)
        command = " ".join(shlex.quote(str(x)) for x in argv)
        shell = "\n".join([
            "set -eu", "cd " + shlex.quote(linux(attempt)),
            "ulimit -v 3145728",
            "export VERILATOR_ROOT=" + shlex.quote(linux(runtime / "share/verilator")),
            "exec /usr/bin/time -v -o " + shlex.quote(name + "_resources.txt") +
            " timeout --signal=TERM --kill-after=5s " + str(limit) + "s " + command +
            " > " + shlex.quote(name + ".log") + " 2>&1",
        ])
        args = ["wsl.exe", "-e", "sh", "-lc", shell] if os.name == "nt" else ["sh", "-lc", shell]
        start = time.monotonic()
        with (attempt / (name + "_launcher.log")).open("xb") as out:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
            record["launcher_pid"] = process.pid
            save(attempt / "status.json", status)
            record["exit"] = process.wait()
        record["wall_seconds"] = time.monotonic() - start
        log = (attempt / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        warnings = re.findall(r"^%Warning-([^:]+):", log, re.M)
        record["warnings"] = {kind: warnings.count(kind) for kind in sorted(set(warnings))}
        record["structural"] = sorted(set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"})
        record["errors"] = re.findall(r"^%Error[^\n]*", log, re.M)
        save(attempt / "status.json", status)
        if record["exit"] != 0 or record["structural"] or record["errors"]:
            raise RuntimeError(name + " failed; original logs retained")
        print(name + ": exit=0", flush=True)
        return log

    try:
        if not re.search(r"Verilator 5\.048\b", stage("version", [executable, "--version"], 30)):
            raise ValueError("expected bundled Verilator 5.048")
        options = ["--timing", "--assert", "--flatten", "-j", "1", "-Wall", "-Wno-fatal",
                   "--top-module", TOP, "--Mdir", "obj", "-CFLAGS", "-O1"]
        options += [linux(inputs / x) for x in ("reference.sv", "candidate.sv", TOP + ".sv")]
        stage("lint", [executable, "--lint-only"] + options, 180)
        stage("build", [executable, "--binary"] + options, 900)
        log = stage("run", [linux(attempt / "obj" / ("V" + TOP))], 120)
        rows = re.findall(r"^ROW_PORT_PASS (.*)$", log, re.M)
        if len(rows) != 1 or re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
            raise ValueError("missing unique completion or runtime assertion")
        summary = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
        required = ("checks", "reads", "writes", "simultaneous_candidate",
                    "collision_blocked", "fallback_cases", "stalls", "resets")
        if any(summary.get(key, 0) <= 0 for key in required):
            raise ValueError("required coverage counter absent or zero")
        if summary.get("simultaneous_old") != 0:
            raise ValueError("old active-pair concurrency unexpectedly nonzero")
        status.update(status="PASS_LOCAL_SCNN_SAME_ROW_PORT_REUSE", summary=summary)
        print(json.dumps(status["summary"], indent=2), flush=True)
    except Exception as error:
        status.update(status="FAIL", error=repr(error))
        raise
    finally:
        save(attempt / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"r[1-9][0-9]*", args.attempt):
        parser.error("attempt must be r followed by a positive integer")
    if args.worker_root:
        worker(args.worker_root)
    else:
        sys.exit(launch(args.attempt))
