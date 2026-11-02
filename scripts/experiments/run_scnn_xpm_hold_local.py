"""Characterize held responses against the installed, unmodified XPM model."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NAME = "scnn_xpm_hold_r1"
TOP = "tb_xpm_hold"
VENDOR = Path("D:/vivado2022.2/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv")


def linux(path):
    value = str(path.resolve())
    return "/mnt/" + value[0].lower() + value[2:].replace("\\", "/") if os.name == "nt" else value


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def launch(attempt_name, engine):
    root = Path(__file__).resolve().parents[2]
    attempt = root / "verification/experiments" / NAME / "runs" / attempt_name
    if not VENDOR.is_file():
        raise FileNotFoundError(VENDOR)
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    sources = {
        "candidate.sv": root / "verification/experiments/scnn_same_row_r1/rtl/scnn_same_row_weight_store_v1.sv",
        TOP + ".sv": root / "verification/experiments" / NAME / (TOP + ".sv"),
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
    info = VENDOR.stat()
    save(attempt / "vendor_dependency.json",
         dict(path=str(VENDOR), bytes=info.st_size, mtime_ns=info.st_mtime_ns,
              copied=False, modified=False))
    return subprocess.call([sys.executable, "-B", str(inputs / "runner.py"),
                            "--worker-root", str(root), "--engine", engine], stdin=subprocess.DEVNULL)


def native_stage(args, cwd, out, limit):
    # A job bounds only this newly spawned, initially suspended tool and children.
    size = ctypes.c_size_t
    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", size), ("max_ws", size),
                    ("active_limit", wintypes.DWORD), ("affinity", size),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Limits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                    ("process_memory", size), ("job_memory", size),
                    ("peak_process", size), ("peak_job", size)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process = None
    assigned = False
    try:
        config = Limits()
        config.basic.flags = 0x2000 | 0x100 | 0x200
        config.process_memory = config.job_memory = 3 * 1024**3
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(config), ctypes.sizeof(config)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT,
                                   creationflags=0x4 | subprocess.CREATE_NO_WINDOW)
        if not kernel.AssignProcessToJobObject(job, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        assigned = True
        if ntdll.NtResumeProcess(process._handle) != 0:
            raise RuntimeError("could not resume owned tool process")
        timed_out = False
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            if not kernel.TerminateJobObject(job, 124):
                raise ctypes.WinError(ctypes.get_last_error())
            code = process.wait()
        usage = Limits()
        if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(usage), ctypes.sizeof(usage), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(exit=code, launcher_pid=process.pid, timed_out=timed_out,
                    peak_job_commit_bytes=usage.peak_job, peak_process_commit_bytes=usage.peak_process)
    finally:
        kernel.CloseHandle(job)
        if process is not None and process.poll() is None:
            if not assigned:
                process.terminate()
            process.wait()


def worker(root, engine):
    inputs = Path(__file__).resolve().parent
    attempt = inputs.parent
    runtime = root / "tools/verilator_5_048_local"
    executable = linux(runtime / "bin/verilator")
    status = dict(status="STARTING",
                  scope="Actual vendor XPM model defect characterization; not implementation acceptance or system performance.",
                  jobs=1, engine=engine, stages={})
    if engine == "xsim":
        status["job_and_process_commit_limit_bytes"] = 3 * 1024**3
    else:
        status["virtual_memory_kib_per_process"] = 3145728
    vendor_before = json.loads((attempt / "vendor_dependency.json").read_text())

    def stage(name, argv, limit):
        record = dict(argv=argv, exit=None)
        status["stages"][name] = record
        status["status"] = name.upper() + "_RUNNING"
        save(attempt / "status.json", status)
        start = time.monotonic()
        if engine == "xsim":
            with (attempt / (name + ".log")).open("xb") as out:
                record.update(native_stage(["cmd.exe", "/d", "/c"] + argv, attempt, out, limit))
        else:
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
        record["errors"] = re.findall(r"^(?:%Error|ERROR:|FATAL_ERROR:|FATAL:|Fatal:)[^\n]*", log, re.M)
        record["native_warnings"] = re.findall(r"^WARNING:[^\n]*", log, re.M)
        save(attempt / "status.json", status)
        if record["exit"] != 0 or record.get("timed_out") or record["structural"] or record["errors"]:
            raise RuntimeError(name + " failed; original logs retained")
        print(name + ": exit=0", flush=True)
        return log

    try:
        if engine == "xsim":
            bins = VENDOR.parents[5] / "bin"
            stage("compile", [str(bins / "xvlog.bat"), "-sv", "-d", "UNIFIED_USE_XPM_MEMORY",
                              str(VENDOR), str(inputs / "candidate.sv"), str(inputs / (TOP + ".sv"))], 180)
            stage("elaborate", [str(bins / "xelab.bat"), TOP, "-s", "hold_test",
                                "--mt", "off", "--timescale", "1ns/1ps"], 300)
            log = stage("run", [str(bins / "xsim.bat"), "hold_test", "-runall"], 60)
        else:
            if not re.search(r"Verilator 5\.048\b", stage("version", [executable, "--version"], 30)):
                raise ValueError("expected bundled Verilator 5.048")
            options = ["--timing", "--assert", "--flatten", "-j", "1", "-Wall", "-Wno-fatal",
                       "-DUNIFIED_USE_XPM_MEMORY", "--top-module", TOP,
                       "--Mdir", "obj", "-CFLAGS", "-O1",
                       linux(VENDOR), linux(inputs / "candidate.sv"), linux(inputs / (TOP + ".sv"))]
            stage("lint", [executable, "--lint-only"] + options, 180)
            stage("build", [executable, "--binary"] + options, 900)
            log = stage("run", [linux(attempt / "obj" / ("V" + TOP))], 30)
        rows = re.findall(r"^XPM_HOLD_CHARACTERIZED (.*)$", log, re.M)
        if len(rows) != 1 or re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
            raise ValueError("missing unique characterization terminal or runtime assertion")
        summary = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
        expected = dict(vulnerable_cases=2, guarded_cases=1, coalesced_safe_cases=1)
        if any(summary.get(k) != v for k, v in expected.items()) or summary.get("checks", 0) <= 0:
            raise ValueError("expected semantic cases not characterized")
        status.update(status="CHARACTERIZED_XPM_HELD_RESPONSE_COUNTEREXAMPLES",
                      candidate_accepted=False, summary=summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        status.update(status="FAIL", error=repr(error))
        raise
    finally:
        info = VENDOR.stat()
        status["vendor_metadata_unchanged"] = (
            info.st_size == vendor_before["bytes"] and
            info.st_mtime_ns == vendor_before["mtime_ns"])
        save(attempt / "status.json", status)
        if not status["vendor_metadata_unchanged"]:
            raise ValueError("vendor dependency metadata changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--engine", choices=("verilator", "xsim"), default="verilator")
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"r[1-9][0-9]*", args.attempt):
        parser.error("attempt must be r followed by a positive integer")
    if args.worker_root:
        worker(args.worker_root, args.engine)
    else:
        sys.exit(launch(args.attempt, args.engine))
