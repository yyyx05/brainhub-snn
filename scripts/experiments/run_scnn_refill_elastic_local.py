"""Local paired SCNN refill slice only; no server, Vivado or production edits."""
import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "verification/experiments/scnn_refill_elastic_r1"
SOURCE = ROOT / "verification/experiments/final_scnn_functional_r1/rtl"
PUBLISHER = "unified_scnn_classifier_cache_refill_publisher_v1.sv"
TB = EXP / "tb_scnn_refill_elastic.sv"
TOP = "tb_scnn_refill_elastic"
STRUCTURAL = {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}


def linux(path):
    value = str(path.resolve())
    if os.name == "nt":
        return "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def diagnostics(log):
    warnings = collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M))
    return dict(warnings=dict(warnings), structural=sorted(STRUCTURAL & warnings.keys()),
                errors=re.findall(r"^%Error[^\n]*", log, re.M))


def fields(line):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}


def parse(log):
    if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
        raise ValueError("runtime failure")
    rows = [fields(x) for x in re.findall(r"^REFILL_CASE (.*)$", log, re.M)]
    terminal = re.findall(r"^REFILL_PASS (.*)$", log, re.M)
    if len(rows) != 99 or len(terminal) != 1 or fields(terminal[0])["cases"] != 99:
        raise ValueError("missing/duplicate completion or cases")
    geometry = [(n, c, m) for m in range(4)
                for n in (1, 2, 3, 5, 64, 512) for c in (1, 4, 5, 12)]
    geometry += [(3, 1, 0), (3, 1, 0), (5, 1, 0)]
    for index, (row, (n, c, m)) in enumerate(zip(rows, geometry)):
        expected = dict(id=index, inputs=n, outputs=c, mode=m, reads=n*9*c,
                        responses=n*9*c, rows=2*c*((n*9+23)//24), max_credit=1)
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("case geometry/count mismatch: " + str(index))
    if len(re.findall(r"^REFILL_FAULT ", log, re.M)) != 2 or \
            len(re.findall(r"^REFILL_ABORT ", log, re.M)) != 1:
        raise ValueError("fault/abort coverage missing")
    end = fields(terminal[0])
    if end.get("faults") != 2 or end.get("aborts") != 1 or end.get("tag_wraps", 0) < 1:
        raise ValueError("terminal coverage mismatch")
    return dict(cases=rows, terminal=end)


def run():
    attempt = EXP / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    attempt.mkdir(parents=True, exist_ok=False)
    runtime = ROOT / "tools/verilator_5_048_local"
    executable = linux(runtime / "bin/verilator")
    state = dict(status="STARTING", attempt=str(attempt),
                 limits=dict(jobs=1, virtual_memory_kib_per_process=3145728),
                 scope="Conv context refill + matrix publisher + original dispatcher; synthetic codes, not whole-core/DDR/physical QoR.",
                 stages={})

    def save():
        write_json(attempt / "status.json", state)

    def stage(name, arguments, seconds):
        log_path = attempt / (name + ".log")
        command = " ".join(shlex.quote(str(x)) for x in arguments)
        shell = "\n".join([
            "set -eu",
            "cd " + shlex.quote(linux(attempt)),
            "ulimit -v 3145728",
            "export VERILATOR_ROOT=" + shlex.quote(linux(runtime / "share/verilator")),
            "exec /usr/bin/time -v -o " + shlex.quote(name + "_resources.txt") +
            " timeout --signal=TERM --kill-after=5s " + str(seconds) + "s " + command +
            " > " + shlex.quote(name + ".log") + " 2>&1",
        ])
        args = ["wsl.exe", "-e", "sh", "-lc", shell] if os.name == "nt" else ["sh", "-lc", shell]
        record = dict(command=args, exit=None)
        state["stages"][name] = record
        state["status"] = name.upper() + "_RUNNING"
        save()
        start = time.monotonic()
        # Linux owns the merged tool log; WSL launcher diagnostics are separate.
        with (attempt / (name + "_launcher.log")).open("xb") as stream:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            record["launcher_pid"] = process.pid
            save()
            record["exit"] = process.wait()
        record["wall_seconds"] = time.monotonic() - start
        log = log_path.read_text(encoding="utf-8", errors="replace")
        record["diagnostics"] = diagnostics(log)
        save()
        if record["exit"] or record["diagnostics"]["structural"] or record["diagnostics"]["errors"]:
            raise RuntimeError(name + " failed; see " + str(log_path))
        print(name + ": exit=0", flush=True)
        return log

    try:
        version = stage("version", [executable, "--version"], 30)
        if not re.search(r"Verilator 5\.048\b", version):
            raise ValueError("expected package Verilator 5.048")
        arms = {}
        for arm, publisher in (
                ("original", SOURCE / "learning" / PUBLISHER),
                ("candidate", EXP / "rtl" / PUBLISHER)):
            sources = [SOURCE / "memory/unified_array16_weight_refill_dispatcher_v1.sv",
                       publisher, SOURCE / "memory/unified_scnn_h40_conv_context_refill_v1.sv", TB]
            options = ["--timing", "--assert", "--flatten", "-j", "1", "-Wall", "-Wno-fatal",
                       "--top-module", TOP, "--Mdir", arm + "_obj", "-CFLAGS", "-O1"]
            options += [linux(path) for path in sources]
            stage(arm + "_lint", [executable, "--lint-only"] + options, 180)
            stage(arm + "_build", [executable, "--binary"] + options, 900)
            log = stage(arm + "_run", [linux(attempt / (arm + "_obj") / ("V" + TOP))], 180)
            arms[arm] = parse(log)
        comparisons = []
        for old, new in zip(arms["original"]["cases"], arms["candidate"]["cases"]):
            if old["row_reads"] or old["row_blocked"]:
                raise ValueError("original unexpectedly overlaps row retirement")
            delta = old["cycles"] - new["cycles"]
            if old["mode"] in (0, 2):
                expected = 2 * old["outputs"] * (((old["inputs"]*9+23)//24)-1)
                if new["row_reads"] != expected or delta != expected:
                    raise ValueError("no-stall row-retirement cycle model mismatch")
            comparisons.append(dict(id=old["id"], inputs=old["inputs"], outputs=old["outputs"],
                                    mode=old["mode"], original_cycles=old["cycles"],
                                    candidate_cycles=new["cycles"], saved_cycles=delta,
                                    row_reads=new["row_reads"], row_blocked=new["row_blocked"]))
        if not any(x["row_blocked"] > 0 for x in comparisons):
            raise ValueError("row-retirement request backpressure not covered")
        if not all(any(c["request_stalls"] > 0 and c["fill_stalls"] > 0 for c in arm["cases"])
                   for arm in arms.values()):
            raise ValueError("backpressure coverage missing")
        state.update(status="PASS_LOCAL_PAIRED_SCNN_REFILL", arms=arms, comparisons=comparisons)
        write_json(attempt / "comparison.json", comparisons)
        print(json.dumps(dict(status=state["status"], attempt=str(attempt),
                              cases_per_arm=99, comparisons=comparisons), indent=2))
    except Exception as error:
        state.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"], nargs="?", default="run")
    parser.parse_args()
    run()
