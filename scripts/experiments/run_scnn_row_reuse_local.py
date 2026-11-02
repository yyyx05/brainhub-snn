"""Measure one address-reuse candidate against accepted row-turnover records, without rerunning the old arm."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NAME = "scnn_row_reuse_r1"
PUBLISHER = "unified_scnn_classifier_cache_refill_publisher_v1.sv"
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


def geometry():
    return [(n, c, m) for m in range(4)
            for n in (1, 2, 3, 5, 64, 512) for c in (1, 4, 5, 12)]


def parse(log):
    if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
        raise ValueError("runtime failure")
    rows = [fields(x) for x in re.findall(r"^REFILL_CASE (.*)$", log, re.M)]
    terminal = re.findall(r"^REFILL_PASS (.*)$", log, re.M)
    if len(rows) != 99 or len(terminal) != 1 or fields(terminal[0])["cases"] != 99:
        raise ValueError("missing/duplicate completion or cases")
    shapes = geometry() + [(3, 1, 0), (3, 1, 0), (5, 1, 0)]
    for index, (row, (n, c, m)) in enumerate(zip(rows, shapes)):
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
    starts = [fields(x) for x in re.findall(r"^REFILL_START (.*)$", log, re.M)]
    expected_starts = [(i, n, c, m, 0) for i, (n, c, m) in enumerate(geometry())]
    expected_starts += [(96, 5, 5, 0, 1), (96, 3, 1, 0, 0),
                        (97, 5, 5, 0, 2), (97, 3, 1, 0, 0),
                        (98, 5, 5, 0, 0), (98, 5, 1, 0, 0)]
    if len(starts) != len(expected_starts):
        raise ValueError("configure phase log count")
    expected_tag = 0
    for index, (start, expected) in enumerate(zip(starts, expected_starts)):
        actual = tuple(start.get(k) for k in ("id", "inputs", "outputs", "mode", "fault"))
        if actual != expected or start.get("phase") != 0:
            raise ValueError("configure phase/geometry mismatch: " + str(index))
        if index < 96:
            if start.get("tag") != expected_tag:
                raise ValueError("normal configure reset/changed tag: " + str(index))
            expected_tag = (expected_tag + start["inputs"]*9*start["outputs"]) % 65536
    return dict(cases=rows, terminal=end, starts=starts)


def archive_and_run():
    root = Path(__file__).resolve().parents[2]
    exp = root / "verification/experiments" / NAME
    prior = root / "verification/experiments/scnn_refill_phase_aligned_r1/runs/r1"
    reference = json.loads((prior / "status.json").read_text())
    if reference["status"] != "PASS_LOCAL_PAIRED_SCNN_REFILL_PHASE_ALIGNED":
        raise ValueError("accepted row-turnover records required")
    source = prior / "inputs/common"
    check = dict(reference_rerun=False, expected_cycle_and_transaction_identity=True,
                 tb_change="added row-retirement subword/address assertions only")
    attempt = exp / "runs/r1"
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    files = {
        "reference.json": prior / "status.json",
        "candidate/" + PUBLISHER: exp / "rtl" / PUBLISHER,
        "common/unified_array16_weight_refill_dispatcher_v1.sv":
            source / "unified_array16_weight_refill_dispatcher_v1.sv",
        "common/unified_scnn_h40_conv_context_refill_v1.sv":
            source / "unified_scnn_h40_conv_context_refill_v1.sv",
        TOP + ".sv": exp / (TOP + ".sv"),
        Path(__file__).name: Path(__file__).resolve(),
    }
    records = []
    for relative, original in files.items():
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if target.read_bytes() != original.read_bytes():
            raise ValueError("archived input differs: " + relative)
        records.append(dict(source=str(original), archived=relative,
                            bytes=target.stat().st_size, byte_equal=True))
    write_json(attempt / "inputs.json", dict(files=records, phase_check=check))
    print(json.dumps(dict(attempt=str(attempt), phase_check=check, archived_inputs=len(records))),
          flush=True)
    # The archived runner, not this live copy, owns compilation and simulation.
    return subprocess.call([sys.executable, "-B", str(inputs / Path(__file__).name),
                            "--worker-root", str(root)], stdin=subprocess.DEVNULL)


def worker(root):
    inputs = Path(__file__).resolve().parent
    attempt = inputs.parent
    runtime = root / "tools/verilator_5_048_local"
    executable = linux(runtime / "bin/verilator")
    state = dict(status="STARTING", attempt=str(attempt), runner=str(Path(__file__).resolve()),
                 limits=dict(jobs=1, virtual_memory_kib_per_process=3145728),
                 scope="One address-reuse candidate, original 99-case phase-aligned oracle; no old arm rerun or whole-core/physical QoR.",
                 stages={}, arms={})

    def save():
        write_json(attempt / "status.json", state)

    def stage(name, arguments, seconds):
        log_path = attempt / (name + ".log")
        command = " ".join(shlex.quote(str(x)) for x in arguments)
        shell = "\n".join([
            "set -eu", "cd " + shlex.quote(linux(attempt)), "ulimit -v 3145728",
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
        for arm in ("candidate",):
            sources = [inputs / "common/unified_array16_weight_refill_dispatcher_v1.sv",
                       inputs / arm / PUBLISHER,
                       inputs / "common/unified_scnn_h40_conv_context_refill_v1.sv",
                       inputs / (TOP + ".sv")]
            options = ["--timing", "--assert", "--flatten", "-j", "1", "-Wall", "-Wno-fatal",
                       "--top-module", TOP, "--Mdir", arm + "_obj", "-CFLAGS", "-O1"]
            options += [linux(path) for path in sources]
            stage(arm + "_lint", [executable, "--lint-only"] + options, 180)
            stage(arm + "_build", [executable, "--binary"] + options, 900)
            log = stage(arm + "_run", [linux(attempt / (arm + "_obj") / ("V" + TOP))], 180)
            state["arms"][arm] = parse(log)
            save()
        reference = json.loads((inputs / "reference.json").read_text())["arms"]["candidate"]
        measured = state["arms"]["candidate"]
        if measured != reference:
            raise ValueError("new candidate changed accepted cases, counts, cycles, starts or tags")
        identity = re.findall(r"^ROW_REUSE_IDENTITY checks=(\d+)$", log, re.M)
        if len(identity) != 1 or int(identity[0]) < sum(row["rows"] for row in measured["cases"]):
            raise ValueError("row retirement invariant coverage missing")
        summary = dict(cases=99, starts=102, unchanged_cycles=99, changed_cycles=0,
                       row_identity_checks=int(identity[0]), reference_rerun=False)
        state.update(status="PASS_LOCAL_SCNN_ROW_ADDRESS_REUSE", summary=summary,
                     exact_reference_records=True)
        print(json.dumps(dict(status=state["status"], attempt=str(attempt),
                              summary=summary), indent=2), flush=True)
    except Exception as error:
        state.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"], nargs="?", default="run")
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_root:
        worker(args.worker_root)
    else:
        sys.exit(archive_and_run())
