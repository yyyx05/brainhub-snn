"""Prepare/test/run one local magnitude-mode slice, never SSH or whole-core EDA."""
import argparse
import collections
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NATIVE_ROOT = Path(str(Path(__file__).resolve()).removeprefix("\\\\?\\")).parents[2]
ROOT = Path("\\\\?\\" + str(NATIVE_ROOT)) if sys.platform == "win32" else NATIVE_ROOT
EXP_REL = Path("verification/experiments/magnitude_rmw_turnover_r1")
OUT_REL = Path("results/paper_evidence/supplement_20261015/performance_optimization/magnitude_rmw_turnover_local_r1")
SOURCE_REL = OUT_REL.parent / "srnn_gradient_retire_core_r1"
EXP, OUT, SOURCE = ROOT / EXP_REL, ROOT / OUT_REL, ROOT / SOURCE_REL
MODULES = (
    "unified_fc_tp_w2_learning_slice", "unified_tp_address_major_sign_reducer",
    "unified_fc_tp_gradient_rmw_path", "unified_fc_tp_gradient_request_adapter",
    "unified_tp_q015_magnitude_scaler", "unified_tp_parameter_rmw_service",
    "unified_tp_active_tensor_update_path", "unified_tp_active_tensor_context",
    "unified_tp_parameter_update_adapter", "unified_tp_direct_residual_update",
    "unified_tp_q015_magnitude_residual_update",
)
TB = "tb_magnitude_rmw_turnover.sv"
STRUCTURAL = {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}


def stamp():
    return dt.datetime.now().astimezone().isoformat()


def identity():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def reference():
    spec = importlib.util.spec_from_file_location("magnitude_reference", EXP / "reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_records():
    records = []
    for name in MODULES:
        original = SOURCE / "rtl/learning" / (name + ".sv")
        replica = EXP / "rtl" / (name + ".sv")
        if original.read_bytes() != replica.read_bytes():
            raise ValueError("source copy differs: " + name)
        info = original.stat()
        records.append(dict(path=(SOURCE_REL / "rtl/learning" / (name + ".sv")).as_posix(),
                            bytes=info.st_size, mtime_ns=info.st_mtime_ns))
    return records


def prepare():
    OUT.mkdir(exist_ok=False)
    (EXP / "rtl").mkdir(exist_ok=False)
    for name in MODULES:
        shutil.copy2(SOURCE / "rtl/learning" / (name + ".sv"), EXP / "rtl" / (name + ".sv"))
    evidence = [
        (SOURCE_REL / "rtl/learning/unified_fc_tp_w2_learning_slice.sv", 133, 144),
        (SOURCE_REL / "rtl/learning/unified_fc_tp_w2_learning_slice.sv", 181, 239),
        (SOURCE_REL / "rtl/learning/unified_fc_tp_gradient_rmw_path.sv", 95, 108),
        (SOURCE_REL / "rtl/learning/unified_fc_tp_gradient_request_adapter.sv", 59, 99),
        (SOURCE_REL / "rtl/learning/unified_tp_q015_magnitude_scaler.sv", 10, 14),
        (SOURCE_REL / "rtl/learning/unified_tp_q015_magnitude_residual_update.sv", 7, 11),
        (OUT_REL.parent / "fc_representative_a_r1/tb_fc_profile_measurement.sv", 1491, 1500),
        (OUT_REL.parent / "scnn_representative_t20_a_r1/tb_scnn_profile_measurement.sv", 1196, 1213),
    ]
    excerpts = []
    for path, first, last in evidence:
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        excerpts.append(dict(path=path.as_posix(), first_line=first, last_line=last,
                             text="\n".join(f"{i}: {lines[i-1]}" for i in range(first, last + 1))))
    write_json(OUT / "numeric_sources.json", excerpts)
    records = source_records()
    write_json(OUT / "source_inputs.json", records)
    fixture = reference().write_fixture(OUT)
    state = dict(status="PREPARED_NOT_RUN", time=stamp(), source_count=len(records),
                 fixture=fixture, scope="Real slice/reducer/adapter/scaler/RMW, synthetic contributions, no softmax/full network or shared-service owner arbiter.",
                 owner_scope="Active tensor/profile context, blocked busy cfg/release, legal drained release and FC/SCNN reconfiguration; no global reset between windows/cases.",
                 model_scope="Small protocol store commits writes at result handshake; no DDR/XPM or whole-core memory test.")
    write_json(OUT / "status.json", state)
    print(json.dumps(state, indent=2))


def software_test(tool_5048=False):
    log_path = OUT / (("software_5048_" if tool_5048 else "software_") + identity() + ".log")
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(EXP), "-p", "test_*.py", "-v"]
    begin = time.monotonic()
    with log_path.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    records = source_records()
    state = dict(status="SOFTWARE_PASS_RTL_NOT_RUN" if result.returncode == 0 else "SOFTWARE_FAIL",
                 time=stamp(), exit=result.returncode, wall_seconds=time.monotonic() - begin,
                 log=log_path.name, source_count_byte_equal=len(records), command=command)
    write_json(OUT / (log_path.stem + ".json"), state)
    write_json(OUT / ("status_5048.json" if tool_5048 else "status.json"), state)
    print(log_path.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(state, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


def diagnostics(log):
    categories = collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M))
    errors = re.findall(r"^%Error[^\n]*", log, re.M)
    info = dict(categories=dict(categories), warnings=sum(categories.values()),
                structural=sorted(STRUCTURAL.intersection(categories)), errors=errors)
    return info


def fields(line):
    return {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line)}


def parse_transcript(log):
    if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", log):
        raise ValueError("runtime diagnostic")
    ref = reference()
    windows = ref.fixtures()
    summaries = [fields(line) for line in re.findall(r"^MR_WINDOW (.*)$", log, re.M)]
    requests = [fields(line) for line in re.findall(r"^MR_REQUEST (.*)$", log, re.M)]
    results = [fields(line) for line in re.findall(r"^MR_RESULT (.*)$", log, re.M)]
    terminal = [fields(line) for line in re.findall(r"^MR_PASS (.*)$", log, re.M)]
    if len(summaries) != 8 or len(terminal) != 1:
        raise ValueError("missing/duplicate window or completion")
    expected_keys = [(w["case"], w["window"]) for w in windows]
    if [(row.get("case"), row.get("window")) for row in summaries] != expected_keys:
        raise ValueError("window order")
    if any((row.get("case"), row.get("window")) not in expected_keys for row in requests + results):
        raise ValueError("unexpected transaction window")
    for w, row in zip(windows, summaries):
        counts = ref.expected_counts(w)
        for key, value in counts.items():
            if row.get(key) != value:
                raise ValueError("oracle count: " + key)
        for key in ("reads", "responses", "results"):
            if row.get(key) != counts["requests"]:
                raise ValueError("transaction accounting: " + key)
        if row.get("profile") != w["profile"]["profile"] or row.get("pattern") != w["pattern"]:
            raise ValueError("profile/pattern mismatch")
        if any(row.get(k, 0) <= 0 for k in ("overlaps", "triples", "result_stalls", "request_stalls")):
            raise ValueError("retirement/backpressure not exercised")
        if w["pattern"] and row.get("read_stalls", 0) <= 0:
            raise ValueError("read backpressure not exercised")
        if (row.get("owner_blocked"), row.get("done"), row.get("max_credit")) != (3, 1, 1):
            raise ValueError("owner/drain/credit coverage")
        belongs = lambda item: (item["case"], item["window"]) == (w["case"], w["window"])
        actual_req = list(filter(belongs, requests))
        actual_result = list(filter(belongs, results))
        expected = [x for x in w["expected"] if x.delta]
        if len(actual_req) != len(expected) or len(actual_result) != len(expected):
            raise ValueError("missing/extra transaction")
        for req, result, ex in zip(actual_req, actual_result, expected):
            if (req.get("addr"), req.get("delta")) != (ex.addr, ex.delta):
                raise ValueError("request address/delta")
            if tuple(result.get(k) for k in ("addr", "code", "residual", "wc", "wr")) != (ex.addr, ex.code, ex.residual, ex.write_code, ex.write_residual):
                raise ValueError("independent writeback mismatch")
            if req.get("overlap") not in (0, 1) or req.get("triple") not in (0, 1) or req["triple"] > req["overlap"]:
                raise ValueError("invalid overlap flags")
        if sum(x["overlap"] for x in actual_req) != row["overlaps"] or sum(x["triple"] for x in actual_req) != row["triples"]:
            raise ValueError("overlap event accounting")
    end = terminal[0]
    totals = dict(windows=8, inputs=sum(x["contributions"] for x in summaries),
                  results=sum(x["results"] for x in summaries), zeros=sum(x["zeros"] for x in summaries),
                  overlaps=sum(x["overlaps"] for x in summaries), triples=sum(x["triples"] for x in summaries),
                  zero_retirements=sum(x["zero_retirements"] for x in summaries), releases=4)
    if end != totals:
        raise ValueError("terminal accounting")
    return dict(totals=totals, windows=summaries)


def linux_path(native):
    text = str(native)
    if len(text) < 3 or text[1] != ":":
        raise ValueError("expected Windows drive path")
    return "/mnt/" + text[0].lower() + text[2:].replace("\\", "/")


def run(flatten_lint=False, tool_5048=False):
    before = source_records()
    attempt_name = ("attempt_5048_" if tool_5048 else "attempt_") + identity()
    attempt = OUT / attempt_name
    attempt.mkdir(exist_ok=False)
    shutil.copytree(EXP / "rtl", attempt / "rtl")
    for name in (TB, "reference.py", "test_reference.py", "test_runner.py"):
        shutil.copy2(EXP / name, attempt / name)
    shutil.copy2(__file__, attempt / "runner.py")
    for name in ("fixture.txt", "initial.txt", "reference.json", "numeric_sources.json", "source_inputs.json"):
        shutil.copy2(OUT / name, attempt / name)
    state = dict(status="LOCAL_STARTING", start=stamp(), attempt=attempt_name,
                 sources=before, source_count=11, protected="Original accepted files are never edited",
                 scope="Synthetic magnitude-mode slice correctness/retirement only, not whole-core performance",
                 limits=dict(jobs=1, virtual_memory_kib_per_process=3145728,
                             verilator="5.048" if tool_5048 else "5.032"))
    runtime = linux_path(NATIVE_ROOT / "tools/verilator_5_048_local")
    executable = shlex.quote(runtime + "/bin/verilator") if tool_5048 else "verilator"
    environment_prefix = ("export VERILATOR_ROOT=" + shlex.quote(runtime + "/share/verilator") + "; ") if tool_5048 else ""

    def save():
        write_json(attempt / "status.json", state)
        write_json(OUT / ("status_5048.json" if tool_5048 else "status.json"), state)

    def stage(name, command, timeout):
        linux = linux_path(NATIVE_ROOT / OUT_REL / attempt_name)
        shell = ("set -e; cd " + shlex.quote(linux) + "; ulimit -v 3145728; "
                 + environment_prefix
                 + f"printf '%s\\n' \"$$\" > {name}_linux.pid; set +e; "
                 + "{ " + command + f"; }} > {name}.log 2>&1; rc=$?; "
                 + f"printf '%s\\n' \"$rc\" > {name}_linux.exit; exit \"$rc\"")
        args = ["wsl.exe", "-e", "sh", "-lc", shell]
        entry = dict(start=stamp(), command=args, exit=None, linux_exit=None)
        state[name] = entry; state["status"] = name.upper() + "_RUNNING"; save()
        start = time.monotonic()
        with (attempt / (name + "_launcher.log")).open("xb") as stream:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            entry["launcher_pid"] = process.pid; save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                entry.update(timeout=True, wall_seconds=time.monotonic() - start)
                state["status"] = "TIMEOUT_PROCESS_LEFT_ALIVE"; save()
                raise
        entry.update(exit=code, wall_seconds=time.monotonic() - start, end=stamp())
        exit_file = attempt / (name + "_linux.exit")
        if exit_file.exists():
            entry["linux_exit"] = int(exit_file.read_text().strip())
        pid_file = attempt / (name + "_linux.pid")
        if pid_file.exists():
            entry["linux_shell_pid"] = int(pid_file.read_text().strip())
        log_path = attempt / (name + ".log")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        entry["diagnostics"] = diagnostics(log); save()
        if code != 0 or entry["linux_exit"] != 0:
            raise RuntimeError(name + " failed")
        return log

    save()
    try:
        if tool_5048:
            previous = OUT / "attempt_20261019_220813_937557"
            compared = ["rtl/" + name + ".sv" for name in MODULES]
            compared += [TB, "reference.py", "fixture.txt", "initial.txt", "reference.json"]
            for name in compared:
                if (previous / name).read_bytes() != (attempt / name).read_bytes():
                    raise ValueError("input differs from 5.032 attempt3: " + name)
            state["same_inputs_as_5032_attempt3"] = compared
            save()
        environment = stage("environment", executable + " --version && test -x /usr/bin/time && ulimit -v", 180)
        version_pattern = r"Verilator 5\.048\b" if tool_5048 else r"Verilator 5\.032\b"
        if not re.search(version_pattern, environment) or not re.search(r"^3145728$", environment, re.M):
            raise ValueError("unexpected tool version or memory limit")
        inputs = " ".join("rtl/" + name + ".sv" for name in MODULES)
        options = " --timing --assert -j 1 -Wall -Wno-fatal --output-split 3000 -CFLAGS -O1 --top-module tb_magnitude_rmw_turnover --Mdir obj " + inputs + " " + TB
        if tool_5048:
            lint = stage("lint", "/usr/bin/time -v -o lint_resources.txt " + executable + " --lint-only --flatten" + options, 180)
            diag = diagnostics(lint)
            if diag["structural"] or diag["errors"]:
                raise ValueError("structural lint diagnostics; no binary build or simulation launched")
        operation = "--lint-only --flatten" if flatten_lint else "--binary"
        build = stage("build", "/usr/bin/time -v -o build_resources.txt " + executable + " " + operation + options, 1800)
        diag = diagnostics(build)
        if diag["structural"] or diag["errors"]:
            raise ValueError("structural build diagnostics; no simulation launched")
        if flatten_lint:
            state["status"] = "FLATTEN_LINT_CLEAN_RTL_NOT_RUN"
            return
        log = stage("run", "/usr/bin/time -v -o run_resources.txt obj/Vtb_magnitude_rmw_turnover +FIXTURE=fixture.txt +INITIAL=initial.txt", 300)
        state["checks"] = parse_transcript(log)
        state["status"] = "PASS_MAGNITUDE_RMW_TURNOVER_LOCAL"
    except Exception as error:
        if state["status"] != "TIMEOUT_PROCESS_LEFT_ALIVE":
            state["status"] = "FAIL"
        state["error"] = repr(error)
        raise
    finally:
        state["end"] = stamp()
        try:
            after = source_records()
            state["source_bytes_equal"] = len(after)
            state["source_metadata_preserved"] = after == before
            state["compiled_archive_bytes_equal"] = all(
                (attempt / "rtl" / (n + ".sv")).read_bytes() ==
                (SOURCE / "rtl/learning" / (n + ".sv")).read_bytes() for n in MODULES)
            if not state["source_metadata_preserved"] or not state["compiled_archive_bytes_equal"]:
                state.update(status="FAIL", error="source/archive changed")
        except Exception as error:
            state.update(status="FAIL", source_check_error=repr(error))
        save()
        print(json.dumps(state, indent=2))
    if state["status"] != "PASS_MAGNITUDE_RMW_TURNOVER_LOCAL":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "software-test", "run", "flatten-lint", "software-test-5048", "run-5048"))
    action = parser.parse_args().action
    {"prepare": prepare, "software-test": software_test, "run": run,
     "flatten-lint": lambda: run(flatten_lint=True),
     "software-test-5048": lambda: software_test(tool_5048=True),
     "run-5048": lambda: run(tool_5048=True)}[action]()
