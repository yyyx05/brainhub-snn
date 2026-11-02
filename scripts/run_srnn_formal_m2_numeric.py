#!/usr/bin/env python3
"""Opt-in, single-job SRNN real-size M2 numeric verification.

Forward mode uses a read-only TB parameter source. Online mode uses the real
production coordinator, Wout-only learner and shared forward/RMW store. Both
use the physical H40 array, but neither is final_composition_top or raw700 RTL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/paper_evidence/supplement_20261004/rtl/srnn_formal_m2"
WORK = ROOT / "work/supplement_20261004/srnn_formal_m2"
TB = ROOT / "tb/tb_srnn_formal_m2_numeric.sv"
TOPS = {"lif-probe": "tb_srnn_formal_m2_lif_probe", "context-probe": "tb_srnn_formal_m2_context_probe", "rearm-probe": "tb_srnn_formal_m2_rearm_probe", "forward": "tb_srnn_formal_m2_numeric", "online": "tb_srnn_formal_m2_numeric"}
MARKERS = {"lif-probe": "SRNN_M2_LIF_Q14_PASS", "context-probe": "SRNN_M2_CONTEXT_PROBE_PASS", "rearm-probe": "SRNN_M2_REARM_PROBE_PASS", "forward": "SRNN_FORMAL_M2_FORWARD_PASS", "online": "SRNN_FORMAL_M2_ONLINE_PASS"}
BOUNDARY = "production engine + adapter + real H40 array; read-only TB parameter memory; NOT final-top / online shared store / raw700 RTL"
ONLINE_BOUNDARY = "production coordinator + shared forward/RMW store + engine + real H40 array; frozen FP32 backbone input; NOT final-top / raw700 RTL"
CONTEXT_BOUNDARY = "production Wout-only learner interface; real B2T2 errors/traces, 5120 parameters; TB backing memory; NOT coordinator / shared-store integration / forward network"
REARM_BOUNDARY = "production Wout-only store wrapper + shared RAM + learner; real B2 errors/traces; cfg/abort/restore control only; NOT H40 forward / final-top"


def owned_output(path, root):
    path = path.resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"output must be a child of {root}")
    return path


def sources_for(mode):
    # The arithmetic reproducer elaborates one real LIF operator, not an old
    # downsized network. The main top elaborates the production 12-tile array.
    testbench = (ROOT / "tb/include" / ("srnn_formal_m2_" + mode.replace("-", "_") + ".svh")
                 if mode in ("context-probe", "rearm-probe") else TB)
    return ([ROOT / "rtl/tiles/unified_dual_role_lif10_arithmetic_candidate_v3.sv"]
            if mode == "lif-probe" else sorted((ROOT / "rtl").rglob("*.sv"))) + [testbench]


def build_command(mode, work, verilator, jobs=1, activity_monitor=None):
    if jobs < 1 or jobs > 12:
        raise ValueError("compiler jobs must be in 1..12")
    cflags = "-O1 -D_GLIBCXX_USE_CXX11_ABI=0" if os.name == "nt" else "-O1"
    return [str(verilator), "--binary", "--timing", "--assert", "-j", str(jobs),
            "-Wall", "-Wno-fatal", "--output-split", "3000", "--output-split-cfuncs", "200",
            "-CFLAGS", cflags, "--top-module", TOPS[mode],
            "--Mdir", work.as_posix(), "-f", (work / "sources.f").as_posix(),
            *(["-GWITH_COORDINATOR=1", "+define+SRNN_FORMAL_M2_ONLINE"] if mode == "online" else []),
            *([str(activity_monitor)] if activity_monitor else [])]


def decode_marker(mode, stdout):
    rows = [line for line in stdout.splitlines() if line.startswith(MARKERS[mode] + " ")]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {MARKERS[mode]} marker, got {len(rows)}")
    values = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", rows[0])}
    if mode == "lif-probe":
        status = "production_lif_formal_q14_minimal_case_pass"
    elif mode == "context-probe":
        if any(values.get(key) != expected for key, expected in {
                "windows": 2, "held_start_cycles": 64, "held_start_accepts": 0,
                "configs": 2, "starts": 2, "gradients": 10240, "commits": 2,
                "contributions": 20480, "store_checks": 10240}.items()):
            raise ValueError("incomplete learner context protocol probe")
        status = "learner_requires_per_window_config_reproduced_not_coordinator_pass"
    elif mode == "rearm-probe":
        required = dict(windows=6, commits=6, detach=6, aborts=3, auto_configs=9,
                        external_configs=6, invalid_cfg=1, cfg_competition=1,
                        cfg_pending_during_rearm=1, initial_cfg_off_cfg=1,
                        learning_off_cfg=1, clean_abort=1, dirty_abort=1, restore=1,
                        first_read_fault=1, legal_cfg_error_clear=1,
                        recovery_blocked_cycles=64, idle_no_rearm_cycles=64)
        if any(values.get(key) != expected for key, expected in required.items()):
            raise ValueError("incomplete wrapper rearm control regression")
        status = "production_wrapper_rearm_cfg_abort_restore_pass_not_forward"
    elif mode == "online":
        status = "production_online_reference_pass_formal_q14_gap" if values.get("formal_pre_differences", 0) else "production_online_and_formal_q14_reference_pass_not_finaltop"
    elif values.get("formal_pre_differences", 0):
        status = "production_reference_pass_formal_q14_gap"
    else:
        status = "production_forward_and_formal_q14_reference_pass_not_online_or_finaltop"
    return status, values


def partial_progress(stdout):
    result = {"completed_windows": []}
    for line in stdout.splitlines():
        if line.startswith("SRNN_FORMAL_M2_ONLINE_STEP "):
            result["completed_windows"].append({key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line)})
        if line.startswith("M2_LEARNER_READY "):
            result["learner_ready"] = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line)}
        if "M2_NO_PROGRESS " in line:
            result["failure_kind"] = "testbench_no_progress_watchdog_not_a_DUT_numeric_assertion"
            result["watchdog"] = line.split("M2_NO_PROGRESS ", 1)[1]
    return result


def expected_id_failure(stdout, kind):
    injection = "M2_ID_INJECTION kind=2 busy=0 done=1" if kind == "done" else "M2_ID_INJECTION kind=1 busy=1"
    return injection in stdout and any("M2_COORDINATOR_ERROR " in line and
        re.search(r"\bstate=6\b", line) and re.search(r"\btimestep_id=1\b", line)
        for line in stdout.splitlines())


def execute(command, log, timeout):
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=tool_environment(),
                                       stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=stream, stderr=subprocess.STDOUT, check=False)
            else:
                process.kill()
            process.wait()
            return {"returncode": None, "timeout": True, "elapsed_seconds": time.perf_counter() - started}
    return {"returncode": returncode, "timeout": False,
            "elapsed_seconds": time.perf_counter() - started}


def tool_environment():
    env = os.environ.copy()
    if os.name == "nt":
        env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
        env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TOPS, default="forward")
    parser.add_argument("--learning", choices=("off", "on"), default="off")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--reuse-executable", type=Path)
    parser.add_argument("--verilator", type=Path, default=Path("C:/msys64/mingw64/bin/verilator_bin.exe"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--activity-monitor", type=Path)
    parser.add_argument("--run", action="store_true", help="default only prints commands; no directories or subprocesses")
    parser.add_argument("--inject-timestep-id", choices=("stream", "done"), help="online-only negative test through TB public response signals")
    args = parser.parse_args(argv)
    if args.mode != "online" and args.learning != "off":
        parser.error("--learning on requires --mode online")
    if args.inject_timestep_id and args.mode != "online":
        parser.error("ID injection requires --mode online")
    output = owned_output(args.output_dir, EVIDENCE)
    work = owned_output(args.work_dir or WORK / output.name, WORK)
    if args.activity_monitor:
        args.activity_monitor = args.activity_monitor.resolve()
        if not args.activity_monitor.is_relative_to(ROOT) or not args.activity_monitor.is_file():
            raise ValueError("activity monitor must exist inside the workspace")
    command = build_command(args.mode, work, args.verilator, args.jobs, args.activity_monitor)
    executable = args.reuse_executable.resolve() if args.reuse_executable else work / ("V" + TOPS[args.mode] + (".exe" if os.name == "nt" else ""))
    run_command = [str(executable)]
    fixture_info = None
    if args.mode in ("forward", "online", "context-probe", "rearm-probe"):
        if args.fixture is None:
            parser.error("this mode requires --fixture")
        fixture = args.fixture.resolve()
        fixture_info = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
        if (fixture_info["batch"], fixture_info["backbone"], fixture_info["hidden"], fixture_info["classes"]) != (2, 450, 256, 20):
            raise ValueError("only real B2 450->256->20 fixture is supported")
        for name in ("input_q14.mem", "expected_pre.mem", "expected_membrane.mem", "expected_spike.mem",
                     "expected_trace.mem", "expected_feature.mem", "expected_logit.mem", "formal_pre.mem"):
            if not (fixture / name).is_file():
                raise FileNotFoundError(fixture / name)
        run_command += [f"+FIXTURE={fixture.as_posix()}", f"+STEPS={fixture_info['steps']}"]
        if args.mode in ("context-probe", "rearm-probe"):
            if fixture_info["steps"] != (2 if args.mode == "context-probe" else 8):
                raise ValueError("context/rearm probe requires its original B2T2/B2T8 prefix")
            for key in ("error", "trace", "gradient", "wout_after", "residual"):
                if not (fixture / f"expected_on_{key}.mem").is_file():
                    raise FileNotFoundError(fixture / f"expected_on_{key}.mem")
        if args.mode == "online":
            prefix = "expected_on" if args.learning == "on" else "expected"
            for key in ("pre", "membrane", "spike", "trace", "feature", "logit", "wout_before", "wout_after", "residual", "gradient"):
                if not (fixture / f"{prefix}_{key}.mem").is_file():
                    raise FileNotFoundError(fixture / f"{prefix}_{key}.mem")
            run_command += [f"+LEARNING={int(args.learning == 'on')}"]
    if args.inject_timestep_id:
        run_command += ["+INJECT_DONE_TIMESTEP_ID_ERROR" if args.inject_timestep_id == "done" else "+INJECT_TIMESTEP_ID_ERROR"]
    report = dict(status="dry_run", mode=args.mode,
                  boundary=REARM_BOUNDARY if args.mode == "rearm-probe" else CONTEXT_BOUNDARY if args.mode == "context-probe" else ONLINE_BOUNDARY if args.mode == "online" else BOUNDARY if args.mode == "forward" else "production 10-lane LIF arithmetic only; lane0 active; NOT a network run",
                  learning="on_interface_only" if args.mode in ("context-probe", "rearm-probe") else args.learning,
                  id_injection=args.inject_timestep_id,
                  compile_command=None if args.reuse_executable else command,
                  simulation_command=run_command, fixture=str(args.fixture) if args.fixture else None,
                  executable=str(executable), output_dir=str(output), work_dir=str(work),
                  online_learning_tested=False, shared_store_tested=False, final_top_tested=False)
    if fixture_info:
        report["input_source"] = fixture_info.get("source")
        if args.mode == "online" and isinstance(fixture_info.get("source"), dict) and "observed_file" in fixture_info["source"]:
            report["boundary"] = "production M2 coordinator + shared store + H40; actual observed backbone spikes replayed across a file boundary; NOT same-DUT raw700-to-M2 or final-top"
    if not args.run:
        print(json.dumps(report, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"

    def save():
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    report["status"] = "started"
    save()
    if args.reuse_executable:
        if not executable.is_relative_to(WORK.resolve()) or not executable.is_file():
            raise ValueError("reuse executable must exist inside this task's work directory")
        report["reuse_note"] = "Explicit reuse; caller must rebuild after any relevant TB/RTL source change. No source identity claim."
    else:
        work.mkdir(parents=True, exist_ok=False)
        (work / "sources.f").write_text("\n".join('"' + x.as_posix() + '"' for x in sources_for(args.mode)) + "\n", encoding="utf-8")
        print(f"Compiling {TOPS[args.mode]} with -j{args.jobs}; log: {output / 'build.log'}", flush=True)
        report["build"] = execute(command, output / "build.log", 900)
        save()
        if report["build"]["returncode"] != 0:
            report["status"] = "build_failed"
            save()
            print(json.dumps(report, indent=2))
            return 1
    print(f"Simulating {TOPS[args.mode]}; log: {output / 'simulation.log'}", flush=True)
    report["simulation"] = execute(run_command, output / "simulation.log", 600)
    stdout = (output / "simulation.log").read_text(encoding="utf-8", errors="replace")
    report["progress"] = partial_progress(stdout)
    if args.inject_timestep_id:
        passed = report["simulation"]["returncode"] not in (0, None) and expected_id_failure(stdout, args.inject_timestep_id)
        report["status"] = "expected_active_timestep_id_error_detected" if passed else "id_error_negative_test_failed"
        save()
        print(json.dumps(report, indent=2))
        return 0 if passed else 1
    if report["simulation"]["returncode"] != 0:
        report["status"] = "simulation_failed"
        save()
        print(json.dumps(report, indent=2))
        return 1
    try:
        report["status"], report["observed"] = decode_marker(args.mode, stdout)
        if fixture_info and args.mode not in ("context-probe", "rearm-probe"):
            report["reference_differences"] = fixture_info["current_vs_formal_q14"]
            expected = fixture_info["steps"]
            if (report["observed"]["steps"], report["observed"]["hidden_checks"], report["observed"]["logit_checks"]) != (expected, expected * 512, expected * 40):
                raise ValueError("simulation did not cover all fixture timesteps")
            if report["observed"]["formal_pre_differences"] != fixture_info["current_vs_formal_q14"]["pre"]["count"]:
                raise ValueError("observed formal pre-state difference count disagrees with reference")
            if args.mode == "online":
                reference = fixture_info["reference_counters"]["production_" + args.learning]
                for observed_key, reference_key in (("rmw_reads", "rmw"), ("code_writes", "code_writes"), ("residual_writes", "residual_writes")):
                    if report["observed"][observed_key] != sum(row[reference_key] for row in reference):
                        raise ValueError(f"{observed_key} differs from independent integer reference")
                report["shared_store_tested"] = True
                report["online_learning_tested"] = args.learning == "on"
                report["reference_counters"] = reference
    except (ValueError, KeyError) as exc:
        report["status"], report["error"] = "invalid_completion", str(exc)
        save()
        print(json.dumps(report, indent=2))
        return 1
    save()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
