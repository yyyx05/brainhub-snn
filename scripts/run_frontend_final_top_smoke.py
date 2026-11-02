"""Bounded DecisionEngine-to-real-final-top replay; no EDA, SSH or model build."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_fc_final_counts import ROOT, build_settings, generate_tb, resolve_verification_store

COMPONENT = ROOT / "software/autonomous_reconfiguration"
sys.path.insert(0, str(COMPONENT))
from snn_reconfig.decision import DecisionEngine, load_json  # noqa: E402
from snn_reconfig.final_top_smoke_bridge import FinalTopReplayTransport, check_transcript  # noqa: E402

OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/software/frontend_final_top_smoke"
TOP = "tb_frontend_final_top_smoke"
EXAMPLES = ("nmnist_query.json", "shd_query.json", "smnist_unsupported_query.json",
            "dvsgesture_query.json", "nmnist_query.json")
OVERRIDES = {
    "PARAM_DEPTH": "128", "MAX_FEATURES": "8", "MAX_CLASSES": "2",
    "MAX_BATCH": "4", "MAX_TIMESTEPS": "20", "SPATIAL": "4", "POOLED_SPATIAL": "2",
    "MAX_INPUT_CHANNELS": "1", "MAX_OUTPUT_CHANNELS": "1",
    "SRNN_M2_PARAM_DEPTH": "128", "SRNN_BACKBONE_PARAM_DEPTH": "128",
    "SRNN_BACKBONE_PARAM_ADDR_WIDTH": "24", "SRNN_M2_BATCH_MAX": "2",
    "SRNN_M2_TIMESTEP_MAX": "2", "SRNN_M2_BACKBONE_MAX": "4",
    "SRNN_M2_HIDDEN_MAX": "4", "SRNN_M2_CLASS_MAX": "4",
    "SRNN_BATCH_MAX": "2", "SRNN_TIMESTEP_MAX": "2",
}
LIMITS = [
    "Actual unmodified unified_h40_final_composition_top_v1 plus actual shared parameter store, not a mock or a subtop.",
    "Opcode is mapped by software to native config/layer/group/graph requests; final top has no opcode/profile-ready decoder port.",
    "Synthetic FC two-neuron and SRNN two-neuron feedforward backbone worksets; no catalog DMA, trained weights, recurrent group, M2 dispatch or H450+M2 accuracy claim.",
    "SRNN captures into the real phase tape then native phase-aborts the undelivered tape; this is not an M2 completion.",
    "SCNN B2T2 eight 2x2 single-channel descriptors per frame and two-class tail; zero weights/features, TB feature/state services, not VGG9 assets or DDR.",
    "Full native descriptor capture, state+trace restore/spill and response backpressure are checked only at this small geometry and all-ones maintenance masks.",
    "Learning controls are off; zero writes are checked at the shared store and zero commits only at SCNN o_commit_pulse. FC/SRNN commit outputs are not monitored; no learning-publication claim.",
    "Feature request stall cycles do not prove request-address stability or feature-update backpressure. Four frames are checked via protocol fields, not an independent frame-done counter.",
    "No optimizer, real asset loading, board, timing/PPA or full-model evidence.",
    "Version alignment is model/parameter visibility, not a profile-ready or learning-publication handshake.",
]


def simulation_sources(tb: Path) -> list[Path]:
    return list(dict.fromkeys([
        *sorted(ROOT.glob("rtl/**/*.sv")), resolve_verification_store(ROOT), tb,
    ]))


def prepare(output: Path) -> tuple[FinalTopReplayTransport, list[dict]]:
    engine = DecisionEngine.from_files(COMPONENT / "catalog/hardware_profiles.json",
                                       COMPONENT / "catalog/fingerprint_prototypes.json")
    transport = FinalTopReplayTransport()
    sequence = []
    for example in EXAMPLES:
        decision = engine.decide(load_json(COMPONENT / "examples" / example), "off")
        before = len(transport.commands)
        if decision["status"] == "hardware_profile_selected":
            transport.plan(decision)
        else:
            try:
                transport.plan(decision)
            except ValueError:
                pass
            else:
                raise ValueError("rejection was accepted by transport")
        added = len(transport.commands) - before
        if decision["status"] != "hardware_profile_selected" and added:
            raise ValueError("rejection emitted DUT commands")
        sequence.append(dict(example=example, status=decision["status"], commands_added=added,
                             selected_profile=decision["selected_profile"]))
    transport.write_program(output / "program.txt")
    (output / "frontend_plan.json").write_text(json.dumps(dict(
        status="simulation_replay_pending", sequence=sequence, selections=transport.selections,
        commands=transport.commands), indent=2) + "\n", encoding="utf-8")
    return transport, sequence


def process_tree_rss(pid: int) -> int:
    """Aggregate live working sets of this runner's command and descendants."""
    if os.name != "nt":
        parents = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    parents[int(entry.name)] = (int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
                except (OSError, ValueError, IndexError):
                    continue
    else:
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                        ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                        ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
                        ("exe", wintypes.WCHAR * 260)]

        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "quota_peak_paged", "quota_paged",
                                                   "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise OSError("cannot monitor process tree")
        parents = {}
        entry = Entry(); entry.size = ctypes.sizeof(entry)
        try:
            available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while available:
                parents[entry.pid] = (entry.parent, 0)
                available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
    selected = {pid}
    while True:
        children = {child for child, (parent, _) in parents.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    total = 0
    for child in selected:
        if os.name != "nt":
            total += parents.get(child, (0, 0))[1]
        else:
            handle = kernel.OpenProcess(0x1010, False, child)
            if not handle:
                continue  # Processes may exit between enumeration and sampling.
            try:
                counters = Counters(); counters.size = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    total += counters.working
            finally:
                kernel.CloseHandle(handle)
    return total


def run_bounded(command: list[str], output: Path, stem: str, env: dict, timeout: int) -> dict:
    record = dict(command=command, cwd=str(ROOT), timeout_seconds=timeout,
                  memory_limit_bytes=2_000_000_000, peak_tree_rss_bytes=0)
    path = output / f"{stem}_command.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    with (output / f"{stem}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        while process.poll() is None:
            try:
                rss = process_tree_rss(process.pid)
                record["peak_tree_rss_bytes"] = max(record["peak_tree_rss_bytes"], rss)
                reason = "memory_limit" if rss >= record["memory_limit_bytes"] else (
                    "time_limit" if time.monotonic() - started >= timeout else None)
            except OSError as exc:
                reason = "resource_monitor_unavailable"
                record["monitor_error"] = str(exc)
            if reason:
                record["stopped_reason"] = reason
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=log, stderr=subprocess.STDOUT, check=False,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    import signal
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.2)
        record["exit_code"] = process.wait()
    record["seconds"] = round(time.monotonic() - started, 3)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="fresh child of this task's evidence directory")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("run-name must be a simple fresh directory name")
    output = OUTPUT / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    transport, sequence = prepare(output)
    report = dict(status="NOT_RUN", evidence_kind="actual_final_top_native_replay",
                  actual_dut="unified_h40_final_composition_top_v1", learning_enable=False,
                  catalog_dma_status="NOT_COVERED", m2_status="NOT_COVERED",
                  learning_publication_status="NOT_COVERED", rejected_task_commands=sequence[2]["commands_added"],
                  limits=LIMITS, parameter_overrides=OVERRIDES)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tb = output / (TOP + ".sv")
    generate_tb(tb, top=TOP, body=ROOT / "tb/tb_frontend_final_top_smoke.sv", parameter_overrides=OVERRIDES)
    # Same elaboration pattern as the existing small scnn_frame_service runner;
    # only the selected top and bounded parameterization are compiled.
    source_list = output / "sources.f"
    source_list.write_text("".join('"' + p.as_posix() + '"\n'
                                    for p in simulation_sources(tb)), encoding="ascii")
    if not args.prepare_only:
        try:
            build, exe, env = build_settings(output / "obj", source_list, jobs=args.jobs, top=TOP)
            # Avoid generating thousands of tiny C++ translation units.
            build[build.index("--output-split") + 1] = "12000"
            build += ["--output-split-cfuncs", "0"]
            print("Focused bridge/parser/resource-monitor unit tests first", flush=True)
            report["unit_tests"] = run_bounded([
                sys.executable, "-m", "unittest", "discover", "-s",
                str(COMPONENT / "tests"), "-p", "test_final_top_smoke_bridge.py", "-v",
            ], output, "unit_tests", env, 20)
            if report["unit_tests"]["exit_code"] != 0:
                report["status"] = "FAIL"
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(dict(status="FAIL", report=str(report_path))))
                return 1
            print(f"Bounded small final-top build: jobs={args.jobs}, <=570s, <2GB tree RSS", flush=True)
            report["status"] = "BUILD_RUNNING"
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["build"] = run_bounded(build, output, "build", env, 570)
            if report["build"]["exit_code"] != 0 or report["build"].get("stopped_reason"):
                report["status"] = "BLOCKED" if report["build"].get("stopped_reason") else "FAIL"
            else:
                report["status"] = "SIMULATION_RUNNING"
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print("One short simulation, no automatic retry", flush=True)
                report["simulation"] = run_bounded([str(exe), "+program=" + (output / "program.txt").as_posix()],
                                                   output, "simulation", env, 30)
                try:
                    report["dut"] = check_transcript((output / "simulation.log").read_text(encoding="utf-8"), transport.commands)
                except (ValueError, IndexError) as exc:
                    report["dut"] = dict(status="FAIL", parse_error=str(exc))
                report["status"] = "PASS" if report["simulation"]["exit_code"] == 0 and report["dut"]["status"] == "PASS" else "FAIL"
        except (OSError, ValueError) as exc:
            report.update(status="BLOCKED", error=str(exc))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], report=str(report_path))))
    return 0 if report["status"] in ("PASS", "NOT_RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
