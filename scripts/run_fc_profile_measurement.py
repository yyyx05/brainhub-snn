#!/usr/bin/env python3
"""Passive FC B16 paired profiling; reuse fixtures, never generate/train models."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import time

try:
    from .run_fc_final_counts import ROOT, build_settings, generate_tb, resolve_verification_store
except ImportError:
    from run_fc_final_counts import ROOT, build_settings, generate_tb, resolve_verification_store

OUTPUT = Path("results/paper_evidence/supplement_20261004/performance/current_source_measurements/fc")
FIXTURES = Path("results/paper_evidence/supplement_20261004/algorithm/fc_final_counts_reference")
SOURCE_LIST = Path("build/ip/unified_snn_native_ip_source_20261009_r2/reachable_sources.f")
BODY = Path("tb/include/fc_final_counts_body.svh")
MONITOR = Path("tb/include/fc_profile_measurement_monitor.svh")
TOP = "tb_fc_profile_measurement"
STAGES = ("outside", "parameter_load", "initial_refill", "initial_state", "runtime",
          "learning_tail_wait", "runtime_refill", "learning_configuration")
BUS_FIELDS = ("parameter_input", "store_load", "forward_request", "forward_response",
              "learning_request", "learning_response", "learning_write", "code_write",
              "residual_write", "state_write", "trace_write", "state_read_request",
              "state_read_response", "trace_read_request", "trace_read_response",
              "source_group", "source_events", "observation", "student_trace", "logit")
BUS_WIDTHS = {"store_load": 32, "forward_response": 16, "learning_response": 32,
              "code_write": 16, "residual_write": 16, "state_read_response": 320,
              "trace_read_response": 320, "student_trace": 17, "logit": 16}


def project_path(path: Path, root: Path = ROOT) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"dependency must remain in the project: {path}")
    return resolved


def same_file_data(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk = a.read(1024 * 1024)
            if chunk != b.read(1024 * 1024):
                return False
            if not chunk:
                return True


def structural_warnings(text: str) -> list[str]:
    return [name for name in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN")
            if "%Warning-" + name in text]


def inspect_pair(on: Path, off: Path, steps: int) -> dict:
    refs = {}
    for arm, directory in (("on", on), ("off", off)):
        ref = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
        if (ref.get("status") != "PASS" or ref.get("learning") is not (arm == "on")
                or ref.get("steps") != steps or ref.get("batch") != 16
                or ref.get("batches", 1) != 1 or ref.get("hidden") != 160
                or ref.get("classes") != 10 or not steps <= ref.get("source_steps", 0) <= 99):
            raise ValueError(f"{arm}: requires an existing single B16 H160 C10 fixture, T<=99")
        if ref.get("numeric", {}).get("scale_to_q6_q16") != 47303:
            raise ValueError(f"{arm}: scale differs from the unchanged TB stimulus")
        for name in ("input_counts", "parameters", "labels", "pre", "trace", "spike",
                     "dot", "q6", "w2_after", "residual_after"):
            path = directory / (name + ".mem")
            metadata = ref.get("files", {}).get(path.name, {})
            if not path.is_file() or path.stat().st_size != metadata.get("bytes"):
                raise ValueError(f"missing/truncated existing fixture: {path}")
        refs[arm] = ref
    for key in ("samples", "source_steps", "numeric", "w2_parameter_base"):
        if refs["on"].get(key) != refs["off"].get(key):
            raise ValueError(f"paired fixtures differ in {key}")
    for name in ("input_counts.mem", "labels.mem", "parameters.mem"):
        if not same_file_data(on / name, off / name):
            raise ValueError(f"paired initial/input payload differs: {name}")
    samples = refs["on"]["samples"]
    if (len(samples) != 16 or len({s["dataset_index"] for s in samples}) != 16
            or any(not 0 <= s["valid_timesteps"] <= refs["on"]["source_steps"] for s in samples)):
        raise ValueError("invalid sample IDs/valid lengths")
    return {"batch": 16, "batches": 1, "steps": steps,
            "source_steps": refs["on"]["source_steps"], "samples": samples,
            "full_sample": steps == refs["on"]["source_steps"],
            "definition": "Complete supplied first-saccade 1ms count frames including batch padding; no forced T100; not full Test.",
            "expected": {arm: {"hidden_checks": steps * 16 * 160,
                                "dot_checks": steps * 16 * 10, "q6_checks": steps * 16 * 10,
                                "commits": steps if arm == "on" else 0} for arm in refs}}


def instrument_body(body: str, monitor: str) -> str:
    # Insert only TB-private assignments, without adding events/delays or DUT writes.
    replacements = (
        ("        load_parameters();", "        fc_measure_stage = 1;\n        load_parameters();"),
        ("        refill(0); refill(1);", "        fc_measure_stage = 2;\n        refill(0); refill(1);"),
        ("        clear_states();\n        init_cycles", "        fc_measure_stage = 3;\n        clear_states();\n        init_cycles"),
        ("        if (learning) begin\n            @(negedge i_clk); i_learning_fc_cfg_valid",
         "        fc_measure_stage = 7;\n        if (learning) begin\n            @(negedge i_clk); i_learning_fc_cfg_valid"),
        ("        for (batch_index = 0;", "        fc_measure_stage = 4;\n        for (batch_index = 0;"),
        ("                learn_start = cycles;", "                fc_measure_stage = 5;\n                learn_start = cycles;"),
        ("                learning_cycles += cycles-learn_start;",
         "                learning_cycles += cycles-learn_start;\n                fc_measure_stage = 4;"),
        ("            if (learning) refill(1);",
         "            if (learning) begin\n                fc_measure_stage = 6;\n                refill(1);\n                fc_measure_stage = 4;\n            end"),
        ("        $finish;", "        fc_measure_complete = 1;\n        $display(\"FC_PROFILE_MEASUREMENT_PASS learning=%0d steps=%0d\", learning, total_steps);\n        $finish;"),
    )
    for original, replacement in replacements:
        if body.count(original) != 1:
            raise ValueError(f"original FC TB marker changed: {original!r}")
        body = body.replace(original, replacement, 1)
    return monitor + "\n" + body


def simulation_sources(root: Path, source_list: Path, tb: Path, activity: bool) -> list[Path]:
    entries = [line.strip().strip('"') for line in source_list.read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith(("#", "//"))]
    sources = [project_path(Path(entry), root) for entry in entries]
    if (len(sources) != 105 or len(set(sources)) != 105
            or any(not p.is_file() or p.suffix != ".sv" or not p.is_relative_to(root / "rtl") for p in sources)
            or root / "rtl/top/unified_h40_final_composition_top_v1.sv" not in sources):
        raise ValueError("use the existing 105-file production closure, not a glob/history tree")
    store = resolve_verification_store(root)
    if store in sources:
        raise ValueError("verification store is not a production closure member")
    sources += [store, tb]
    if activity:
        sources.append(root / "tb/monitors/supplement_h40_activity_monitor.sv")
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("simulation dependency is missing")
    return sources


def parse_log(text: str, arm: str, scope: dict, activity: bool = False) -> dict:
    if re.search(r"%Error|Assertion failed|\$fatal|Aborting", text):
        raise ValueError("simulation failed; retain the raw log")
    def rows(prefix):
        return [{k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
                for line in text.splitlines() if line.startswith(prefix + " ")]
    legacy, complete = rows("FC_FINAL_COUNTS_PASS"), rows("FC_PROFILE_MEASUREMENT_PASS")
    if len(legacy) != 1 or len(complete) != 1:
        raise ValueError("both original and instrumented completion markers are required")
    expected = scope["expected"][arm]
    if (legacy[0].get("learning") != int(arm == "on") or legacy[0].get("batch") != 16
            or legacy[0].get("batches") != 1 or legacy[0].get("steps") != scope["steps"]
            or complete[0] != {"learning": int(arm == "on"), "steps": scope["steps"]}
            or any(legacy[0].get(key) != value for key, value in expected.items())):
        raise ValueError("completion scope/check counts differ from the paired fixture")
    raw_stages = rows("FC_PROFILE_STAGE")
    if len(raw_stages) != 8 or {r.get("id") for r in raw_stages} != set(range(8)):
        raise ValueError("missing or duplicate stage counters")
    stages = {STAGES[r["id"]]: r for r in raw_stages}
    windows = rows("FC_PROFILE_WINDOW")
    if len(windows) != expected["commits"] or [w.get("index") for w in windows] != list(range(len(windows))):
        raise ValueError("incomplete commit windows")
    required = {*BUS_FIELDS, "id", "cycles", "window_busy", "work_busy", "work_overlap_forward",
                "work_only", "work_other", "forward_scope", "refill_busy", "commit_present",
                "commit_stall", "state_write_mask_bytes", "trace_write_mask_bytes"}
    for row in raw_stages:
        if not required <= row.keys():
            raise ValueError("incomplete stage fields")
        if any(value < 0 for value in row.values()):
            raise ValueError("negative stage counter")
        if row["work_busy"] != row["work_overlap_forward"] + row["work_only"] + row["work_other"]:
            raise ValueError("learning work overlap accounting is inconsistent")
        if any(row[key] > row["cycles"] for key in ("window_busy", "work_busy", "forward_scope", "refill_busy")):
            raise ValueError("busy counter exceeds stage clocks")
        row["payload_bits"] = {key: row[key] * width for key, width in BUS_WIDTHS.items()}
        row["payload_bits"]["learning_write"] = row["code_write"] * 16 + row["residual_write"] * 16
        row["payload_bits"]["state_write"] = row["state_write_mask_bytes"] * 8
        row["payload_bits"]["trace_write"] = row["trace_write_mask_bytes"] * 8
    for window in windows:
        if (window["observation"] < 0 or window["done"] < window["observation"]
                or window["commit_first"] < window["done"] or window["commit"] < window["commit_first"]):
            raise ValueError("missing/inconsistent observation-to-commit timestamps")
        window["observation_to_commit_clocks"] = window["commit"] - window["observation"]
        window["done_to_commit_clocks"] = window["commit"] - window["done"]
        window["commit_valid_to_accept_clocks"] = window["commit"] - window["commit_first"]
        window["update_emit_to_done_clocks"] = (window["done"] - window["update_start"]
                                                  if window["update_start"] >= 0 else None)
        window["last_parameter_write_to_commit_clocks"] = (window["commit"] - window["last_write"]
                                                            if window["last_write"] >= 0 else None)
        window["work_only"] = window["work_busy"] - window["work_overlap_forward"] - window["work_other"]
        if window["work_only"] < 0:
            raise ValueError("negative per-window learning-only work")
    totals = {key: sum(row[key] for row in raw_stages) for key in BUS_FIELDS}
    if (totals["forward_request"] != legacy[0]["forward_reads"]
            or totals["learning_request"] != legacy[0]["learning_reads"]
            or totals["code_write"] != legacy[0]["code_writes"]
            or totals["residual_write"] != legacy[0]["residual_writes"]
            or totals["store_load"] != 371520 or totals["parameter_input"] != 371520
            or totals["state_write"] != 320 or totals["trace_write"] != 320
            or totals["forward_response"] != totals["forward_request"]
            or totals["learning_response"] != totals["learning_request"]):
        raise ValueError("passive bus counters do not match original checks/initialization")
    runtime_cycles = sum(stages[key]["cycles"] for key in STAGES[4:7])
    if runtime_cycles != legacy[0]["cycles"]:
        raise ValueError("runtime interval differs from the unchanged original TB")
    if arm == "off" and (windows or totals["learning_request"] or totals["learning_write"]
                         or sum(r["window_busy"] + r["work_busy"] for r in raw_stages)):
        raise ValueError("learning-off must not execute learning work")
    payload_totals = {key: sum(row["payload_bits"][key] for row in raw_stages)
                      for key in raw_stages[0]["payload_bits"]}
    result = {"status": "PASS", "arm": arm, "scope": scope, "stages": stages,
              "windows": windows, "bus_handshakes": totals, "legacy": legacy[0],
              "bus_payload_bits": payload_totals,
              "runtime_cycles": runtime_cycles,
              "projected_runtime_ms_at_15ns": runtime_cycles * 15e-6,
              "amortized_cycles_per_supplied_sample": runtime_cycles / 16 if scope["full_sample"] else None,
              "projected_samples_per_second_at_15ns": 16 / (runtime_cycles * 15e-9) if scope["full_sample"] and runtime_cycles else None,
              "scope_limit": "Fixed B16 supplied samples; setup excluded, conservative W2 refill retained; 15ns projection not board throughput; behavioral simulation memories, not XPM/STA validation.",
              "board_measured": False, "full_dataset_accuracy": False}
    if activity:
        try:
            from .summarize_h40_activity import parse_activity
        except ImportError:
            from summarize_h40_activity import parse_activity
        result["h40_activity"] = parse_activity(text, "FC_PROFILE_MEASUREMENT_PASS")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--steps", type=int, choices=range(1, 100), default=99)
    parser.add_argument("--fixture-on", type=Path, default=FIXTURES / "t99_on")
    parser.add_argument("--fixture-off", type=Path, default=FIXTURES / "t99_off")
    parser.add_argument("--source-list", type=Path, default=SOURCE_LIST)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verilator", default="/root/autodl-tmp/verilator_5_048_install/bin/verilator")
    parser.add_argument("--activity-monitor", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("positive --jobs and a simple new --run-name are required")
    fixtures = {arm: project_path(path) for arm, path in (("on", args.fixture_on), ("off", args.fixture_off))}
    scope = inspect_pair(fixtures["on"], fixtures["off"], args.steps)
    source_list = project_path(args.source_list)
    out = ROOT / OUTPUT / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    body_path = out / "fc_profile_measurement_body.svh"
    body_path.write_text(instrument_body((ROOT / BODY).read_text(encoding="utf-8"),
                                         (ROOT / MONITOR).read_text(encoding="utf-8")), encoding="ascii")
    tb = out / (TOP + ".sv")
    generate_tb(tb, top=TOP, body=body_path)
    sources = simulation_sources(ROOT, source_list, tb, args.activity_monitor)
    rom = ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
    if not rom.is_file():
        raise FileNotFoundError(f"missing original surrogate ROM: {rom}")
    filelist = out / "sources.f"
    filelist.write_text("".join('"' + p.as_posix() + '"\n' for p in sources), encoding="ascii")
    build, exe, env = build_settings(out / "obj", filelist, verilator=args.verilator, jobs=args.jobs, top=TOP)
    runs = {arm: [str(exe), "+FIXTURE=" + fixtures[arm].relative_to(ROOT).as_posix(),
                  f"+STEPS={args.steps}", f"+LEARNING={int(arm == 'on')}", "+BATCHES=1"] for arm in fixtures}
    dependencies = [ROOT / "scripts/run_fc_profile_measurement.py", ROOT / "scripts/run_fc_final_counts.py",
                    ROOT / BODY, ROOT / MONITOR, source_list,
                    rom]
    if args.activity_monitor:
        dependencies.append(ROOT / "scripts/summarize_h40_activity.py")
    dependencies += [p for p in sources if p != tb]
    dependencies += [p for directory in fixtures.values() for p in sorted(directory.iterdir())
                     if p.suffix == ".mem" or p.name == "reference.json"]
    record = {"status": "PREPARED_NOT_SIMULATED", "scope": scope, "cwd": str(ROOT),
              "build": build, "runs": runs, "production_sv_count": 105,
              "simulation_source_count": len(sources), "activity_monitor": args.activity_monitor,
              "memory_backend": "unchanged legacy FC behavioral memories; UNIFIED_USE_XPM_MEMORY not defined",
              "minimal_dependencies": sorted({p.relative_to(ROOT).as_posix() for p in dependencies}),
              "stages": {}}
    def save():
        (out / "commands.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save()
    print(f"Prepared new passive paired measurement: {out}", flush=True)
    if args.prepare_only:
        return 0
    record["status"] = "RUNNING"
    save()
    results = {}
    for stage, command in [("build", build), *runs.items()]:
        started = time.monotonic()
        with (out / (stage + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["stages"][stage] = {"exit_code": process.returncode, "wall_seconds": round(time.monotonic() - started, 3)}
        if process.returncode:
            record["status"] = "FAILED"
            save()
            return process.returncode
        if stage == "build":
            warnings = structural_warnings((out / "build.log").read_text(encoding="utf-8", errors="replace"))
            if warnings:
                record["status"] = "FAILED_STRUCTURAL_BUILD"
                record["stages"][stage]["structural_warnings"] = warnings
                save()
                print("Structural build warnings; neither arm started: " + ", ".join(warnings), flush=True)
                return 1
        save()
        if stage != "build":
            try:
                results[stage] = parse_log((out / (stage + ".log")).read_text(encoding="utf-8"), stage, scope, args.activity_monitor)
            except ValueError:
                record["status"] = "FAILED_MEASUREMENT_PARSE"
                save()
                raise
            (out / (stage + ".json")).write_text(json.dumps(results[stage], indent=2) + "\n", encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps({"status": "PASS", "scope": scope,
        "runtime_on_minus_off_cycles": results["on"]["runtime_cycles"] - results["off"]["runtime_cycles"],
        "interpretation": "Paired runtime delta includes learning, scheduling and W2 refills; not isolated pure learning latency.",
        "arms": {arm: arm + ".json" for arm in results}}, indent=2) + "\n", encoding="utf-8")
    record["status"] = "PASS"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
