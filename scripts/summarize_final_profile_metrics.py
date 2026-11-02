#!/usr/bin/env python3
"""Read archived local counters; never execute a model, simulator or EDA tool."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

from summarize_h40_activity import BUCKETS, parse_activity
from run_fc_profile_measurement import parse_log as parse_fc_measurement
from run_srnn_profile_measurement import parse_log as parse_srnn_measurement
from run_scnn_profile_measurement import parse_result as parse_scnn_measurement
from run_scnn_profile_measurement import check_build_log as check_scnn_build_log

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = Path("results/paper_evidence/supplement_20261004")
OUTPUT = SUPPLEMENT / "performance/final_profile_metrics_20261009"
FINAL_TOP = "rtl/top/unified_h40_final_composition_top_v1.sv"
CURRENT_FC = SUPPLEMENT / "performance/current_source_measurements/fc/server_pair_r1"
CURRENT_SRNN = SUPPLEMENT / "performance/current_source_measurements/srnn/server_pair_r1"
SRNN_RAW = Path("build/current_profile_measurements_20261009/srnn_complete_pair_r1_raw")
CURRENT_SCNN = SUPPLEMENT / "performance/current_source_measurements/scnn/server_pair_r2"
SCNN_RAW = Path("build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw")
SCNN_RAW_FILES = ("scripts/run_scnn_profile_measurement.py",
                  "tb/include/scnn_profile_measurement_monitor.svh",
                  "tb/include/scnn_profile_measurement_body.svh")
FAILURE = re.compile(r"%Error|Assertion failed|\$fatal|Aborting")
PROFILE_CSV_FIELDS = (
    "id", "topology", "learning", "batch", "timesteps_per_sample", "samples", "batches",
    "training_samples", "query_samples", "query_timesteps", "cycles",
    "initial_load_and_refill_cycles", "initial_state_cycles", "cycles_including_initialization",
    "projected_interval_ms", "projected_including_initialization_ms",
    "bounded_support_interval_samples_s_projection", "full_sample_throughput_samples_s",
    "simulation_wall_seconds", "source_family", "evidence_status", "rtl_entry", "evidence",
    "measured_interval_boundary", "model", "period_ns",
)


def normal_path(path):
    value = os.fspath(path)
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(value)


def filesystem_path(path):
    path = Path(path)
    if os.name != "nt":
        return path
    value = os.path.abspath(os.path.normpath(os.fspath(path)))
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def read_json(path):
    return json.loads(filesystem_path(path).read_text(encoding="utf-8-sig"))


def counter_rows(text, marker, allow_signed=False):
    rows = []
    for line in text.splitlines():
        if line.startswith(marker + " "):
            row = {key: int(value) for key, value in
                   re.findall(r"(\w+)=(-?\d+)(?=\s|$)", line)}
            if not allow_signed and any(value < 0 for value in row.values()):
                raise ValueError(f"negative counter: {marker}")
            rows.append(row)
    return rows


def terminal(text, marker):
    rows = counter_rows(text, marker)
    if FAILURE.search(text) or len(rows) != 1:
        raise ValueError(f"missing, duplicate or failed terminal {marker}")
    return rows[0]


def step_deltas(rows, expected, key):
    if len(rows) != expected or [row[key] for row in rows] != list(range(expected)):
        raise ValueError("incomplete or duplicate step sequence")
    previous = 0
    result = []
    for row in rows:
        delta = row["cycles"] - previous
        if delta <= 0:
            raise ValueError("nonmonotonic cumulative cycles")
        result.append({**row, "interval_cycles": delta})
        previous = row["cycles"]
    return result


def parse_fc(text):
    final = terminal(text, "FC_FINAL_COUNTS_PASS")
    load = terminal(text, "FC_FINAL_COUNTS_LOAD")
    steps = counter_rows(text, "FC_FINAL_COUNTS_STEP")
    per_batch, remainder = divmod(final["steps"], final["batches"])
    if remainder or [(r["batch_index"], r["t"]) for r in steps] != [
            (b, t) for b in range(final["batches"]) for t in range(per_batch)]:
        raise ValueError("incomplete FC batch/step sequence")
    if steps[-1]["cycles"] != final["cycles"]:
        raise ValueError("FC terminal cycles disagree with final step")
    return final, load


def parse_current_fc(text, archived, arm, scope):
    measured = parse_fc_measurement(text, arm, scope, activity=True)
    if measured != archived:
        raise ValueError("current FC summary disagrees with its own passive/raw activity log")
    final, initial = parse_fc(text)
    if (final != measured["legacy"] or
            initial["cycles"] != measured["stages"]["parameter_load"]["cycles"] +
            measured["stages"]["initial_refill"]["cycles"] or
            initial["init_cycles"] != measured["stages"]["initial_state"]["cycles"]):
        raise ValueError("current FC initialization/legacy boundary disagrees with passive counters")
    return measured, initial


def parse_srnn(text):
    final = terminal(text, "SRNN_SAME_DUT_PASS")
    steps = step_deltas(counter_rows(text, "SRNN_SAME_DUT_STEP"), final["steps"], "t")
    if steps[-1]["cycles"] != final["cycles"]:
        raise ValueError("SRNN terminal cycles disagree with final step")
    for key in ("bone", "hidden", "logits", "code", "residual", "commits", "learning"):
        if steps[-1][key] != final[key]:
            raise ValueError(f"SRNN terminal mismatch: {key}")
    return final, steps


def validate_current_srnn_commands(commands, metadata):
    if (commands["status"] != "PASS_two_runtime_arms_one_fixed_sample_not_Test"
            or any(commands[stage + "_exit"] != 0 for stage in ("build", "off", "on"))
            or commands["build_warnings"]["blocking"]
            or commands["production_source_count"] != 105
            or not commands["same_executable_for_on_off"] or commands["runner_modifies_rtl"]
            or not commands["activity_monitor_requested"] or commands["period_ns_projection"] != 15
            or commands["sample_definition"] != metadata
            or (metadata["batch"], metadata["samples"], metadata["sample_index"], metadata["steps"],
                metadata["native_steps"], metadata["padded_steps"], metadata["timestep_policy"],
                metadata["complete_sample"]) != (1, 1, 0, 73, 73, 0, "native", True)):
        raise ValueError("current SRNN paired execution/native full-sample scope did not pass")
    paired = []
    for arm in ("off", "on"):
        command = commands["run"][arm]
        fixture = local_fixture({"run": command}).as_posix()
        if (fixture != commands["fixture"] or "+RAW=" + fixture not in command
                or [arg for arg in command if arg.startswith("+LEARNING=")] != [f"+LEARNING={int(arm == 'on')}"]):
            raise ValueError("current SRNN recorded input/learning arguments differ")
        paired.append([arg for arg in command if not arg.startswith("+LEARNING=")])
    if paired[0] != paired[1]:
        raise ValueError("current SRNN paired inputs/executables differ")


def parse_current_srnn(text, archived, archived_activity, commands, arm, metadata):
    validate_current_srnn_commands(commands, metadata)
    final, steps = parse_srnn(text)
    measured = parse_srnn_measurement(text, metadata, commands["period_ns_projection"])
    measured["simulation_wall_seconds"] = commands[arm + "_wall_seconds"]
    if measured != archived or measured["status"] != "PASS_complete_fixed_sample_not_Test":
        raise ValueError("current SRNN summary disagrees with its own complete passive log")
    end, counts = measured["terminal"], measured["counters"]
    if ((final["batch"], final["steps"], final["learning"]) != (1, 73, int(arm == "on"))
            or end["learning"] != final["learning"] or final["cycles"] != end["end"] - end["initialized"]
            or final["store"] != end["store_checks"]):
        raise ValueError("current SRNN functional/passive completion differs")
    for field, counter in (("bone", "phase_actual_capture_accepted"), ("hidden", "m2_hidden_accepted"),
                           ("logits", "m2_logit_accepted"), ("gradient", "gradient_accepted"),
                           ("code", "code_write_accepted"), ("residual", "residual_write_accepted"),
                           ("commits", "commit_pulse"), ("detach", "detach_pulse")):
        if final[field] != counts[counter]:
            raise ValueError(f"current SRNN functional/passive counter differs: {field}")
    hardware = [r for r in measured["intervals"] if r["kind"] == "hardware"]
    for phase, expected in (("m2_prepare", 73), ("m2_forward", 73),
                            ("pure_learning_busy", 73 if arm == "on" else 0)):
        rows = [r for r in hardware if r["phase"] == phase]
        if [r["timestep"] for r in rows] != list(range(expected)):
            raise ValueError(f"current SRNN hardware interval incomplete: {phase}")
    pure = sum(r["cycles"] for r in hardware if r["phase"] == "pure_learning_busy")
    events = measured["events"]
    if (pure != counts["pure_learning_busy_cycles"] or pure > counts["learner_busy_cycles"]
            or [(r["name"], r["timestep"]) for r in events] !=
            [("commit", t) for t in range(73 if arm == "on" else 0)]):
        raise ValueError("current SRNN learning busy/commit ledger incomplete")
    activity = parse_activity(text, "SRNN_PROFILE_COMPLETE", 12)
    activity["sample_definition"] = {key: metadata[key] for key in
                                      ("batch", "sample_index", "steps", "native_steps", "timestep_policy")}
    activity["learning"] = int(arm == "on")
    if (activity != archived_activity or activity["cycles"] != end["end"] - counts["reset_cycles"]
            or activity["totals"]["cycles"] != 12 * activity["cycles"]):
        raise ValueError("current SRNN activity disagrees with its own reset-released window")
    return measured, final, steps, activity


def parse_scnn(text):
    final = terminal(text, "SCNN_TEMPORAL_FINAL_PASS")
    initialization = terminal(text, "SUPPLEMENT_FULLGRAPH_INITIALIZATION")
    layers = counter_rows(text, "SCNN_TEMPORAL_LAYER")
    frames = counter_rows(text, "SCNN_TEMPORAL_FRAME_STATUS")
    windows = counter_rows(text, "SCNN_TEMPORAL_WINDOW_START")
    commits = counter_rows(text, "SCNN_TEMPORAL_COMMIT")
    ddr = counter_rows(text, "SCNN_TEMPORAL_DDR")
    traffic = []
    for line in text.splitlines():
        if line.startswith("SCNN_TEMPORAL_FRAME_PASS "):
            row = {}
            for key in ("requests", "contexts", "scalars"):
                match = re.search(rf"\b{key}=([\d+]+)(?=\s|$)", line)
                if not match:
                    raise ValueError(f"SCNN frame traffic missing: {key}")
                row[key] = [int(value) for value in match[1].split("+")]
                if len(row[key]) != 8:
                    raise ValueError("SCNN frame traffic needs eight layers")
            row["frame"] = int(re.search(r"\bframe=(\d+)", line)[1])
            traffic.append(row)
    if traffic and [row["frame"] for row in traffic] != list(range(final["frames"])):
        raise ValueError("SCNN frame traffic sequence incomplete")
    if [(r["frame"], r["layer"]) for r in layers] != [
            (f, layer) for f in range(final["frames"]) for layer in range(1, 9)]:
        raise ValueError("incomplete or duplicate SCNN eight-layer frames")
    if (len(layers) != final["descriptors"] or
            [r["descriptors"] for r in layers] != list(range(1, len(layers) + 1))):
        raise ValueError("SCNN descriptor count mismatch")
    if [r["frame"] for r in frames] != list(range(final["frames"])):
        raise ValueError("incomplete SCNN frame intervals")
    stages = []
    for frame in frames:
        selected = layers[frame["frame"] * 8:(frame["frame"] + 1) * 8]
        if any((r["sample"], r["timestep"]) != (frame["sample"], frame["timestep"])
               for r in selected):
            raise ValueError("SCNN layer/frame identity mismatch")
        extra = frame["cycles"] - sum(r["cycles"] for r in selected)
        if extra < 0:
            raise ValueError("SCNN layers exceed enclosing frame interval")
        for r in selected:
            layer_row = {**r, "kind": "layer", "stage": f"Conv{r['layer']}" +
                         ("+pool+classifier" if r["layer"] == 8 else ""),
                         "boundary": "descriptor start-valid assertion through observed layer/frame done; TB waits included"}
            if traffic:
                for key in ("requests", "contexts", "scalars"):
                    current = traffic[r["frame"]][key][r["layer"] - 1]
                    previous = traffic[r["frame"] - 1][key][r["layer"] - 1] if r["frame"] else 0
                    if current < previous:
                        raise ValueError("nonmonotonic SCNN cumulative traffic")
                    layer_row[key] = current - previous
                    layer_row["traffic_boundary"] = "delta of per-layer cumulative FRAME_PASS counters; not DDR bytes or model MACs"
            stages.append(layer_row)
        stages.append({**frame, "kind": "frame", "stage": "enclosing_frame",
                       "boundary": "contains all eight layers; final training frame also waits for commit/checks"})
        stages.append({**frame, "cycles": extra, "kind": "derived_remainder",
                       "stage": "frame_minus_layers",
                       "boundary": "frame interval minus layer intervals; not pure learning/commit cost"})
    if [r["window"] for r in windows] != list(range(1, len(windows) + 1)):
        raise ValueError("SCNN window sequence is incomplete")
    for index, window in enumerate(windows):
        end = windows[index + 1]["cycle"] if index + 1 < len(windows) else final["cycles"]
        if end <= window["cycle"]:
            raise ValueError("nonmonotonic SCNN window boundaries")
        stages.append({**window, "cycles": end - window["cycle"], "kind": "window",
                       "stage": "query4" if window["query"] else "B4T4_train",
                       "boundary": "window-start log to next window-start or final PASS; includes intervening TB/setup"})
    if (len(commits) != final["commits"] or
            sum(r["rmw"] for r in commits) != final["rmw"] or
            sum(r["code_changes"] for r in commits) != final["code_changes"]):
        raise ValueError("SCNN commit counters disagree with terminal")
    if sorted(r["bank"] for r in ddr) != list(range(9)):
        raise ValueError("SCNN DDR bank counters are incomplete")
    stages.append({**initialization, "kind": "initialization", "stage": "initialization",
                   "boundary": "initial setup and classifier publication, not a standalone full-model load measurement"})
    remainder = final["cycles"] - initialization["cycles"] - sum(r["cycles"] for r in frames)
    if remainder < 0:
        raise ValueError("SCNN initialization/frames exceed total cycles")
    stages.append({"cycles": remainder, "kind": "derived_remainder", "stage": "outside_frames",
                   "boundary": "total minus initialization and enclosing frames; unclassified TB/setup/check time"})
    return final, initialization, stages, commits, ddr


def parse_timing(text):
    slack = re.search(r"WNS\(ns\)[^\n]*\n[^\n]*\n\s*"
                      r"(-?[\d.]+)\s+(-?[\d.]+)\s+\d+\s+\d+\s+"
                      r"(-?[\d.]+)\s+(-?[\d.]+)", text)
    clock = re.search(r"^core_clk\s+\{[^}]+\}\s+([\d.]+)\s+([\d.]+)", text, re.M)
    if not slack or not clock:
        raise ValueError("timing summary or core clock missing")
    wns, tns, whs, ths = map(float, slack.groups())
    return {"period_ns": float(clock[1]), "wns_ns": wns, "tns_ns": tns,
            "whs_ns": whs, "ths_ns": ths,
            "constraints_met": "All user specified timing constraints are met." in text,
            "boundary": "OOC core only; external I/O/DDR timing, board Fmax and measured hardware time unproven"}


def project_cycles(cycles, timing):
    if (not timing["constraints_met"] or
            min(timing["wns_ns"], timing["whs_ns"], timing["tns_ns"], timing["ths_ns"]) < 0):
        return None
    return cycles * timing["period_ns"] / 1_000_000


def local_fixture(commands):
    matches = [arg.split("=", 1)[1] for arg in commands["run"] if arg.startswith("+FIXTURE=")]
    if len(matches) != 1:
        raise ValueError("one recorded project-local fixture required")
    path = Path(matches[0].replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("fixture must be project-local")
    return path


def validate_current_scnn_records(preparation, reference, statuses, launcher):
    if (sorted(re.findall(r"^(off_exit|on_exit|exit_code)=(-?\d+)$", launcher, re.M)) !=
            [("exit_code", "0"), ("off_exit", "0"), ("on_exit", "0")]
            or any(statuses[key]["exit_code"] != 0 for key in ("elab", "build", "off", "on"))
            or preparation["top"] != "unified_h40_final_composition_top_v1"
            or (preparation["sample"], preparation["batch"], preparation["timesteps"],
                preparation["production_sv_files"], preparation["source_files"],
                preparation["activity_monitor"]) != (0, 1, 20, 105, 108, True)
            or preparation["windows"] != [dict(base=t, count=4, batch=1) for t in (0, 4, 8, 12, 16)]
            or (reference["sample"], reference["batch"], reference["timesteps"],
                reference["window_timesteps"], reference["multiplier"], reference["time_bases"]) !=
                (0, 1, 20, 4, 640, [0, 4, 8, 12, 16])
            or reference["corruption"] != "none; original clean sample0"):
        raise ValueError("current SCNN paired execution/fixed clean sample0 scope did not pass")
    expected_schedule = [dict(frame=t, sample=0, timestep=t, time_base=t//4*4,
                              local_timestep=t % 4, count=4, batch=1, trace_reset=t < 4,
                              sample_reset=t == 0, wait_for_window=t % 4 == 3) for t in range(20)]
    if reference["frame_schedule"] != expected_schedule:
        raise ValueError("current SCNN reference is not continuous five-window T20")
    paired = []
    for arm in ("off", "on"):
        command = statuses[arm]["command"]
        fixtures = [arg.split("=", 1)[1] for arg in command if arg.startswith("+SCNN_PROFILE_FIXTURE=")]
        if fixtures != [preparation["fixture"]] or [arg for arg in command if arg.startswith("+SCNN_LEARNING=")] != [f"+SCNN_LEARNING={int(arm == 'on')}"]:
            raise ValueError("current SCNN recorded fixture/arm arguments differ")
        paired.append([arg for arg in command if not arg.startswith("+SCNN_LEARNING=")])
    if paired[0] != paired[1]:
        raise ValueError("current SCNN paired inputs/executables differ")


def parse_current_scnn(text, archived, reference, status, arm):
    final = terminal(text, "SCNN_PROFILE_PASS")
    measured = parse_scnn_measurement(text, reference, arm == "on", activity=True)
    measured.update(host_simulator_wall_seconds=status["seconds"],
                    host_file_load_seconds=None, host_golden_seconds=None)
    if measured != archived or measured["final"] != final or status["exit_code"] != 0:
        raise ValueError("current SCNN summary/status disagrees with its own complete passive log")
    if final["cycles"] != measured["timing"]["sample_end"]:
        raise ValueError("current SCNN terminal/timing boundary differs")
    traffic = measured["traffic"]
    if (traffic["code_write_words"] != sum(w["code_changes"] for w in reference["windows"][arm])
            or traffic["residual_write_words"] != sum(w["residual_changes"] for w in reference["windows"][arm])):
        raise ValueError("current SCNN write-enable counts disagree with reference")
    activity = parse_activity(text, "SCNN_PROFILE_PASS", 12)
    if (any(value != measured["h40_summary"][key] for key, value in activity["totals"].items())
            or activity["cycles"] != measured["timing"]["included_setup_cycles"]):
        raise ValueError("current SCNN activity disagrees with its own reset-released window")
    frames = counter_rows(text, "SCNN_PROFILE_FRAME")
    if sum(r["cycles"] for r in frames) > measured["timing"]["sample_cycles"]:
        raise ValueError("current SCNN frames exceed sample service interval")

    # The learner toggles in many short bursts. Aggregate within each window, not into one contiguous interval.
    learning = [r for r in measured["intervals"] if r["kind"] == "learning"]
    starts = [r for r in measured["events"] if r["kind"] == "learning_start"]
    if [(r["window"], r["cycle"]) for r in starts] != [(r["window"], r["start"]) for r in learning]:
        raise ValueError("current SCNN learner start/interval ledger differs")
    busy = []
    for window in range(1, 6):
        selected = [r for r in learning if int(r["window"]) == window]
        if not selected:
            continue
        cycles = [int(r["end"]) - int(r["start"]) for r in selected]
        busy.append(dict(window=window, interval_count=len(selected), cycles=sum(cycles),
                         first_start=int(selected[0]["start"]), last_end=int(selected[-1]["end"]),
                         min_interval_cycles=min(cycles), max_interval_cycles=max(cycles)))
    if (sum(r["interval_count"] for r in busy) != len(learning)
            or sum(r["cycles"] for r in busy) != measured["active"]["learning"]):
        raise ValueError("current SCNN learner intervals disagree with active clocks")
    intervals = [{key: value if key == "kind" else int(value) for key, value in row.items()}
                 for row in measured["intervals"] if row["kind"] != "learning"]
    events = [{key: value if key == "kind" else int(value) for key, value in row.items()}
              for row in measured["events"] if row["kind"] != "learning_start"]
    if sum(r["end"]-r["start"] for r in intervals if r["kind"] == "publication" and r["window"] > 0) != measured["active"]["publication"]:
        raise ValueError("current SCNN publication intervals disagree with active clocks")
    return measured, frames, busy, intervals, events, activity


def current_scnn_stages(name, measured, frames, busy, intervals, events):
    rows = []
    for descriptor in measured["descriptors"]:
        rows.append(dict(profile=name, **descriptor, kind="descriptor",
                         stage=f"Conv{descriptor['layer']}" + ("+pool+classifier" if descriptor["layer"] == 8 else ""),
                         boundary="accepted descriptor start to observed done; includes refill/state service; Conv8 contains pool/classifier and control subintervals overlap"))
    for frame in frames:
        children = [r for r in measured["descriptors"] if r["frame"] == frame["frame"]]
        remainder = frame["cycles"] - sum(r["cycles"] for r in children)
        if remainder < 0:
            raise ValueError("current SCNN descriptors exceed enclosing frame")
        rows.append(dict(profile=name, **frame, kind="frame", stage="enclosing_frame",
                         boundary="TB frame interval contains descriptors and final-window commit/check waits; not additive with children"))
        rows.append(dict(profile=name, frame=frame["frame"], cycles=remainder,
                         kind="derived_remainder", stage="frame_minus_descriptors",
                         boundary="frame minus eight descriptor intervals; unclassified waits, not pure learning cost"))
    rows.append(dict(profile=name, kind="setup", stage="initial_setup", cycles=measured["timing"]["setup_cycles"],
                     boundary="reset release to first window request, includes initial classifier publication; not full model DMA load"))
    rows.append(dict(profile=name, kind="derived_remainder", stage="outside_frames",
                     cycles=measured["timing"]["sample_cycles"] - sum(r["cycles"] for r in frames),
                     boundary="service interval minus enclosing frames; scheduling/boundary clocks, not isolated computation"))
    for control in measured["control"]:
        for stage in ("conv8", "pool", "classifier"):
            start, end = control[stage + "_first"], control[stage + "_end"]
            rows.append(dict(profile=name, kind="control_interval", stage=stage, frame=control["frame"],
                             start=start, end=end, cycles=end-start,
                             boundary="original first-to-end control edges; pool includes collection and overlaps Conv8 writeback, classifier excludes tail capture; not exclusive costs"))
    for interval in intervals:
        rows.append({"profile": name, **interval, "stage": interval["kind"], "kind": "phase_interval",
                     "cycles": interval["end"]-interval["start"],
                     "boundary": "original signal edges; window0 publication is setup, commit-pending-to-pulse contains publication and drain, never add both as exclusive latency"})
    for item in busy:
        rows.append(dict(profile=name, **item, kind="busy_interval_aggregate", stage="learning_busy",
                         boundary="sum of fragmented gradient/update busy intervals within this window; first_start/last_end span gaps, not contiguous latency or all learning support cost"))
    for event in events:
        rows.append({"profile": name, **event, "stage": event["kind"], "kind": "phase_event",
                     "boundary": "accepted request or observed done/commit pulse edge; not independent elapsed cost"})
    for window in range(1, 6):
        start = next(r["cycle"] for r in events if r["kind"] == "window_request" and r["window"] == window)
        end = next(r["cycle"] for r in events if r["kind"] == "window_done" and r["window"] == window)
        rows.append(dict(profile=name, kind="window_interval", stage="B1T4_window", window=window,
                         start=start, end=end, cycles=end-start,
                         boundary="accepted window request to observed done; parent of four frames, contains learning/publication waits in on arm"))
        if measured["learning"]:
            start = next(r["cycle"] for r in events if r["kind"] == "publish_request" and r["window"] == window)
            end = next(r["cycle"] for r in events if r["kind"] == "publish_done" and r["window"] == window)
            if end <= start:
                raise ValueError("current SCNN publication request/done ordering")
            rows.append(dict(profile=name, kind="publication_boundary", stage="publish_request_to_done", window=window,
                             start=start, end=end, cycles=end-start,
                             boundary="accepted backend commit-publish request to observed done; overlaps publication busy and commit-pending interval, not an additive platform latency"))
    for key, value in measured["active"].items():
        rows.append(dict(profile=name, kind="active_counter", stage=key, cycles=value,
                         boundary="named sample-period busy predicate; overlaps descriptors/control/other predicates, not additive latency"))
    for group in ("traffic", "payload"):
        for key, value in measured[group].items():
            rows.append(dict(profile=name, kind="flow_counter", stage=key, value=value,
                             units="port_rows_bytes_or_cycles_as_named" if group == "traffic" else "payload_bytes_or_scalar16_words_as_named",
                             boundary="TB printed constant, not a passive port counter" if key == "store_loads" else
                             "actual native-model handshakes/masks, named service-high clocks or derived payload; bytes exclude address/tag/control and are not measured board DDR traffic"))
    rows.extend(dict(profile=name, **bank, kind="bank_flow_counter", stage="DDR_model_bank",
                     boundary="actual accepted native forward reads/responses per bank, not board bandwidth")
                for bank in measured["banks"])
    return rows


def collect(root=ROOT):
    root = Path(root)
    sources = set()
    profiles, stages, activity, accuracy, missing = [], [], [], [], []

    def load(path, json_file=True):
        path = Path(path)
        sources.add(path.as_posix())
        return read_json(root / path) if json_file else filesystem_path(root / path).read_text(encoding="utf-8-sig")

    def gap(profile, counter, reason):
        missing.append({"profile": profile, "counter": counter, "reason": reason})

    timing_paths = {
        "cache_r3": Path("results/paper_evidence/physical/run131_cache_residency_return_impl/final_composition_impl/timing_summary_post_route.rpt"),
        "frame_service": Path("results/paper_evidence/physical/run137_scnn_frame_service_impl/timing_summary_post_route.rpt"),
    }
    timing = {key: {**parse_timing(load(path, False)), "source": path.as_posix()}
              for key, path in timing_paths.items()}
    # These are documented archived-source associations, not a check of live RTL.
    timing["cache_r3"]["source_boundary"] = "RUN131 cache-family-r3 snapshot only; not the later frame-service source"
    timing["frame_service"]["source_boundary"] = "RUN137 exact RUN136 DCP; documented current frame-service RTL for SCNN B4T12 R2, FC current-source B16T99, SRNN current-source native B1T73 and SCNN current-source original B1T20 R2; behavioral simulation memories are not XPM/STA validation"
    for path in ("rtl/fc_final_counts/README.md", "rtl/srnn_same_dut/README.md",
                 "rtl/scnn_real_final_graph/README.md", "performance/h40_activity/README.md"):
        load(SUPPLEMENT / path, False)

    def profile(name, topology, arm, batch, steps, cycles, family, evidence, input_data, **extra):
        row = {"id": name, "topology": topology, "learning": arm, "batch": batch,
               "timesteps_per_sample": steps, "cycles": cycles, "source_family": family,
               "evidence_status": "archived_PASS_not_new_experiment",
               "rtl_entry": FINAL_TOP, "evidence": evidence.as_posix(), "input": input_data,
               "timing": timing[family], "projected_interval_ms": project_cycles(cycles, timing[family]),
               "simulation_wall_seconds": None, "full_sample_throughput_samples_s": None,
               "hardware_time_measured": False, "rtl_dataset_accuracy_percent": None, **extra}
        profiles.append(row)
        return row

    comparison = load(SUPPLEMENT / "performance/fc_final_counts/cache_r3_support160_r2/comparison.json")
    fc_commands, fc_refs = {}, {}
    for arm in ("off", "on"):
        evidence = SUPPLEMENT / f"rtl/fc_final_counts/cache_r3_support160_r2_{arm}"
        commands = load(evidence / "commands.json")
        measurement = load(evidence / "measurements.json")
        final, initial = parse_fc(load(evidence / "run.log", False))
        reference_path = local_fixture(commands) / "reference.json"
        reference = load(reference_path)
        if commands["run_exit"] != 0 or measurement["status"] != "PASS" or comparison["status"] != "PASS":
            raise ValueError("FC archived execution did not pass")
        if initial != measurement["rows"][0]:
            raise ValueError("FC initial load counters disagree with raw log")
        for key, value in final.items():
            if value != measurement["rows"][-1][key] or value != comparison["arms"][arm][key]:
                raise ValueError(f"FC summary disagrees with raw log: {key}")
        if (final["learning"] != int(arm == "on") or final["batch"] != reference["batch"] or
                final["batches"] != reference["batches"] or final["steps"] != reference["total_steps"]):
            raise ValueError("FC profile/reference identity mismatch")
        for key, refkey in (("input_events", "input_events"), ("hidden_events", "hidden_spikes"),
                            ("code_writes", "code_changes"), ("residual_writes", "residual_changes")):
            if final[key] != reference["totals"][refkey]:
                raise ValueError(f"FC reference counter mismatch: {key}")
        name = f"FC_support160_cache_r3_{arm}"
        row = profile(name, "FC", arm, final["batch"], reference["steps"], final["cycles"],
                      "cache_r3", evidence, {"fixture": reference_path.parent.as_posix(),
                      "episode": reference["episode"], "support_indices": reference["support_indices"],
                      "samples": reference["samples"], "numeric_source": reference["numeric_source"],
                      "boundary": "original seed38472 support order, +2 pixel shift, multiplicity preserved; padded T99"},
                      model="2312->160->10; W0 INT5/shift6, W2 INT8; W2-only",
                      execution={"command": commands["run"], "cwd": commands["cwd"],
                                 "build_executed_in_this_record": "build_exit" in commands,
                                 "source_scope_document": (SUPPLEMENT / "rtl/fc_final_counts/README.md").as_posix()},
                      samples=final["batch"] * final["batches"], batches=final["batches"],
                      counters=final, initial_load_and_refill_cycles=initial["cycles"],
                      initial_state_cycles=initial["init_cycles"], simulation_wall_seconds=commands["run_seconds"],
                      measured_interval_boundary=comparison["cycle_boundary"],
                      final_changed_w2=comparison["final_changed_w2"] if arm == "on" else 0,
                      visibility=comparison["updated_prediction_visibility"] if arm == "on" else None)
        row["cycles_including_initialization"] = final["cycles"] + initial["cycles"] + initial["init_cycles"]
        row["projected_including_initialization_ms"] = project_cycles(row["cycles_including_initialization"], timing["cache_r3"])
        row["bounded_support_interval_samples_s_projection"] = row["samples"] * 1000 / row["projected_interval_ms"] if row["projected_interval_ms"] else None
        for key, boundary in (("w0_cycles", "W0 TB forward interval"), ("w2_cycles", "W2 TB forward interval"),
                              ("learning_cycles", "wait after last sample; overlaps other learner activity; not pure arithmetic"),
                              ("refill_cycles", "includes first refill; overlaps load/measured interval; do not add to total")):
            stages.append({"profile": name, "kind": "aggregate_counter", "stage": key,
                           "cycles": final[key], "boundary": boundary})
        gap(name, "pure_learning_active_cycles/commit_cycles", "只测末样本后的等待，学习与观察回放可重叠，未独立打印学习活跃/提交周期")
        gap(name, "lane_fires/enabled_terms/exclusive_tile_buckets", "该support160执行未挂活动监测；B16T99是另一条独立执行")
        fc_commands[arm], fc_refs[arm] = commands, reference
    if fc_commands["on"]["run"][0] != fc_commands["off"]["run"][0]:
        raise ValueError("FC arms use different executables")
    if fc_refs["on"]["samples"] != fc_refs["off"]["samples"]:
        raise ValueError("FC arms use different input samples")

    srnn_commands = {}
    for arm in ("off", "on"):
        evidence = SUPPLEMENT / f"rtl/srnn_same_dut/{arm}_cache_family_r3"
        commands = load(evidence / "commands.json")
        final, steps = parse_srnn(load(evidence / "run.log", False))
        reference_path = local_fixture(commands) / "reference.json"
        reference = load(reference_path)
        raw_path = Path(reference["raw_reference"].replace("\\", "/"))
        raw = load(raw_path / "summary.json")
        if (commands["run_exit"] != 0 or final["learning"] != int(arm == "on") or
                (final["batch"], final["steps"]) != (reference["batch"], reference["steps"])):
            raise ValueError("SRNN archived execution/profile mismatch")
        name = f"SRNN_raw700_M2_cache_r3_{arm}"
        profile(name, "SRNN", arm, final["batch"], final["steps"], final["cycles"], "cache_r3", evidence,
                {"fixture": reference_path.parent.as_posix(), "raw_fixture": raw_path.as_posix(),
                 "sample_index": reference["sample_index"], "labels": reference["labels"],
                 "preprocessing": raw["source"]["preprocessing"], "dependencies": raw["dependencies"],
                 "boundary": "official sample0 prefix, DUT backbone spikes feed M2 in the same final top; fixed /64 gradient"},
                model="700->450 recurrent + 450->256->20 M2; INT8 backbone/Q2.14 M2; Wout-only",
                execution={"command": commands["run"], "cwd": commands["cwd"],
                           "build_executed_in_this_record": "build_exit" in commands,
                           "source_scope_document": (SUPPLEMENT / "rtl/srnn_same_dut/README.md").as_posix()},
                samples=1, counters=final, initial_load_and_refill_cycles=None, initial_state_cycles=None,
                simulation_wall_seconds=commands["run_seconds"],
                measured_interval_boundary="cycles-initialized_cycle: excludes initial parameter/state setup, includes workset refills and TB readback/check waits")
        stages.extend({"profile": name, **step, "cycles": step["interval_cycles"], "kind": "step",
                       "stage": f"t{step['t']}", "cumulative_cycles": step["cycles"],
                       "boundary": "delta of cumulative end-step logs; backbone+M2+learning/refill/checks combined"}
                      for step in steps)
        for counter, reason in (
            ("initial_parameter_load_cycles/initial_state_cycles", "initialized_cycle没有打印，无法恢复已排除的初始设置耗时"),
            ("backbone_forward_cycles/M2_forward_cycles/refill_cycles/learning_cycles/commit_cycles", "只打印合并的每步累计周期，没有分段起止计数"),
            ("forward_read_handshakes/learning_read_handshakes/refill_payload_bytes", "store=962560是TB参数检查项，不是总线流量"),
            ("zero_skip_count/saturation_count", "参考JSON有软件期望，但当前RTL终态未打印这两个计数，不冒充实测"),
            ("lane_fires/enabled_terms/exclusive_tile_buckets", "B2骨干与观察后M2监测来自独立DUT，不是此B1同DUT执行"),
            ("full_sequence_cycles", "仅官方sample0的前8步，不能算完整SHD样本吞吐")):
            gap(name, counter, reason)
        srnn_commands[arm] = commands
    if srnn_commands["on"]["run"][0] != srnn_commands["off"]["run"][0]:
        raise ValueError("SRNN arms use different executables")
    if ([arg for arg in srnn_commands["on"]["run"] if not arg.startswith("+LEARNING=")] !=
            [arg for arg in srnn_commands["off"]["run"] if not arg.startswith("+LEARNING=")]):
        raise ValueError("SRNN paired inputs differ")

    evidence = SUPPLEMENT / "rtl/scnn_real_final_graph/server_b4t12_r2"
    status = load(evidence / "run.status.json")
    summary = load(evidence / "summary.json")
    preparation = load(evidence / "preparation.json")
    load(evidence / "sources.f", False)
    load(evidence / "dependencies.txt", False)
    scnn_text = load(evidence / "run.log", False)
    final, initial, scnn_stages, commits, ddr = parse_scnn(scnn_text)
    if status["exit_code"] != 0 or summary["status"] != "PASS" or summary["measured"] != final:
        raise ValueError("SCNN archived summary/status disagrees with raw log")
    fixture_args = [arg.split("=", 1)[1] for arg in status["command"] if arg.startswith("+SCNN_TEMPORAL_FIXTURE=")]
    manifest_path = Path(fixture_args[0]) / "manifest.json"
    manifest = load(manifest_path)
    if final["frames"] != manifest["total_graph_frames"]:
        raise ValueError("SCNN manifest frame count disagrees with measured execution")
    if len(commits) != len(manifest["commits"]):
        raise ValueError("SCNN manifest commit count mismatch")
    for actual, expected in zip(commits, manifest["commits"]):
        if any(actual[key] != expected[key] for key in ("code_changes", "residual_changes")):
            raise ValueError("SCNN measured commit disagrees with reference")
    name = "SCNN_real_final_B4T12_query4"
    profile(name, "SCNN", "on_train/off_query", manifest["batch_count"], manifest["timesteps"],
            final["cycles"], "frame_service", evidence,
            {"fixture": manifest_path.parent.as_posix(), "seed": manifest["seed"],
             "support_subject": manifest["support_subject"], "query_subject": manifest["query_subject"],
             "support_ids": manifest["support_ids"], "labels": manifest["labels"],
             "query_id": manifest["query_id"], "temporal_jitter": manifest["temporal_jitter"],
             "frame_schedule": manifest["frame_schedule"], "boundary": preparation["stimulus"]},
            model="eight-Conv VGG9 + 2048->11; Q0.15 banks/Q8.8 observations; classifier-only",
            execution={"command": status["command"],
                       "source_list": (evidence / "sources.f").as_posix(),
                       "dependency_list": (evidence / "dependencies.txt").as_posix(),
                       "source_scope_document": (SUPPLEMENT / "rtl/scnn_real_final_graph/README.md").as_posix()},
            training_samples=4, query_samples=1, query_timesteps=4,
            counters=final, commits=commits, ddr_bank_reads=ddr,
            crossings=counter_rows(scnn_text, "SCNN_TEMPORAL_CROSSING", allow_signed=True),
            initialization_cycles=initial["cycles"], initial_publication_words=initial["weight_fill_words"],
            simulation_wall_seconds=status["seconds"],
            measured_interval_boundary="total cycle_count at final PASS; includes setup, 48 train frames, 4 query frames and TB checks")
    stages.extend({"profile": name, **stage} for stage in scnn_stages)
    for counter, reason in (
        ("standalone_full_model_load_cycles/state_restore_cycles/state_spill_cycles", "71595为合并的初始设置/发布；状态搬运在层周期内，没有独立时延计数"),
        ("state_restore_words/state_spill_words/trace_transfer_words", "有完整搬运一致性断言，但最新长测未独立打印这些端口的握手字数"),
        ("Conv8_cycles/pool_cycles/classifier_cycles", "layer8 descriptor区间合并Conv8、pool和classifier，不能伪拆分"),
        ("pure_learning_cycles/commit_latency_cycles/publication_latency_cycles", "RMW/commit/publication次数已知，独立阶段起止时间戳缺失"),
        ("lane_fires/enabled_terms/exclusive_tile_buckets", "最新B4T12执行无被动活动计数，不能继承旧B1T2图监测"),
        ("learning_off_B4T12_cycles", "没有同输入独立learning-off B4T12臂；B1T2为不同工作负载"),
        ("full_T20_sample_cycles", "B4T12+queryT4是有界窗口，不能算完整DVS样本吞吐")):
        gap(name, counter, reason)

    # Append new measured rows; the five prior workload records remain intact.
    current_comparison = load(CURRENT_FC / "comparison.json")
    current_commands = load(CURRENT_FC / "commands.json")
    current_scope = current_comparison["scope"]
    if (current_comparison["status"] != "PASS" or current_commands["status"] != "PASS"
            or current_commands["scope"] != current_scope
            or current_commands["production_sv_count"] != 105
            or not current_commands["activity_monitor"]
            or any(current_commands["stages"][stage]["exit_code"] != 0 for stage in ("build", "on", "off"))
            or (current_scope["batch"], current_scope["batches"], current_scope["steps"],
                current_scope["source_steps"], current_scope["full_sample"]) != (16, 1, 99, 99, True)
            or current_commands["runs"]["on"][0] != current_commands["runs"]["off"][0]):
        raise ValueError("current FC paired execution/source/full-sample scope did not pass")
    current_profiles, current_activity = {}, []
    for arm in ("off", "on"):
        if current_comparison["arms"][arm] != arm + ".json":
            raise ValueError("unexpected current FC arm summary path")
        archived = load(CURRENT_FC / (arm + ".json"))
        measured, initial = parse_current_fc(load(CURRENT_FC / (arm + ".log"), False),
                                             archived, arm, current_scope)
        reference_path = local_fixture({"run": current_commands["runs"][arm]}) / "reference.json"
        reference = load(reference_path)
        if (reference["samples"] != current_scope["samples"] or reference["steps"] != 99
                or reference["source_steps"] != 99 or reference["explicit_debug_prefix"]
                or reference["learning"] != (arm == "on")):
            raise ValueError("current FC input/reference scope mismatch")
        name = f"FC_current_source_B16T99_{arm}"
        activity_id = f"fc_current_source_b16t99_{arm}"
        row = profile(name, "FC", arm, 16, 99, measured["runtime_cycles"], "frame_service", CURRENT_FC,
                      {"fixture": reference_path.parent.as_posix(), "samples": current_scope["samples"],
                       "shape": [16, 99, 2312], "full_sample": True,
                       "valid_timesteps_min": min(s["valid_timesteps"] for s in current_scope["samples"]),
                       "valid_timesteps_max": max(s["valid_timesteps"] for s in current_scope["samples"]),
                       "numeric": reference["numeric"], "boundary": current_scope["definition"]},
                      model="2312->160->10; W0 INT5/shift6, W2 INT8; W2-only",
                      evidence_status="accepted_current_source_PASS",
                      samples=16, batches=1, counters=measured["legacy"],
                      passive_stages=measured["stages"], bus_handshakes=measured["bus_handshakes"],
                      bus_payload_bits=measured["bus_payload_bits"], activity_id=activity_id,
                      commits=[{key: (None if key in ("observation", "update_start", "done", "commit_first",
                                  "commit", "work_first", "last_forward", "last_write") and value < 0 else value)
                                for key, value in window.items()} for window in measured["windows"]],
                      initial_load_and_refill_cycles=initial["cycles"],
                      initial_load_cycles=measured["stages"]["parameter_load"]["cycles"],
                      initial_refill_cycles=measured["stages"]["initial_refill"]["cycles"],
                      initial_state_cycles=initial["init_cycles"],
                      initial_learning_configuration_cycles=measured["stages"]["learning_configuration"]["cycles"],
                      simulation_wall_seconds=current_commands["stages"][arm]["wall_seconds"],
                      measured_interval_boundary="complete supplied B16T99 runtime; excludes parameter load, first refill, state/trace initialization and initial learning configuration; includes conservative post-commit W2 refills",
                      full_sample_throughput_is_projection=True,
                      full_sample_throughput_boundary="16 supplied first-saccade T99 samples / runtime at 15ns; not full Test, board throughput, isolated sample latency or optimal scheduling",
                      execution={"command": current_commands["runs"][arm], "cwd": current_commands["cwd"],
                                 "build_command": current_commands["build"], "production_sv_count": 105,
                                 "memory_backend": current_commands["memory_backend"],
                                 "source_scope_document": (SUPPLEMENT / "performance/current_source_measurements/fc/README.md").as_posix()})
        projected = row["projected_interval_ms"]
        row["full_sample_throughput_samples_s"] = 16 * 1000 / projected if projected else None
        if (projected is None or abs(projected - measured["projected_runtime_ms_at_15ns"]) > 1e-9
                or abs(row["full_sample_throughput_samples_s"] - measured["projected_samples_per_second_at_15ns"]) > 1e-9):
            raise ValueError("current FC archived 15ns projection disagrees with RUN137 period")
        row["cycles_including_initialization"] = sum(s["cycles"] for s in measured["stages"].values())
        row["projected_including_initialization_ms"] = project_cycles(row["cycles_including_initialization"], timing["frame_service"])
        row["isolated_pure_learning_latency_cycles"] = None
        row["platform_publication_latency_cycles"] = None
        for stage, values in measured["stages"].items():
            stages.append({"profile": name, "kind": "passive_stage_partition", "stage": stage,
                           **{key: value for key, value in values.items() if key != "payload_bits"},
                           **{f"payload_bits_{key}": value for key, value in values["payload_bits"].items()},
                           "boundary": "pre-NBA posedge stage partition; busy/forward/refill counts are overlapping subsets, not extra latency; off logit is a read-only tap, not a learner handshake"})
        for window in row["commits"]:
            stages.append({"profile": name, "kind": "learning_window_timestamps", "stage": "FC_window_commit",
                           **window, "boundary": "edge-index differences; may overlap stage/forward intervals; FC window handshake, not platform publication"})
        measured_activity = measured["h40_activity"]
        current_activity.append({"id": activity_id, "topology": "FC", "learning": arm, "batch": 16,
                         "timesteps_per_sample": 99, "source_log": (CURRENT_FC / (arm + ".log")).as_posix(),
                         "source_summary": (CURRENT_FC / (arm + ".json")).as_posix(),
                         "source_family": "frame_service", "profile_relation": name,
                         "tile_cycle_denominator": measured_activity["totals"]["cycles"],
                         "physical_lane_cycle_denominator": measured_activity["cycles"] * 480,
                         "physical_p4_term_cycle_denominator": measured_activity["cycles"] * 1920,
                         "accepted_issue_lane_denominator": measured_activity["totals"]["contributions"] * 40,
                         "accepted_lane_source_denominator": measured_activity["totals"]["lane_fires"] * 4,
                         **measured_activity})
        gap(name, "isolated_pure_learning_latency_cycles/platform_publication_latency_cycles",
            "已有学习忙并集、前向重叠和FC窗口commit边沿；非重叠忙时钟不是反事实纯学习时延，未测平台发布")
        current_profiles[arm] = row
    if (current_profiles["on"]["cycles"] - current_profiles["off"]["cycles"] !=
            current_comparison["runtime_on_minus_off_cycles"]):
        raise ValueError("current FC paired runtime difference disagrees with comparison")

    srnn_commands = load(CURRENT_SRNN / "commands.json")
    fixture = local_fixture({"run": srnn_commands["run"]["off"]})
    # Resolve the downloaded original fixture under its archive, not the live project or server cwd.
    srnn_reference = load(SRNN_RAW / fixture / "reference.json")
    validate_current_srnn_commands(srnn_commands, srnn_reference)
    launcher = load(CURRENT_SRNN / "srnn_launcher.status", False)
    if re.findall(r"^exit_code=(-?\d+)$", launcher, re.M) != ["0"]:
        raise ValueError("current SRNN launcher did not exit successfully")
    srnn_dependencies = load(CURRENT_SRNN / "dependencies.json")
    load(CURRENT_SRNN / "sources.f", False)
    load(SRNN_RAW / "scripts/run_srnn_profile_measurement.py", False)
    for arm in ("off", "on"):
        evidence = CURRENT_SRNN / arm
        measured, final, steps, measured_activity = parse_current_srnn(
            load(evidence / "run.log", False), load(evidence / "measurements.json"),
            load(evidence / "activity.json"), srnn_commands, arm, srnn_reference)
        name = f"SRNN_current_source_B1T73_native_{arm}"
        activity_id = f"srnn_current_source_b1t73_native_{arm}"
        counts = measured["counters"]
        handshakes = {key.removesuffix("_accepted"): {"accepted": value,
                       "stalled_cycles": counts[key.removesuffix("_accepted") + "_stalled_cycles"]}
                      for key, value in counts.items() if key.endswith("_accepted")}
        row = profile(name, "SRNN", arm, 1, 73,
                      measured["sample_service_cycles_including_refill_excluding_tb_checks"],
                      "frame_service", evidence,
                      {"fixture": (SRNN_RAW / fixture).as_posix(),
                       "recorded_fixture": fixture.as_posix(), "sample_definition": srnn_reference,
                       "shape": [1, 73, 700], "full_sample": True,
                       "gradient_divisor": 64,
                       "boundary": "unshifted official sample0 native73, raw700 backbone spikes feed M2 on the same DUT; not official first-batch B64 padded89 or full Test"},
                      model="700->450 recurrent + 450->256->20 M2; INT8 backbone/Q2.14 M2; Wout-only",
                      evidence_status="accepted_current_source_PASS", samples=1, batches=1,
                      counters=final, passive_counters=counts, passive_stages=measured["stage_cycles"],
                      passive_intervals=measured["intervals"], commits=measured["events"],
                      bus_handshakes=handshakes, bus_payload_bytes=None, activity_id=activity_id,
                      initial_load_and_refill_cycles=measured["stage_cycles"]["parameter_load"],
                      initial_load_cycles=measured["stage_cycles"]["parameter_load"],
                      initial_state_cycles=measured["stage_cycles"]["initial_state"],
                      initial_load_and_state_cycles=measured["initial_load_and_state_cycles"],
                      initial_reset_and_configuration_cycles=measured["stage_cycles"]["reset"] + measured["stage_cycles"]["config"],
                      tb_store_check_cycles=measured["tb_store_check_cycles"],
                      tb_store_check_host_seconds=measured["tb_store_check_host_seconds"],
                      simulation_wall_seconds=measured["simulation_wall_seconds"],
                      measured_interval_boundary="one native73 sample; end-initialized-tb_store_check cycles, includes every backbone/M2 refill and TB scheduling, excludes initial parameter/state/reset/config",
                      full_sample_throughput_is_projection=True,
                      full_sample_throughput_boundary="one complete supplied sample0 native B1T73 / service interval at 15ns; not Test accuracy, official B64 padded89 throughput or board measurement",
                      isolated_pure_learning_latency_cycles=None, commit_latency_cycles=None,
                      platform_publication_latency_cycles=None,
                      measured_hardware_throughput_samples_s=None,
                      learner_busy_excluding_pure_predicate_cycles=counts["learner_busy_cycles"] - counts["pure_learning_busy_cycles"],
                      learning_busy_boundary=srnn_commands["busy_and_event_definitions"]["pure_learning_busy_cycles"] + "; observed clocks, not isolated causal learning latency; hardware intervals overlap stage intervals and can overlap by an edge",
                      boundaries=measured["boundaries"],
                      execution={"command": srnn_commands["run"][arm], "cwd": srnn_commands["cwd"],
                                 "build_command": srnn_commands["build"], "production_sv_count": 105,
                                 "build_exit": srnn_commands["build_exit"], "run_exit": srnn_commands[arm + "_exit"],
                                 "launcher_exit": 0, "build_warnings": srnn_commands["build_warnings"],
                                 "source_list": (CURRENT_SRNN / "sources.f").as_posix(),
                                 "dependency_list": (CURRENT_SRNN / "dependencies.json").as_posix(),
                                 "recorded_dependencies": srnn_dependencies,
                                 "archived_runner": (SRNN_RAW / "scripts/run_srnn_profile_measurement.py").as_posix(),
                                 "counter_definitions": srnn_commands["counter_definitions"],
                                 "busy_and_event_definitions": srnn_commands["busy_and_event_definitions"],
                                 "memory_backend": "behavioral simulation memories; no UNIFIED_USE_XPM_MEMORY"})
        projected = row["projected_interval_ms"]
        row["full_sample_throughput_samples_s"] = 1000 / projected if projected else None
        if (projected is None or abs(projected - measured["projected_sample_interval_ms"]) > 1e-9
                or abs(row["full_sample_throughput_samples_s"] - measured["full_sample_throughput_samples_s_projection"]) > 1e-12):
            raise ValueError("current SRNN archived 15ns projection disagrees with RUN137 period")
        row["cycles_including_initialization"] = measured["terminal"]["end"]
        row["projected_including_initialization_ms"] = project_cycles(row["cycles_including_initialization"], timing["frame_service"])
        for interval in measured["intervals"]:
            stages.append({"profile": name, **interval, "stage": interval["phase"],
                           "boundary": "original stage partition or hardware signal edge interval; hardware subsets overlap stages; in the on arm m2_forward and pure_learning_busy overlap by 1 clock per step, 73 clocks total; never sum these as exclusive costs"})
        stages.extend({"profile": name, **event, "kind": "commit_event", "stage": event["name"],
                       "boundary": "observed commit pulse clock, not standalone commit/publication latency"}
                      for event in measured["events"])
        for counter, value in counts.items():
            stages.append({"profile": name, "kind": "passive_counter", "stage": counter, "value": value,
                           "units": "accepted_transfers" if counter.endswith("_accepted") else "observed_clocks_or_pulses",
                           "boundary": "actual pre-edge accepted valid&&ready, valid&&!ready stalls, or named signal; whole measured run, not DDR bytes or additive latency"})
        current_activity.append({"id": activity_id, "topology": "SRNN", "learning": arm, "batch": 1,
                         "timesteps_per_sample": 73, "source_log": (evidence / "run.log").as_posix(),
                         "source_summary": (evidence / "activity.json").as_posix(),
                         "source_family": "frame_service", "profile_relation": name,
                         "tile_cycle_denominator": measured_activity["totals"]["cycles"],
                         "physical_lane_cycle_denominator": measured_activity["cycles"] * 480,
                         "physical_p4_term_cycle_denominator": measured_activity["cycles"] * 1920,
                         "accepted_issue_lane_denominator": measured_activity["totals"]["contributions"] * 40,
                         "accepted_lane_source_denominator": measured_activity["totals"]["lane_fires"] * 4,
                         **measured_activity})
        gap(name, "isolated_pure_learning_latency_cycles/commit_latency_cycles/platform_publication_latency_cycles",
            "已有predicate学习忙区间和commit脉冲边沿；不能将重叠区间或两臂差当独立学习/提交/平台发布时延")
        gap(name, "bus_payload_bytes/tb_store_check_host_seconds",
            "保留真实accepted/stalled握手字数；未独立计payload字节或分离TB检查主机耗时，0仿真检查周期不等于0主机耗时")

    scnn_preparation = load(CURRENT_SCNN / "preparation.json")
    scnn_fixture = Path(scnn_preparation["fixture"])
    if scnn_fixture.is_absolute() or ".." in scnn_fixture.parts:
        raise ValueError("current SCNN fixture must be recorded project-local")
    scnn_reference = load(scnn_fixture / "manifest.json")
    scnn_statuses = {key: load(CURRENT_SCNN / filename) for key, filename in
                     (("elab", "elab.status.json"), ("build", "build.status.json"),
                      ("off", "run_off.status.json"), ("on", "run_on.status.json"))}
    validate_current_scnn_records(scnn_preparation, scnn_reference, scnn_statuses,
                                  load(CURRENT_SCNN / "scnn_launcher.status", False))
    scnn_summary = load(CURRENT_SCNN / "summary.json")
    if (scnn_summary["status"] != "PASS" or not scnn_summary["actual_final_top"]
            or not scnn_summary["fixed_sample_only"]):
        raise ValueError("current SCNN archived summary did not pass")
    for path in SCNN_RAW_FILES:
        load(SCNN_RAW / path, False)
    scnn_source_list = load(CURRENT_SCNN / "sources.f", False)
    if len([line for line in scnn_source_list.splitlines() if line.strip()]) != scnn_preparation["source_files"]:
        raise ValueError("current SCNN recorded source count differs")
    scnn_dependencies = load(CURRENT_SCNN / "dependencies.txt", False).splitlines()
    scnn_warnings = {}
    for key in ("elab", "build"):
        text = load(CURRENT_SCNN / (key + ".log"), False)
        check_scnn_build_log(text)
        codes = re.findall(r"%Warning-([A-Z0-9_]+):", text)
        scnn_warnings[key] = {code: codes.count(code) for code in sorted(set(codes))}
    for arm in ("off", "on"):
        measured, frames, busy, intervals, events, measured_activity = parse_current_scnn(
            load(CURRENT_SCNN / ("run_" + arm + ".log"), False),
            scnn_summary["measurements"][arm], scnn_reference, scnn_statuses[arm], arm)
        name = f"SCNN_current_source_B1T20_{arm}"
        activity_id = f"scnn_current_source_b1t20_{arm}"
        scnn_stages = current_scnn_stages(name, measured, frames, busy, intervals, events)
        row = profile(name, "SCNN", arm, 1, 20, measured["timing"]["sample_cycles"], "frame_service", CURRENT_SCNN,
                      {"fixture": scnn_fixture.as_posix(), "sample_index": 0, "full_sample": True,
                       "input_words_per_frame": 2048, "label": scnn_reference["label"],
                       "multiplier": 640, "corruption": scnn_reference["corruption"],
                       "state": scnn_reference["state"], "protocol": scnn_reference["protocol"],
                       "frame_schedule": scnn_reference["frame_schedule"],
                       "boundary": "one original clean sample0; continuous five underfilled B1T4 windows, no intermediate reset/reload or fresh query; not shifted 52-frame adaptation/full Test"},
                      model="eight-Conv VGG9 + 2048->11; Q0.15 banks/Q8.8 observations; classifier-only",
                      evidence_status="accepted_current_source_PASS", samples=1, batches=1,
                      counters=measured["final"], passive_timing=measured["timing"],
                      active_counters=measured["active"], flow_counts=measured["traffic"],
                      bus_payload=measured["payload"], ddr_bank_reads=measured["banks"],
                      descriptor_intervals=measured["descriptors"], control_intervals=measured["control"],
                      phase_intervals=intervals, phase_events=events, learning_busy_by_window=busy,
                      publication_request_to_done=[s for s in scnn_stages if s["kind"] == "publication_boundary"],
                      reference_windows=scnn_reference["windows"][arm],
                      commits=[e for e in events if e["kind"] == "commit"],
                      setup_cycles=measured["timing"]["setup_cycles"],
                      initial_load_and_refill_cycles=None, initial_state_cycles=None,
                      initial_publication_words=1892,
                      runtime_publication_words=measured["final"]["publications"] - 1892,
                      cycles_including_initialization=measured["timing"]["included_setup_cycles"],
                      simulation_wall_seconds=measured["host_simulator_wall_seconds"],
                      host_file_load_seconds=measured["host_file_load_seconds"],
                      host_golden_seconds=measured["host_golden_seconds"],
                      full_sample_throughput_is_projection=True,
                      full_sample_throughput_boundary="one complete fixed sample0 T20 / service interval at 15ns; excludes setup, not board throughput, full Test, B4-normalized adaptation or optimal scheduling",
                      measured_interval_boundary="sample_end-first accepted window request; includes all twenty frames, runtime refill/state/trace/learning/publication/check scheduling; excludes initial setup/publication",
                      isolated_pure_learning_latency_cycles=None,
                      platform_publication_latency_cycles=None,
                      source_throughput=measured["throughput"], activity_id=activity_id,
                      caveats=measured["caveats"],
                      execution={"command": scnn_statuses[arm]["command"], "production_sv_count": 105,
                                 "stage_statuses": scnn_statuses, "launcher_exit": 0,
                                 "warnings": scnn_warnings, "source_list": (CURRENT_SCNN / "sources.f").as_posix(),
                                 "dependency_list": (CURRENT_SCNN / "dependencies.txt").as_posix(),
                                 "recorded_dependencies": scnn_dependencies,
                                 "archived_sources": [(SCNN_RAW / path).as_posix() for path in SCNN_RAW_FILES],
                                 "memory_backend": "behavioral native DDR/store models; no UNIFIED_USE_XPM_MEMORY"})
        row["timing"] = {**timing["frame_service"], "source_boundary": "RUN137 exact RUN136 DCP; documented current frame-service RTL for SCNN current-source original B1T20 R2; behavioral simulation is not XPM/STA validation"}
        projected = row["projected_interval_ms"]
        row["full_sample_throughput_samples_s"] = 1000 / projected if projected else None
        row["projected_including_initialization_ms"] = project_cycles(row["cycles_including_initialization"], timing["frame_service"])
        if (projected is None or abs(row["full_sample_throughput_samples_s"] - measured["throughput"]["projected_15ns_samples_per_second_excluding_setup"]) > 1e-12
                or abs(1000 / row["projected_including_initialization_ms"] - measured["throughput"]["projected_15ns_samples_per_second_including_setup"]) > 1e-12):
            raise ValueError("current SCNN archived 15ns projection disagrees with RUN137 period")
        stages.extend(scnn_stages)
        current_activity.append({"id": activity_id, "topology": "SCNN", "learning": arm, "batch": 1,
                         "timesteps_per_sample": 20, "source_log": (CURRENT_SCNN / ("run_" + arm + ".log")).as_posix(),
                         "source_summary": (CURRENT_SCNN / "summary.json").as_posix(),
                         "source_family": "frame_service", "profile_relation": name,
                         "tile_cycle_denominator": measured_activity["totals"]["cycles"],
                         "physical_lane_cycle_denominator": measured_activity["cycles"] * 480,
                         "physical_p4_term_cycle_denominator": measured_activity["cycles"] * 1920,
                         "accepted_issue_lane_denominator": measured_activity["totals"]["contributions"] * 40,
                         "accepted_lane_source_denominator": measured_activity["totals"]["lane_fires"] * 4,
                         **measured_activity})
        gap(name, "standalone_full_model_load_cycles/initial_state_cycles/host_file_load_seconds/host_golden_seconds",
            "setup为初始化/首次classifier发布合并区间，readmemh和golden仿真周期为0，独立主机成本及完整模型DMA装载仍未知")
        gap(name, "isolated_pure_learning_latency_cycles/platform_publication_latency_cycles/board_DDR_time",
            "忙信号、commit-pending和发布request-to-done已计数但互相包含，非反事实独立学习或端到端平台时延；原生DDR模型不能替代板测")

    activity_runs = (
        ("fc_b16t99_on", "FC", "on", 16, 99, "rtl/fc_final_counts/server_activity_b16_r2_on/run.log", "FC_FINAL_COUNTS_PASS"),
        ("fc_b16t99_off", "FC", "off", 16, 99, "rtl/fc_final_counts/server_activity_b16_r2_off/run.log", "FC_FINAL_COUNTS_PASS"),
        ("srnn_raw700_b2t8", "SRNN_backbone", "off", 2, 8, "rtl/srnn_raw700/server_run1/run.log", "SRNN_RAW700_BACKBONE_PASS"),
        ("srnn_m2_observed_on", "SRNN_M2_replay", "on", 2, 8, "rtl/srnn_formal_m2/server_observed_b2t8_on/simulation.log", "SRNN_FORMAL_M2_ONLINE_PASS"),
        ("srnn_m2_observed_off", "SRNN_M2_replay", "off", 2, 8, "rtl/srnn_formal_m2/server_observed_b2t8_off/simulation.log", "SRNN_FORMAL_M2_ONLINE_PASS"),
        ("scnn_graph_b1t2_r2", "SCNN_graph", "off", 1, 2, "rtl/scnn_graph_closure/server_run2_signed_capture/graph_run.log", "SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS"),
        ("scnn_classifier_cold_b1t2", "SCNN_classifier_replay", "on", 1, 2, "rtl/scnn_graph_closure/server_run2_signed_capture/learner_run.log", "SCNN_GRAPH_STAGED_LEARN_RECOMPUTE_PASS"),
    )
    for identifier, topology, arm, batch, steps, log_path, marker in activity_runs:
        log = SUPPLEMENT / log_path
        archived = SUPPLEMENT / f"performance/h40_activity/{identifier}.json"
        measured = parse_activity(load(log, False), marker)
        prior = load(archived)
        if any(measured[k] != prior[k] for k in ("cycles", "totals", "modes")):
            raise ValueError(f"activity archive disagrees with log: {identifier}")
        activity.append({"id": identifier, "topology": topology, "learning": arm,
                         "batch": batch, "timesteps_per_sample": steps, "source_log": log.as_posix(),
                         "source_summary": archived.as_posix(), "source_family": "earlier independent activity execution; not latest frame-service source",
                         "profile_relation": "not interchangeable with support160, B1 same-DUT or real-final B4T12",
                         "tile_cycle_denominator": measured["totals"]["cycles"],
                         "physical_lane_cycle_denominator": measured["cycles"] * 480,
                         "physical_p4_term_cycle_denominator": measured["cycles"] * 1920,
                         "accepted_issue_lane_denominator": measured["totals"]["contributions"] * 40,
                         "accepted_lane_source_denominator": measured["totals"]["lane_fires"] * 4,
                         **measured})

    activity.extend(current_activity)

    def accuracy_row(topology, arm, correct, samples, path, boundary):
        accuracy.append({"topology": topology, "arm": arm, "correct": correct, "samples": samples,
                         "accuracy_percent": correct * 100 / samples, "execution": "software",
                         "rtl_dataset_accuracy_percent": None, "source": path.as_posix(), "boundary": boundary})

    path = SUPPLEMENT / "algorithm/fc_formal_test_a1/accuracy.json"
    data = load(path)
    if data["status"] != "PASS":
        raise ValueError("FC software accuracy incomplete")
    for arm, metric in data["metrics"].items():
        accuracy_row("FC", arm, metric["correct"], metric["samples"], path,
                     "official Test10000, same quantized weights; floats are not original FP32 checkpoint; separate from support160 RTL")
    path = SUPPLEMENT / "algorithm/srnn_full_integer/official2264_cuda_20261008/result.json"
    data = load(path)
    if data["status"] != "completed_software_full_integer_not_full_test_RTL":
        raise ValueError("SRNN software accuracy incomplete")
    accuracy_row("SRNN", "full_integer", data["correct"], data["samples"], path, "official Test2264; not B1T8 RTL accuracy")
    accuracy_row("SRNN", "formal_fp32", data["fp32_correct"], data["samples"], path, "official FP32 reference; not M2-weight-only quantization accuracy")
    path = SUPPLEMENT / "algorithm/scnn_full_integer/fixed264_cuda_20261008/result.json"
    data = load(path)
    if data["status"] != "pass":
        raise ValueError("SCNN software accuracy incomplete")
    for arm in ("full_integer", "same_banks_fp32"):
        accuracy_row("SCNN", arm, data["arms"][arm]["correct"], data["samples"], path,
                     "fixed balanced 264x20 clean subset, same hardware banks/strict execution; not full DVS Test or B4T12 RTL accuracy")
    gap("all", "board_time/DDR_bandwidth/measured_power/full_dataset_RTL_accuracy", "均未测量；仿真墙钟只表示主机成本，软件准确率独立列出")
    return {"scope": "read-only local consolidation of existing bounded evidence; no new experiments or live-source PPA",
            "profiles": profiles, "stages": stages, "activity": activity, "software_accuracy": accuracy,
            "missing_counters": missing, "sources": sorted(sources)}


def profile_csv_rows(profiles):
    rows = []
    for profile in profiles:
        row = {field: profile.get(field) for field in PROFILE_CSV_FIELDS}
        row["period_ns"] = (profile.get("timing") or {}).get("period_ns")
        rows.append(row)
    return rows


def activity_csv_rows(activity):
    return [{**{key: value for key, value in row.items() if not isinstance(value, (dict, list))},
             **{f"total_{key}": value for key, value in row["totals"].items()}}
            for row in activity]


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with filesystem_path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if any(isinstance(value, (list, dict, tuple, set)) for value in row.values()):
                raise TypeError("CSV cells must be scalar; nested details belong in metrics.json")
            writer.writerow(row)


def chinese_readme(report, output=ROOT / OUTPUT, root=ROOT):
    root, output = normal_path(root), normal_path(output)
    lines = ["# 最终固定画像指标：本地有界汇总（2026-10-09）", "",
             "仅解析已有本地日志/JSON/时序报告，不调用 SSH、EDA、GPU 或仿真，不重跑实验，不修改 RTL/TB/共享文档。",
             "输出不是新实验 PASS，也不是当前 live RTL 的重新资格认证；来源关联沿用归档说明，不新建 hash/基线/门禁。", "",
             "## 复现", "", "在项目根执行（仅 Python 标准库）：", "", "```powershell",
             "python -B scripts/summarize_final_profile_metrics.py",
             "python -B -m unittest discover -s tests -p test_final_profile_metrics.py -v", "```", "",
             "默认写本目录；可用 --output-dir 指定本目录内的子目录，不能写其他共享路径。",
             "Windows文件读写自动处理长路径、UNC及已有扩展前缀；JSON来源字段和README链接仍保持原相对路径，不写入扩展前缀。",
             "metrics.json 是详细版本，完整保留 input/execution/counters/commits/ddr_bank_reads、原始样本ID及全部来源，不因CSV精简而删除。",
             "profiles.csv 是论文表格/Excel用的精简版本：固定核心标量字段，另从timing提取period_ns，不把嵌套JSON塞进单元格。",
             "activity.csv 只保留标量元数据及total_前缀总计数列；逐tile和mode明细留在metrics.json。stages.csv、software_accuracy.csv、missing_counters.csv同样只含标量字段，可独立读取。",
             "JSON null / CSV 空字段表示未测量，不等于零；stages 中父区间与子区间不可重复相加。", "",
             "## 主工作负载", "",
             "| 记录 | B / T | 实测 cycles | 区间时间投影 ms | 仿真主机墙钟 s |",
             "|---|---|---:|---:|---:|"]
    for row in report["profiles"]:
        projection = row["projected_interval_ms"]
        projected = f"{projection:.6f}" if projection is not None else "未投影"
        lines.append(f"| {row['id']} | {row['batch']} / {row['timesteps_per_sample']} | {row['cycles']} | "
                     f"{projected} | {row['simulation_wall_seconds']:.3f} |")
    lines.extend(["", "### 输入、装载与学习边界", "",
                  "- 历史 FC cache_r3：2312→160→10，cache_r3_support160_r2 两臂，同一 exe，原 seed38472/+2 像素 support160 顺序，10×B16T99。保留事件重数和 padding，不截断为49槽；未覆盖为当前源码新记录。",
                  "- 历史 FC 区间排除首次装载/补载1542534 cycles 与状态初始化641 cycles，包含后续补载和批间初始化。含初始化时间另列 JSON/CSV；只提供该固定 support 区间的等效速率，不称板级或稳态吞吐。",
                  "- 历史 FC W2-only：开启30次 code 写、397621次 residual 写、990 commit，最终26地址变化；两臂各自匹配整数参考，t49首次可见。learning_cycles 是末样本后等待；refill_cycles 含首次补载，不能整体再加进区间。",
                  "- 历史 SRNN cache_r3：700→450循环骨干→450→256→20 M2，同一 production final top，官方 sample0 前8步，B1；实际 DUT spike 驱动 M2/循环反馈，不拼接两个独立 DUT 的周期。Wout-only，梯度仍固定 /64，不是B64准确率。",
                  "- 历史 SRNN 开启1680 code/4940 residual 写、8 commit/detach；cycles 从 initialized_cycle 后起算，包含工作集补载和 TB 读回检查。store=962560 是存储检查项数，不是访存握手。初次装载和各阶段时间未打印，不能恢复或外推完整 SHD samples/s。",
                  "- SCNN：真实最终组合顶层，seed38472、subject29→28、jitter1、support IDs 0..3，三个B4T4连续窗口构成B4T12，再对 query22 执行B1T4，即48训练帧+4查询帧，不是52个完整样本。模型为八卷积VGG9+2048→11，仅classifier可写。",
                  "- SCNN 三窗 residual 变化20041/20011/20221，code变化0/0/662，3 commit；地址18433由870变869，query current/logit各4组变化。fresh_reads=36868608 仅查询重读；ddr_bank_reads 是各bank整个执行的实际TB握手，不能用fresh_reads替代全程流量或称板级DDR带宽。",
                  ""])
    current_fc = [r for r in report["profiles"] if r["id"].startswith("FC_current_source_B16T99_")]
    lines.extend(["### 当前源码 FC 完整给定样本", "",
                  "独立 server_pair_r1 两臂均PASS，同一新编译exe、105生产SV加测试store和monitor。真实 first-saccade 输入为16×99×2312，有效长度94–99；全部99步含padding执行，不强补T100，不代表full Test或板级。旧cache_r3行及旧活动记录原样保留，数值偶然相同也不替代新来源。",
                  "", "| 当前 FC 臂 | runtime cycles | 15 ns投影 samples/s |", "|---|---:|---:|"])
    for row in current_fc:
        rate = row["full_sample_throughput_samples_s"]
        lines.append(f"| {row['learning']} | {row['cycles']} | " + (f"{rate:.7f}" if rate is not None else "未投影") + " |")
    lines.extend(["", "上述full_sample_throughput_samples_s字段仅表示16个给定完整T99样本的15 ns区间投影，不是测得的板级吞吐、独立单样本时延或最优调度。runtime排除参数装载743040、首次补载799494、state/trace初始化641，以及初始学习配置（off 0/on 2）cycles；开启臂仍含逐commit保守W2补载。",
                  "", "| 被动计数 | off | on |", "|---|---:|---:|"])
    by_arm = {row["learning"]: row for row in current_fc}
    for key, label in (("work_busy", "学习工作忙并集时钟"), ("work_overlap_forward", "其中与前向重叠"),
                       ("work_only", "其中非前向/补载的忙时钟")):
        values = [sum(s[key] for s in by_arm[arm]["passive_stages"].values()) for arm in ("off", "on")]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    for key, label in (("learning_reads", "learning读请求握手"), ("residual_writes", "16-bit residual写使能握手"),
                       ("code_writes", "16-bit code写使能握手"), ("commits", "FC窗口commit握手")):
        lines.append(f"| {label} | {by_arm['off']['counters'][key]} | {by_arm['on']['counters'][key]} |")
    lines.extend(["", "学习busy是frontend/reducer/update忙信号的并集，不是各项之和；非重叠忙时钟也不是反事实纯学习因果时延。末样本后等待7288871与非重叠忙7288079不是同一指标，612216重叠忙不可再加到前向周期。两臂差7701008 cycles还含调度及W2补载，不称纯学习成本。",
                  "commit的observation/update/done/valid/accept边沿及其差值保留在JSON与stages.csv，起止差不加1；只是FC学习窗握手，不是平台发布。缺失的last_write等时间戳转为null，off没有学习窗，不补成0时延。本批on有99 commit但code写为0，不继承support160的30次code写或新权重可见性结论。",
                  "各stage总线握手、mask有效payload位数保留；store/维护总线不是板级DDR字数，off logit是只读tap而非learner握手。当前12tile活动仅解析同一on/off.log，monitor包含初始化，不能套runtime分母。仿真沿用行为存储，不定义XPM宏，不是XPM原语仿真或新STA。"])
    current_srnn = [r for r in report["profiles"] if r["id"].startswith("SRNN_current_source_B1T73_native_")]
    lines.extend(["", "### 当前源码 SRNN 完整 native sample0", "",
                  "独立server_pair_r1的launcher/build/off/on退出均0；同一exe、105生产SV加验证store/monitor，原fixture从完整回传归档读取，不按服务器cwd去匹配本地工作区。两臂完整日志以现有run_srnn_profile_measurement.parse_log重解析，并核对原measurements/activity JSON和73步数值完结。",
                  "输入是未偏移的官方sample0，B1 native T73，raw700→循环骨干→M2在同一DUT执行；不是旧T8前缀，也不是官方firstbatch B64 padded T89。Wout-only和固定梯度/64保持，即使执行B1也不改为/1。单个给定完整样本通过，不代表full Test准确率、B64吞吐或板测。",
                  "", "| 当前 SRNN 臂 | sample service cycles | 15 ns投影 samples/s |", "|---|---:|---:|"])
    for row in current_srnn:
        rate = row["full_sample_throughput_samples_s"]
        lines.append(f"| {row['learning']} | {row['cycles']} | " + (f"{rate:.12f}" if rate is not None else "未投影") + " |")
    lines.extend(["", "服务区间=终态end−initialized−TB检查仿真周期；含全部骨干/M2工作集refill和TB调度，不含初始120426 cycles（reset6+config2+参数装载120321+state初始化97）。TB存储检查8783360项使用层次peek，不推进仿真时钟，检查周期为0；主机检查耗时未知null，仿真墙钟不能当硬件时间。",
                  "", "| 真实阶段 / 信号计数 | off | on |", "|---|---:|---:|"])
    srnn_by_arm = {row["learning"]: row for row in current_srnn}
    for key, label in (("backbone_refill_ff", "骨干前馈补载阶段"), ("backbone_refill_r", "骨干循环补载阶段"),
                       ("backbone", "骨干阶段"), ("m2", "M2含准备/前向/学习阶段")):
        lines.append(f"| {label} | {srnn_by_arm['off']['passive_stages'][key]} | {srnn_by_arm['on']['passive_stages'][key]} |")
    for key, label in (("learner_busy_cycles", "learner busy高时钟"),
                       ("pure_learning_busy_cycles", "learner busy且M2 engine空闲且处于M2阶段"),
                       ("backbone_parameter_read_accepted", "骨干参数读握手"),
                       ("m2_refill_read_accepted", "M2补载读握手"),
                       ("store_learning_read_accepted", "学习读握手"),
                       ("code_write_accepted", "code写使能握手"),
                       ("residual_write_accepted", "residual写使能握手"),
                       ("learner_commit_accepted", "learner commit握手")):
        lines.append(f"| {label} | {srnn_by_arm['off']['passive_counters'][key]} | {srnn_by_arm['on']['passive_counters'][key]} |")
    lines.extend(["", "stage区间连续分账，hardware的m2_prepare/m2_forward/pure_learning_busy是重叠子区间，不可再加到stage总数；on臂每步m2_forward与pure_learning_busy边沿重叠1周期，73步合计73周期，不能相加为互斥耗时。保留原始边沿、区间值和总周期，不修正为互斥。1906488 learner忙时钟中1635671满足上述predicate，余270817只表示不满足predicate，不能直接命名为前向重叠或额外学习耗时。两臂差1635671恰与predicate计数相同，仍不冒充反事实纯学习时延。",
                  "73个commit脉冲时间戳、每步准备/前向/学习忙区间、全程真实accepted/stalled通道握手均保留在JSON/stages.csv；没有独立commit或平台发布起止时延，留null。bus握手以通道分列，同一传输可经过多个边界，不跨通道加成DDR总流量；未独立测payload字节。fixture中的zero_skips/saturations是软件期望，不提升为新RTL计数。",
                  "本轮12tile只用同臂run.log，monitor覆盖reset释放后的初始化/补载/学习/调度时钟：off110572522、on112208193；与服务区间分母不同。行为存储仿真不定义XPM宏，不是XPM原语仿真、新STA或板测。",
                  "", "### SCNN 真实阶段", "",
                  "stages.csv 由 SCNN_TEMPORAL_LAYER 和 FRAME_STATUS 汇总416个descriptor与52帧。逐层周期是真实日志值，不借用旧T2实验。FRAME_PASS的八层累计requests/contexts/scalars按相邻帧差分，还原每descriptor的计数，不能当DDR字节或模型MAC数。",
                  "Conv8区间含pool+classifier；帧末还可能等待学习、发布及TB检查，frame_minus_layers 不标纯学习耗时。window 行用实际开始时间戳差，包含中间控制/设置。初始化71595 cycles含初始classifier publication 1892字，不能叫整模型独立装载时延。",
                  ""])
    scnn = next(row for row in report["profiles"] if row["topology"] == "SCNN")
    selected = [s for s in report["stages"] if s["profile"] == scnn["id"]]
    lines.extend(["| 互斥记账项 | cycles |", "|---|---:|"])
    for layer in range(1, 9):
        total = sum(s["cycles"] for s in selected if s["kind"] == "layer" and s["layer"] == layer)
        lines.append(f"| Conv{layer}" + ("+pool+classifier" if layer == 8 else "") + f" | {total} |")
    for label, value in (("全部frame内未分段余量", sum(s["cycles"] for s in selected if s["stage"] == "frame_minus_layers")),
                         ("初始设置", scnn["initialization_cycles"]),
                         ("frame外未归因余量", next(s["cycles"] for s in selected if s["stage"] == "outside_frames"))):
        lines.append(f"| {label} | {value} |")
    current_scnn = [r for r in report["profiles"] if r["id"].startswith("SCNN_current_source_B1T20_")]
    scnn_by_arm = {row["learning"]: row for row in current_scnn}
    lines.extend(["", "### 当前源码 SCNN 原始 sample0 完整T20", "",
                  "server_pair_r2的launcher/elab/build/off/on退出均0，同一新exe、105生产SV加验证store/TB/12tile被动monitor。现有run_scnn_profile_measurement.parse_result只读重解析两臂原始日志，与原summary及固定reference核对；不调用producer summarize、不重写主证据。elab/build各18条普通Verilator告警保留，结构阻断项为0；准备文件中的估时/not_measured是执行前元数据，不当作终态。",
                  "这是原始clean sample0，B1连续五个既有T4窗口构成完整T20，仍用固定640缩放；窗口间不reset/reload，没有fresh query，不是旧shifted 52帧B4T12+query4。每臂20帧/160 descriptor、40960 feature、220 current/logit和480 probe检查通过。on另有40960 classifier trace值检查、85600 RMW、0 code写、85600 residual写和5 commit，不等于全Test准确率或已证明适应收益。",
                  "", "| 当前 SCNN 臂 | runtime cycles | 15 ns投影 samples/s |", "|---|---:|---:|"])
    for row in current_scnn:
        rate = row["full_sample_throughput_samples_s"]
        lines.append(f"| {row['learning']} | {row['cycles']} | " + (f"{rate:.12f}" if rate is not None else "未投影") + " |")
    lines.extend(["", "两臂均排除71594 cycles setup（reset释放至首个request，含初始classifier publication）。包含setup总周期off781728973/on784597050另列JSON/CSV。模型/oracle readmemh和165次golden检查在仿真中不推进时钟，golden_cycles=0；host_file_load_seconds/host_golden_seconds仍null，不能称0主机成本、完整模型DMA装载或板级吞吐。原报告的10ns仿真比率只保留在source_throughput，统一表只填RUN137关联的15ns投影，不是新STA。",
                  "", "| 被动计数（按各自范围） | off | on |", "|---|---:|---:|"])
    for key, label in (("learning", "runtime梯度/update busy时钟"), ("learning_exclusive", "其中排除forward/publication重叠"),
                       ("publication", "runtime publisher busy时钟"), ("forward_learning_overlap", "forward/learning重叠时钟"),
                       ("learning_publication_overlap", "learning/publication重叠时钟")):
        lines.append(f"| {label} | {scnn_by_arm['off']['active_counters'][key]} | {scnn_by_arm['on']['active_counters'][key]} |")
    for key, label in (("forward_reads", "全监测窗口native参数读握手"),
                       ("learning_reads", "全监测窗口learning读握手"), ("code_write_words", "code写使能握手"),
                       ("residual_write_words", "residual写使能握手"),
                       ("state_restore_rows", "320-bit state restore行"), ("trace_restore_rows", "320-bit trace restore行"),
                       ("state_spill_rows", "320-bit state spill行"), ("trace_spill_rows", "320-bit trace spill行")):
        lines.append(f"| {label} | {scnn_by_arm['off']['flow_counts'][key]} | {scnn_by_arm['on']['flow_counts'][key]} |")
    lines.extend(["", "初始publication两臂均1892个原生fill握手字；on后续五次共9460字，总11352字，off总1892字。真实状态/trace搬运每行320bit=20个16bit标量=40byte；restore mask计数及各DDR-model bank read/response、code/residual有效写payload分列保留，不加address/tag/control，也不称板级DDR带宽。traffic为reset释放后的全窗口计数，active为首request后的sample计数，不能混用区间。store_loads=0是monitor打印常量，不升级为被动端口测量。",
                  "descriptor/Conv8/pool/classifier控制区间、frame/window父区间互相包含，不能相加。pool_control含Conv8 writeback收集，classifier_execute不含tail capture；不能把它们拆成互斥卷积/池化/分类器耗时。sample周期可以用descriptor+frame余量+frame外余量分账，但busy/控制/提交发布计数不能再加进去。",
                  "on有91393段碎片化学习忙区间，按五窗汇总段数、busy时钟、first_start/last_end及最小/最大段长；首尾范围含空隙，不伪装为连续时延。原始逐段learning_start/interval仍在本轮run_on.log/summary，已全部重解析核对，不在统一表重复展开九万行。learning_exclusive611436只是指定predicate忙时钟，不是所有学习支持或反事实纯学习成本；两臂差2868077 cycles也不是纯学习耗时。",
                  "五个commit_pending_to_pulse区间和backend publish request-to-done边沿均保留，前者包含发布与drain，不能与publication busy再加成独立时延。端到端平台发布/板级DDR时间仍null。两臂自己12tile monitor覆盖初始化至finish，off781728973/on784597050，不套runtime分母；不继承旧T2或52帧活动记录。"])
    lines.extend(["", "## 时序投影范围", "",
                  "投影公式为 cycles×period_ns/10^6 ms，独立于 simulator walltime；不是仿真器输出的虚拟时间或板测硬件时间。",
                  "历史FC support160/SRNN前8步仅关联cache_r3快照的RUN131 15 ns OOC核心报告；不能称为后来frame-service live源码PPA。旧SCNN B4T12 R2、新FC B16T99、SRNN native B1T73及新SCNN B1T20关联RUN136 exact-DCP的RUN137 15 ns OOC核心报告，仍只是区间投影。",
                  "没有移植旧V2、RUN125、旧单模式或私有学习器资源百分比，没有基于裕量重新估算Fmax。学习关闭是同一exe的运行时开关，不等于ENABLE_ONLINE_LEARNING=0物理裁剪。RUN139 R2/R3历史负setup保留；2026-10-10 R5学习关和RUN142私有臂已各自15 ns OOC实际布线闭合，性能计数不改。R5与RUN137策略不同，不称严格同recipe学习支持布线成本；RUN137/RUN142另有匹配布线资源表。",
                  "所有行保留普通告警、未约束板级I/O/DDR、无板测时间/功耗边界；历史短SRNN/SCNN窗口的full_sample_throughput始终留空，新FC/SRNN/SCNN完整给定样本仅填投影。", "",
                  "## lane / P4", "",
                  "七项历史记录、两项当前FC、两项当前SRNN及两项当前SCNN均重新解析各自被动bind日志，并与对应JSON核对。当前记录仅关联各自本轮主记录；历史监测不继承为当前源码，旧SCNN 52帧活动缺口也不借新T20补填。",
                  "全窗口lane分母=480×monitor cycles；P4分母=1920×monitor cycles；发射内lane分母=40×contributions；lane内source分母=4×lane_fires。使能项含零权重，不是非零MAC/DSP/FLOPs。",
                  "互斥桶分母=12×cycles，优先级abort→compute→issue_wait→region_wait→observation_wait→fill→internal→idle。internal/idle不是完整瓶颈因果解释。", "",
                  "| 独立记录 | monitor cycles | 全窗口lane % | 全窗口P4 % | 发射内lane % |", "|---|---:|---:|---:|---:|"])
    for row in report["activity"]:
        lines.append(f"| {row['id']} | {row['cycles']} | {100*row['whole_array_destination_lane_occupancy']:.5f} | "
                     f"{100*row['whole_array_p4_term_occupancy']:.5f} | {100*row['accepted_issue_destination_occupancy']:.4f} |")
    lines.extend(["", "## 软件准确率与RTL一致性分列", "",
                  "主表是有界RTL数值/事务一致性，不填写数据集RTL准确率。下表只读取已完成软件报告，不运行CPU/GPU推理。", "",
                  "| 拓扑 | 软件臂 | 正确/样本 | % |", "|---|---|---:|---:|"])
    for row in report["software_accuracy"]:
        lines.append(f"| {row['topology']} | {row['arm']} | {row['correct']}/{row['samples']} | {row['accuracy_percent']:.6f} |")
    lines.extend(["", "FC同量化权重float不是原始FP32 checkpoint；SRNN完整整数不是仅M2权重量化；SCNN是固定264×20子集的严格同bank对照，不是完整DVS Test。软件输入范围也不同于主表在线窗口。", "",
                  "## 明确缺失的计数", "",
                  "以下是现有日志的精确缺口，不是假设为0，也不授权补跑实验。", ""])
    grouped = {}
    for row in report["missing_counters"]:
        grouped.setdefault((row["counter"], row["reason"]), []).append(row["profile"])
    for (counter, reason), names in grouped.items():
        lines.append(f"- `{', '.join(names)}`：`{counter}`。{reason}。")
    lines.extend(["", "## 原始依据", "",
                  "主表evidence列给出精确执行目录；JSON sources列出所有读取文件。以下入口保留各源码范围：", ""])
    for path in ("rtl/fc_final_counts/README.md", "rtl/srnn_same_dut/README.md", "rtl/scnn_real_final_graph/README.md", "performance/h40_activity/README.md"):
        link = Path(os.path.relpath(root / SUPPLEMENT / path, output)).as_posix()
        lines.append(f"- [{path}]({link})")
    link = Path(os.path.relpath(root / CURRENT_FC / "comparison.json", output)).as_posix()
    lines.append(f"- [当前FC B16T99独立配对]({link})")
    link = Path(os.path.relpath(root / CURRENT_SRNN / "commands.json", output)).as_posix()
    lines.append(f"- [当前SRNN native B1T73独立配对]({link})")
    lines.extend(["", "### SRNN归档的最小读取文件", "",
                  "轻包复算只需保留metrics.json sources列出的实际读取文件，路径按原项目相对目录放置；不是承诺重新仿真全部算法/EDA。SRNN build raw内仅读取以下两个原样文件，不需要镜像整个raw、obj、生产RTL、fixture mem或记录的source_files/dependencies树。解析端scripts需保留run_srnn_profile_measurement.py、build_srnn_profile_measurement_reference.py及已复用的run_fc_final_counts.py；其余既有FC/activity解析依赖不变。归档runner只作provenance，不从归档执行。", ""])
    fixture = current_srnn[0]["input"]["fixture"]
    for path in (Path(fixture) / "reference.json", SRNN_RAW / "scripts/run_srnn_profile_measurement.py"):
        link = Path(os.path.relpath(root / path, output)).as_posix()
        lines.append(f"- [{path.as_posix()}]({link})")
    lines.extend(["", "### SCNN归档的最小读取文件", "",
                  "SCNN reference使用原样复制的performance/current_source_measurements/scnn/fixed_sample0_b1t20_r1/manifest.json。run_off/on.log、原summary、preparation、四个阶段status、launcher.status、elab/build.log及sources/dependencies清单来自server_pair_r2，实际读取路径均列在metrics.json sources。清单中服务器绝对路径和依赖名仅记录provenance，不要求它们等于本地cwd，也不据此复制整个依赖树。",
                  "SCNN_RAW内仅读取以下三个原样源码文件，不执行归档脚本，不镜像raw/obj/105 RTL或全部fixture。解析端scripts需保留run_scnn_profile_measurement.py及其标准库/本地导入：prepare_scnn_profile_measurement.py、run_scnn_real_final_graph.py、generate_scnn_real_final_graph.py、generate_scnn_real_final_b4t12.py、supplement_scnn_same_dut_reference.py、supplement_scnn_graph_reference.py、analyze_scnn_b4t4_rtl_update_contract.py及既有run_fc_final_counts.py；只调用parse_result，不运行这些模块的prepare/build/reference入口。", ""])
    for path in SCNN_RAW_FILES:
        path = SCNN_RAW / path
        link = Path(os.path.relpath(root / path, output)).as_posix()
        lines.append(f"- [{path.as_posix()}]({link})")
    for path in (CURRENT_SCNN / "summary.json", Path(current_scnn[0]["input"]["fixture"]) / "manifest.json"):
        link = Path(os.path.relpath(root / path, output)).as_posix()
        lines.append(f"- [{path.as_posix()}]({link})")
    return "\n".join(lines) + "\n"


def write_report(report, output, root=ROOT):
    readme = chinese_readme(report, output, root)
    output = filesystem_path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "profiles.csv", profile_csv_rows(report["profiles"]), PROFILE_CSV_FIELDS)
    write_csv(output / "activity.csv", activity_csv_rows(report["activity"]))
    for name, key in (("stages", "stages"), ("software_accuracy", "software_accuracy"),
                      ("missing_counters", "missing_counters")):
        write_csv(output / f"{name}.csv", report[key])
    (output / "README.md").write_text(readme, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args(argv)
    output = normal_path(filesystem_path(args.output_dir).resolve())
    if not output.is_relative_to(normal_path(filesystem_path(ROOT / OUTPUT).resolve())):
        parser.error(f"output must remain under {OUTPUT.as_posix()}")
    report = collect()
    write_report(report, output)
    print(f"Wrote {len(report['profiles'])} profiles, {len(report['stages'])} stages, "
          f"{len(report['activity'])} activity runs and {len(report['software_accuracy'])} software rows to {output}")


if __name__ == "__main__":
    main()
