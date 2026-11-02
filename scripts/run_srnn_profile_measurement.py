#!/usr/bin/env python3
"""Prepare/run a complete-sample SRNN same-DUT measurement without RTL edits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import time

from run_fc_final_counts import ROOT, build_settings, generate_tb, resolve_verification_store
from build_srnn_profile_measurement_reference import OUTPUT

SOURCE_LIST = ROOT / "build/ip/unified_snn_native_ip_source_20261009_r2/reachable_sources.f"
VERILATOR = "/root/autodl-tmp/verilator_5_048_install/bin/verilator"
BODY = ROOT / "tb/include/srnn_same_dut_body.svh"
MONITOR = ROOT / "tb/include/srnn_profile_measurement_monitor.svh"
ACTIVITY_MONITOR = ROOT / "tb/monitors/supplement_h40_activity_monitor.sv"
TOP = "tb_srnn_profile_measurement"
PURE_BUSY = '`RED.o_busy && !`ENG.o_busy && profile_phase == "m2"'

# Separate accepted transfers from valid-but-not-ready cycles on each boundary.
CHANNELS = (
    ("parameter_load", "i_learning_srnn_parameter_load_valid", "o_learning_srnn_parameter_load_ready"),
    ("store_load", "o_store_load_valid", "i_store_load_ready"),
    ("backbone_refill_request", "i_srnn_backbone_refill_valid", "o_srnn_backbone_refill_ready"),
    ("backbone_parameter_read", "o_srnn_backbone_forward_read_valid", "i_srnn_backbone_forward_read_ready"),
    ("backbone_parameter_response", "i_srnn_backbone_forward_response_valid", "o_srnn_backbone_forward_response_ready"),
    ("backbone_layer", "i_fc_srnn_layer_valid", "o_fc_srnn_layer_ready"),
    ("backbone_group", "i_fc_srnn_group_valid", "o_fc_srnn_group_ready"),
    ("backbone_observation", "o_fc_srnn_observation_valid", "i_fc_srnn_observation_ready"),
    ("phase_capture_config", "i_srnn_phase_capture_cfg_valid", "o_srnn_phase_capture_cfg_ready"),
    ("phase_start", "i_srnn_phase_start_valid", "o_srnn_phase_start_ready"),
    ("phase_actual_capture", "u_final.link_srnn_actual_valid", "u_final.link_srnn_actual_ready"),
    ("phase_backbone_replay", "u_final.link_srnn_plane_backbone_valid", "u_final.link_srnn_plane_backbone_ready"),
    ("m2_prepare", "u_final.link_plane_to_system_srnn_h40_prepare_valid", "u_final.link_system_to_plane_srnn_h40_prepare_ready"),
    ("m2_refill_read", "u_final.link_system_to_plane_srnn_h40_forward_read_valid", "u_final.link_plane_to_system_srnn_h40_forward_read_ready"),
    ("m2_refill_response", "u_final.link_plane_to_system_srnn_h40_forward_response_valid", "u_final.link_system_to_plane_srnn_h40_forward_response_ready"),
    ("m2_timestep_start", "u_final.link_plane_to_system_srnn_h40_timestep_start", "u_final.link_system_to_plane_srnn_h40_timestep_start_ready"),
    ("m2_hidden", "u_final.link_system_to_plane_srnn_h40_hidden_valid", "u_final.link_plane_to_system_srnn_h40_hidden_ready"),
    ("m2_logit", "o_learning_srnn_logit_valid", "i_learning_srnn_logit_ready"),
    ("label", "i_learning_srnn_label_valid", "o_learning_srnn_label_ready"),
    ("store_forward_read", "o_store_forward_read_valid", "i_store_forward_read_ready"),
    ("store_forward_response", "i_store_forward_response_valid", "o_store_forward_response_ready"),
    ("store_learning_read", "o_store_learning_read_valid", "i_store_learning_read_ready"),
    ("store_learning_response", "i_store_learning_response_valid", "o_store_learning_response_ready"),
    ("store_learning_write", "o_store_learning_write_valid", "i_store_learning_write_ready"),
    ("code_write", "o_store_learning_write_valid && o_store_learning_write_code_enable", "i_store_learning_write_ready"),
    ("residual_write", "o_store_learning_write_valid && o_store_learning_write_residual_enable", "i_store_learning_write_ready"),
    ("gradient", "`RED.reducer_gradient_valid", "`RED.reducer_gradient_ready"),
    ("learner_start", "`RED.i_start_window", "`RED.o_start_ready"),
    ("learner_commit", "`RED.o_commit_valid", "`RED.i_commit_ready"),
    ("external_state_write", "i_state_maintenance_write_valid", "o_state_maintenance_write_ready"),
    ("external_trace_write", "i_trace_maintenance_write_valid", "o_trace_maintenance_write_ready"),
    ("external_state_read", "i_state_maintenance_read_valid", "o_state_maintenance_read_ready"),
    ("external_trace_read", "i_trace_maintenance_read_valid", "o_trace_maintenance_read_ready"),
    ("external_state_result", "o_state_maintenance_result_valid", "i_state_maintenance_result_ready"),
    ("external_trace_result", "o_trace_maintenance_result_valid", "i_trace_maintenance_result_ready"),
    ("m2_maintenance_acquire", "`ENG.o_maintenance_start_valid", "`ENG.i_maintenance_start_ready"),
    ("m2_maintenance_release", "`ENG.o_maintenance_release_valid", "`ENG.i_maintenance_release_ready"),
    ("m2_state_write", "`ENG.o_state_write_valid", "`ENG.i_state_write_ready"),
    ("m2_trace_write", "`ENG.o_trace_write_valid", "`ENG.i_trace_write_ready"),
    *((f"m2_cache_bank{bank}", f"`ENG.o_weight_fill_valid[{bank}]", f"`ENG.i_weight_fill_ready[{bank}]") for bank in range(4)),
)
EVENTS = (
    ("reset_cycles", "!i_rst_n"),
    ("backbone_refill_done", "o_srnn_backbone_refill_done"),
    ("backbone_layer_done", "o_fc_srnn_layer_done"),
    ("capture_done", "o_fc_srnn_learning_capture_done"),
    ("backbone_replay_done", "o_fc_srnn_learning_done"),
    ("m2_forward_done", "u_final.link_system_to_plane_srnn_h40_timestep_done"),
    ("m2_prepare_done", "u_final.link_system_to_plane_srnn_h40_prepare_done"),
    ("timestep_done", "o_learning_srnn_timestep_done"),
    ("commit_pulse", "o_learning_srnn_learning_commit"),
    ("detach_pulse", "o_learning_srnn_detach"),
    ("learner_busy_cycles", "`RED.o_busy"),
    ("pure_learning_busy_cycles", PURE_BUSY),
    ("m2_refill_busy_cycles", "`ENG.refill_busy"),
    ("m2_state_clear_busy_cycles", "`ENG.clear_busy"),
    ("m2_adapter_busy_cycles", "`ENG.adapter_busy"),
    ("m2_engine_busy_cycles", "`ENG.o_busy"),
    ("phase_busy_cycles", "o_srnn_phase_busy"),
    ("backbone_busy_cycles", "o_fc_srnn_busy"),
)
COUNTER_NAMES = tuple(name+suffix for name, _, _ in CHANNELS for suffix in ("_accepted", "_stalled_cycles")) + tuple(name for name, _ in EVENTS)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"same-DUT template anchor changed: {old[:70]!r}")
    return text.replace(old, new, 1)


def render_monitor(template=None):
    text = MONITOR.read_text(encoding="utf-8") if template is None else template
    indexes = {name: index for index, name in enumerate(COUNTER_NAMES)}
    updates = []
    for name, valid, ready in CHANNELS:
        updates += [f"            if (({valid}) && ({ready})) profile_count[{indexes[name+'_accepted']}]++;",
                    f"            if (({valid}) && !({ready})) profile_count[{indexes[name+'_stalled_cycles']}]++;"]
    updates += [f"            if ({expression}) profile_count[{indexes[name]}]++;"
                for name, expression in EVENTS if name != "reset_cycles"]
    dump = [f'        $display("SRNN_PROFILE_COUNT name={name} value=%0d",profile_count[{index}]);'
            for name, index in indexes.items()]
    for old, new in (("@COUNT_LAST@", str(len(indexes)-1)), ("@RESET_COUNTER@", str(indexes["reset_cycles"])),
                     ("@COUNTER_UPDATES@", "\n".join(updates)), ("@COUNTER_DUMP@", "\n".join(dump)),
                     ("@PURE_BUSY@", PURE_BUSY)):
        text = text.replace(old, new)
    if re.search(r"@[A-Z_]+@", text):
        raise ValueError("unexpanded monitor placeholder")
    return text


def render_body(metadata, source=None):
    if (metadata["batch"], metadata["sample_index"], metadata["complete_sample"]) != (1, 0, True):
        raise ValueError("expected one complete official sample0")
    steps, words = metadata["steps"], metadata["feedforward_words"]
    if not 1 <= steps <= 128 or words < 1:
        raise ValueError("invalid fixture dimensions")
    text = BODY.read_text(encoding="utf-8") if source is None else source
    text = replace_once(text, "T = 8,", f"T = {steps},")
    text = replace_once(text, "ff_groups[0:8191]", f"ff_groups[0:{words-1}]")
    text = re.sub(r"\bT\*2\b", "T", text)
    text = re.sub(r"\btimestep\*2\b", "timestep", text).replace("timestep*960", "timestep*480")
    text = replace_once(text, "cycles > 24000000", f"cycles > {2_000_000+steps*5_000_000}")
    text = replace_once(text, "repeat(6) @(negedge i_clk); i_rst_n = 1;",
                        'repeat(6) @(negedge i_clk); i_rst_n = 1; profile_enter("config");')
    text = replace_once(text, "        for (int a=0; a<120320; a++) begin",
                        '        profile_enter("parameter_load");\n        for (int a=0; a<120320; a++) begin')
    text = replace_once(text, "        for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin",
                        '        profile_enter("initial_state");\n        for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin')
    text = replace_once(text, "        initialized_cycle = cycles;",
                        '        initialized_cycle = cycles; profile_enter("boundary");')
    text = replace_once(text, "            refill(0); refill(1); next_spikes = 0;",
                        '            profile_enter("backbone_refill_ff"); refill(0);\n'
                        '            profile_enter("backbone_refill_r"); refill(1);\n'
                        '            profile_enter("backbone"); next_spikes = 0;')
    text = replace_once(text, "            @(negedge i_clk); i_learning_srnn_timestep_id =",
                        '            profile_enter("m2");\n            @(negedge i_clk); i_learning_srnn_timestep_id =')
    text = replace_once(text, "            for(int a=0; a<120320; a++) begin",
                        '            profile_enter("tb_store_check");\n            for(int a=0; a<120320; a++) begin')
    text = replace_once(text, '            $display("SRNN_SAME_DUT_STEP',
                        '            profile_enter("boundary");\n            $display("SRNN_SAME_DUT_STEP')
    text = replace_once(text, "steps=8 learning=", f"steps={steps} learning=")
    text = replace_once(text, "        $finish;", "        profile_finish();\n        $finish;")
    text = replace_once(text, "    `undef RED", render_monitor()+"\n    `undef RED")
    return text


def simulation_sources(source_list, tb, activity_monitor=False):
    sources = []
    for line in source_list.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path = (ROOT / line.strip().strip('"')).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or path in sources:
            raise ValueError(f"invalid or duplicate reachable source: {line}")
        sources.append(path)
    if not sources:
        raise ValueError("empty source list")
    store = resolve_verification_store()
    if store not in sources:
        sources.append(store)
    return sources + [tb] + ([ACTIVITY_MONITOR] if activity_monitor else [])


def rom_dependencies(sources):
    paths = set()
    for source in sources:
        text = source.read_text(encoding="utf-8")
        paths.update(re.findall(r'\$readmemh\(\s*"([^"]+)"', text))
        paths.update(re.findall(r'parameter\s+(?:string\s+)?ROM_FILE\s*=\s*"([^"]+)"', text))
    return [ROOT / path for path in sorted(paths)]


def inspect_build_log(text):
    codes = re.findall(r"%Warning-([A-Z0-9_]+):", text)
    blocking = ("UNOPTFLAT", "LATCH", "MULTIDRIVEN")
    return dict(blocking={code: codes.count(code) for code in blocking if code in codes},
                ordinary={code: codes.count(code) for code in sorted(set(codes)-set(blocking))})


def parse_log(text, metadata, period_ns=None):
    if period_ns is not None and period_ns <= 0:
        raise ValueError("projection period must be positive")
    counters = dict.fromkeys(COUNTER_NAMES)
    intervals, events, terminal = [], [], None
    functional = False
    for line in text.splitlines():
        if line.startswith("SRNN_SAME_DUT_PASS "):
            functional = True
        if not line.startswith("SRNN_PROFILE_"):
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if line.startswith("SRNN_PROFILE_COUNT "):
            name = fields["name"]
            if name not in counters or counters[name] is not None:
                raise ValueError(f"unknown/duplicate counter: {name}")
            value = int(fields["value"])
            if value < 0:
                raise ValueError("negative counter")
            counters[name] = value
        elif line.startswith("SRNN_PROFILE_INTERVAL "):
            row = {key: int(value) if key in ("timestep", "start", "end", "cycles") else value
                   for key, value in fields.items()}
            if row["cycles"] != row["end"]-row["start"] or row["cycles"] < 0:
                raise ValueError("invalid interval boundaries")
            intervals.append(row)
        elif line.startswith("SRNN_PROFILE_EVENT "):
            events.append({key: int(value) if key in ("timestep", "cycle") else value for key, value in fields.items()})
        elif line.startswith("SRNN_PROFILE_COMPLETE "):
            if terminal is not None:
                raise ValueError("duplicate completion marker")
            terminal = {key: int(value) for key, value in fields.items()}
    missing = [key for key, value in counters.items() if value is None]
    complete = functional and terminal is not None and not missing
    if terminal and (terminal["batch"], terminal["steps"]) != (1, metadata["steps"]):
        raise ValueError("completion does not match prepared sample")
    stages = [row for row in intervals if row["kind"] == "stage"]
    if complete:
        if not stages or stages[0]["start"] != 0 or stages[-1]["end"] != terminal["end"]:
            raise ValueError("stage ledger does not span the measured run")
        if any(a["end"] != b["start"] for a, b in zip(stages, stages[1:])):
            raise ValueError("stage intervals have a gap or overlap")
        for phase in ("backbone_refill_ff", "backbone_refill_r", "backbone", "m2", "tb_store_check"):
            rows = [row for row in stages if row["phase"] == phase]
            if [row["timestep"] for row in rows] != list(range(metadata["steps"])):
                raise ValueError(f"incomplete per-timestep interval: {phase}")
        for name in ("backbone_layer_done", "capture_done", "backbone_replay_done", "m2_forward_done", "timestep_done"):
            if counters[name] != metadata["steps"]:
                raise ValueError(f"incomplete DUT boundary: {name}")
        if terminal["store_checks"] != metadata["steps"]*120320:
            raise ValueError("incomplete verification-store checks")
    stage_totals = {phase: sum(row["cycles"] for row in stages if row["phase"] == phase)
                    for phase in dict.fromkeys(row["phase"] for row in stages)}
    checks = stage_totals.get("tb_store_check")
    sample_cycles = terminal["end"]-terminal["initialized"]-checks if complete and checks is not None else None
    projected_ms = sample_cycles*period_ns/1e6 if sample_cycles is not None and period_ns else None
    return dict(status="PASS_complete_fixed_sample_not_Test" if complete else "incomplete_measurement",
                sample_definition=metadata, learning=terminal["learning"] if terminal else None,
                counters=counters, missing_counters=missing, intervals=intervals, events=events,
                stage_cycles=stage_totals, terminal=terminal,
                sample_service_cycles_including_refill_excluding_tb_checks=sample_cycles,
                initial_load_and_state_cycles=terminal["initialized"] if complete else None,
                tb_store_check_cycles=checks, tb_store_check_host_seconds=None,
                period_ns_projection=period_ns, projected_sample_interval_ms=projected_ms,
                full_sample_throughput_samples_s_projection=1000/projected_ms if projected_ms and complete else None,
                measured_hardware_throughput_samples_s=None, simulation_wall_seconds=None,
                boundaries=["One complete fixed official sample, not complete official Test or adaptation accuracy.",
                    "All port counters are accepted valid&&ready or valid&&!ready stalls, not inferred traffic.",
                    "Stage intervals include TB scheduling; hardware forward and pure learner-busy intervals may overlap stages and must not be added twice.",
                    "Pure learning busy is learner o_busy while M2 engine is idle in the M2 stage; config/reset busy is separate.",
                    "Verification-store checks use hierarchical TB peeks without advancing simulation clocks; host overhead is unknown, not zero.",
                    "Simulator walltime is not hardwaretime; optional period is a projection, never inherited old RUN131 timing."])


def write_measurements(output, measurements):
    (output / "measurements.json").write_text(json.dumps(measurements, indent=2)+"\n", encoding="utf-8")
    for name, rows, fields in (
        ("intervals.csv", measurements["intervals"], ("kind", "phase", "timestep", "start", "end", "cycles")),
        ("counters.csv", [{"name": k, "value": v} for k, v in measurements["counters"].items()], ("name", "value"))):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: "null" if row.get(key) is None else row[key] for key in fields} for row in rows)


def prepare(fixture, run_name, *, source_list=SOURCE_LIST, verilator=VERILATOR, jobs=2, period_ns=None,
            activity_monitor=False):
    fixture = fixture.resolve()
    if not fixture.is_relative_to(OUTPUT.resolve()):
        raise ValueError("fixture must remain in the SRNN measurement directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("run name must be a simple new directory name")
    metadata = json.loads((fixture / "reference.json").read_text(encoding="utf-8"))
    body_text = render_body(metadata)
    output = OUTPUT / run_name
    output.mkdir(parents=True, exist_ok=False)
    body = output / "srnn_profile_measurement_generated_body.svh"
    body.write_text(body_text, encoding="ascii")
    tb = output / (TOP+".sv")
    generate_tb(tb, top=TOP, body=body)
    sources = simulation_sources(source_list, tb, activity_monitor)
    listing = output / "sources.f"
    listing.write_text("".join('"'+path.relative_to(ROOT).as_posix()+'"\n' for path in sources), encoding="ascii")
    build, exe, env = build_settings(output.relative_to(ROOT) / "obj", listing.relative_to(ROOT),
                                     verilator=verilator, jobs=jobs, platform="posix", top=TOP)
    run = {mode:[exe.as_posix(), "+RAW="+fixture.relative_to(ROOT).as_posix(),
                 "+FIXTURE="+fixture.relative_to(ROOT).as_posix(), f"+LEARNING={int(mode=='on')}"]
           for mode in ("off", "on")}
    record = dict(status="prepared_only_not_simulated", build=build, run=run, cwd="project root",
                  fixture=fixture.relative_to(ROOT).as_posix(), source_list=source_list.relative_to(ROOT).as_posix(),
                  production_source_count=len(sources)-2-int(activity_monitor), verification_store=resolve_verification_store().relative_to(ROOT).as_posix(),
                  period_ns_projection=period_ns, same_executable_for_on_off=True, runner_modifies_rtl=False,
                  template_reused=BODY.relative_to(ROOT).as_posix(), sample_definition=metadata,
                  activity_monitor_requested=activity_monitor,
                  activity_monitor_scope="All reset-released clocks, including initialization/refill/learning/TB scheduling gaps; not compute-only occupancy.",
                  counter_definitions={name:dict(valid=valid, ready=ready) for name,valid,ready in CHANNELS},
                  busy_and_event_definitions=dict(EVENTS))
    dependencies = [Path(__file__).resolve(), ROOT / "scripts/run_fc_final_counts.py",
                    ROOT / "scripts/build_srnn_profile_measurement_reference.py", BODY, MONITOR,
                    source_list, resolve_verification_store(), ROOT / "tb/data/srnn_m2_v28_wh1_initial_q14.mem",
                    ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem"]
    dependencies += sources + sorted(fixture.iterdir())
    dependencies += rom_dependencies([path for path in sources if path not in (tb, ACTIVITY_MONITOR)])
    if activity_monitor:
        dependencies.append(ROOT / "scripts/summarize_h40_activity.py")
    relative = sorted({path.relative_to(ROOT).as_posix() for path in dependencies})
    if any(not (ROOT / path).is_file() for path in relative):
        raise FileNotFoundError("missing declared dependency")
    (output / "dependencies.json").write_text(json.dumps(relative, indent=2)+"\n", encoding="utf-8")
    (output / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return output, record, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--source-list", type=Path, default=SOURCE_LIST)
    parser.add_argument("--verilator", default=VERILATOR)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--period-ns", type=float)
    parser.add_argument("--activity-monitor", action="store_true", help="passive existing H40 lane/P4 bind; no stimulus changes")
    parser.add_argument("--summarize", type=Path, help="parse an existing new-measurement run directory, never simulate")
    args = parser.parse_args(argv)
    if args.period_ns is not None and args.period_ns <= 0:
        parser.error("--period-ns must be positive")
    if args.summarize:
        directory = args.summarize.resolve()
        if not directory.is_relative_to(OUTPUT.resolve()):
            parser.error("summary output must stay in the SRNN measurement directory")
        metadata = json.loads((args.fixture / "reference.json").read_text(encoding="utf-8"))
        previous_path = directory / "measurements.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
        period_ns = args.period_ns if args.period_ns is not None else previous.get("period_ns_projection")
        measurements = parse_log((directory / "run.log").read_text(encoding="utf-8"), metadata, period_ns)
        # Host walltime is recorded by the runner, not recoverable from the RTL log.
        measurements["simulation_wall_seconds"] = previous.get("simulation_wall_seconds")
        write_measurements(directory, measurements)
        print(measurements["status"])
        return
    if not args.run_name or args.jobs < 1:
        parser.error("a new --run-name and positive --jobs are required")
    output, record, env = prepare(args.fixture, args.run_name, source_list=args.source_list.resolve(),
                                   verilator=args.verilator, jobs=args.jobs, period_ns=args.period_ns,
                                   activity_monitor=args.activity_monitor)
    if args.prepare_only:
        print(f"Prepared only: {output}")
        return
    for stage, command in [("build", record["build"])] + list(record["run"].items()):
        record["status"] = stage+"_running"
        (output / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        stage_output = output / stage
        stage_output.mkdir()
        started = time.monotonic()
        with (stage_output / ("build.log" if stage == "build" else "run.log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.monotonic()-started
        record[stage+"_exit"] = process.returncode
        record[stage+"_wall_seconds"] = seconds
        record["status"] = stage+"_completed" if process.returncode == 0 else stage+"_failed"
        (output / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        if process.returncode:
            raise SystemExit(f"Preserved {stage} failure: {stage_output}")
        if stage == "build":
            warnings = inspect_build_log((stage_output / "build.log").read_text(encoding="utf-8"))
            record["build_warnings"] = warnings
            if warnings["blocking"]:
                record["status"] = "build_rejected_structural_warnings_not_simulated"
                (output / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
                raise SystemExit(f"Structural warnings {warnings['blocking']}; no simulation started: {stage_output}")
        if stage != "build":
            measurements = parse_log((stage_output / "run.log").read_text(encoding="utf-8"), record["sample_definition"], args.period_ns)
            measurements["simulation_wall_seconds"] = seconds
            write_measurements(stage_output, measurements)
            if not measurements["status"].startswith("PASS_"):
                raise SystemExit(f"Incomplete {stage} measurements preserved: {stage_output}")
            if args.activity_monitor:
                from summarize_h40_activity import parse_activity
                activity = parse_activity((stage_output / "run.log").read_text(encoding="utf-8"), "SRNN_PROFILE_COMPLETE", 12)
                activity["sample_definition"] = {key: record["sample_definition"][key] for key in
                                                   ("batch", "sample_index", "steps", "native_steps", "timestep_policy")}
                activity["learning"] = int(stage == "on")
                (stage_output / "activity.json").write_text(json.dumps(activity, indent=2)+"\n", encoding="utf-8")
    record["status"] = "PASS_two_runtime_arms_one_fixed_sample_not_Test"
    (output / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
