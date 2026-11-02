#!/usr/bin/env python3
"""Reparse the completed FC experiment and compare unchanged workload/counters."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

from run_srnn_resident_remote import ROOT, RESULT, local_path


def path(value):
    return Path(local_path(value))


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def summarize(name, reference_run="fc_compat_r1", output_stem=None):
    original = ROOT / RESULT / reference_run
    current = ROOT / RESULT / name
    state = read_json(current / "status.json")
    if state["status"] != "PASS_FC_B16T99_pair_scanner_pipeline_not_physical_QoR":
        raise ValueError("candidate has not completed both arms")
    if any(state[stage]["exit"] != 0 for stage in ("build", "run_off", "run_on")):
        raise ValueError("nonzero actual process exit")
    for filename in ("tb_fc_profile_measurement.sv", "compatibility_input.json", "scripts/run_fc_profile_measurement.py"):
        if path(original / filename).read_bytes() != path(current / filename).read_bytes():
            raise ValueError("comparison input changed: " + filename)
    old_sources = {p.relative_to(path(original)).as_posix(): p for p in path(original / "rtl").rglob("*.sv")}
    new_sources = {p.relative_to(path(current)).as_posix(): p for p in path(current / "rtl").rglob("*.sv")}
    if old_sources.keys() != new_sources.keys():
        raise ValueError("RTL source set changed")
    differences = [key for key in old_sources if old_sources[key].read_bytes() != new_sources[key].read_bytes()]
    if differences != ["rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv"]:
        raise ValueError("unexpected RTL differences: " + str(differences))
    overlay = read_json(current / "fc_scanner_overlay.json")
    candidate_name = overlay["fc_scanner_candidate"]
    if not re.fullmatch(r"[a-zA-Z0-9_]+", candidate_name):
        raise ValueError("invalid candidate name")
    prepared = ROOT / "verification/experiments/fc_contribution_pipeline" / candidate_name / differences[0]
    if path(prepared).read_bytes() != new_sources[differences[0]].read_bytes():
        raise ValueError("actual scheduler does not match the unit-tested candidate")
    config = read_json(current / "compatibility_input.json")
    for dependency in config["minimal_dependencies"]:
        if dependency.startswith("rtl/"):
            continue
        if path(original / dependency).read_bytes() != path(current / dependency).read_bytes():
            raise ValueError("fixture/parser dependency changed: " + dependency)
    sys.path.insert(0, str(path(current / "scripts")))
    from run_fc_profile_measurement import parse_log
    rows = []
    counters = {}
    for arm in ("off", "on"):
        measurements = []
        for directory in (original, current):
            text = path(directory / ("run_" + arm + ".log")).read_text(encoding="utf-8")
            if re.search(r"%Error|Assertion failed|out of memory|Killed process", text, re.IGNORECASE):
                raise ValueError("runtime diagnostic in " + arm)
            parsed = parse_log(text, arm, config["scope"], config["activity_monitor"])
            saved = read_json(directory / (arm + "_measurements.json"))
            saved.pop("simulation_wall_seconds", None)
            if parsed != saved:
                raise ValueError("saved measurement differs from actual log: " + str(directory))
            measurements.append(parsed)
        old, new = measurements
        if old["scope"] != new["scope"] or old["bus_handshakes"] != new["bus_handshakes"]:
            raise ValueError("workload or transfer counts changed")
        for key in ("hidden_checks", "dot_checks", "q6_checks", "commits", "code_writes", "residual_writes"):
            if old["legacy"][key] != new["legacy"][key]:
                raise ValueError("numeric check/write count changed: " + key)
        old_tail = old["stages"]["learning_tail_wait"]["cycles"]
        new_tail = new["stages"]["learning_tail_wait"]["cycles"]
        setup = sum(new["stages"][key]["cycles"] for key in
                    ("parameter_load", "initial_refill", "initial_state", "learning_configuration"))
        if setup != sum(old["stages"][key]["cycles"] for key in
                        ("parameter_load", "initial_refill", "initial_state", "learning_configuration")):
            raise ValueError("setup changed")
        row = dict(arm=arm, batch=config["scope"]["batch"], steps=config["scope"]["steps"],
                   old_cycles=old["runtime_cycles"], new_cycles=new["runtime_cycles"],
                   cycle_speedup=old["runtime_cycles"] / new["runtime_cycles"],
                   old_learning_tail=old_tail, new_learning_tail=new_tail,
                   learning_tail_speedup=old_tail / new_tail if new_tail else None,
                   setup_cycles=setup,
                   setup_inclusive_cycle_speedup=(old["runtime_cycles"] + setup) / (new["runtime_cycles"] + setup),
                   new_samples_per_second_if_15ns=config["scope"]["batch"] / (new["runtime_cycles"] * 15e-9),
                   physical_timing_validated=False)
        rows.append(row)
        counters[arm] = dict(bus_handshakes=new["bus_handshakes"],
                             stage_cycles={key: value["cycles"] for key, value in new["stages"].items()},
                             legacy=new["legacy"], physical_tiles=new["h40_activity"]["physical_tiles"])
    build = path(current / "build.log").read_text(encoding="utf-8", errors="replace")
    warnings = re.findall(r"^%Warning-([^:]+):", build, re.MULTILINE)
    result = dict(status="PASS_reparsed_fixed_FC_comparison_not_physical_QoR", candidate_run=name,
        reference_run=reference_run, source_count=len(new_sources), changed_sources=differences,
        testbench_and_non_RTL_dependencies_unchanged=True, rows=rows, counters=counters,
        warnings={key: warnings.count(key) for key in sorted(set(warnings))},
        limitations=["Same supplied B16T99 only, not a full Test or multi-batch service distribution.",
                     "No new synthesis/routing; 15ns values are conditional projections, not inherited STA.",
                     "FC has residual writes but zero code changes; nonzero-update SCNN queries remain untested.",
                     "Shared contribution scheduler changes require SCNN compatibility before production merge."])
    if output_stem is None:
        output_stem = ("fc_scanner_pipeline_comparison" if name == "fc_scanner_pipeline_r1"
                       and reference_run == "fc_compat_r1" else name + "_vs_" + reference_run)
    if not re.fullmatch(r"[a-zA-Z0-9_]+", output_stem):
        raise ValueError("invalid output stem")
    destination = ROOT / RESULT / (output_stem + ".json")
    path(destination).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with path(destination.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fc_scanner_pipeline_r1")
    parser.add_argument("--reference-run", default="fc_compat_r1")
    parser.add_argument("--output-stem")
    args = parser.parse_args()
    summarize(args.name, args.reference_run, args.output_stem)
