#!/usr/bin/env python3
"""Compare accepted SRNN cycles with a terminal, numerically verified candidate."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
EVIDENCE = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization"
BASELINE = ROOT / "results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/server_pair_r1"


def followup(include_compatibility=False):
    control = json.loads((EVIDENCE / "control_r1/status.json").read_text(encoding="utf-8"))
    modes = json.loads((EVIDENCE / "store_modes_r1/status.json").read_text(encoding="utf-8"))
    steady = json.loads((EVIDENCE / "steady_off_r1/status.json").read_text(encoding="utf-8"))
    if (control["status"], modes["status"], steady["status"]) != (
        "PASS_cache_control_only", "PASS_store_modes_only", "PASS_two_inference_sequences_same_profile_not_Test"
    ):
        raise ValueError("follow-up experiment incomplete")
    for name, record in (("control_r1",control), ("store_modes_r1",modes), ("steady_off_r1",steady)):
        if any(record[stage]["exit"] != 0 for stage in ("build", "run")):
            raise ValueError("follow-up stage failure: "+name)
    log = (EVIDENCE / "steady_off_r1/run.log").read_text(encoding="utf-8")
    if len(re.findall(r"^SRNN_SAME_DUT_PASS ", log, re.MULTILINE)) != 2:
        raise ValueError("two numerical completions required")
    if len(re.findall(r"^RESIDENCY_PASS backbone_loads=2 wh1_loads=1 wout_loads=1 wout_reloads=0$",log,re.MULTILINE)) != 2:
        raise ValueError("weights were not retained over the two sequences")
    initial = re.search(r"^SRNN_PROFILE_COMPLETE .*initialized=(\d+)", log, re.MULTILINE)
    if initial is None:
        raise ValueError("missing original initialization interval")
    report = dict(status="PASS_focused_followup_evidence", control_scenarios=7, store_term_checks=1120,
                  initial_parameter_load_and_state_cycles=int(initial.group(1)),
                  steady_inference_samples=steady["samples"], candidate_post_route_15ns_verified=False,
                  boundaries=["Control tests use the actual candidate M2 engine/refill and a synthetic parameter response server.",
                              "Store-mode tests do not replace full FC/SCNN integration or profile-switch tests.",
                              "The same fixed sample0 is repeated as two independent sequences; no learning-on steady-state or diverse-batch claim.",
                              "Initial setup, per-sequence state clearing and runtime are separate quantities."])
    if include_compatibility:
        fc_root = EVIDENCE / "fc_compat_r1"
        frontend_root = EVIDENCE / "frontend_compat_r1"
        fc_status = json.loads((fc_root / "status.json").read_text(encoding="utf-8"))
        frontend_status = json.loads((frontend_root / "status.json").read_text(encoding="utf-8"))
        specialized_modes = json.loads((EVIDENCE / "store_modes_specialized_r1/status.json").read_text(encoding="utf-8"))
        if (fc_status["status"], frontend_status["status"], specialized_modes["status"]) != (
            "PASS_FC_B16T99_pair_candidate_compatibility", "PASS_small_final_top_profile_switch_only", "PASS_store_modes_only"
        ):
            raise ValueError("compatibility experiment incomplete")
        for record, stages in ((fc_status, ("build", "run_off", "run_on")),
                               (frontend_status, ("build", "run")), (specialized_modes, ("build", "run"))):
            if any(record[stage]["exit"] != 0 for stage in stages) or record["structural_warnings"]:
                raise ValueError("compatibility stage/structural failure")
        fc_rows = {}
        for arm in ("off", "on"):
            measured = json.loads((fc_root / (arm+"_measurements.json")).read_text(encoding="utf-8"))
            if measured["status"] != "PASS" or measured["runtime_cycles"] != fc_status[arm+"_cycles"]:
                raise ValueError("FC compatibility counters differ from terminal status")
            fc_rows[arm] = dict(runtime_cycles=measured["runtime_cycles"], numerical_checks=measured["legacy"],
                                h40_activity=measured.get("h40_activity"))
        transcript = json.loads((frontend_root / "transcript_checks.json").read_text(encoding="utf-8"))
        if transcript["status"] != "PASS" or not all(transcript["checks"].values()):
            raise ValueError("native-switch transcript checks failed")
        report["specialized_candidate_compatibility"] = dict(
            source_candidate="candidate_r4", fc_B16T99=fc_rows,
            small_final_top_switch=dict(commands=frontend_status["commands"], checks=transcript["checks"], rows=transcript["rows"]),
            specialized_store_term_checks=1120)
        report["boundaries"].append("FC full fixture and small synthetic learning-off profile switching use the specialized candidate; they do not establish real-asset SCNN or formal M2 switching/abort recovery.")
    (EVIDENCE / "srnn_resident_followup_summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (EVIDENCE / "srnn_resident_steady_inference.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(steady["samples"][0]))
        writer.writeheader()
        writer.writerows(steady["samples"])
    print(json.dumps(report, indent=2))


def native_pair(name):
    root = EVIDENCE / name
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    expected = json.loads((root / "multi_fixture/reference.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS_two_different_native_sequences_pair_not_Test":
        raise ValueError("native pair is not terminal PASS")
    if any(status[s]["exit"] != 0 for s in ("build", "run_off", "run_on")) or status["structural_warnings"]:
        raise ValueError("native pair stage/structural failure")
    rows = []
    initial_intervals = []
    for arm in ("off", "on"):
        log = (root / ("run_"+arm+".log")).read_text(encoding="utf-8")
        raw = re.findall(r"^RESIDENT_NATIVE_SEQUENCE (.+)$", log, re.MULTILINE)
        functional = re.findall(r"^SRNN_SAME_DUT_PASS (.+)$",log,re.MULTILINE)
        stored_rows = status[arm+"_sequences"]
        if len(raw) != 2 or len(functional) != 2 or len(stored_rows) != 2:
            raise ValueError("two numerical completions are required per arm")
        for counter in ("parameter_load_accepted", "store_load_accepted"):
            values = re.findall(r"^SRNN_PROFILE_COUNT name="+counter+r" value=(\d+)$", log, re.MULTILINE)
            if values != ["120320", "120320"]:
                raise ValueError("model parameter loads did not remain constant: "+counter)
        initialized = re.search(r"^SRNN_PROFILE_COMPLETE .*initialized=(\d+)", log, re.MULTILINE)
        if initialized is None:
            raise ValueError("missing original initialization interval")
        initial_intervals.append(int(initialized.group(1)))
        for i, (raw_row, row) in enumerate(zip(raw, stored_rows)):
            observed = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", raw_row)}
            if any(row.get(k) != v for k,v in observed.items()):
                raise ValueError("native status/log counters differ")
            wanted = expected["expected"][arm][i]
            numerical = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",functional[i])}
            if numerical.get("gradient") != (wanted["steps"]*5120 if arm=="on" else 0) or numerical.get("detach") != wanted["commits"]:
                raise ValueError("gradient/detach completion differs from native scope")
            for key in ("sequence", "steps", "bone", "hidden", "logits", "store", "code", "residual", "commits"):
                if row[key] != wanted[key]:
                    raise ValueError("native reference counter mismatch: "+key)
            rows.append(dict(arm=arm, dataset_index=wanted["dataset_index"], label=wanted["label"],
                native_steps=row["steps"], runtime_cycles=row["cycles"], state_setup_cycles=row["state_setup_cycles"],
                samples_s_including_state_setup_at_15ns_projection=1e9/(15*(row["cycles"]+row["state_setup_cycles"])),
                bone=row["bone"], hidden=row["hidden"], logits=row["logits"], store=row["store"],
                code_writes=row["code"], residual_writes=row["residual"], commits=row["commits"],
                gradients=numerical["gradient"],
                backbone_loads_cumulative=row["backbone_loads"], wh1_loads_cumulative=row["wh1_loads"],
                wout_loads_cumulative=row["wout_loads"]))
    if len(set(initial_intervals)) != 1:
        raise ValueError("paired initialization intervals differ")
    report = dict(status="PASS_two_official_native_sequences_pair", run=name, native_steps=expected["native_steps"],
        initial_parameter_and_state_cycles=initial_intervals[0], rows=rows, candidate_post_route_15ns_verified=False,
        boundaries=["Two official samples only, not Test accuracy or a sustained-throughput distribution.",
                    "Each native sequence executes its own length; array padding is not executed.",
                    "Learning codes and residuals persist, while logical dynamic state clears between samples.",
                    "Projection excludes initial setup, includes separately listed per-sequence state clearing; no board claim.",
                    "Different sample0/sample1 latencies are not a matched-input speedup comparison.",
                    "R1 legacy SRNN_PROFILE_COMPLETE steps prints capacity=78; native sequence markers and numerical completions give the actual 73/78 steps. Original logs are retained."])
    (EVIDENCE / "srnn_resident_native_sequences.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (EVIDENCE / "srnn_resident_native_sequences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="srnn_r5")
    parser.add_argument("--output-stem", default="srnn_resident_comparison")
    parser.add_argument("--followup-only", action="store_true")
    parser.add_argument("--include-compatibility", action="store_true")
    parser.add_argument("--native-pair-only", action="store_true")
    parser.add_argument("--native-run", default="multi_native_r1")
    args = parser.parse_args()
    if args.native_pair_only:
        if not args.native_run.replace("_", "").isalnum():
            parser.error("simple native run name required")
        native_pair(args.native_run)
        return
    if args.followup_only:
        followup(args.include_compatibility)
        return
    if not args.run.replace("_", "").isalnum():
        parser.error("simple run name required")
    if not args.output_stem.replace("_", "").isalnum():
        parser.error("simple output stem required")
    candidate = EVIDENCE / args.run
    status = json.loads((candidate / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS_candidate_fixed_sample_pair_not_physical_QoR":
        raise ValueError("candidate is not a terminal accepted pair")
    result = []
    for arm in ("off", "on"):
        old = json.loads((BASELINE / arm / "measurements.json").read_text(encoding="utf-8"))
        new = json.loads((candidate / arm / "measurements.json").read_text(encoding="utf-8"))
        if old["sample_definition"] != new["sample_definition"]:
            raise ValueError("candidate sample differs from accepted baseline")
        log = (candidate / ("run_"+arm+".log")).read_text(encoding="utf-8")
        functional = re.search(r"^SRNN_SAME_DUT_PASS (.+)$", log, re.MULTILINE)
        residency = re.search(r"^RESIDENCY_PASS (.+)$", log, re.MULTILINE)
        if functional is None or residency is None:
            raise ValueError("missing numerical or residency completion")
        checks = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", functional.group(1))}
        counts = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", residency.group(1))}
        expected = dict(batch=1, steps=73, learning=int(arm == "on"), bone=32850, hidden=18688, logits=1460,
                        code=68040 if arm == "on" else 0, residual=145500 if arm == "on" else 0,
                        commits=73 if arm == "on" else 0)
        if any(checks.get(key) != value for key, value in expected.items()):
            raise ValueError("candidate numerical counts differ: " + str(checks))
        field = "sample_service_cycles_including_refill_excluding_tb_checks"
        old_cycles, new_cycles = old[field], new[field]
        row = dict(arm=arm, sample="sample0_B1T73", baseline_cycles=old_cycles, candidate_cycles=new_cycles,
                   cycle_speedup=old_cycles/new_cycles, cycle_reduction_percent=100*(1-new_cycles/old_cycles),
                   baseline_samples_s_at_15ns=1e9/(15*old_cycles), candidate_samples_s_at_15ns=1e9/(15*new_cycles),
                   backbone_loads=counts["backbone_loads"], wh1_loads=counts["wh1_loads"],
                   wout_loads=counts["wout_loads"], wout_reloads=counts["wout_reloads"],
                   simulation_wall_seconds=new["simulation_wall_seconds"], candidate_post_route_15ns_verified=False)
        result.append(row)
    source = json.loads((candidate / "source_comparison.json").read_text(encoding="utf-8"))
    report = dict(status="PASS_measured_candidate_cycles_only", candidate=args.run, rows=result,
                  production_unchanged=True, source_candidate=source.get("candidate_name", "candidate_r3"), source_overlay_count=len(source["changed_sources"]),
                  boundaries=["Native fixed sample0 B1T73, not complete Test or multi-sample steady state.",
                              "Initial profile cache refill included in runtime; parameter loading and state initialization excluded.",
                              "15 ns throughput is a projection pending this candidate's physical implementation.",
                              "Learning arithmetic, TP /64, original model and masks preserved.",
                              "Candidate is isolated pending compatibility/edge cases and physical QoR."])
    (EVIDENCE / (args.output_stem+".json")).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (EVIDENCE / (args.output_stem+".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
