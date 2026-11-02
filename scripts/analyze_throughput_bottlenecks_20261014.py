#!/usr/bin/env python3
"""Reproduce the throughput diagnosis without changing RTL or raw evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PERIOD_NS = 15.0
DEVICE_BRAM36 = 1470.0
CURRENT_BRAM36 = 514.5


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rate(cycles: float, units: float, period_ns: float = PERIOD_NS) -> float:
    if cycles <= 0 or units <= 0:
        raise ValueError(f"invalid rate inputs: cycles={cycles}, units={units}")
    return units / (cycles * period_ns * 1e-9)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_close(name: str, actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance * max(1.0, abs(expected)):
        raise ValueError(f"{name}: actual={actual} expected={expected}")


def collect_current(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    base = root / "results/paper_evidence/supplement_20261004/performance/current_source_measurements"
    baseline_rows: list[dict] = []
    phase_rows: list[dict] = []

    fc_dir = base / "fc/server_pair_r1"
    for arm in ("off", "on"):
        data = load_json(fc_dir / f"{arm}.json")
        if data["status"] != "PASS":
            raise ValueError(f"FC {arm} is not PASS")
        cycles = int(data["runtime_cycles"])
        samples = int(data["scope"]["batch"])
        steps = int(data["scope"]["steps"])
        measured_rate = float(data["projected_samples_per_second_at_15ns"])
        require_close(f"FC {arm} sample rate", rate(cycles, samples), measured_rate)
        baseline_rows.append({
            "profile": "FC",
            "arm": arm,
            "workload": f"B{samples}T{steps}",
            "cycles": cycles,
            "period_ns": PERIOD_NS,
            "complete_samples_per_s": f"{measured_rate:.6f}",
            "frames_or_timesteps_per_s": f"{rate(cycles, samples * steps):.6f}",
            "normalization_unit": "sample-timestep",
            "measurement_scope": "runtime only; setup excluded",
            "evidence_status": "measured RTL cycles; frequency projection",
        })
        stage_total = 0
        for stage in ("runtime", "learning_tail_wait", "runtime_refill"):
            value = int(data["stages"][stage]["cycles"])
            stage_total += value
            phase_rows.append({
                "profile": "FC",
                "arm": arm,
                "phase": stage,
                "cycles": value,
                "share_of_runtime": f"{value / cycles:.9f}",
                "source": f"fc/server_pair_r1/{arm}.json",
            })
        if stage_total != cycles:
            raise ValueError(f"FC {arm} exclusive stage sum {stage_total} != {cycles}")

    srnn_dir = base / "srnn/server_pair_r1"
    srnn_records: dict[str, dict] = {}
    for arm in ("off", "on"):
        data = load_json(srnn_dir / arm / "measurements.json")
        if not str(data["status"]).startswith("PASS"):
            raise ValueError(f"SRNN {arm} is not PASS")
        srnn_records[arm] = data
        cycles = int(data["sample_service_cycles_including_refill_excluding_tb_checks"])
        samples = int(data["sample_definition"]["samples"])
        steps = int(data["sample_definition"]["steps"])
        measured_rate = float(data["full_sample_throughput_samples_s_projection"])
        require_close(f"SRNN {arm} sample rate", rate(cycles, samples), measured_rate)
        baseline_rows.append({
            "profile": "SRNN",
            "arm": arm,
            "workload": f"B{samples}T{steps}",
            "cycles": cycles,
            "period_ns": PERIOD_NS,
            "complete_samples_per_s": f"{measured_rate:.6f}",
            "frames_or_timesteps_per_s": f"{rate(cycles, samples * steps):.6f}",
            "normalization_unit": "sample-timestep",
            "measurement_scope": "service incl. runtime refill; initial load/state excluded",
            "evidence_status": "measured RTL cycles; frequency projection",
        })
        stage_names = ("backbone_refill_ff", "backbone_refill_r", "backbone", "m2")
        stage_total = 0
        for stage in stage_names:
            value = int(data["stage_cycles"][stage])
            stage_total += value
            phase_rows.append({
                "profile": "SRNN",
                "arm": arm,
                "phase": stage,
                "cycles": value,
                "share_of_runtime": f"{value / cycles:.9f}",
                "source": f"srnn/server_pair_r1/{arm}/measurements.json",
            })
        if stage_total != cycles:
            raise ValueError(f"SRNN {arm} stage sum {stage_total} != {cycles}")

    scnn = load_json(base / "scnn/server_pair_r2/summary.json")
    if scnn["status"] != "PASS":
        raise ValueError("SCNN pair is not PASS")
    for arm in ("off", "on"):
        data = scnn["measurements"][arm]
        cycles = int(data["timing"]["sample_cycles"])
        samples = int(data["final"]["samples"])
        frames = int(data["final"]["frames"])
        baseline_rows.append({
            "profile": "SCNN",
            "arm": arm,
            "workload": f"B{samples}T{frames}",
            "cycles": cycles,
            "period_ns": PERIOD_NS,
            "complete_samples_per_s": f"{rate(cycles, samples):.6f}",
            "frames_or_timesteps_per_s": f"{rate(cycles, frames):.6f}",
            "normalization_unit": "frame",
            "measurement_scope": "20-frame service; setup excluded",
            "evidence_status": "measured RTL cycles; frequency projection",
        })
        layer_cycles: dict[int, int] = {}
        for descriptor in data["descriptors"]:
            layer = int(descriptor["layer"])
            layer_cycles[layer] = layer_cycles.get(layer, 0) + int(descriptor["cycles"])
        for layer, value in sorted(layer_cycles.items()):
            phase_rows.append({
                "profile": "SCNN",
                "arm": arm,
                "phase": f"conv{layer}",
                "cycles": value,
                "share_of_runtime": f"{value / cycles:.9f}",
                "source": "scnn/server_pair_r2/summary.json",
            })

    candidate_rows: list[dict] = []
    for arm, data in srnn_records.items():
        steps = int(data["sample_definition"]["steps"])
        baseline_cycles = int(data["sample_service_cycles_including_refill_excluding_tb_checks"])
        backbone_refill = int(data["stage_cycles"]["backbone_refill_ff"]) + int(
            data["stage_cycles"]["backbone_refill_r"]
        )
        m2_refill = int(data["counters"]["m2_refill_busy_cycles"])
        repeated_refill = backbone_refill + m2_refill
        first_sample_cycles = baseline_cycles - repeated_refill * (steps - 1) / steps
        steady_state_cycles = baseline_cycles - repeated_refill
        for candidate, cycles, scope in (
            ("measured_baseline", float(baseline_cycles), "current measured complete sample"),
            (
                "resident_backbone_and_m2_first_sample",
                first_sample_cycles,
                "model: keep one profile-load share, remove 72/73 repeated refills",
            ),
            (
                "resident_backbone_and_m2_steady_state",
                steady_state_cycles,
                "model: same-profile next sample; profile loads amortized outside sample",
            ),
        ):
            candidate_rows.append({
                "arm": arm,
                "candidate": candidate,
                "cycles": f"{cycles:.3f}",
                "projected_complete_samples_per_s": f"{rate(cycles, 1):.6f}",
                "speedup_vs_measured": f"{baseline_cycles / cycles:.6f}",
                "evidence_status": "measured" if candidate == "measured_baseline" else "analytical model; not yet RTL-verified",
                "scope": scope,
            })

    return baseline_rows, phase_rows, candidate_rows


def collect_old_calibration(old_root: Path) -> list[dict]:
    path = old_root / (
        "best_versions/01_R21_H4080_current_candidate/source_and_local_evidence/"
        "worktree/analysis/cycle_calibration_20260823.json"
    )
    data = load_json(path)
    rows = []
    for point in data["points"]:
        cycles = float(point["mean_cycles"])
        timesteps = int(point["timesteps"])
        rows.append({
            "timesteps": timesteps,
            "mean_cycles": f"{cycles:.3f}",
            "nominal_samples_per_s_at_unverified_50MHz": f"{50_000_000 / cycles:.6f}",
            "nominal_timesteps_per_s_at_unverified_50MHz": f"{50_000_000 * timesteps / cycles:.6f}",
            "physical_lif_lanes": data["physical_lif_lanes"],
            "synaptic_lanes": data["synaptic_lanes"],
            "workload": "2 tiny layers; 3x16 edges/timestep; 12 W2 coeffs; 2 LIF updates",
            "evidence_status": data["claim_boundary"],
        })
    return rows


def architecture_delta_rows() -> list[dict]:
    return [
        {"dimension": "target_topologies", "old_project": "FC + SRNN", "current_project": "FC + SRNN + 8-conv SCNN", "throughput_effect": "current SCNN adds a full spatial descriptor service path"},
        {"dimension": "physical_destinations", "old_project": "R4D: 12x40 = 480", "current_project": "12x40 = 480", "throughput_effect": "the routed comparison is not explained by neuron count"},
        {"dimension": "learning", "old_project": "inference only", "current_project": "selective integer TP-derived output update", "throughput_effect": "commit, residual/code write and cache publication add serial work"},
        {"dimension": "weights", "old_project": "compile-time resident ROM", "current_project": "addressable parameter plane + shared writable cache", "throughput_effect": "runtime refill and version coherence dominate SRNN"},
        {"dimension": "synaptic_issue", "old_project": "K8 event stream in retained R4D FC path", "current_project": "P4 shared mixed-precision issue", "throughput_effect": "current path trades peak issue width for cross-topology reuse"},
        {"dimension": "state", "old_project": "resident inference state", "current_project": "state/trace maintenance and topology-safe switching", "throughput_effect": "extra service traffic; SCNN restore+spill is measurable but not the main bottleneck"},
        {"dimension": "measurement_denominator", "old_project": "tiny T1-T4 calibration", "current_project": "FC B16T99, SRNN B1T73, SCNN B1T20", "throughput_effect": "old nominal samples/s cannot be compared directly with full-model samples/s"},
        {"dimension": "clock_evidence", "old_project": "R4D routed at 50 MHz; tiny calibration clock unverified", "current_project": "RUN137 routed at 15 ns with WNS +0.102 ns", "throughput_effect": "current frequency is higher but has little timing margin for blind widening"},
    ]


def resource_rows() -> list[dict]:
    wh1_bits = 450 * 256 * 16
    wout_bits = 256 * 20 * 16
    payload_bram = (wh1_bits + wout_bits) / 36_864
    geometry_bits = 7 * 4 * 200 * 450 + 1 * 4 * 200 * 256
    geometry_bram = geometry_bits / 36_864
    planning_bram = 96.0
    return [
        {"item": "current_RUN136", "bram36eq": f"{CURRENT_BRAM36:.3f}", "device_share": f"{CURRENT_BRAM36 / DEVICE_BRAM36:.6f}", "status": "measured synthesis"},
        {"item": "private_M2_payload_lower_bound", "bram36eq": f"{payload_bram:.3f}", "device_share": f"{payload_bram / DEVICE_BRAM36:.6f}", "status": "information-capacity lower bound"},
        {"item": "private_M2_4x200_geometry", "bram36eq": f"{geometry_bram:.3f}", "device_share": f"{geometry_bram / DEVICE_BRAM36:.6f}", "status": "logical bank geometry before primitive quantization"},
        {"item": "private_M2_primitive_planning_ceiling", "bram36eq": f"{planning_bram:.3f}", "device_share": f"{planning_bram / DEVICE_BRAM36:.6f}", "status": "conservative planning estimate; synthesis required"},
        {"item": "current_plus_planning_ceiling", "bram36eq": f"{CURRENT_BRAM36 + planning_bram:.3f}", "device_share": f"{(CURRENT_BRAM36 + planning_bram) / DEVICE_BRAM36:.6f}", "status": "capacity appears feasible; timing and placement unverified"},
    ]


def reviewer_plan_rows() -> list[dict]:
    return [
        {"priority": "P0", "experiment": "denominator and phase audit", "question": "are the low rates an arithmetic-core failure or service overhead?", "required_evidence": "existing-log exact phase shares and normalized timestep/frame rates", "falsifier": "stage totals do not close to runtime", "server_needed": "no", "status": "completed by this analysis"},
        {"priority": "P1", "experiment": "SRNN resident multi-context weight cache", "question": "can backbone and M2 weights remain resident across T73?", "required_evidence": "same B1T73 off/on values, commits and counters; before/after cycles; synth and route resources/timing", "falsifier": "refill reduction does not survive full RTL or cache coherence breaks", "server_needed": "yes for full RTL and Vivado", "status": "not run"},
        {"priority": "P2", "experiment": "SCNN descriptor prefetch/command overlap A-B", "question": "can feature fetch for command n+1 overlap command n?", "required_evidence": "short exact fixture then complete B1T20; descriptor/layer cycles and all golden counters", "falsifier": "backpressure or context hazards prevent sustained overlap", "server_needed": "yes", "status": "not run"},
        {"priority": "P3", "experiment": "FC dirty-row/direct cache update", "question": "can the 412137-cycle post-learning refill be reduced without changing commit semantics?", "required_evidence": "B16T99 on same updates/commits and lower refill cycles", "falsifier": "updated cache version diverges from parameter store", "server_needed": "yes", "status": "not run"},
        {"priority": "P4", "experiment": "selective P8 or dual-P4 issue", "question": "does wider issue improve useful utilization after refill removal?", "required_evidence": "cycle reduction plus routed 15 ns timing and resource delta", "falsifier": "critical path/congestion erases throughput gain", "server_needed": "yes", "status": "defer until P1/P2"},
        {"priority": "P5", "experiment": "multi-sample same-profile steady-state", "question": "what is sustained throughput after one-time profile load?", "required_evidence": "at least two consecutive complete samples with setup/profile-load separated", "falsifier": "cache is invalidated or reloads recur between samples", "server_needed": "yes", "status": "not run"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-root", type=Path, default=Path(r"D:\Desktop\后仿-无SCNN-无在线学习"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to results/paper_evidence/supplement_20261014/performance_optimization",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/paper_evidence/supplement_20261014/performance_optimization"

    baseline, phases, srnn_candidates = collect_current(root)
    old_rows = collect_old_calibration(args.old_root)
    write_csv(output / "baseline_normalized.csv", baseline)
    write_csv(output / "phase_attribution.csv", phases)
    write_csv(output / "srnn_residency_model.csv", srnn_candidates)
    write_csv(output / "old_tiny_calibration_normalized.csv", old_rows)
    write_csv(output / "architecture_delta.csv", architecture_delta_rows())
    write_csv(output / "resident_cache_resource_model.csv", resource_rows())
    write_csv(output / "reviewer_experiment_plan.csv", reviewer_plan_rows())

    summary = {
        "status": "PASS_analysis_only",
        "period_ns": PERIOD_NS,
        "baseline_rows": len(baseline),
        "phase_rows": len(phases),
        "srnn_candidate_rows": len(srnn_candidates),
        "old_calibration_rows": len(old_rows),
        "boundaries": [
            "No RTL, simulation, synthesis, implementation, board or power result was produced.",
            "Candidate throughput is an analytical cycle-removal model, not a measured result.",
            "The old tiny calibration is not a full-model throughput comparison.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
