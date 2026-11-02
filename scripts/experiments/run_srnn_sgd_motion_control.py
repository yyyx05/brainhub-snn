"""One development-motion-calibrated SGD control; no query-based rate search.

The scalar rate is interpolated once from accepted +16 development motion.
Actual motion is measured, never forced equal or interpreted as RTL cost.
"""
from __future__ import annotations

import csv
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import run_srnn_integer_optimizer_comparison as base
from run_srnn_sgd_expanded import Inputs, now, peak_working_set, support, write_json
from screen_srnn_m2_direct_optimizer import read_signed_hex
from supplement_srnn_magnitude_sgd import REPORT_SEEDS

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization"
OLD = ROOT / "results/paper_evidence/supplement_20261012/srnn_integer_optimizer"
TRANSFER = PERF / "srnn_shift_transfer_r1"
OUT = PERF / "srnn_sgd_motion_control_r1"
SHIFTS = (-16, 32)


def calibrate(rows):
    # Only method, rate and cumulative code motion enter the interpolation.
    direct = [r for r in rows if r["method"] == "direct_response"]
    sgd = [r for r in rows if r["method"] == "magnitude_sgd"]
    if len(direct) != 1 or len({r["learning_rate"] for r in sgd}) != len(sgd):
        raise ValueError("one direct endpoint and unique SGD rates required")
    target = direct[0]["counters"]["code_lsb_steps"]
    points = sorted((r["counters"]["code_lsb_steps"], Fraction(str(r["learning_rate"]))) for r in sgd)
    below = [p for p in points if p[0] < target]
    above = [p for p in points if p[0] > target]
    if not below or not above or target <= 0:
        raise ValueError("accepted positive bracketing motion required")
    low, high = below[-1], above[0]
    if not 0 < low[1] < high[1]:
        raise ValueError("bracketing motion and rate order disagree")
    rate = low[1] + (high[1]-low[1]) * Fraction(target-low[0], high[0]-low[0])
    return dict(target_code_lsb_steps=target,
                lower=dict(rate=float(low[1]), code_lsb_steps=low[0]),
                upper=dict(rate=float(high[1]), code_lsb_steps=high[0]),
                rate_fraction=str(rate), rate=float(rate),
                rule="one linear interpolation in accepted development cumulative code-LSB motion; no retuning")


def load_transfer_support(inputs, shift, seed):
    indices = inputs.json(OLD / "report_r1" / f"seed{seed}_indices.json")["support"]
    tag = f"seed{seed}_shift{shift}"
    meta = inputs.json(TRANSFER / "features" / (tag+".json"))
    if meta["indices"] != indices or meta["shift"] != shift or len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError("saved complete support order/condition mismatch")
    batches = []
    for b in range(2):
        data = inputs.npz(TRANSFER / "features" / f"{tag}_batch{b}.npz")
        saved = meta["batches"][b]
        if (saved["indices"] != indices[64*b:64*(b+1)] or
                data["spikes"].shape != (saved["padded_timesteps"], 64, 256) or data["labels"].shape != (64,)):
            raise ValueError("saved support shape/order mismatch")
        batches.append((data["spikes"], data["labels"]))
    return batches


def learn(tag, batches, initial, rate):
    code, residual = initial.copy(), np.zeros_like(initial)
    totals, records = {}, []
    for b, (spikes, labels) in enumerate(batches):
        value = base.adapt(spikes, labels, code, residual, "magnitude_sgd", rate, verify=True)
        code, residual = value["codes"], value["residual"]
        for key, count in value["counters"].items():
            totals[key] = totals.get(key, 0) + count
        records.append(dict(batch=b, timesteps=len(spikes), labels=labels.tolist(),
                            counters=value["counters"], rows=value["rows"],
                            independent_checked_values=value["independent_checked_values"]))
        np.savez_compressed(base.native_path(OUT / f"{tag}_batch{b+1}_state.npz"),
                            initial=initial, codes=code, residual=residual)
    return code, dict(counters=totals, support=records,
                     final_changed_addresses=int(np.count_nonzero(code != initial)),
                     final_l1_code_delta=int(np.abs(code-initial).sum()))


def measure_pair(inputs, shift, seed, initial, rate, clean):
    batches = load_transfer_support(inputs, shift, seed)
    shifted_meta = inputs.json(TRANSFER / "features" / f"query_shift{shift}.json")
    if shifted_meta["indices"] != list(range(2264)) or shifted_meta["shift"] != shift:
        raise ValueError("complete saved official query required")
    data = inputs.npz(TRANSFER / "features" / f"query_shift{shift}.npz")
    shifted = data["counts"], data["labels"]
    np.testing.assert_array_equal(clean[1], shifted[1])
    old_rows = {m: inputs.json(TRANSFER / f"shift{shift}_seed{seed}_{m}.json")
                for m in ("direct_response", "magnitude_sgd")}
    for old in old_rows.values():
        if old["seed"] != seed or old["shift"] != shift:
            raise ValueError("paired old record identity mismatch")
        for b, (_, labels) in enumerate(batches):
            if old["support"][b]["labels"] != labels.tolist():
                raise ValueError("paired support labels differ")
    tag = f"shift{shift}_seed{seed}_motion_sgd"
    code, record = learn(tag, batches, initial, rate)
    before = dict(clean=base.evaluate(*clean, initial), shifted=base.evaluate(*shifted, initial))
    after = dict(clean=base.evaluate(*clean, code), shifted=base.evaluate(*shifted, code))
    for old in old_rows.values():
        for view in before:
            if old["baseline_"+view] != before[view]:
                raise ValueError("saved and new frozen query baseline disagree")
    direct_motion = old_rows["direct_response"]["counters"]["code_lsb_steps"]
    record.update(seed=seed, shift=shift, method="motion_calibrated_magnitude_sgd", rate=rate,
                  before=before, after=after,
                  clean_change_pp=after["clean"]["accuracy_percent"]-before["clean"]["accuracy_percent"],
                  shifted_gain_pp=after["shifted"]["accuracy_percent"]-before["shifted"]["accuracy_percent"],
                  motion_to_paired_direct=record["counters"]["code_lsb_steps"]/direct_motion)
    write_json(OUT / (tag+".json"), record)
    flat = dict(shift=shift, seed=seed, sgd_rate=rate, query=len(clean[1]),
                frozen_clean=before["clean"]["correct"], frozen_shifted=before["shifted"]["correct"],
                motion_sgd_clean=after["clean"]["correct"], motion_sgd_shifted=after["shifted"]["correct"],
                motion_sgd_clean_change_pp=record["clean_change_pp"], motion_sgd_shifted_gain_pp=record["shifted_gain_pp"],
                motion_sgd_code_lsb_steps=record["counters"]["code_lsb_steps"],
                motion_sgd_code_events=record["counters"]["code_changes"],
                motion_sgd_residual_events=record["counters"]["residual_changes"],
                motion_sgd_saturations=record["counters"]["saturations"],
                motion_to_direct_ratio=record["motion_to_paired_direct"])
    for name, old in (("direct", old_rows["direct_response"]), ("old_sgd", old_rows["magnitude_sgd"])):
        for view in before:
            flat[f"{name}_{view}_correct"] = old[view+"_after"]["correct"]
        for key in ("code_lsb_steps", "code_changes", "residual_changes", "saturations"):
            flat[f"{name}_{key}"] = old["counters"][key]
    print(f"PAIR shift={shift} seed={seed} clean={flat['motion_sgd_clean']} shifted={flat['motion_sgd_shifted']} "
          f"motion_ratio={flat['motion_to_direct_ratio']:.6f}", flush=True)
    return record, flat


def main():
    Path(base.native_path(OUT)).mkdir(parents=True, exist_ok=False)
    started, inputs, rows, comparison = time.perf_counter(), Inputs(), [], []
    result = {}
    try:
        dev = inputs.json(PERF / "srnn_sgd_expanded_r1/dev/summary.json")
        if dev["status"] != "PASS_expanded_integer_SGD_dev":
            raise ValueError("accepted development source required")
        calibration = calibrate(dev["episodes"])
        protocol = dict(started=now(), calibration=calibration, shifts=list(SHIFTS), seeds=list(REPORT_SEEDS),
            support="same saved 128 samples, two B64 windows; no early stopping", query="all 2264, already exposed historically",
            arithmetic="unchanged signed16 Q2.14 head, Q28 logits/Q14 trace/Q24 LUT/Q40 gradient, fixed /64, FP64 SGD rate product and Q0.24 LSB residual",
            purpose="test whether previous weak SGD control was merely near-frozen; approximate cumulative code-motion control",
            limits="not exact per-episode motion matching, equal hardware cost, query-optimal SGD, independent test, universal optimizer ranking or new distribution",
            historical="direct and 3e-5 SGD endpoints imported, not adapted again; old accuracy-selected rate not replaced",
            resource=dict(device="cpu", threads=1, python=platform.python_version(), numpy=np.__version__))
        write_json(OUT / "protocol.json", protocol)
        sources = [Path(__file__), Path(__file__).with_name("test_srnn_sgd_motion_control.py"),
                   Path(base.__file__), Path(sys.modules["run_srnn_sgd_expanded"].__file__),
                   Path(base.integer_head.__file__), Path(base.reference.__file__),
                   ROOT / "scripts/screen_srnn_m2_direct_optimizer.py", ROOT / "scripts/supplement_srnn_magnitude_sgd.py",
                   ROOT / "rtl/learning/unified_tp_row_softmax_q12_q24.sv"]
        for path in sources:
            inputs.record(path)
            target = OUT / "source" / path.relative_to(ROOT)
            Path(base.native_path(target.parent)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base.native_path(path), base.native_path(target))
        weight = ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem"
        inputs.record(weight)
        initial = read_signed_hex(weight, 16, 5120).reshape(20, 256)
        indices = inputs.json(OLD / "dev_r1/indices.json")
        base.validate_dev_split(indices["support"], indices["query"])
        batches = support(inputs, OLD / "dev_r1", "dev_support", indices["support"])
        _, development = learn("dev_motion_sgd", batches, initial, calibration["rate"])
        development["motion_to_direct_ratio"] = development["counters"]["code_lsb_steps"]/calibration["target_code_lsb_steps"]
        write_json(OUT / "development_motion.json", development)
        print(f"CALIBRATION rate={calibration['rate']:.17g} realized_ratio={development['motion_to_direct_ratio']:.6f}; no retuning", flush=True)
        clean_data = inputs.npz(OLD / "report_r1/features/official_query_clean.npz")
        clean_meta = inputs.json(OLD / "report_r1/features/official_query_clean.json")
        if clean_meta["indices"] != list(range(2264)):
            raise ValueError("all original official clean queries required")
        clean = clean_data["counts"], clean_data["labels"]
        for shift in SHIFTS:
            for seed in REPORT_SEEDS:
                record, flat = measure_pair(inputs, shift, seed, initial, calibration["rate"], clean)
                rows.append(record)
                comparison.append(flat)
        with open(base.native_path(OUT / "comparison.csv"), "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        inputs.check_unchanged()
        result = dict(status="PASS_motion_calibrated_SGD_control", calibration=calibration,
                      development_motion_ratio=development["motion_to_direct_ratio"], completed_pairs=len(rows),
                      independent_probability_gradient_values=sum(s["independent_checked_values"] for r in [development]+rows for s in r["support"]),
                      inputs_unchanged=True, episodes=rows)
    except Exception:
        result = dict(status="FAIL", traceback=traceback.format_exc(), completed_pairs=len(rows))
        raise
    finally:
        write_json(OUT / "input_files.json", inputs.used)
        result.update(ended=now(), elapsed_seconds=time.perf_counter()-started, process_peak_working_set_bytes=peak_working_set())
        write_json(OUT / "summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
