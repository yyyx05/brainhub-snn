"""New -16/+32 input conditions with unchanged +16-selected head rules.

All official queries and the three saved support resamples are retained. This
tests condition transfer, not a new dataset, per-condition tuning, or RTL cost.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import h5py
import torch
import run_srnn_integer_optimizer_comparison as base
from run_srnn_sgd_expanded import Inputs, select_expanded, peak_working_set
import evaluate_srnn_full_integer as full
import supplement_srnn_backbone_numeric as bb
from supplement_srnn_magnitude_sgd import REPORT_SEEDS, update_regime

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/paper_evidence/supplement_20261012/srnn_integer_optimizer/report_r1"
OUT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_shift_transfer_r1"
SHIFTS = (-16, 32)
METHODS = ("direct_response", "magnitude_sgd")


def save(path, value):
    with open(base.native_path(path), "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def shift_counts(counts, shift):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 700 or counts.dtype.kind not in "iu":
        raise ValueError("integer T,700 counts required")
    if shift not in SHIFTS or np.any((counts < 0) | (counts > 255)):
        raise ValueError("only prescribed new conditions without count clipping")
    changed = np.zeros_like(counts)
    if shift > 0:
        changed[:, shift:] = counts[:, :-shift]
    else:
        changed[:, :shift] = counts[:, -shift:]
    return changed


def independent_new_sequences(counts, captured, matrices):
    inputs = counts[:, :2]
    backbone = bb.backbone_reference(inputs, matrices[0], matrices[1])
    batch = inputs.shape[1]
    state = np.zeros((batch, 256), dtype=np.int64)
    trace, feature = state.copy(), state.copy()
    logit = np.zeros((batch, 20), dtype=np.int64)
    pre, spike, logits = [], [], []
    for b_spike in backbone["spike"]:
        value = base.reference.forward_step(b_spike*16384, matrices[2].T, matrices[3].T,
                                           state, trace, feature, logit)
        state, trace, feature, logit = (value[k] for k in ("membrane", "trace", "feature", "logit"))
        pre.append(value["pre"])
        spike.append(value["spike"])
        logits.append(logit.copy())
    expected = dict(backbone_pre=backbone["pre"], backbone_spike=backbone["spike"],
                    m2_pre=np.stack(pre), m2_spike=np.stack(spike), logit=np.stack(logits))
    checked = {}
    for key, value in expected.items():
        np.testing.assert_array_equal(captured[key][:, :2], value, err_msg=key)
        checked[key] = int(value.size)
    return dict(padded_timesteps=len(inputs), samples_in_first_batch=[0, 1], checked_values=checked)


def features(inputs, source, indices, shift, tag, weights, matrices, initial, keep_spikes=False, verify=False):
    inputs.record(source)
    sums, labels, batches, supports = [], [], [], []
    verification = None
    with h5py.File(base.native_path(source), "r") as dataset:
        for offset in range(0, len(indices), 64):
            selected = indices[offset:offset+64]
            original = [full.frame_counts(dataset["spikes/times"][i], dataset["spikes/units"][i]) for i in selected]
            lengths = [len(x) for x in original]
            counts = np.zeros((max(lengths), len(selected), 700), dtype=np.int64)
            lost, original_mass, shifted_mass = [], [], []
            for b, raw in enumerate(original):
                changed = shift_counts(raw, shift)
                counts[:len(raw), b] = changed
                original_mass.append(int(raw.sum()))
                shifted_mass.append(int(changed.sum()))
                lost.append(int(raw.sum()-changed.sum()))
            y = np.asarray([dataset["labels"][i] for i in selected], dtype=np.int64)
            frozen, captured = full.forward(counts, weights, "cuda", capture=True)
            if verify and offset == 0:
                verification = independent_new_sequences(counts, captured, matrices)
                verification["official_indices"] = selected[:2]
            spikes = captured["m2_spike"].astype(np.uint8)
            summed = spikes.sum(axis=0, dtype=np.int64)
            np.testing.assert_array_equal(base.frozen_logits(summed, initial), frozen)
            sums.append(summed)
            labels.extend(y.tolist())
            batches.append(dict(indices=selected, native_timesteps=lengths, padded_timesteps=max(lengths),
                                raw_count_mass=original_mass, shifted_count_mass=shifted_mass, dropped_count_mass=lost))
            if keep_spikes:
                supports.append((spikes, y))
                np.savez_compressed(base.native_path(OUT / "features" / f"{tag}_batch{offset//64}.npz"),
                                    spikes=spikes, labels=y)
            del captured, frozen, counts, original
    counts, y = np.concatenate(sums), np.asarray(labels, dtype=np.int64)
    np.savez_compressed(base.native_path(OUT / "features" / f"{tag}.npz"), counts=counts, labels=y)
    save(OUT / "features" / f"{tag}.json", dict(source=str(source.relative_to(ROOT)), indices=indices,
        shift=shift, batches=batches, checked_frozen_logits=len(indices)*20,
        independent_full_sequence_check=verification))
    print(f"FEATURES {tag} samples={len(indices)} dropped_mass={sum(sum(b['dropped_count_mass']) for b in batches)}", flush=True)
    return (counts, y), supports, verification


def episode(seed, shift, method, rate, initial, batches, clean, shifted):
    start = time.perf_counter()
    code, residual = initial.copy(), np.zeros_like(initial)
    totals, records = {}, []
    before_clean, before_shift = base.evaluate(*clean, code), base.evaluate(*shifted, code)
    curve = [dict(support_samples=0, clean=before_clean, shifted=before_shift, counters={})]
    tag = f"shift{shift}_seed{seed}_{method}"
    for b, (spikes, y) in enumerate(batches):
        result = base.adapt(spikes, y, code, residual, method, rate if method == "magnitude_sgd" else None, verify=True)
        code, residual = result["codes"], result["residual"]
        for key, value in result["counters"].items():
            totals[key] = totals.get(key, 0)+value
        records.append(dict(batch=b, labels=y.tolist(), timesteps=len(spikes), rows=result["rows"],
                            counters=result["counters"], independent_checked_values=result["independent_checked_values"]))
        curve.append(dict(support_samples=64*(b+1), clean=base.evaluate(*clean, code),
                          shifted=base.evaluate(*shifted, code), counters=totals.copy()))
        np.savez_compressed(base.native_path(OUT / f"{tag}_batch{b+1}_state.npz"), initial=initial, codes=code, residual=residual)
    last = curve[-1]
    row = dict(seed=seed, shift=shift, method=method, learning_rate=rate if method == "magnitude_sgd" else None,
        baseline_clean=before_clean, baseline_shifted=before_shift,
        clean_after=last["clean"], shifted_after=last["shifted"],
        clean_change_pp=last["clean"]["accuracy_percent"]-before_clean["accuracy_percent"],
        shifted_gain_pp=last["shifted"]["accuracy_percent"]-before_shift["accuracy_percent"],
        counters=totals, changed_code_addresses=int(np.count_nonzero(code != initial)),
        update_regime=update_regime(totals, code), support=records, curve=curve,
        elapsed_seconds=time.perf_counter()-start)
    save(OUT / (tag+".json"), row)
    print(f"ADAPT {tag} clean={last['clean']['correct']} shifted={last['shifted']['correct']} "
          f"shift_gain_pp={row['shifted_gain_pp']:.6f} code_events={totals['code_changes']}", flush=True)
    return row


def tables(rows):
    flat, curves, transitions = [], [], []
    for row in rows:
        flat.append(dict(shift=row["shift"], seed=row["seed"], method=row["method"], rate=row["learning_rate"],
            query=row["clean_after"]["total"], clean_before=row["baseline_clean"]["correct"],
            clean_after=row["clean_after"]["correct"], shift_before=row["baseline_shifted"]["correct"],
            shift_after=row["shifted_after"]["correct"], clean_change_pp=row["clean_change_pp"], shifted_gain_pp=row["shifted_gain_pp"],
            code_events=row["counters"]["code_changes"], residual_events=row["counters"]["residual_changes"],
            saturations=row["counters"]["saturations"], final_changed_addresses=row["changed_code_addresses"]))
        for c in row["curve"]:
            curves.append(dict(shift=row["shift"], seed=row["seed"], method=row["method"], support_samples=c["support_samples"],
                clean_correct=c["clean"]["correct"], shifted_correct=c["shifted"]["correct"],
                code_events=c["counters"].get("code_changes", 0), residual_events=c["counters"].get("residual_changes", 0)))
    for name, data in (("comparison.csv", flat), ("curves.csv", curves)):
        with open(base.native_path(OUT / name), "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "features").mkdir()
    inputs, rows, checks = Inputs(), [], []
    started = time.perf_counter()
    result = {}
    try:
        # No new condition is evaluated before reading the existing development decision.
        dev = inputs.json(OUT.parent / "srnn_sgd_expanded_r1/dev/summary.json")
        decision = select_expanded(dev["episodes"])
        if dev["status"] != "PASS_expanded_integer_SGD_dev" or decision != dev["selection"] or decision["status"] != "selected":
            raise ValueError("existing completed development selection required")
        rate = decision["selected_rate"]
        protocol = dict(started=datetime.now(timezone.utc).astimezone().isoformat(), shifts=list(SHIFTS),
            choice="Prespecified opposite direction (-16) and doubled original amplitude (+32); no condition dropped or tuned after evaluation",
            mask="Wout-only", direct_rule=dict(increment=31, threshold=65), sgd_rate=rate,
            selection_source="srnn_sgd_expanded_r1/dev/summary.json (+16 only)",
            query_indices=list(range(2264)), support_seeds=list(REPORT_SEEDS), support_per_seed=128,
            forward="Unchanged production integer equations in existing GPU FP64 exact-integer carrier",
            arithmetic="Unchanged logit/trace/LUT softmax/Q40 fixed /64; SGD existing FP64 rate product, integer residual",
            learning_scope="Condition transfer of fixed hyperparameters, not per-condition best SGD or full TP",
            statistical_scope="Old raw records with new zero-fill transforms; fixed checkpoint/shared historical query, not new IID dataset or training seeds",
            cost_scope="Software update events, not physical writes/cycles or GPU-vs-FPGA timing",
            resource=dict(cpu_threads=1, gpu_memory_fraction_limit=0.25, streaming_batch=64))
        save(OUT / "protocol.json", protocol)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this bounded existing reference, no automatic backend substitution")
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        for path in (bb.MODEL / "w0_int8.mem", bb.MODEL / "r_int8.mem",
                     ROOT / "tb/data/srnn_m2_v28_wh1_initial_q14.mem", ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem"):
            inputs.record(path)
        weights = full.load_weights("cuda")
        matrices = [w.cpu().numpy().astype(np.int64) for w in weights]
        initial = matrices[3].T.copy()
        cached = inputs.npz(OLD / "features/official_query_clean.npz")
        clean = cached["counts"], cached["labels"]
        indices_by_seed = {seed: inputs.json(OLD / f"seed{seed}_indices.json")["support"] for seed in REPORT_SEEDS}
        for shift in SHIFTS:
            shifted, _, checked = features(inputs, ROOT / "imports/srnn/dataset_cache/SHD/shd_test.h5",
                list(range(2264)), shift, f"query_shift{shift}", weights, matrices, initial, verify=True)
            np.testing.assert_array_equal(clean[1], shifted[1])
            checks.append(dict(shift=shift, **checked))
            for seed, indices in indices_by_seed.items():
                if len(indices) != 128 or len(set(indices)) != 128:
                    raise ValueError("original complete support128 required")
                _, batches, _ = features(inputs, ROOT / "imports/srnn/dataset_cache/SHD/shd_train.h5",
                    indices, shift, f"seed{seed}_shift{shift}", weights, matrices, initial, keep_spikes=True)
                for method in METHODS:
                    rows.append(episode(seed, shift, method, rate, initial, batches, clean, shifted))
        inputs.check_unchanged()
        tables(rows)
        save(OUT / "input_files.json", inputs.used)
        result = dict(status="PASS_two_new_shift_conditions_integer_adaptation", protocol=protocol,
            episodes=rows, independent_forward_checks=checks,
            independent_probability_gradient_values=sum(s["independent_checked_values"] for r in rows for s in r["support"]),
            gpu_device=torch.cuda.get_device_name(), torch_version=torch.__version__, numpy_version=np.__version__)
    except Exception:
        result = dict(status="FAIL", traceback=traceback.format_exc(), completed_episodes=len(rows))
        raise
    finally:
        result.update(ended=datetime.now(timezone.utc).astimezone().isoformat(), elapsed_seconds=time.perf_counter()-started,
                      process_peak_working_set_bytes=peak_working_set(),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)
        save(OUT / "summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("episodes", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
