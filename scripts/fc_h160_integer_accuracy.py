#!/usr/bin/env python3
"""Frozen-weight CPU reference accuracy on the imported 1024 x T100 snapshot."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_fc_w2_numeric_reference import (
    accumulate_logits, lif18, trace16, w2_dot,
)

INPUT = ROOT / "imports/fc/p3a_h160/data/eval_1024_t100"
MODEL = ROOT / "imports/fc/p3a_h160/data/weights/p3a_nmnist_h160_k8_int5_w2_int8_hash8/weights_h160_int5_w2_int8.npz"
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_h160_integer_a1"
N, T, SLOTS, H, CLASSES = 1024, 100, 49, 160, 10
ARMS = ("integer_lif18", "same_weight_float32_leak098", "same_weight_float32_fixed_leak")


def wrap(values, bits):
    return (values + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def load_hex(path):
    with path.open(encoding="ascii") as stream:
        return np.fromiter((int(line.strip(), 16) for line in stream), dtype=np.int64)


def run_batch(features, w0, w2, scale):
    batch = len(features)
    voltage = np.zeros((batch, H), dtype=np.int64)
    trace = np.zeros_like(voltage)
    logits = np.zeros((batch, CLASSES), dtype=np.int64)
    float_voltage = np.zeros((2, batch, H), dtype=np.float32)
    float_logits = np.zeros((2, batch, CLASSES), dtype=np.float32)
    float_w2 = w2.astype(np.float32) * np.float32(scale)
    leaks = np.array([0.98, 32113 / 32768], dtype=np.float32)
    spike_totals = np.zeros((3, batch), dtype=np.int64)
    disagreements = np.zeros(2, dtype=np.int64)
    overflow = dict(current_wrap=0, pre_state_wrap=0, post_state_wrap=0,
                    trace_saturation=0, cumulative_logit_wrap=0, cumulative_q6_saturation=0)
    raw_logit_max = 0
    first_steps = []
    for timestep in range(T):
        raw_current = w0[features[:, timestep]].sum(axis=1, dtype=np.int64)
        current = wrap(raw_current << 6, 18)
        overflow["current_wrap"] += int(np.count_nonzero(current != raw_current << 6))
        decay = wrap((voltage * 32113 + 16384) >> 15, 18)
        pre_unwrapped = decay + current
        pre = wrap(pre_unwrapped, 18)
        overflow["pre_state_wrap"] += int(np.count_nonzero(pre != pre_unwrapped))
        spikes = pre >= 1024
        post_unwrapped = pre - spikes * 1024
        voltage = wrap(post_unwrapped, 18)
        overflow["post_state_wrap"] += int(np.count_nonzero(voltage != post_unwrapped))
        trace_unclamped = (((trace * 64225 + 32768) >> 16) & 65535) + spikes * 1024
        overflow["trace_saturation"] += int(np.count_nonzero(trace_unclamped > 65535))
        trace = np.minimum(trace_unclamped, 65535)
        dot = wrap(spikes.astype(np.int64) @ w2, 24)
        next_logits = logits + dot
        logits = wrap(next_logits, 24)
        overflow["cumulative_logit_wrap"] += int(np.count_nonzero(logits != next_logits))
        raw_logit_max = max(raw_logit_max, int(np.abs(logits).max()))
        product = logits * 47303
        q, r = np.abs(product) >> 16, np.abs(product) & 65535
        q = q + ((r > 32768) | ((r == 32768) & ((q & 1) != 0)))
        q = np.where(product < 0, -q, q)
        overflow["cumulative_q6_saturation"] += int(np.count_nonzero((q < -32768) | (q > 32767)))
        spike_totals[0] += spikes.sum(1)
        for arm in range(2):
            float_pre = float_voltage[arm] * leaks[arm] + raw_current.astype(np.float32) * np.float32(0.0625)
            float_spike = float_pre >= np.float32(1.0)
            float_voltage[arm] = float_pre - float_spike.astype(np.float32)
            float_logits[arm] += float_spike.astype(np.float32) @ float_w2
            spike_totals[arm + 1] += float_spike.sum(1)
            disagreements[arm] += np.count_nonzero(float_spike != spikes)
        first_steps.append({"timestep": timestep, "hidden_spikes": int(spikes[0].sum()),
                            "dot_raw24": dot[0].tolist(), "cumulative_raw24": logits[0].tolist()})
    return dict(logits=[logits, float_logits[0], float_logits[1]], voltage=voltage,
                trace=trace, spike_totals=spike_totals, disagreements=disagreements,
                overflow=overflow, raw_logit_abs_max=raw_logit_max, first_steps=first_steps)


def check_scalar(sample_id, features, w0, w2, batch_result):
    voltage, trace, logits = [0] * H, [0] * H, [0] * CLASSES
    spike_count = 0
    weights = w2.tolist()
    for timestep in range(T):
        current = w0[features[timestep]].sum(axis=0, dtype=np.int64) << 6
        _, spikes, voltage = map(list, zip(*[lif18(int(i), v) for i, v in zip(current, voltage)]))
        trace = [trace16(e, s, fractional_bits=10) for e, s in zip(trace, spikes)]
        dot = w2_dot(weights, spikes)
        logits = accumulate_logits(logits, dot)
        spike_count += sum(spikes)
        observed = batch_result["first_steps"][timestep]
        assert sum(spikes) == observed["hidden_spikes"]
        assert dot == observed["dot_raw24"]
        assert logits == observed["cumulative_raw24"]
    np.testing.assert_array_equal(voltage, batch_result["voltage"][0])
    np.testing.assert_array_equal(trace, batch_result["trace"][0])
    np.testing.assert_array_equal(logits, batch_result["logits"][0][0])
    assert spike_count == int(batch_result["spike_totals"][0, 0])
    return dict(sample_id=sample_id, timesteps=T, mismatches=0, hidden_spikes=spike_count,
                cumulative_logits_raw24=logits, final_membrane18=voltage, final_trace_uq6_10=trace)


def verify_source_float(output):
    import torch
    from scripts.run_fc_tp_h160_mask_ablation import build_h160_model

    torch.set_num_threads(1)
    ids = [0, 478, 1023]
    words = load_hex(INPUT / "eval1024_events.hex").reshape(N, T, SLOTS)[ids]
    features = words >> 8
    features = np.where(features == 4095, 2312, features)
    with np.load(MODEL, allow_pickle=False) as packed:
        w0 = np.vstack([packed["W0"].astype(np.int64), np.zeros((1, H), dtype=np.int64)])
        w2 = packed["W2"].astype(np.float32) * np.float32(float(packed["W2_scale"]))
    with (output / "predictions.csv").open(encoding="ascii", newline="") as stream:
        saved = {int(row["sample_id"]): row for row in csv.DictReader(stream) if int(row["sample_id"]) in ids}
    traces = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload/traces_propagation-main"
    model = build_h160_model(traces, MODEL, len(ids), torch.device("cpu"))
    checks = []
    for arm, leak in zip(ARMS[1:], (0.98, 32113 / 32768)):
        model.network[0].leak_m = leak
        model.reset_potential()
        voltage = np.zeros((len(ids), H), dtype=np.float32)
        logits = np.zeros((len(ids), CLASSES), dtype=np.float32)
        spike_differences = 0
        max_membrane_error = max_logit_error = 0.0
        with torch.no_grad():
            for timestep in range(T):
                inputs = np.zeros((len(ids), 2313), dtype=np.float32)
                np.add.at(inputs, (np.arange(len(ids))[:, None], features[:, timestep]), 1)
                actual = model.inference(torch.from_numpy(inputs[:, :2312].copy()))
                raw = w0[features[:, timestep]].sum(axis=1, dtype=np.int64)
                pre = voltage * np.float32(leak) + raw.astype(np.float32) * np.float32(0.0625)
                spike = pre >= np.float32(1)
                voltage = pre - spike.astype(np.float32)
                logits += spike.astype(np.float32) @ w2
                spike_differences += int(np.count_nonzero(spike != model.hs[-1].numpy()))
                max_membrane_error = max(max_membrane_error, float(np.abs(voltage - model.network[0].membrane_potential.numpy()).max()))
                max_logit_error = max(max_logit_error, float(np.abs(logits - actual.numpy()).max()))
        predicted = actual.argmax(1).tolist()
        expected = [int(saved[i][f"{arm}_prediction"]) for i in ids]
        assert predicted == expected and spike_differences == 0 and max_membrane_error == 0
        saved_logits = np.array([[float(saved[i][f"{arm}_logit_{c}"]) for c in range(CLASSES)] for i in ids])
        checks.append(dict(arm=arm, predictions=predicted, saved_predictions=expected,
                           spike_mismatches=spike_differences, membrane_max_abs_error=max_membrane_error,
                           all_steps_logit_max_abs_error=max_logit_error,
                           saved_final_logit_max_abs_error=float(np.abs(saved_logits - actual.numpy()).max())))
    report = dict(status="pass", scope="three complete CPU float source checks, not a rerun of the 1024-sample report",
                  samples=ids, timesteps=T, device="cpu", threads=1, torch=torch.__version__,
                  model_source=inspect.getsourcefile(type(model)), neuron_source=inspect.getsourcefile(type(model.network[0])),
                  builder_source=inspect.getsourcefile(build_h160_model), checks=checks)
    with (output / "formal_float_source_check.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify-source-float", action="store_true",
                        help="check samples 0,478,1023 against formal TP_MLP; requires existing results")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error("batch-size must be in [1,64]")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must remain inside the project")
    if args.verify_source_float:
        verify_source_float(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    np.seterr(over="raise", invalid="raise")
    with np.load(MODEL, allow_pickle=False) as model:
        assert model["W0"].shape == (2312, H) and model["W2"].shape == (H, CLASSES)
        assert float(model["W0_scale"]) == 0.0625 and int(model["input_shift"]) == 6
        assert int(model["lif_threshold_fixed"]) == 1024
        w0 = np.vstack([model["W0"].astype(np.int64), np.zeros((1, H), dtype=np.int64)])
        w2, scale = model["W2"].astype(np.int64), float(model["W2_scale"])
    words = load_hex(INPUT / "eval1024_events.hex")
    labels = load_hex(INPUT / "eval1024_labels.hex")
    assert words.size == N * T * SLOTS and labels.size == N
    assert np.all((labels >= 0) & (labels < CLASSES))
    label_histogram = np.bincount(labels, minlength=CLASSES)
    unique_classes = np.flatnonzero(label_histogram).tolist()
    majority_class = int(label_histogram.argmax())
    label_distribution = dict(histogram_class0_to9=label_histogram.tolist(),
                              unique_classes=unique_classes, unique_class_count=len(unique_classes),
                              majority_class_baseline=dict(prediction=majority_class,
                                  correct=int(label_histogram[majority_class]), total=N,
                                  accuracy_percent=100 * int(label_histogram[majority_class]) / N),
                              boundary="Imbalanced imported snapshot, not a ten-class balanced sample or the full N-MNIST test set")
    assert np.all((words >= 0) & (words <= 0xFFFFF))
    features = words >> 8
    valid = features != 4095
    assert np.all(features[valid] < 2312) and np.all((words[valid] & 255) == 1)
    events = int(valid.sum())
    features = np.where(valid, features, 2312).astype(np.int32).reshape(N, T, SLOTS)
    del words, valid
    arrays = [np.empty((N, CLASSES), dtype=np.int64)] + [np.empty((N, CLASSES), dtype=np.float32) for _ in range(2)]
    spikes = np.empty((3, N), dtype=np.int64)
    totals, spike_differences, scalar_checks = {}, np.zeros(2, dtype=np.int64), []
    raw_logit_max = 0
    # Three complete scalar samples check the vectorized state, dots and ordering.
    for sample_id in (0, 511, 1023):
        result = run_batch(features[sample_id:sample_id + 1], w0, w2, scale)
        scalar_checks.append(check_scalar(sample_id, features[sample_id], w0, w2, result))
    for start in range(0, N, args.batch_size):
        stop = min(N, start + args.batch_size)
        result = run_batch(features[start:stop], w0, w2, scale)
        for arm in range(3):
            arrays[arm][start:stop] = result["logits"][arm]
        spikes[:, start:stop] = result["spike_totals"]
        spike_differences += result["disagreements"]
        raw_logit_max = max(raw_logit_max, result["raw_logit_abs_max"])
        for name, value in result["overflow"].items():
            totals[name] = totals.get(name, 0) + value
        if start == 0:
            sample0_steps = result["first_steps"]
        if stop % 128 == 0 or stop == N:
            print(f"FC_INTEGER_A1 samples={stop}/{N}", flush=True)
    predictions = np.stack([a.argmax(1) for a in arrays])
    metrics = {}
    for arm, name in enumerate(ARMS):
        confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
        np.add.at(confusion, (labels, predictions[arm]), 1)
        correct = int(np.count_nonzero(predictions[arm] == labels))
        metrics[name] = dict(correct=correct, total=N, accuracy_percent=100 * correct / N,
                             hidden_spikes=int(spikes[arm].sum()), confusion_true_rows=confusion.tolist(),
                             final_logit_ties=int(np.count_nonzero((arrays[arm] == arrays[arm].max(1)[:, None]).sum(1) > 1)))
    comparisons = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        l_correct, r_correct = predictions[left] == labels, predictions[right] == labels
        changed = predictions[left] != predictions[right]
        comparisons[f"{ARMS[left]}_vs_{ARMS[right]}"] = dict(
            prediction_changes=int(changed.sum()), changed_sample_ids=np.flatnonzero(changed).tolist(),
            left_correct_right_wrong=int(np.count_nonzero(l_correct & ~r_correct)),
            left_wrong_right_correct=int(np.count_nonzero(~l_correct & r_correct)),
            accuracy_left_minus_right_points=metrics[ARMS[left]]["accuracy_percent"] - metrics[ARMS[right]]["accuracy_percent"])
    report = dict(status="pass", scope="CPU reference on an imbalanced two-class imported snapshot only; not full N-MNIST test accuracy, formal-protocol closure, RTL accuracy, or weight-quantization loss",
        label_distribution=label_distribution,
        command=[sys.executable, *sys.argv], numpy=np.__version__, cpu_math_threads=1,
        inputs=dict(events=str(INPUT / "eval1024_events.hex"), labels=str(INPUT / "eval1024_labels.hex"), weights=str(MODEL)),
        protocol=dict(samples=N, timesteps=T, slots_per_step=SLOTS, events=events, batch_size=args.batch_size,
                      sample_order="file order 0..1023", reset="each sample", labels="used for scoring only",
                      learn=False, shift=False, duplicate_slots="sum every valid event; no deduplication", argmax_tie="first class index",
                      relation_to_formal_query="separate imported snapshot; no inferred original dataset IDs, no support/query or A2 reuse"),
        input_mapping=dict(formal_source="scripts/run_fc_tp_supervised_shift_adaptation.py:121",
                           formal_preprocessing="NMNIST first_saccade_only=True; ToFrame(time_window=1000), reshape, FP32 cast; no clipping/binarization in load_dataset_batches",
                           snapshot_source="imports/fc/p3a_h160/tb/new_architecture/tb_p2c_hx_matched_official.sv:184",
                           snapshot_preprocessing="100 steps, 49 slots/step, feature4095 padding, valid event amplitude1",
                           count_representation="Repeated feature slots add repeated W0 contributions, representing positive counts only when export preserves all multiplicities; no deduplication",
                           exact_original_event_export_mapping="not established in inspected imported scripts/result metadata",
                           boundary="Do not infer matching framing, count preservation, truncation policy or sample selection from the parent TB snapshot; no formal-protocol closure"),
        integer=dict(input_shift=6, membrane_bits=18, membrane_fractional_bits=10, leak_q15=32113, threshold=1024,
                     rounding="add-half then arithmetic shift, ties toward +infinity", trace_decay_q16=64225,
                     trace_format="UQ6.10", logit_bits=24, temporal_logit="raw sum, wrap24; no Q6 conversion for prediction", overflow_counts=totals),
        float_controls=dict(weight_source="same NPZ code*scale, not original prequantization checkpoint", dtype="float32",
                            interpretation="Same-weight forward arithmetic-format quantization and leak-coefficient comparison, not weight-quantization loss",
                            leak_coefficients=[0.98, 32113 / 32768], threshold=1.0, reset="soft", output_leak=1.0,
                            source_equations="models/neuron_layers.py LIFLayer/LILayer, models/spike_activation.py >= threshold; screen_fc_tp_update_candidates._snn_params",
                            numeric_boundary="NumPy source-equation control, not a claim of bitwise PyTorch GEMM identity",
                            w2_scale=scale, spike_mismatches_vs_integer=spike_differences.tolist(),
                            final_logit_max_abs_error_vs_integer_real_units=[float(np.abs(arrays[0] * scale - a).max()) for a in arrays[1:]]),
        metrics=metrics, comparisons=comparisons, scalar_validation=dict(samples=[0, 511, 1023], timesteps_per_sample=100, mismatches=0),
        maximum_cumulative_raw_abs=raw_logit_max, wall_seconds=time.perf_counter() - started)
    for path, data in (("accuracy.json", report), ("scalar_validation.json", scalar_checks), ("sample0_steps.json", sample0_steps)):
        with (output / path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
    with (output / "predictions.csv").open("x", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label"] + [f"{a}_prediction" for a in ARMS] + [f"{a}_spikes" for a in ARMS] + [f"{a}_logit_{c}" for a in ARMS for c in range(CLASSES)])
        for i in range(N):
            writer.writerow([i, int(labels[i]), *predictions[:, i].tolist(), *spikes[:, i].tolist(), *[v for a in arrays for v in a[i].tolist()]])
    print(json.dumps({"output": str(output), "status": "pass", "metrics": {k: {n: v[n] for n in ("correct", "total", "accuracy_percent", "hidden_spikes")} for k, v in metrics.items()}, "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
