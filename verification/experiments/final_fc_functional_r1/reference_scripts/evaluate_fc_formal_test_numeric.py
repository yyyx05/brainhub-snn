#!/usr/bin/env python3
"""Evaluate all local formal N-MNIST Test counts with same-weight CPU controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_fc_w2_numeric_reference import (
    MODEL, accumulate_logits, lif18, load_model, scale_raw_to_q6, trace16, w2_dot,
)

OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_formal_test_a1"
H, C, F = 160, 10, 2312
ARMS = ("integer_raw24", "integer_q6", "same_weight_fp32_leak098", "same_weight_fp32_fixed_leak")
OVERFLOWS = ("current_wrap18", "decay_wrap18", "pre_wrap18", "post_wrap18",
             "trace_saturation16", "dot_wrap24", "cumulative_wrap24", "q6_saturation16")


def wrap(values, bits):
    return (values + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def to_q6(logits, scale):
    product = np.asarray(logits, dtype=np.int64) * scale
    quotient, remainder = np.divmod(np.abs(product), 65536)
    quotient += (remainder > 32768) | ((remainder == 32768) & ((quotient & 1) != 0))
    rounded = np.where(product < 0, -quotient, quotient)
    return np.clip(rounded, -32768, 32767), (rounded < -32768) | (rounded > 32767)


def integer_step(raw_current, w2, voltage, trace, logits, scale):
    raw_current, w2, voltage, trace, logits = (
        np.asarray(value, dtype=np.int64) for value in (raw_current, w2, voltage, trace, logits)
    )
    current_unwrapped = raw_current * 64
    current = wrap(current_unwrapped, 18)
    decay_unwrapped = (voltage * 32113 + 16384) >> 15
    decay = wrap(decay_unwrapped, 18)
    pre_unwrapped = decay + current
    pre = wrap(pre_unwrapped, 18)
    spike = (pre >= 1024).astype(np.int64)
    post_unwrapped = pre - spike * 1024
    voltage = wrap(post_unwrapped, 18)
    trace_unclamped = (((trace * 64225 + 32768) >> 16) & 65535) + spike * 1024
    trace = np.minimum(trace_unclamped, 65535)
    dot_unwrapped = spike @ w2
    dot = wrap(dot_unwrapped, 24)
    cumulative_unwrapped = logits + dot
    logits = wrap(cumulative_unwrapped, 24)
    q6, q6_saturation = to_q6(logits, scale)
    masks = (current != current_unwrapped, decay != decay_unwrapped, pre != pre_unwrapped,
             voltage != post_unwrapped, trace_unclamped > 65535, dot != dot_unwrapped,
             logits != cumulative_unwrapped, q6_saturation)
    return dict(pre=pre, voltage=voltage, trace=trace, spike=spike, dot=dot, logits=logits, q6=q6,
                overflow={key: np.count_nonzero(mask, axis=1) for key, mask in zip(OVERFLOWS, masks)})


def validate_counts(counts):
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[0] < 1 or counts.shape[1] < 1 or counts.shape[2] != F:
        raise ValueError("expected nonempty [B,T,2312] counts")
    if counts.dtype.kind not in "iuf" or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite real integers")
    if np.any(counts < 0) or np.any(counts > 32767) or np.any(counts != np.floor(counts)):
        raise ValueError("counts outside the nonnegative formal ToFrame integer range")
    return counts


def run_batch(counts, model, capture=()):
    counts = validate_counts(counts)
    batch, steps, _ = counts.shape
    w0, w2 = model["W0"].astype(np.int64), model["W2"].astype(np.int64)
    scale = float(model["W2_scale"])
    scale_q16 = round(scale * 64 * 65536)
    float_w2 = w2.astype(np.float32) * np.float32(scale)
    voltage = np.zeros((batch, H), dtype=np.int64)
    trace = np.zeros_like(voltage)
    logits = np.zeros((batch, C), dtype=np.int64)
    float_voltage = np.zeros((2, batch, H), dtype=np.float32)
    float_logits = np.zeros((2, batch, C), dtype=np.float32)
    leaks = np.array([0.98, 32113 / 32768], dtype=np.float32)
    spikes_total = np.zeros((3, batch), dtype=np.int64)
    spike_differences = np.zeros((2, batch), dtype=np.int64)
    overflow = {key: np.zeros(batch, dtype=np.int64) for key in OVERFLOWS}
    maxima = dict(current_shift6_abs=0, voltage_abs=0, trace=0, cumulative_raw_abs=0,
                  input_abs_sum_bound=0)
    history = {int(sample): [] for sample in capture}
    for t in range(steps):
        frame = counts[:, t].astype(np.int64)
        # This bound also proves exact W0 accumulation in the FP32 source:
        # every partial code sum fits the exact integer range of float32.
        bound = int(frame.sum(axis=1).max()) * 16
        if bound >= 1 << 24:
            raise ValueError("W0 FP32 exact-current bound exceeded")
        maxima["input_abs_sum_bound"] = max(maxima["input_abs_sum_bound"], bound)
        raw = np.zeros((batch, H), dtype=np.int64)
        for sample, row in enumerate(frame):
            active = np.flatnonzero(row)
            raw[sample] = row[active] @ w0[active]
        state = integer_step(raw, w2, voltage, trace, logits, scale_q16)
        voltage, trace, logits = (state[key] for key in ("voltage", "trace", "logits"))
        for key in OVERFLOWS:
            overflow[key] += state["overflow"][key]
        spikes_total[0] += state["spike"].sum(axis=1)
        float_spikes = []
        for arm in range(2):
            pre = float_voltage[arm] * leaks[arm] + raw.astype(np.float32) * np.float32(0.0625)
            spike = (pre >= np.float32(1.0)).astype(np.float32)
            float_voltage[arm] = pre - spike
            float_logits[arm] += spike @ float_w2
            if not np.all(np.isfinite(float_logits[arm])) or not np.all(np.isfinite(pre)):
                raise ValueError("nonfinite float-forward state")
            spikes_total[arm + 1] += spike.sum(axis=1).astype(np.int64)
            spike_differences[arm] += np.count_nonzero(spike != state["spike"], axis=1)
            float_spikes.append(spike)
        maxima["current_shift6_abs"] = max(maxima["current_shift6_abs"], int(np.abs(raw * 64).max()))
        maxima["voltage_abs"] = max(maxima["voltage_abs"], int(np.abs(voltage).max()))
        maxima["trace"] = max(maxima["trace"], int(trace.max()))
        maxima["cumulative_raw_abs"] = max(maxima["cumulative_raw_abs"], int(np.abs(logits).max()))
        for sample in history:
            row = {key: state[key][sample].copy() for key in ("pre", "voltage", "trace", "spike", "dot", "logits", "q6")}
            row["float_spike"] = np.stack([s[sample] for s in float_spikes])
            row["float_voltage"] = float_voltage[:, sample].copy()
            row["float_logits"] = float_logits[:, sample].copy()
            history[sample].append(row)
    return dict(logits=[logits, state["q6"], float_logits[0], float_logits[1]],
                spikes=spikes_total, spike_differences=spike_differences,
                overflow=overflow, maxima=maxima, history=history)


def check_scalar(counts, model, observed):
    voltage, trace, logits = [0] * H, [0] * H, [0] * C
    w2 = model["W2"].tolist()
    scale = round(float(model["W2_scale"]) * 64 * 65536)
    comparisons = 0
    for t, frame in enumerate(counts):
        active = np.flatnonzero(frame)
        raw = [sum(int(frame[f]) * int(model["W0"][f, h]) for f in active) for h in range(H)]
        pre, spike, voltage = map(list, zip(*[lif18(value * 64, old) for value, old in zip(raw, voltage)]))
        trace = [trace16(old, s, fractional_bits=10) for old, s in zip(trace, spike)]
        dot = w2_dot(w2, spike)
        logits = accumulate_logits(logits, dot)
        q6 = [scale_raw_to_q6(value, scale) for value in logits]
        for key, expected in (("pre", pre), ("spike", spike), ("voltage", voltage),
                              ("trace", trace), ("dot", dot), ("logits", logits), ("q6", q6)):
            np.testing.assert_array_equal(observed[t][key], expected, err_msg=f"scalar t={t} {key}")
            comparisons += len(expected)
    return {"timesteps": len(counts), "compared_values": comparisons, "mismatches": 0}


def check_float_source(counts, model_path, traces_root, observed):
    import torch
    from scripts.run_fc_tp_h160_mask_ablation import build_h160_model
    source = build_h160_model(traces_root, model_path, 1, torch.device("cpu"))
    checks = []
    with torch.inference_mode():
        for arm, leak in enumerate((0.98, 32113 / 32768)):
            source.network[0].leak_m = leak
            source.reset_potential()
            max_error = 0.0
            for t, frame in enumerate(counts):
                result = source.inference(torch.from_numpy(frame[None].astype(np.float32)))
                expected = observed[t]
                np.testing.assert_array_equal(source.hs[-1].numpy()[0], expected["float_spike"][arm])
                np.testing.assert_array_equal(source.network[0].membrane_potential.numpy()[0], expected["float_voltage"][arm])
                actual = result.numpy()[0]
                max_error = max(max_error, float(np.abs(actual - expected["float_logits"][arm]).max()))
                np.testing.assert_allclose(actual, expected["float_logits"][arm], rtol=2e-5, atol=2e-4)
            if int(actual.argmax()) != int(expected["float_logits"][arm].argmax()):
                raise AssertionError("source float prediction mismatch")
            checks.append({"arm": ARMS[arm + 2], "spike_mismatches": 0, "membrane_mismatches": 0,
                           "logit_max_abs_error": max_error, "prediction_mismatches": 0})
    return checks


def inventory(data_root, traces_root):
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic
    with patch.object(tonic.datasets.NMNIST, "download", side_effect=RuntimeError("read-only data; download/extraction disabled")):
        dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=False, first_saccade_only=True)
    histogram = np.bincount(dataset.targets, minlength=10)
    if len(dataset) != 10000 or histogram.shape != (10,) or np.any(histogram == 0):
        raise ValueError("local official Test must contain 10000 samples covering all ten classes")
    scalar_ids = [next(i for i, label in enumerate(dataset.targets) if label == c) for c in range(10)]
    scalar_ids.append(9999)
    return dataset, histogram.tolist(), sorted(set(scalar_ids))


def load_batch(data_root, traces_root, indices):
    import tonic
    from scripts.run_fc_tp_supervised_shift_adaptation import load_dataset_batches
    records = []
    original = tonic.datasets.NMNIST.__getitem__

    def observe(dataset, index):
        frames, label = original(dataset, index)
        if frames.ndim != 2 or frames.shape[1] != F or np.any(frames < 0):
            raise ValueError("invalid formal ToFrame counts")
        records.append({"dataset_index": index, "label": int(label), "valid_timesteps": len(frames),
                        "rawfilename": Path(dataset.data[index]).resolve().relative_to(data_root.resolve()).as_posix()})
        return frames, label

    with patch.object(tonic.datasets.NMNIST, "download", side_effect=RuntimeError("read-only data; download/extraction disabled")), \
            patch.object(tonic.datasets.NMNIST, "__getitem__", observe):
        batches = load_dataset_batches(traces_root=traces_root, data_root=data_root, train=False,
                                       indices=indices, batch_size=64)
    if len(batches) != 1 or [r["dataset_index"] for r in records] != indices:
        raise ValueError("formal loader changed sample order")
    frames, labels = batches[0]
    np.testing.assert_array_equal(labels.numpy(), [r["label"] for r in records])
    return frames.numpy(), labels.numpy(), records


def metrics_for(labels, logits):
    predicted = np.asarray(logits).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    correct = int(np.count_nonzero(predicted == labels))
    return {"samples": len(labels), "correct": correct, "accuracy_percent": 100 * correct / len(labels),
            "confusion_true_rows": confusion.tolist(),
            "argmax_ties": int(np.count_nonzero((logits == logits.max(axis=1, keepdims=True)).sum(axis=1) > 1))}


def estimate_full_seconds(first_batch_seconds, samples=10000, batch=64):
    return first_batch_seconds * ((samples + batch - 1) // batch) * 2


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/datasets"))
    parser.add_argument("--traces-root", type=Path, default=Path("D:/Desktop/traces_propagation-main"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-estimated-seconds", type=float, default=7200)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        parser.error("output must stay within algorithm/fc_formal_test_a1")
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("accuracy.json", "predictions.csv", "timing.json")):
        raise FileExistsError("use a fresh output directory; do not overwrite evidence")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    dataset, histogram, scalar_ids = inventory(args.data_root, args.traces_root)
    model = load_model()
    arrays = [np.empty((10000, C), dtype=np.int64) for _ in range(2)] + [np.empty((10000, C), dtype=np.float32) for _ in range(2)]
    all_labels = np.array(dataset.targets, dtype=np.int64)
    overflows = {key: 0 for key in OVERFLOWS}
    maxima = {}
    comparisons = []
    spike_differences = np.zeros(2, dtype=np.int64)
    input_stats = dict(events=0, nonzero_features=0, repeated_extra_events=0, frames_gt49=0,
                       empty_frames=0, padding_frames=0, max_feature_count=0, max_frame_events=0,
                       valid_T_min=100000, valid_T_max=0)
    fields = ["dataset_index", "label", "rawfilename", "valid_timesteps", "batch_timesteps", "events", "repeated_extra_events"]
    fields += [f"{arm}_{field}" for arm in ARMS for field in ("prediction", "correct")]
    fields += ["integer_hidden_spikes", "float098_hidden_spikes", "float_fixed_hidden_spikes",
               "float098_spike_differences", "float_fixed_spike_differences", *OVERFLOWS]
    with (out / "predictions.csv").open("x", encoding="utf-8", newline="") as csv_file, \
            (out / "batch_metrics.jsonl").open("x", encoding="utf-8") as batch_log:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for start in range(0, 10000, 64):
            tick = time.perf_counter()
            stop = min(start + 64, 10000)
            frames, labels, records = load_batch(args.data_root, args.traces_root, list(range(start, stop)))
            np.testing.assert_array_equal(labels, all_labels[start:stop])
            capture = [index - start for index in scalar_ids if start <= index < stop]
            result = run_batch(frames, model, capture)
            for local in capture:
                scalar = check_scalar(frames[local], model, result["history"][local])
                source = check_float_source(frames[local], MODEL, args.traces_root, result["history"][local])
                comparisons.append({"dataset_index": start + local, "scalar": scalar, "float_source": source})
            for arm in range(4):
                arrays[arm][start:stop] = result["logits"][arm]
            for key in OVERFLOWS:
                overflows[key] += int(result["overflow"][key].sum())
            for key, value in result["maxima"].items():
                maxima[key] = max(maxima.get(key, 0), value)
            spike_differences += result["spike_differences"].sum(axis=1)
            frame_events = frames.sum(axis=2, dtype=np.int64)
            events = frame_events.sum(axis=1)
            nonzero = np.count_nonzero(frames, axis=(1, 2))
            input_stats["events"] += int(events.sum())
            input_stats["nonzero_features"] += int(nonzero.sum())
            input_stats["repeated_extra_events"] += int((events - nonzero).sum())
            input_stats["frames_gt49"] += int(np.count_nonzero(frame_events > 49))
            input_stats["empty_frames"] += int(np.count_nonzero(frame_events == 0))
            input_stats["padding_frames"] += sum(frames.shape[1] - r["valid_timesteps"] for r in records)
            input_stats["max_feature_count"] = max(input_stats["max_feature_count"], int(frames.max()))
            input_stats["max_frame_events"] = max(input_stats["max_frame_events"], int(frame_events.max()))
            input_stats["valid_T_min"] = min(input_stats["valid_T_min"], min(r["valid_timesteps"] for r in records))
            input_stats["valid_T_max"] = max(input_stats["valid_T_max"], max(r["valid_timesteps"] for r in records))
            for local, record in enumerate(records):
                row = {**record, "batch_timesteps": int(frames.shape[1]), "events": int(events[local]),
                       "repeated_extra_events": int(events[local] - nonzero[local])}
                for arm, name in enumerate(ARMS):
                    prediction = int(result["logits"][arm][local].argmax())
                    row[name + "_prediction"] = prediction
                    row[name + "_correct"] = int(prediction == int(labels[local]))
                row.update(zip(("integer_hidden_spikes", "float098_hidden_spikes", "float_fixed_hidden_spikes"),
                               map(int, result["spikes"][:, local])))
                row.update(zip(("float098_spike_differences", "float_fixed_spike_differences"),
                               map(int, result["spike_differences"][:, local])))
                row.update({key: int(result["overflow"][key][local]) for key in OVERFLOWS})
                writer.writerow(row)
            elapsed = time.perf_counter() - tick
            batch_log.write(json.dumps({"start": start, "stop": stop, "timesteps": int(frames.shape[1]),
                                        "seconds": elapsed, "events": int(events.sum())}) + "\n")
            batch_log.flush()
            csv_file.flush()
            if start == 0:
                estimate = estimate_full_seconds(elapsed)
                timing = {"first_batch_samples": 64, "first_batch_seconds": elapsed,
                          "estimate_with_2x_margin_seconds": estimate, "budget_seconds": args.max_estimated_seconds,
                          "includes": "正式loader、三个前向数值状态、逐样本记录及首样本标量/原模型核查",
                          "decision": "continue" if estimate <= args.max_estimated_seconds else "stop_over_budget"}
                _json(out / "timing.json", timing)
                print("FIRST_BATCH " + json.dumps(timing, ensure_ascii=False), flush=True)
                if estimate > args.max_estimated_seconds:
                    return 2
            if stop % 1024 == 0 or stop == 10000:
                print(f"FULL_TEST samples={stop}/10000 elapsed={time.perf_counter()-started:.2f}s", flush=True)
            del result, frames
    metrics = {name: metrics_for(all_labels, values) for name, values in zip(ARMS, arrays)}
    differences = {}
    integer_prediction = arrays[0].argmax(axis=1)
    for arm in range(1, 4):
        prediction = arrays[arm].argmax(axis=1)
        changed = prediction != integer_prediction
        left_correct, right_correct = integer_prediction == all_labels, prediction == all_labels
        differences[ARMS[arm]] = {
            "prediction_changes_vs_integer_raw24": int(changed.sum()),
            "changed_dataset_indices": np.flatnonzero(changed).tolist(),
            "integer_correct_other_wrong": int(np.count_nonzero(left_correct & ~right_correct)),
            "integer_wrong_other_correct": int(np.count_nonzero(~left_correct & right_correct)),
            "integer_minus_other_accuracy_points": metrics[ARMS[0]]["accuracy_percent"] - metrics[ARMS[arm]]["accuracy_percent"],
        }
        if arm >= 2:
            differences[ARMS[arm]]["final_logit_max_abs_error_real_units"] = float(np.abs(arrays[0] * float(model["W2_scale"]) - arrays[arm]).max())
            differences[ARMS[arm]]["hidden_spike_mismatches"] = int(spike_differences[arm - 2])
    class_rows = []
    for label in range(10):
        selected = all_labels == label
        row = {"class": label, "samples": int(selected.sum())}
        for name, values in zip(ARMS, arrays):
            correct = int(np.count_nonzero(values[selected].argmax(axis=1) == label))
            row[name + "_correct"] = correct
            row[name + "_accuracy_percent"] = 100 * correct / row["samples"]
        row["integer_vs_float098_prediction_changes"] = int(np.count_nonzero(arrays[0][selected].argmax(1) != arrays[2][selected].argmax(1)))
        class_rows.append(row)
    with (out / "per_class.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    np.savez_compressed(out / "final_logits.npz", dataset_indices=np.arange(10000), labels=all_labels,
                        **dict(zip(ARMS, arrays)))
    import tonic
    report = {
        "status": "PASS", "scope": "官方N-MNIST Test全10000的同权重软件数值对照，不是RTL全测试集精度",
        "command": [sys.executable, "-B", *sys.argv], "device": "cpu", "cpu_threads": 1,
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "tonic": tonic.__version__},
        "tf32": False, "model": str(MODEL), "data_root": str(args.data_root.resolve()),
        "traces_root": str(args.traces_root.resolve()), "class_histogram": histogram,
        "protocol": {"samples": 10000, "batch_size": 64, "last_batch_size": 16, "sample_order": "Tonic Test枚举顺序0..9999",
                     "first_saccade": "t<100000 us", "frame_window_us": 1000,
                     "framing": "原load_dataset_batches；Tonic从首事件起分窗、include_incomplete=False；每batch PadTensors至最长真实T",
                     "counts": "保重数，无49槽裁剪/二值化，无固定T100补齐或截断", "reset": "每batch独立零状态；各样本状态互相独立",
                     "learning": False, "shift": False, "argmax_tie": "最小类别索引", "scalar_ids_preselected": scalar_ids},
        "numeric": {"integer": "INT5 W0 count-sum<<6/wrap18；LIF18 leak32113 threshold1024；UQ6.10 trace decay64225；INT8 W2 raw24累计",
                    "q6_readout": "raw24乘47303，RNE/sat16；单独报告Q6读出的argmax与tie",
                    "float": "相同NPZ的code*scale，不是原始FP32 checkpoint；FP32 LIF阈值1、soft reset，输出LI累加",
                    "float_leaks": [0.98, 32113 / 32768], "w2_scale": float(model["W2_scale"]),
                    "interpretation": "前向算术/泄漏系数差异，不是权重量化损失；CPU整数计算未使用GPU浮点替代"},
        "input_statistics": input_stats, "overflow_counts": overflows, "maxima": maxima,
        "metrics": metrics, "differences": differences, "per_class": class_rows,
        "validation": {"scalar_samples": len(comparisons), "scalar_mismatches": 0,
                       "source_float_checks": "两个float控制均逐时步核对原TP_MLP：隐藏spike/vmem精确相同，logit按记录误差和容差核查，最终预测相同"},
        "wall_seconds": time.perf_counter() - started,
    }
    _json(out / "accuracy.json", report)
    _json(out / "scalar_validation.json", comparisons)
    lines = ["# FC 官方 Test 同权重数值对照", "", "已完成本地官方N-MNIST Test全部10000样本、10类。仅软件参考，非RTL全测试集精度。", "",
             "同一现有INT5 W0/INT8 W2的code*scale作为float权重，不是原始FP32 checkpoint；不训练、不改权重、不选样。",
             "原Tonic first-saccade<100ms、1ms分帧、每B64 pad至最长真实T，末批B16；无49事件裁剪，不丢counts。", "",
             "| 数值臂 | 正确数/10000 | 准确率 |", "|---|---:|---:|"]
    for name in ARMS:
        lines.append(f"| {name} | {metrics[name]['correct']} | {metrics[name]['accuracy_percent']:.4f}% |")
    lines += ["", "主整数结果用raw24 argmax；Q6读出另列，平分时选最小类别。两个float臂分别使用0.98和32113/32768的leak。",
              "这比较同权重下的前向数值与leak差异，不能解释为原始FP32权重量化损失。", "",
              f"原始帧长{input_stats['valid_T_min']}..{input_stats['valid_T_max']}；事件{input_stats['events']}，额外重复事件{input_stats['repeated_extra_events']}，超过49事件的帧{input_stats['frames_gt49']}。",
              f"整数与主float预测差异{differences[ARMS[2]]['prediction_changes_vs_integer_raw24']}例；整数减主float准确率{differences[ARMS[2]]['integer_minus_other_accuracy_points']:+.4f}个百分点。",
              "越界/饱和按所有timestep、样本、元素累计：" + json.dumps(overflows, ensure_ascii=False), "",
              f"固定各类首样本及最后样本共{len(comparisons)}例，完整batch时间轴逐步标量核对零失配；两float臂另核对原TP_MLP。",
              "首B64耗时和两倍裕量预算估算见timing.json；该批直接计入全量，没有重跑来挑结果。",
              f"CPU单线程，无GPU/TF32；程序内部墙钟{report['wall_seconds']:.3f}秒。进程树采样资源见run/resources.json。", "",
              "逐样本身份、标签、预测、正确性、脉冲差异及越界：predictions.csv。每类准确率：per_class.csv。",
              "所有最终logit：final_logits.npz。完整配置、混淆矩阵、差异索引：accuracy.json。",
              "独立标量与原模型核查：scalar_validation.json；各批耗时：batch_metrics.jsonl。", "",
              "复现（已有输出须指定本目录内新的--output-dir）：", "", "```powershell",
              "python -B scripts/evaluate_fc_formal_test_numeric.py --data-root D:/datasets --traces-root D:/Desktop/traces_propagation-main",
              "python -B -m unittest discover -s tests -p test_fc_formal_test_numeric.py -v", "```", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": {name: metrics[name]['correct'] for name in ARMS},
                      "samples": 10000, "wall_seconds": report['wall_seconds']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
