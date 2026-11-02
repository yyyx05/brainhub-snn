#!/usr/bin/env python3
"""Build time-major B16 FC integer references, without running RTL or training."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fc_tp_b16_softmax_rtl_fixture import _quantized_row
from scripts.build_fc_w2_numeric_reference import MODEL, load_model, scale_raw_to_q6

OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_final_counts_reference"
DEFAULT_COUNTS = OUTPUT.parent / "fc_formal_counts/support_shifted.npz"
BATCH, INPUTS, HIDDEN, CLASSES = 16, 2312, 160, 10
W2_BASE = INPUTS * HIDDEN
RESIDUAL_UNIT = 32768
DELTA_MAX = 131071
MEM_LAYOUT = {
    "input_counts": (32, False, "t*B*2312+s*2312+f", BATCH * INPUTS),
    "pre": (32, True, "t*B*H+s*H+h; sign-extended pre18", BATCH * HIDDEN),
    "trace": (32, False, "t*B*H+s*H+h; UQ6.10 in low16", BATCH * HIDDEN),
    "spike": (1, False, "t*B*H+s*H+h", BATCH * HIDDEN),
    "dot": (32, True, "t*B*C+s*C+c; sign-extended raw24", BATCH * CLASSES),
    "q6": (16, True, "t*B*C+s*C+c", BATCH * CLASSES),
    "cumulative": (32, True, "t*B*C+s*C+c; sign-extended raw24", BATCH * CLASSES),
    "w2_after": (16, True, "t*H*C+h*C+c; sign-extended INT8", HIDDEN * CLASSES),
    "residual_after": (16, True, "t*H*C+h*C+c", HIDDEN * CLASSES),
}


def _integer_array(value, name, low, high, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or np.any(array < low) or np.any(array > high):
        raise ValueError(f"{name} must contain integers in [{low}, {high}]")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    return array.astype(np.int64, copy=False)


def _wrap(values, bits):
    return (values + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def scale_gradient(gradient):
    """Integer RNE(-gradient*11/2**26), symmetrically saturated to signed18."""
    gradient = _integer_array(gradient, "gradient", -(1 << 47), (1 << 47) - 1)
    product = gradient * 11
    quotient, remainder = np.divmod(np.abs(product), 1 << 26)
    rounded = quotient + ((remainder > (1 << 25)) |
                          ((remainder == (1 << 25)) & ((quotient & 1) != 0)))
    saturated = rounded > DELTA_MAX
    magnitude = np.minimum(rounded, DELTA_MAX)
    return np.where(product < 0, magnitude, -magnitude), saturated


def apply_residual_update(code, residual, delta):
    code = _integer_array(code, "code", -127, 127)
    residual = _integer_array(residual, "residual", -32767, 32767, code.shape)
    delta = _integer_array(delta, "delta", -DELTA_MAX, DELTA_MAX, code.shape)
    total = residual + delta
    # Signed normalization truncates toward zero, not Python's negative //.
    code_delta = np.where(total < 0, -(np.abs(total) // RESIDUAL_UNIT),
                          np.abs(total) // RESIDUAL_UNIT)
    candidate = code + code_delta
    saturated = (candidate < -127) | (candidate > 127)
    code_next = np.clip(candidate, -127, 127)
    residual_next = np.where(saturated, 0, total - code_delta * RESIDUAL_UNIT)
    return code_next, residual_next, saturated


def quantized_error(q6, labels):
    q6 = _integer_array(q6, "q6", -32768, 32767, (BATCH, CLASSES))
    labels = _integer_array(labels, "labels", 0, CLASSES - 1, (BATCH,))
    probability = np.empty_like(q6)
    for sample in range(BATCH):
        # Every q6/64 value is exactly representable. Reuse the existing LUT
        # quantizer, not a floating softmax or software-model gradient.
        recovered, probability[sample] = _quantized_row(q6[sample] / 64)
        if not np.array_equal(recovered, q6[sample]):
            raise AssertionError("softmax quantizer changed the Q6 input")
    error = probability.copy()
    error[np.arange(BATCH), labels] -= 65536
    return probability, error


def batch_gradient(trace, error):
    trace = _integer_array(trace, "trace", 0, 65535, (BATCH, HIDDEN))
    error = _integer_array(error, "error", -65536, 65536, (BATCH, CLASSES))
    # B16 * uint16 * signed Q0.16 is bounded by 2**36, so all partial sums
    # fit the production signed48 reducer. There is no batch division.
    return trace.T @ error


def forward_step(counts, w0, w2, voltage, trace, cumulative, scale_q16):
    counts = _integer_array(counts, "counts", 0, (1 << 32) - 1, (BATCH, INPUTS))
    w0 = _integer_array(w0, "W0", -16, 15, (INPUTS, HIDDEN))
    w2 = _integer_array(w2, "W2", -127, 127, (HIDDEN, CLASSES))
    voltage = _integer_array(voltage, "voltage", -131072, 131071, (BATCH, HIDDEN))
    trace = _integer_array(trace, "trace", 0, 65535, (BATCH, HIDDEN))
    cumulative = _integer_array(cumulative, "cumulative", -(1 << 23), (1 << 23) - 1,
                                (BATCH, CLASSES))
    raw_current = np.zeros((BATCH, HIDDEN), dtype=np.int64)
    for sample, row in enumerate(counts):
        active = np.flatnonzero(row)
        raw_current[sample] = row[active] @ w0[active]
    current = _wrap(raw_current * 64, 18)
    decayed = _wrap((voltage * 32113 + 16384) >> 15, 18)
    pre = _wrap(decayed + current, 18)
    spike = (pre >= 1024).astype(np.int64)
    voltage_next = _wrap(pre - spike * 1024, 18)
    trace_next = np.minimum(65535, (((trace * 64225 + 32768) >> 16) & 65535) + spike * 1024)
    dot = _wrap(spike @ w2, 24)
    cumulative_next = _wrap(cumulative + dot, 24)
    q6 = np.array([[scale_raw_to_q6(int(value), scale_q16) for value in row]
                   for row in cumulative_next], dtype=np.int64)
    return {"pre": pre, "trace": trace_next, "spike": spike, "dot": dot,
            "cumulative": cumulative_next, "q6": q6, "voltage": voltage_next}


def load_counts(path, steps):
    with np.load(path, allow_pickle=False) as data:
        counts = data["dense_counts"].copy()
        labels = data["labels"].copy()
        lengths = data["lengths"].copy()
        indices = data["dataset_indices"].copy()
    if counts.ndim != 3 or counts.shape[0] != BATCH or counts.shape[2] != INPUTS:
        raise ValueError("expected dense_counts[B16,T,2312]")
    source_steps = counts.shape[1]
    if not 1 <= source_steps <= 100:
        raise ValueError("source T must be in [1,100]; even a debug prefix cannot hide a longer source")
    if type(steps) is not int or not 1 <= steps <= source_steps:
        raise ValueError("steps must be an integer within the available source T")
    _integer_array(counts, "counts", 0, (1 << 32) - 1)
    labels = _integer_array(labels, "labels", 0, CLASSES - 1, (BATCH,))
    lengths = _integer_array(lengths, "lengths", 0, source_steps, (BATCH,))
    indices = _integer_array(indices, "dataset_indices", 0, (1 << 32) - 1, (BATCH,))
    if np.unique(indices).size != BATCH:
        raise ValueError("duplicate dataset_indices")
    for sample, length in enumerate(lengths):
        if np.any(counts[sample, length:]):
            raise ValueError("nonzero counts in padding after valid length")
    return counts, labels, lengths, indices


def _write_mem(stream, values, bits):
    flat = np.asarray(values).reshape(-1)
    mask, digits = (1 << bits) - 1, (bits + 3) // 4
    for start in range(0, flat.size, 4096):
        stream.write("".join(f"{int(value) & mask:0{digits}x}\n"
                             for value in flat[start:start + 4096]))
    return int(flat.size)


def build_fixture(out, steps=99, learning=True, counts_npz=DEFAULT_COUNTS):
    """Start from initial weights/zero state for every call; return reference.json."""
    started = time.perf_counter()
    if type(learning) is not bool:
        raise ValueError("learning must be a bool")
    out = Path(out)
    out = (out if out.is_absolute() else ROOT / out).resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        raise ValueError(f"output must remain under {OUTPUT}")
    source = Path(counts_npz)
    source = (source if source.is_absolute() else ROOT / source).resolve()
    counts, labels, lengths, indices = load_counts(source, steps)
    model = load_model()
    w0 = model["W0"].astype(np.int64)
    initial_w2 = model["W2"].astype(np.int64)
    _integer_array(initial_w2, "initial W2", -127, 127, (HIDDEN, CLASSES))
    w2 = initial_w2.copy()
    scale_q16 = round(float(model["W2_scale"]) * 64 * 65536)
    voltage = np.zeros((BATCH, HIDDEN), dtype=np.int64)
    trace = np.zeros_like(voltage)
    cumulative = np.zeros((BATCH, CLASSES), dtype=np.int64)
    residual = np.zeros_like(w2)
    out.mkdir(parents=True, exist_ok=True)
    names = [f"{name}.mem" for name in MEM_LAYOUT] + ["parameters.mem", "labels.mem", "reference.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("choose a fresh fixture directory; existing outputs are not overwritten")
    words = {}
    records = []
    with ExitStack() as stack:
        streams = {name: stack.enter_context((out / f"{name}.mem").open("x", encoding="utf-8", newline="\n"))
                   for name in (*MEM_LAYOUT, "parameters", "labels")}
        words["parameters"] = _write_mem(streams["parameters"], np.concatenate([w0.ravel(), initial_w2.ravel()]), 16)
        words["labels"] = _write_mem(streams["labels"], labels, 8)
        for timestep in range(steps):
            frame = counts[:, timestep, :]
            forward = forward_step(frame, w0, w2, voltage, trace, cumulative, scale_q16)
            voltage, trace, cumulative = (forward[key] for key in ("voltage", "trace", "cumulative"))
            probability, error = quantized_error(forward["q6"], labels)
            gradient = batch_gradient(trace, error)
            delta, scale_saturated = scale_gradient(gradient)
            if learning:
                w2_next, residual_next, code_saturated = apply_residual_update(w2, residual, delta)
            else:
                w2_next, residual_next = w2.copy(), residual.copy()
                code_saturated = np.zeros_like(w2, dtype=bool)
            record = {
                "t": timestep, "input_events": int(frame.sum(dtype=np.uint64)),
                "nonzero_input_features": int(np.count_nonzero(frame)),
                "hidden_spikes": int(forward["spike"].sum()),
                "gradient_nonzero": int(np.count_nonzero(gradient)),
                "gradient_min": int(gradient.min()), "gradient_max": int(gradient.max()),
                "gradient_abs_max": int(np.abs(gradient).max()),
                "gradient_sum": int(gradient.sum()),
                "nonzero_contributions": int(np.sum(np.count_nonzero(trace, axis=1) * np.count_nonzero(error, axis=1))),
                "scaled_delta_nonzero": int(np.count_nonzero(delta)),
                "scaled_delta_abs_max": int(np.abs(delta).max()),
                "scaler_saturations_diagnostic": int(np.count_nonzero(scale_saturated)),
                "update_requests": int(np.count_nonzero(delta)) if learning else 0,
                "code_changes": int(np.count_nonzero(w2_next != w2)),
                "code_lsb_steps": int(np.abs(w2_next - w2).sum()),
                "residual_changes": int(np.count_nonzero(residual_next != residual)),
                "code_saturations": int(np.count_nonzero(code_saturated)),
                "residual_abs_max_after": int(np.abs(residual_next).max()),
                "w2_changed_from_initial": int(np.count_nonzero(w2_next != initial_w2)),
                "probability_sum_min": int(probability.sum(axis=1).min()),
                "probability_sum_max": int(probability.sum(axis=1).max()),
            }
            records.append(record)
            tensors = {"input_counts": frame, **forward, "w2_after": w2_next, "residual_after": residual_next}
            for name, (bits, _, _, _) in MEM_LAYOUT.items():
                words[name] = words.get(name, 0) + _write_mem(streams[name], tensors[name], bits)
            w2, residual = w2_next, residual_next
    files = {}
    for name, (bits, signed, layout, per_step) in MEM_LAYOUT.items():
        if words[name] != steps * per_step:
            raise AssertionError(f"wrong word count: {name}")
        files[name + ".mem"] = {"bits": bits, "signed": signed, "words": words[name],
                                "layout": layout, "bytes": (out / f"{name}.mem").stat().st_size}
    for name, bits, signed, layout in (("parameters", 16, True, "W0.ravel() then initial W2.ravel(); W2 base=369920"),
                                       ("labels", 8, False, "s")):
        files[name + ".mem"] = {"bits": bits, "signed": signed, "words": words[name],
                                "layout": layout, "bytes": (out / f"{name}.mem").stat().st_size}
    totals = {key: sum(record[key] for record in records) for key in (
        "input_events", "hidden_spikes", "gradient_nonzero", "nonzero_contributions",
        "scaled_delta_nonzero", "update_requests", "code_changes", "code_lsb_steps",
        "residual_changes", "code_saturations", "scaler_saturations_diagnostic")}
    summary = {
        "status": "PASS", "scope": "整数数值fixture参考；未执行RTL/EDA，不是模型精度或最终顶层通过证据",
        "learning": learning, "steps": steps, "source_steps": int(counts.shape[1]),
        "explicit_debug_prefix": steps < counts.shape[1], "batch": BATCH, "hidden": HIDDEN, "classes": CLASSES,
        "counts_npz": str(source), "model": str(MODEL), "w2_parameter_base": W2_BASE,
        "samples": [{"sample_index": s, "dataset_index": int(indices[s]), "label": int(labels[s]),
                     "valid_timesteps": int(lengths[s])} for s in range(BATCH)],
        "numeric": {
            "W0": "INT5, count-weighted current <<6, wrap18", "W2": "INT8 [-127,127], hidden spike dot",
            "leak_q15": 32113, "threshold": 1024, "trace_decay_q16": 64225,
            "trace": "UQ6.10, unsigned16", "cumulative": "per-sample signed raw24, wrap24",
            "scale_to_q6_q16": scale_q16, "q6": "model-scale integer RNE, sat signed16",
            "error": "_quantized_row(q6/64) Q0.16 probability minus onehot*65536",
            "gradient": "int64 trace.T @ error, B16 SUM; no division by batch; fits signed48",
            "delta": "RNE(-gradient*11/2^26), saturation [-131071,131071]",
            "residual": "32768 per code; truncation toward zero; clear only on code saturation",
        },
        "ordering": "每t先完成16样本前向，再统一W2-only更新一次；每样本vmem/trace/raw24保留到下一t",
        "initialization": "每次函数调用独立加载相同初始参数，vmem/trace/raw24/residual均为零",
        "off_boundary": "off不更新W2/residual；仍计算梯度和缩放候选供诊断，不代表off执行了学习",
        "padding": "按完整batch T执行空帧与padding；仅显式steps可取调试前缀，源T>100总是拒绝",
        "learner_source_audit": {
            "shared_service": "rtl/learning/unified_tp_shared_output_learning_service_v1.sv",
            "slice": "rtl/learning/unified_fc_tp_w2_learning_slice.sv",
            "reducer": "rtl/learning/unified_tp_address_major_sign_reducer.sv",
            "scaler": "rtl/learning/unified_tp_q015_magnitude_scaler.sv",
            "residual": "rtl/learning/unified_tp_q015_magnitude_residual_update.sv",
            "finding": "生产归约器逐地址累加trace*error并直送scaler；无额外batch平均",
        },
        "counter_boundary": "软件整数参考的逐t地址事件，不是RTL接受次数、周期或硬件流量",
        "files": files, "records": records, "totals": totals,
        "final_changed_w2": int(np.count_nonzero(w2 != initial_w2)),
        "final_nonzero_residual": int(np.count_nonzero(residual)),
        "first_code_change_t": next((r["t"] for r in records if r["code_changes"]), None),
        "learning_windows": steps if learning else 0,
        "wall_seconds": time.perf_counter() - started,
    }
    with (out / "reference.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=99)
    parser.add_argument("--learning", choices=("on", "off", "both"), default="both")
    parser.add_argument("--counts-npz", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for arm in (("on", "off") if args.learning == "both" else (args.learning,)):
        out = args.output_dir / f"t{args.steps}_{arm}"
        result = build_fixture(out, args.steps, arm == "on", args.counts_npz)
        print(json.dumps({"output": str(out), "status": result["status"], "steps": result["steps"],
                          "learning": result["learning"], "totals": result["totals"],
                          "first_code_change_t": result["first_code_change_t"],
                          "final_changed_w2": result["final_changed_w2"],
                          "wall_seconds": result["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
