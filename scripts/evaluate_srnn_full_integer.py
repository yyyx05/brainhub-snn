#!/usr/bin/env python3
"""SHD official-test inference with production integer equations, no adaptation.

FP64 is only the host carrier for exactly representable integers. Weights,
rounding, saturation and reset remain the integer hardware equations. This is
software accuracy, not full-test RTL or a hardware FP64 datapath.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import h5py
import numpy as np
import torch
import supplement_srnn_backbone_numeric as bb
from export_shd_h5_events import tonic_slice_indices

ROOT = bb.ROOT
BASE = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_full_integer"


def frame_counts(times, units):
    times = (np.asarray(times) * 1_000_000).astype(np.int64)
    units = np.asarray(units, dtype=np.int64)
    if times.shape != units.shape or np.any(np.diff(times) < 0) or np.any((units < 0) | (units >= 700)):
        raise ValueError("invalid ordered SHD events")
    starts, ends = tonic_slice_indices(times, 10000)
    counts = np.stack([np.bincount(units[a:b], minlength=700) for a, b in zip(starts, ends)])
    if counts.max() > 255:
        raise ValueError("uint8 input would clip; do not silently change the dataset")
    return counts


def load_weights(device):
    def tensor(value):
        return torch.as_tensor(value.copy(), dtype=torch.float64, device=device)
    return (
        tensor(bb.read_signed(bb.MODEL / "w0_int8.mem", 8, (700, 480))[:, :450]),
        tensor(bb.read_signed(bb.MODEL / "r_int8.mem", 8, (480, 480))[:450, :450]),
        tensor(bb.read_signed(ROOT / "tb/data/srnn_m2_v28_wh1_initial_q14.mem", 16, (256, 450)).T),
        tensor(bb.read_signed(ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem", 16, (20, 256)).T),
    )


@torch.no_grad()
def forward(counts, weights, device, capture=False):
    if counts.ndim != 3 or counts.shape[2] != 700 or len(counts) >= 65536 or np.any((counts < 0) | (counts > 255)):
        raise ValueError("unsupported input or FP64 integer-exactness bound")
    w0, rc, wh1, wout = weights
    data = torch.as_tensor(counts, dtype=torch.float64, device=device)
    batch = data.shape[1]
    membrane = torch.zeros((batch, 450), dtype=torch.float64, device=device)
    previous = membrane.clone()
    hidden = torch.zeros((batch, 256), dtype=torch.float64, device=device)
    logits = torch.zeros((batch, 20), dtype=torch.float64, device=device)
    rows = {key: [] for key in ("backbone_pre", "backbone_spike", "m2_pre", "m2_spike", "logit")}
    for frame in data:
        current = (frame @ w0 + previous @ rc).clamp(-(2**31), 2**31-1)
        pre = (torch.floor(membrane * 27853 / 32768) + current).clamp(-(2**31), 2**31-1)
        previous = (pre >= 32).to(torch.float64)
        membrane = pre - previous * 32
        hidden_current = (previous @ wh1).clamp(-(2**31), 2**31-1)
        hidden_pre = torch.round(hidden * 13926 / 16384) + hidden_current
        spike = (hidden_pre >= 8192).to(torch.float64)
        hidden = (hidden_pre - spike * 8192).clamp(-(2**31), 2**31-1)
        logits += (spike @ wout).clamp(-(2**31), 2**31-1) * 16384
        if capture:
            for key, value in zip(rows, (pre, previous, hidden_pre.clamp(-(2**31), 2**31-1), spike, logits)):
                rows[key].append(value.cpu().numpy().astype(np.int64))
    return logits.cpu().numpy().astype(np.int64), {key: np.stack(value) for key, value in rows.items()} if capture else {}


def check_prefix(weights, device):
    fixture = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_backbone_numeric/official01_b2t8_r2_20261007"
    with np.load(fixture / "backbone_reference.npz", allow_pickle=False) as data:
        _, values = forward(data["counts"], weights, device, capture=True)
        expected = {"backbone_pre": data["pre"][:, :, :450], "backbone_spike": data["spike"][:, :, :450]}
    with np.load(fixture / "m2_count_preserving/production_off.npz", allow_pickle=False) as m2:
        expected.update(m2_pre=m2["pre"], m2_spike=m2["spike"], logit=m2["logit"])
    for key in expected:
        np.testing.assert_array_equal(values[key], expected[key], err_msg=key)
    return {key: int(value.size) for key, value in expected.items()}


def verify_full_samples(weights, device, output):
    import build_srnn_formal_m2_fixture as m2
    fp = json.loads((ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_weight_only_a1/fp32.json").read_text())
    matrices = [x.cpu().numpy().astype(np.int64) for x in weights]
    rows = []
    started = time.perf_counter()
    with h5py.File(ROOT / "imports/srnn/dataset_cache/SHD/shd_test.h5", "r") as dataset:
        for index in (0, 1, 63, 64, 1000, 2263):
            frame = frame_counts(dataset["spikes/times"][index], dataset["spikes/units"][index])
            steps = fp["padded_timesteps_per_batch"][index // 64]
            inputs = np.zeros((steps, 1, 700), dtype=np.int64)
            inputs[:len(frame), 0] = frame
            _, actual = forward(inputs, weights, device, capture=True)
            backbone = bb.backbone_reference(inputs, matrices[0], matrices[1])
            state = np.zeros((1, 256), dtype=np.int64)
            trace, feature = state.copy(), state.copy()
            logit = np.zeros((1, 20), dtype=np.int64)
            m2_rows = {key: [] for key in ("pre", "spike", "logit")}
            for spike in backbone["spike"]:
                value = m2.forward_step(spike * 16384, matrices[2].T, matrices[3].T, state, trace, feature, logit)
                state, trace, feature, logit = (value[key] for key in ("membrane", "trace", "feature", "logit"))
                for key in m2_rows:
                    m2_rows[key].append(value[key].copy())
            expected = {"backbone_pre": backbone["pre"], "backbone_spike": backbone["spike"],
                        **{"m2_"+key if key != "logit" else key: np.stack(value) for key, value in m2_rows.items()}}
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"sample{index}/{key}")
            rows.append(dict(sample_index=index, padded_steps=steps, checked_values=sum(x.size for x in expected.values())))
    result = dict(status="PASS", rows=rows, total_checked_values=sum(row["checked_values"] for row in rows),
                  device=str(device), seconds=time.perf_counter()-started,
                  scope="Six complete padded official sequences; GPU FP64 integer carrier equals independent NumPy integer backbone/M2 reference at every timestep")
    output.mkdir(parents=True)
    (output / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--verify-samples", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("choose a new result directory within srnn_full_integer")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    weights = load_weights(device)
    if args.verify_samples:
        verify_full_samples(weights, device, output)
        return
    checks = check_prefix(weights, device)
    output.mkdir(parents=True)
    (output / "prefix_check.json").write_text(json.dumps({"status": "PASS", "device": str(device), "checked_values": checks}, indent=2)+"\n")
    if args.prefix_only:
        print("SRNN_INTEGER_PREFIX_PASS", checks)
        return
    fp_path = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_weight_only_a1/fp32.json"
    fp = json.loads(fp_path.read_text(encoding="utf-8"))
    predictions, labels, steps, batch_seconds = [], [], [], []
    start = time.perf_counter()
    h5_path = ROOT / "imports/srnn/dataset_cache/SHD/shd_test.h5"
    max_count = 0
    with h5py.File(h5_path, "r") as dataset:
        if len(dataset["labels"]) != 2264:
            raise ValueError("expected official test2264")
        for first in range(0, 2264, 64):
            batch_start = time.perf_counter()
            indices = range(first, min(first+64, 2264))
            frames = [frame_counts(dataset["spikes/times"][i], dataset["spikes/units"][i]) for i in indices]
            size = max(len(x) for x in frames)
            inputs = np.zeros((size, len(frames), 700), dtype=np.int64)
            for b, frame in enumerate(frames):
                inputs[:len(frame), b] = frame
                max_count = max(max_count, int(frame.max()))
            logits, _ = forward(inputs, weights, device)
            predictions.extend(logits.argmax(axis=1).tolist())
            labels.extend(int(dataset["labels"][i]) for i in indices)
            steps.append(size)
            batch_seconds.append(time.perf_counter()-batch_start)
            (output / "progress.json").write_text(json.dumps({"status": "running", "samples": len(labels), "correct": sum(a==b for a,b in zip(predictions,labels)), "elapsed_seconds": time.perf_counter()-start}, indent=2)+"\n")
            print(f"SRNN_FULL_INTEGER batch={len(steps)}/36 samples={len(labels)} seconds={batch_seconds[-1]:.3f}", flush=True)
    if labels != fp["labels"] or steps != fp["padded_timesteps_per_batch"]:
        raise ValueError("sample order/labels/padded timestep policy differs from FP32; do not compare")
    correct = sum(a == b for a, b in zip(predictions, labels))
    result = dict(status="completed_software_full_integer_not_full_test_RTL", samples=len(labels), correct=correct,
        accuracy_percent=100*correct/len(labels), predictions=predictions, labels=labels, padded_timesteps_per_batch=steps,
        fp32_correct=fp["correct"], accuracy_change_points=100*(correct-fp["correct"])/len(labels),
        predictions_changed=sum(a != b for a,b in zip(predictions,fp["predictions"])),
        input_maximum_count=max_count, batch=64, learning=False, device=str(device),
        elapsed_seconds=time.perf_counter()-start, batch_seconds=batch_seconds, prefix_checks=checks,
        source_h5=str(h5_path.relative_to(ROOT)), fp32_report=str(fp_path.relative_to(ROOT)),
        arithmetic="INT8 W0/R, floor Q15 leak27853 threshold32; M2 Q2.14 Wh1/Wout RNE14 leak13926 threshold8192; signed32 saturation, cumulative Q28 logit",
        host_carrier="FP64 exactly represents all bounded integer multiply/add intermediates; not a hardware precision change",
        limits="Software official-test accuracy only; no online adaptation, full-test RTL, timing, throughput or board claims.",
        peak_process_memory=bb.peak_memory(), torch=torch.__version__, numpy=np.__version__)
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    (output / "progress.json").write_text(json.dumps({"status": "completed", "samples": len(labels)}, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "correct", "accuracy_percent", "accuracy_change_points", "predictions_changed", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
