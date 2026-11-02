"""Bounded full-integer SCNN inference with the existing Q0.15 hardware banks.

FP64 convolutions carry exact integers, not hardware floating-point arithmetic.
The fixed 264-sample fixture is not the complete DVS-Gesture test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_scnn_vgg9_full_frame_t2_reference import (
    LAYER_CHANNELS, MODEL_DIR, OUTPUT_DIR as PREFIX_DIR, POOL_MODES,
)
from scripts.supplement_scnn_weight_quantization import (
    CHECKPOINT, PAYLOAD, input_batches,
)


def read_hex(path, bits=16):
    with Path(path).open(encoding="ascii") as stream:
        values = np.fromiter((int(line, 16) for line in stream if line.strip()),
                             dtype=np.int64)
    if np.any((values < 0) | (values >= 1 << bits)):
        raise ValueError(f"invalid {bits}-bit hex: {path}")
    return (values ^ (1 << (bits - 1))) - (1 << (bits - 1))


def wrap(values, bits):
    return ((values + (1 << (bits - 1))) & ((1 << bits) - 1)) - (1 << (bits - 1))


def lif(current, state):
    pre17 = wrap(current + ((state * 135) >> 8), 17)
    pre16 = wrap(pre17, 16)
    spike = pre17 >= 256
    after = wrap(pre17 - 256, 17)
    post = torch.where(spike, after.clamp(-32768, 32767), pre16)
    return pre16, post, spike, (pre17 < -32768) | (pre17 > 32767), after


def pool_q8(spike, mode):
    if mode == "IDENTITY":
        return spike.to(torch.int64) * 256
    b, c, h, w = spike.shape
    if h % 2 or w % 2:
        raise ValueError("pooling requires even spatial dimensions")
    windows = spike.reshape(b, c, h // 2, 2, w // 2, 2)
    if mode == "MAX2":
        return windows.any(dim=5).any(dim=3).to(torch.int64) * 256
    if mode == "AAVG2":
        return windows.sum(dim=(3, 5), dtype=torch.int64) * 64
    raise ValueError(mode)


def equal(actual, expected, context):
    a = np.asarray(actual).reshape(-1)
    e = np.asarray(expected).reshape(-1)
    if a.shape != e.shape:
        raise AssertionError(f"{context}: shape {a.shape} != {e.shape}")
    mismatches = np.flatnonzero(a != e)
    if len(mismatches):
        i = int(mismatches[0])
        raise AssertionError(f"{context}: {len(mismatches)} mismatches; first {i}: {a[i]} != {e[i]}")
    return len(a)


class IntegerNetwork:
    def __init__(self, device):
        self.device = torch.device(device)
        self.codes = [read_hex(MODEL_DIR / f"bank_{i:02d}.hex").reshape(co, ci, 3, 3)
                      for i, (ci, co) in enumerate(LAYER_CHANNELS, 1)]
        self.codes.append(read_hex(MODEL_DIR / "bank_12.hex").reshape(11, 2048))
        self.weights = [torch.tensor(w, dtype=torch.float64, device=device) for w in self.codes]
        self.layers = [dict(normalizer_clips=0, pre16_wraps=0, post_clips=0,
                            spikes=0, values=0, max_abs_raw=0) for _ in range(8)]
        self.classifier = dict(normalizer_clips=0, centered_clips=0, max_abs_raw=0)
        self.checks = dict(cpu_conv_dots=0, cpu_lif_values=0, cpu_classifier_dots=0,
                           cpu_pooled_values=0)

    def cpu_layer_checks(self, features, state, raw, result, pooled, layer, selected):
        weight = self.codes[layer]
        for sample in selected:
            x = features[sample].cpu().numpy()
            old = state[sample].cpu().numpy()
            dot = raw[sample].cpu().numpy()
            padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
            co, h, w = dot.shape
            for c, y, z in ((0, 0, 0), (co - 1, h - 1, w - 1), (co // 2, h // 2, w // 2)):
                expected = np.sum(padded[:, y:y + 3, z:z + 3] * weight[c], dtype=np.int64)
                equal([dot[c, y, z]], [expected], f"CPU convolution layer {layer + 1}")
                self.checks["cpu_conv_dots"] += 1
            current = np.clip(dot // 32768, -32768, 32767)
            pre17 = ((current + old * 135 // 256 + 65536) % 131072) - 65536
            pre = ((pre17 + 32768) % 65536) - 32768
            spikes = pre17 >= 256
            after = ((pre17 - 256 + 65536) % 131072) - 65536
            post = np.where(spikes, np.clip(after, -32768, 32767), pre)
            for actual, expected in zip(result[:4], (pre, post, spikes, (pre17 < -32768) | (pre17 > 32767))):
                self.checks["cpu_lif_values"] += equal(actual[sample].cpu().numpy(), expected, "CPU LIF")
            if POOL_MODES[layer] == "IDENTITY":
                pooled_expected = spikes.astype(np.int64) * 256
            else:
                windows = spikes.reshape(co, h // 2, 2, w // 2, 2)
                pooled_expected = (windows.sum(axis=(2, 4)) * 64 if POOL_MODES[layer] == "AAVG2"
                                   else windows.max(axis=(2, 4)).astype(np.int64) * 256)
            self.checks["cpu_pooled_values"] += equal(pooled[sample].cpu().numpy(), pooled_expected, "CPU pool")

    @torch.no_grad()
    def forward(self, frames, *, sample_ids, verify_ids=(), capture=None):
        if frames.dtype != torch.int64 or frames.ndim != 5 or tuple(frames.shape[2:]) != (2, 32, 32):
            raise ValueError("expected B,T,2,32,32 int64 Q8.8 input")
        if bool(((frames < 0) | (frames > 32767)).any()):
            raise ValueError("input must be nonnegative signed16 Q8.8")
        batch, steps = frames.shape[:2]
        if len(sample_ids) != batch:
            raise ValueError("sample IDs must match batch")
        selected = [i for i, sample_id in enumerate(sample_ids) if sample_id in verify_ids]
        states = [torch.zeros((batch, co, 32 // (2 ** (i // 2)), 32 // (2 ** (i // 2))),
                              dtype=torch.int64, device=self.device)
                  for i, (_, co) in enumerate(LAYER_CHANNELS)]
        cumulative = torch.zeros((batch, 11), dtype=torch.int64, device=self.device)
        for t in range(steps):
            features = frames[:, t]
            for layer in range(8):
                # Even the conservative 4608 * 32768^2 bound is below 2^43.
                raw_float = F.conv2d(features.to(torch.float64), self.weights[layer], padding=1)
                if not torch.equal(raw_float, raw_float.round()):
                    raise AssertionError("nonintegral FP64 integer-carrier convolution")
                raw = raw_float.to(torch.int64)
                shifted = raw >> 15
                current = shifted.clamp(-32768, 32767)
                result = lif(current, states[layer])
                pre, post, spikes, ranges, after = result
                pooled = pool_q8(spikes, POOL_MODES[layer])
                if selected:
                    self.cpu_layer_checks(features, states[layer], raw, result, pooled, layer, selected)
                stats = self.layers[layer]
                stats["normalizer_clips"] += int((current != shifted).sum())
                stats["pre16_wraps"] += int(ranges.sum())
                stats["post_clips"] += int((spikes & ((after < -32768) | (after > 32767))).sum())
                stats["spikes"] += int(spikes.sum())
                stats["values"] += spikes.numel()
                stats["max_abs_raw"] = max(stats["max_abs_raw"], int(raw.abs().max()))
                if capture is not None:
                    capture(t, layer, dict(pre=pre, next=post, spike=spikes, range=ranges, pooled=pooled))
                states[layer] = post
                features = pooled
            raw_float = features.flatten(1).to(torch.float64) @ self.weights[-1].T
            if not torch.equal(raw_float, raw_float.round()):
                raise AssertionError("nonintegral classifier dot")
            raw = raw_float.to(torch.int64)
            for sample in selected:
                expected = self.codes[-1] @ features[sample].flatten().cpu().numpy()
                self.checks["cpu_classifier_dots"] += equal(raw[sample].cpu().numpy(), expected, "CPU classifier")
            current = (raw >> 15).clamp(-32768, 32767)
            cumulative = wrap(cumulative + current, 32)
            centered_raw = wrap(cumulative - cumulative.max(dim=1, keepdim=True).values, 32)
            centered = centered_raw.clamp(-32768, 32767)
            self.classifier["normalizer_clips"] += int(((raw >> 15) != current).sum())
            self.classifier["centered_clips"] += int((centered_raw != centered).sum())
            self.classifier["max_abs_raw"] = max(self.classifier["max_abs_raw"], int(raw.abs().max()))
            if capture is not None:
                capture(t, 8, dict(raw=raw, current=current, centered=centered))
        return cumulative


def check_prefix(network):
    frames = read_hex(PREFIX_DIR / "input_sample0_t0_t1_q8_8.hex").reshape(1, 2, 2, 32, 32)
    checks = 0

    def capture(t, layer, values):
        nonlocal checks
        if layer < 8:
            names = {"pre": ("pre_q8_8", 16), "next": ("next_q8_8", 16),
                     "spike": ("spike", 2), "range": ("range", 2)}
            prefix = f"conv{layer + 1}_t{t}_expected_"
            for field, (name, bits) in names.items():
                checks += equal(values[field].cpu().numpy(), read_hex(PREFIX_DIR / f"{prefix}{name}.hex", bits), prefix + name)
            checks += equal((values["pooled"] != 0).cpu().numpy(),
                            read_hex(PREFIX_DIR / f"{prefix}pooled_occupancy.hex", 2), prefix + "pool")
            if layer == 7:
                checks += equal(values["pooled"].cpu().numpy(),
                                read_hex(PREFIX_DIR / f"{prefix}pooled_q8_8.hex"), prefix + "pool_q8")
        else:
            for field, name, bits in (("raw", "raw_q15", 48), ("current", "current_q8_8", 16),
                                      ("centered", "centered_cumulative_q8_8", 16)):
                path = PREFIX_DIR / f"classifier_t{t}_expected_{name}.hex"
                checks += equal(values[field].cpu().numpy(), read_hex(path, bits), path.name)

    network.forward(torch.tensor(frames, device=network.device), sample_ids=[-1], verify_ids=[-1], capture=capture)
    return dict(status="pass", compared_values=checks,
                fixture=str(PREFIX_DIR.relative_to(ROOT)), timesteps=2,
                cpu_checks=dict(network.checks))


def metric(predictions, labels, subjects):
    p, y, s = map(np.asarray, (predictions, labels, subjects))
    correct = int((p == y).sum())
    groups = {}
    for subject in sorted(set(s.tolist())):
        selected = s == subject
        hits, n = int((p[selected] == y[selected]).sum()), int(selected.sum())
        groups[str(subject)] = dict(correct=hits, total=n, accuracy_percent=100 * hits / n)
    return dict(correct=correct, total=len(p), accuracy_percent=100 * correct / len(p),
                predictions=p.tolist(), by_subject=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--limit", type=int, default=264)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or (output / "result.json").exists():
        raise ValueError("use a new result path inside the project")
    if not 1 <= args.limit <= 264:
        raise ValueError("limit must be in [1,264]")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(38472)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start_time = time.perf_counter()
    network = IntegerNetwork(args.device)
    prefix = check_prefix(network)
    print(f"SCNN_INTEGER_PREFIX_PASS values={prefix['compared_values']}", flush=True)
    result = dict(status="pass", device=args.device, prefix=prefix)
    if not args.prefix_only:
        from scripts import run_scnn_tp_selected_layer_shift_adaptation as api
        from scripts import run_scnn_tp_subject_pair_ablation as subjects_api
        data = PAYLOAD / "data"
        subjects, count = subjects_api.read_subjects(data / "test_264x20_metadata.json")
        if count != 264:
            raise ValueError("unexpected fixture size")
        labels = api.read_labels(data / "test_264_labels.hex", count)[:args.limit]
        subjects = subjects[:args.limit]
        # Explicitly install hardware bank codes in the FP32 control arm.
        model = api.build_model(PAYLOAD / "traces_propagation-main", CHECKPOINT,
                                batch_size=4, device=torch.device(args.device), weight_fractional_bits=15)
        model.requires_grad_(False)
        weight_audit = []
        with torch.no_grad():
            for i, codes in enumerate(network.codes):
                param = model.network[i].fc.weight
                old_codes = (param * 32768).round().to(torch.int64).cpu().numpy()
                weight_audit.append(dict(layer=i + 1, values=codes.size,
                                         differing_codes=int((old_codes != codes).sum()),
                                         max_abs_code_difference=int(np.abs(old_codes - codes).max())))
                param.copy_(torch.tensor(codes / 32768.0, dtype=param.dtype, device=param.device))
        # Prefix validation statistics are kept separate from dataset statistics.
        for stats in network.layers + [network.classifier, network.checks]:
            for key in stats:
                stats[key] = 0
        integer_predictions, float_predictions, integer_logits = [], [], []
        verified_samples = [i for i in (0, 1, 63, 64, 132, 263) if i < args.limit]
        with (data / "test_264x20_inputs_q8_8.hex").open(encoding="ascii") as stream:
            for offset, frames in input_batches(stream, count, 4, (20, 2, 32, 32)):
                if offset >= args.limit:
                    break
                frames = frames[:args.limit - offset]
                x = torch.tensor(frames, device=args.device)
                ids = list(range(offset, offset + len(frames)))
                logits = network.forward((x * 256).to(torch.int64), sample_ids=ids, verify_ids=verified_samples)
                integer_predictions.extend(logits.argmax(dim=1).cpu().tolist())
                integer_logits.extend(logits.cpu().tolist())
                y = torch.tensor(labels[offset:offset + len(frames)], device=args.device)
                m = api.evaluate(model, x, y, list(range(len(frames))), batch_size=4, dx=0, dy=0)
                float_predictions.extend(m["predictions"])
                print(f"SCNN_INTEGER_PROGRESS samples={len(integer_predictions)}/{args.limit}", flush=True)
        integer = metric(integer_predictions, labels, subjects)
        floating = metric(float_predictions, labels, subjects)
        differences = [i for i, (a, b) in enumerate(zip(integer_predictions, float_predictions)) if a != b]
        result.update(
            samples=args.limit, timesteps=20, batch_size=4, adaptation=False,
            inputs=str((data / "test_264x20_inputs_q8_8.hex").relative_to(ROOT)),
            model_banks=str(MODEL_DIR.relative_to(ROOT)), weight_audit=weight_audit,
            arms=dict(full_integer=integer, same_banks_fp32=floating), labels=labels.tolist(),
            integer_cumulative_logits_q8_8=integer_logits,
            full_integer_minus_same_banks_fp32_pp=integer["accuracy_percent"] - floating["accuracy_percent"],
            differing_sample_ids=differences, independent_cpu_sample_ids=verified_samples,
            independent_cpu_checks=network.checks, layers=network.layers, classifier=network.classifier,
            numeric_contract="Q0.15 bank weights; Q8.8 inputs/state; raw48 dot, arithmetic >>15 saturated16; leak*135>>8; signed17 comparison; threshold256 soft reset; MAX2/AAVG2; cumulative wrap32",
            exact_integer_carrier="FP64 multiply-add of integers bounded by 4608*32768^2 < 2^43; all outputs integral; selected independent NumPy int64 dot checks. Not hardware FP64.",
            claim_boundary="Software full-integer inference on the existing balanced 264x20 fixed fixture, including development/report subjects. Not full DVS-Gesture test accuracy, full-test RTL, online adaptation accuracy, or FPGA throughput.",
        )
    result["wall_seconds"] = time.perf_counter() - start_time
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SCNN_INTEGER_COMPLETE {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
