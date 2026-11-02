#!/usr/bin/env python3
"""Real B2 M2 input and integer reference; no RTL or external-package writes.

The primary reference describes the corrected production H40 M2 path. The
formal_q14 lane describes the existing numeric consumer's RNE14 membrane leak.
legacy_h40 is retained explicitly for the pre-fix Q15-leak reproduction only.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
from screen_srnn_m2_direct_optimizer import read_signed_hex, round_shift_even, direct_update_tensor

EVIDENCE = ROOT / "results/paper_evidence/supplement_20261004/rtl/srnn_formal_m2"
HANDOFF = ROOT.parent / "SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829"
EPISODE = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_magnitude_sgd_a3/pilot_cpu_20261007/seed128474_direct_response.json"
WH1 = ROOT / "tb/data/srnn_m2_v28_wh1_initial_q14.mem"
WOUT = ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem"
LEAK = 13926
THRESHOLD = 8192
SPIKE = 16384


def sat32(value):
    return np.clip(value, -(1 << 31), (1 << 31) - 1).astype(np.int64)


def read_exp_lut():
    text = (ROOT / "rtl/learning/unified_tp_row_softmax_q12_q24.sv").read_text(encoding="utf-8")
    body = text.split("EXP_LUT [0:128] = '{", 1)[1].split("};", 1)[0]
    values = [int(x) for x in re.findall(r"21'd(\d+)", body)]
    if len(values) != 129:
        raise ValueError("expected 129 entries from production EXP_LUT")
    return np.asarray(values, dtype=np.int64)


def softmax_q24(logit_q28, lut=None):
    lut = read_exp_lut() if lut is None else lut
    metric = round_shift_even(logit_q28, 16)
    if np.any((metric < -(1 << 31)) | (metric >= (1 << 31))):
        raise OverflowError("softmax Q12 metric overflows signed32")
    delta = metric.max(axis=1, keepdims=True) - metric
    index = np.minimum(round_shift_even(delta, 8), 128)
    exponent = lut[index]
    numerator = exponent << 24
    denominator = exponent.sum(axis=1, keepdims=True)
    quotient, remainder = np.divmod(numerator, denominator)
    return quotient + ((2 * remainder > denominator) |
                       ((2 * remainder == denominator) & ((quotient & 1) != 0)))


def wout_gradient(logit, trace, labels, lut=None):
    probability = softmax_q24(logit, lut)
    error = probability.copy()
    error[np.arange(len(labels)), labels] -= 1 << 24
    error <<= 16
    # Production reducer is fixed RNE20, even when cfg_batch_count is two.
    # object arithmetic preserves its 128-bit accumulation without int64 overflow.
    gradient = round_shift_even(error.astype(object).T @ trace.astype(object), 20)
    return probability, error, gradient


def forward_step(input_spikes, wh1, wout, membrane, trace, feature, logit, arithmetic="production"):
    if arithmetic not in ("production", "formal_q14", "legacy_h40"):
        raise ValueError(arithmetic)
    if not np.isin(input_spikes, [0, SPIKE]).all():
        raise ValueError("M2 physical adapter accepts only 0 or 16384 spikes")
    input_spikes, wh1, wout, membrane, trace, feature, logit = (
        np.asarray(value, dtype=np.int64)
        for value in (input_spikes, wh1, wout, membrane, trace, feature, logit))
    current = (input_spikes // SPIKE) @ wh1.T
    if arithmetic != "formal_q14":
        current = sat32(current)
    if arithmetic == "legacy_h40":
        leaked = (membrane * LEAK) >> 15
        pre = sat32(leaked + current)
        spike = pre >= THRESHOLD
        # With this positive threshold, subtract-on-spike fits signed32.
        next_membrane = pre - spike.astype(np.int64) * THRESHOLD
    else:
        leaked = round_shift_even(membrane * LEAK, 14)
        pre_wide = leaked + current
        spike = pre_wide >= THRESHOLD
        pre = sat32(pre_wide)
        next_membrane = sat32(pre_wide - spike.astype(np.int64) * THRESHOLD)
    next_trace = sat32(round_shift_even(trace * LEAK, 14) + spike * SPIKE)
    next_feature = sat32(feature + spike * SPIKE)
    dot = spike.astype(np.int64) @ wout.T
    if arithmetic != "formal_q14":
        dot = sat32(dot)
    # Adapter accumulates current spike dot, not cumulative feature @ new Wout.
    total = logit.astype(object) + (dot.astype(object) << 14)
    if np.any((total < -(1 << 63)) | (total >= (1 << 63))):
        raise OverflowError("cumulative logit overflows signed64")
    return dict(current=current, pre=pre, membrane=next_membrane,
                spike=spike.astype(np.int64), trace=next_trace, feature=next_feature,
                dot=dot, logit=np.asarray(total, dtype=np.int64))


def reference(inputs, labels, wh1, initial_wout, *, learning=False, arithmetic="production"):
    timesteps, batch, backbone = inputs.shape
    hidden, checked_backbone = wh1.shape
    classes, checked_hidden = initial_wout.shape
    if backbone != checked_backbone or hidden != checked_hidden or len(labels) != batch:
        raise ValueError("incompatible reference tensor shapes")
    membrane = np.zeros((batch, hidden), dtype=np.int64)
    trace, feature = membrane.copy(), membrane.copy()
    logit = np.zeros((batch, classes), dtype=np.int64)
    code = initial_wout.copy()
    residual = np.zeros_like(code)
    records = []
    traffic = []
    lut = read_exp_lut()
    for timestep in range(timesteps):
        values = forward_step(inputs[timestep], wh1, code, membrane, trace, feature, logit, arithmetic)
        membrane, trace, feature, logit = (values[k] for k in ("membrane", "trace", "feature", "logit"))
        probability, error, gradient = wout_gradient(logit, trace, labels, lut)
        values.update(probability=probability, error=error, gradient=gradient, wout_before=code.copy())
        if learning:
            code, residual, stats = direct_update_tensor(code, residual, gradient, 31, 65)
        else:
            stats = {key: 0 for key in ("rmw", "code_writes", "residual_writes", "saturations", "zero_skips")}
        values.update(wout_after=code.copy(), residual=residual.copy())
        records.append(values)
        traffic.append(dict(timestep=timestep, **stats))
    arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
    return arrays, traffic


def source_inputs(steps, handoff=HANDOFF, dataset_root=Path("D:/datasets"), episode=EPISODE):
    import torch
    import tonic
    import compare_srnn_shd_shifted_optimizers as original

    saved = json.loads(episode.read_text(encoding="utf-8"))
    if saved["seed"] != 128474 or saved["phase"] != "pilot":
        raise ValueError("expected saved development seed128474 support order")
    indices = saved["support_indices"][:2]
    sys.path.insert(0, str(handoff / "tp_exp/python"))
    trainer = importlib.import_module("multilayer_tp_train")
    base = importlib.import_module("m1_m2_train")
    config = json.loads((handoff / "tp_exp/configs/B64_FORMAL_SEED_A.json").read_text(encoding="utf-8"))
    config["threads"] = 1
    config["dataset_root_relative"] = str(dataset_root.resolve())
    with patch.object(tonic.datasets.SHD, "download", side_effect=RuntimeError("external data is read-only")):
        train, _, collate = trainer.datasets(config)
    dataset = original.FrequencyShiftDataset(train, 16)
    data, labels = next(iter(base.indexed_loader(dataset, collate, [indices])))
    if steps > data.shape[1]:
        raise ValueError(f"requested {steps} steps but B2 padded sequence has {data.shape[1]}")
    model, optimizers = original.fresh_model(handoff, config, trainer, base)
    del optimizers
    model.eval()
    model.reset_potential()
    spike_rows = []
    with torch.no_grad():
        for step in range(steps):
            spikes, _, _ = model.network[0](data[:, step, :].float())
            spike_rows.append((spikes * SPIKE).to(torch.int64).numpy())
    return np.stack(spike_rows), labels.numpy(), dict(
        input_boundary="M2-only: frozen formal FP32 backbone produces H450 spikes; NOT raw700 RTL chain",
        dataset="SHD training-validation support", episode_seed=128474,
        support_indices=indices, labels=labels.tolist(), shift_channels=16,
        timestep_indices=list(range(steps)), complete_padded_timesteps=int(data.shape[1]),
        preprocessing="formal ToFrame(time_window=10000), zero-fill +16, PadTensors B2",
        handoff=str(handoff), dataset_root=str(dataset_root), episode=str(episode),
        torch=torch.__version__, numpy=np.__version__, device="cpu", threads=1)


def write_hex(path, values, bits):
    mask = (1 << bits) - 1
    with path.open("x", encoding="ascii") as stream:
        for value in np.asarray(values).reshape(-1):
            stream.write(f"{int(value) & mask:0{bits // 4}x}\n")


def build_fixture(output, steps=2, handoff=HANDOFF, dataset_root=Path("D:/datasets"), episode=EPISODE, production_arithmetic="production"):
    if production_arithmetic not in ("production", "legacy_h40"):
        raise ValueError(production_arithmetic)
    output = output.resolve()
    if not output.is_relative_to(EVIDENCE.resolve()) and not output.is_relative_to(
            ROOT / "work/supplement_20261004/srnn_formal_m2"):
        raise ValueError("fixture output must stay in this task's evidence/work directories")
    if not 2 <= steps <= 128:
        raise ValueError("use an unskipped prefix of 2..128 timesteps")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    wh1 = read_signed_hex(WH1, 16, 256 * 450).reshape(256, 450)
    wout = read_signed_hex(WOUT, 16, 20 * 256).reshape(20, 256)
    inputs, labels, metadata = source_inputs(steps, handoff, dataset_root, episode)
    write_hex(output / "input_q14.mem", inputs, 32)
    write_hex(output / "labels.mem", labels, 8)
    references = {}
    for arithmetic in ("production", "formal_q14"):
        for learning in (False, True):
            tag = arithmetic + ("_on" if learning else "_off")
            arrays, counters = reference(inputs, labels, wh1, wout, learning=learning,
                arithmetic=production_arithmetic if arithmetic == "production" else arithmetic)
            np.savez(output / f"{tag}.npz", **arrays)
            references[tag] = (arrays, counters)
            if arithmetic == "production":
                prefix = "expected_on" if learning else "expected"
                for key in ("pre", "membrane", "spike", "trace", "feature", "dot", "logit"):
                    write_hex(output / f"{prefix}_{key}.mem", arrays[key], 64 if key == "logit" else 32)
                for key in ("wout_before", "wout_after", "residual", "gradient", "error"):
                    write_hex(output / f"{prefix}_{key}.mem", arrays[key], 64 if key in ("gradient", "error") else 16)
    production, formal = references["production_off"][0], references["formal_q14_off"][0]
    write_hex(output / "formal_pre.mem", formal["pre"], 32)
    differences = {}
    for key in ("pre", "membrane", "spike", "trace", "feature", "dot", "logit"):
        locations = np.argwhere(production[key] != formal[key])
        first = tuple(locations[0]) if len(locations) else None
        differences[key] = dict(count=len(locations),
            first_index=list(map(int, first)) if first is not None else None,
            production=int(production[key][first]) if first is not None else None,
            formal_q14=int(formal[key][first]) if first is not None else None)
    summary = dict(status="reference_complete_not_rtl", batch=2, steps=steps, backbone=450, hidden=256, classes=20,
        production_arithmetic=production_arithmetic,
        extension_order=[2, 8, 16, 32, 61],
        extension_rule="Same two saved support samples, contiguous prefixes only. Extend only if no real code change. No sample or gradient selection.",
        source=metadata, wh1_file=str(WH1), wout_file=str(WOUT),
        input_nonzero_per_step=[int(np.count_nonzero(x)) for x in inputs],
        rules=dict(hidden_current="sum binary spikes * signed16 Q14 weights; production sat32",
                   production_membrane=("sat32((old_membrane*13926) arithmetic>>15 + current); soft subtract8192" if production_arithmetic == "legacy_h40" else
                       "wide_pre=RNE(old_membrane*13926/2**14)+sat32(current); spike compares wide_pre>=8192; report sat32(pre); state=sat32(wide_pre-spike*8192)"),
                   adapter_next_observation="adapter reconstructs sat32(reported_pre-spike*8192); equals resident state for this non-saturating real fixture, not all overflow inputs",
                   formal_membrane="RNE(old_membrane*13926/2**14) + current; existing numeric consumer",
                   trace="sat32(RNE(old_trace*13926/2**14) + spike*16384)",
                   output_feature="sat32(cumulative spike*16384), not decaying trace",
                   logits="cumulative signed64 Q28 += current_spike @ current_Wout * 16384",
                   softmax="RNE Q28->Q12; production EXP_LUT Q20, 1/16 index RNE/clamp128; division RNE Q24",
                   gradient="(prob_Q24 - onehot_Q24)<<16; RNE(sum_B(error_Q40 * trace_Q14)/2**20), fixed /64 even B2",
                   update="original direct_update_tensor, increment31 threshold65 sat16, clear residual on saturation"),
        current_vs_formal_q14=differences,
        reference_counters={tag: counters for tag, (_, counters) in references.items()},
        first_code_change_timestep={tag: next((r["timestep"] for r in counters if r["code_writes"]), None)
                                   for tag, (_, counters) in references.items()},
        elapsed_seconds=time.perf_counter() - started,
        claim_boundary="Integer Python reference plus real FP32-backbone input only; no RTL simulation or final-top claim")
    (output / "fixture.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--handoff", type=Path, default=HANDOFF)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/datasets"))
    parser.add_argument("--legacy-h40", action="store_true", help="historical pre-fix Q15-leak reference only; do not use against corrected RTL")
    args = parser.parse_args(argv)
    result = build_fixture(args.output_dir, args.steps, args.handoff, args.dataset_root,
                           production_arithmetic="legacy_h40" if args.legacy_h40 else "production")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
