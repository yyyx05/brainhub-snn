#!/usr/bin/env python3
"""Small integer reference for a staged, cold-state real SCNN B1T2 window.

Uses the existing production fixed-SGD integer scale (640), not a new B1
normalization. This underfilled window is a continuity diagnostic, not the
signed-off B4 adaptation protocol. No fabricated residual/code crossings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_scnn_b4t4_rtl_update_contract import (
    round_nearest_even_ratio, saturate16, softmax_error_q016,
    read_signed_hex, write_signed_hex,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES, CLASSES, TIMESTEPS = 2048, 11, 2
PARAMS = FEATURES * CLASSES
GRAPH_PASS = "SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS"
REFERENCE = ROOT / "results/paper_evidence/rtl/scnn_full_graph_same_array_20260927/t2_reference"
WEIGHTS = ROOT / "models/scnn_dvsgesture_vgg9_q0_15/bank_12.hex"
LABELS = ROOT / "imports/scnn/datasets/DVSGesture/hardware/test_10_labels.hex"


def read_capture(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    tables = {"feature": [[None] * FEATURES for _ in range(TIMESTEPS)],
              "logit": [[None] * CLASSES for _ in range(TIMESTEPS)]}
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["kind", "timestep", "index", "value"]:
            raise ValueError("invalid graph capture header")
        for row in reader:
            kind = row["kind"]
            t, i, value = (int(row[k]) for k in ("timestep", "index", "value"))
            if kind not in tables or not 0 <= t < TIMESTEPS:
                raise ValueError(f"invalid capture coordinate: {row}")
            table = tables[kind]
            if not 0 <= i < len(table[t]) or table[t][i] is not None:
                raise ValueError(f"duplicate/out-of-range capture: {row}")
            if not -32768 <= value <= 32767:
                raise ValueError(f"non-int16 capture: {row}")
            if kind == "feature" and value not in (0, 64, 128, 192, 256):
                raise ValueError(f"non-AAVG2 feature: {row}")
            table[t][i] = value
    if any(value is None for table in tables.values() for row in table for value in row):
        raise ValueError("incomplete graph capture")
    return tables["feature"], tables["logit"]


def forward(features: list[list[int]], weights: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    currents, logits = [], []
    cumulative = [0] * CLASSES
    for frame in features:
        current = [saturate16(sum(x * w for x, w in zip(
            frame, weights[c * FEATURES:(c + 1) * FEATURES])) >> 15)
            for c in range(CLASSES)]
        cumulative = [a + b for a, b in zip(cumulative, current)]
        maximum = max(cumulative)
        currents.append(current)
        logits.append([saturate16(x - maximum) for x in cumulative])
    return currents, logits


def gradients_for_window(features, logits, label):
    if not 0 <= label < CLASSES:
        raise ValueError("invalid class label")
    if len(features) != len(logits) or any(len(row) != FEATURES for row in features):
        raise ValueError("invalid window shape")
    trace = [0] * FEATURES
    gradients = [0] * PARAMS
    for frame, row in zip(features, logits):
        trace = [((old * 250) >> 8) + x for old, x in zip(trace, frame)]
        if any(not 0 <= x <= 32767 for x in trace):
            raise ValueError("window exceeds nonnegative trace adapter range")
        errors = softmax_error_q016(row, label)
        for c, error in enumerate(errors):
            for i, value in enumerate(trace):
                gradients[c * FEATURES + i] += error * (value * 2)
    return gradients


def apply_update(weights, residuals, gradients):
    codes, next_residuals = [], []
    for weight, residual, gradient in zip(weights, residuals, gradients):
        delta = round_nearest_even_ratio(-gradient * 640, 1 << 26)
        if not -(1 << 17) <= delta < (1 << 17):
            raise ValueError("window exceeds production signed 18-bit delta")
        total = residual + delta
        sign = -1 if total < 0 else 1
        candidate = weight + sign * (abs(total) >> 15)
        code = saturate16(candidate)
        codes.append(code)
        next_residuals.append(0 if code != candidate else sign * (abs(total) & 32767))
    return codes, next_residuals


def make_fixture(capture: Path, graph_log: Path, output: Path) -> dict:
    if GRAPH_PASS not in graph_log.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("graph simulation has no terminal fullgraph PASS")
    features, captured_logits = read_capture(capture)
    for t, frame in enumerate(features):
        expected = read_signed_hex(REFERENCE / f"conv8_t{t}_expected_pooled_q8_8.hex", FEATURES)
        if frame != expected:
            raise ValueError(f"actual RTL pooled features differ from independent graph reference at t={t}")
    weights = read_signed_hex(WEIGHTS, PARAMS)
    label = int(LABELS.read_text(encoding="ascii").splitlines()[0], 16)
    current, logits = forward(features, weights)
    if logits != captured_logits:
        raise ValueError("actual graph cumulative logits disagree with captured-feature dot products")
    gradients = gradients_for_window(features, logits, label)
    codes, residuals = apply_update(weights, [0] * PARAMS, gradients)
    updated_current, updated_logits = forward(features, codes)
    code_changes = sum(a != b for a, b in zip(codes, weights))
    residual_changes = sum(x != 0 for x in residuals)
    changed_current = sum(a != b for aa, bb in zip(current, updated_current) for a, b in zip(aa, bb))
    changed_logits = sum(a != b for aa, bb in zip(logits, updated_logits) for a, b in zip(aa, bb))
    output.mkdir(parents=True, exist_ok=True)
    flat = lambda rows: [x for row in rows for x in row]
    for filename, values in {
        "initial_code_q15.hex": weights,
        "initial_residual_q15.hex": [0] * PARAMS,
        "expected_code_after_commit_q15.hex": codes,
        "expected_residual_after_commit_q15.hex": residuals,
        "features_q8_8.hex": flat(features),
        "labels.hex": [label],
        "baseline_current_q8_8.hex": flat(current),
        "baseline_temporal_logits_q8_8.hex": flat(logits),
        "updated_current_q8_8.hex": flat(updated_current),
        "updated_temporal_logits_q8_8.hex": flat(updated_logits),
        "expected_counts.hex": [code_changes, residual_changes, changed_current, changed_logits],
    }.items():
        write_signed_hex(output / filename, values)
    summary = {
        "status": "reference_generated_not_learner_RTL_PASS",
        "scope": "staged real sample0 B1T2, cold pretrained classifier, zero residual",
        "batch": 1, "timesteps": TIMESTEPS, "sample": 0, "label": label,
        "multiplier": 640, "warm_windows": 0,
        "normalization": "unchanged B4 multiplier; underfilled diagnostic, not nominal B1-normalized or B4 episode",
        "capture": str(capture), "graph_log": str(graph_log),
        "features_from_RTL": TIMESTEPS * FEATURES,
        "logits_from_RTL": TIMESTEPS * CLASSES,
        "expected_code_changes": code_changes,
        "expected_residual_changes": residual_changes,
        "expected_changed_currents": changed_current,
        "expected_changed_logits": changed_logits,
        "same_DUT_graph_learning": False,
        "full_graph_after_update": False,
    }
    (output / "reference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
