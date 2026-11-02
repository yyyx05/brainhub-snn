#!/usr/bin/env python3
"""Independent expectations for cold B1T2 learning plus one fresh graph frame.

No DUT capture is stimulus. Existing independently computed Conv1..8 reference
frames are scoreboard data; only raw input frames and fixed weight banks drive
the graph. Keep the existing B4T4 multiplier 640 for this underfilled window.
"""
from __future__ import annotations

import json
from pathlib import Path

from supplement_scnn_graph_reference import (
    CLASSES, FEATURES, LABELS, PARAMS, REFERENCE, ROOT, WEIGHTS,
    apply_update, forward, gradients_for_window, read_signed_hex, write_signed_hex,
)
from analyze_scnn_b4t4_rtl_update_contract import round_nearest_even_ratio


def expectations(learning: bool = True) -> dict:
    features = [read_signed_hex(REFERENCE / f"conv8_t{t}_expected_pooled_q8_8.hex", FEATURES)
                for t in range(2)]
    weights = read_signed_hex(WEIGHTS, PARAMS)
    label = int(LABELS.read_text(encoding="ascii").splitlines()[0], 16)
    currents, logits = forward(features, weights)
    for t in range(2):
        for actual, filename in (
            (currents[t], f"classifier_t{t}_expected_current_q8_8.hex"),
            (logits[t], f"classifier_t{t}_expected_centered_cumulative_q8_8.hex"),
        ):
            if actual != read_signed_hex(REFERENCE / filename, CLASSES):
                raise ValueError(f"independent classifier/reference mismatch: {filename}")
    gradients = gradients_for_window(features, logits, label)
    rmw_mask = [int(learning and round_nearest_even_ratio(-g * 640, 1 << 26) != 0) for g in gradients]
    codes, residuals = apply_update(weights, [0] * PARAMS, gradients) if learning else (weights[:], [0] * PARAMS)
    next_current, next_logit = forward([features[0]], codes)
    trace = [0] * FEATURES
    traces = []
    for frame in features:
        trace = [(old * 250 >> 8) + value for old, value in zip(trace, frame)]
        traces.extend(trace)
    return dict(
        features=features + [features[0]], currents=currents + next_current,
        logits=logits + next_logit, codes=codes, residuals=residuals, traces=traces,
        weights=weights, label=label, rmw_mask=rmw_mask,
        counts=[sum(a != b for a, b in zip(weights, codes)), sum(x != 0 for x in residuals), sum(rmw_mask)],
        learning=learning,
    )


def write_fixture(output: Path, learning: bool) -> dict:
    expected = expectations(learning)
    output.mkdir(parents=True, exist_ok=False)
    flat = lambda rows: [value for row in rows for value in row]
    files = {
        "expected_features.hex": flat(expected["features"]),
        "expected_currents.hex": flat(expected["currents"]),
        "expected_logits.hex": flat(expected["logits"]),
        "expected_traces.hex": expected["traces"],
        "expected_codes.hex": expected["codes"],
        "expected_residuals.hex": expected["residuals"],
        "expected_counts.hex": expected["counts"],
        "expected_rmw_mask.hex": expected["rmw_mask"],
        "label.hex": [expected["label"]],
    }
    for filename, values in files.items():
        write_signed_hex(output / filename, values)
    summary = {
        "status": "independent_expectations_only_not_RTL_PASS",
        "learning": learning, "sample": 0, "label": expected["label"],
        "model": "Conv1..8 plus 2048x11 classifier; Q0.15 weights/Q8.8 activity",
        "learning_window": "cold B1T2; unchanged B4T4 scale 640; classifier only",
        "next_forward": "same DUT, fresh sample0/frame0, sample reset only; no parameter reload",
        "graph_frames": 3, "multiplier": 640, "warm_windows": 0,
        "expected_code_changes": expected["counts"][0],
        "expected_residual_changes": expected["counts"][1],
        "expected_rmw_transactions": expected["counts"][2],
        "expected_next_current_changes": sum(a != b for a, b in zip(expected["currents"][0], expected["currents"][2])),
        "expected_next_logit_changes": sum(a != b for a, b in zip(expected["logits"][0], expected["logits"][2])),
        "not_covered": ["full B4T4 episode", "whole dataset", "final composition top",
                        "changed-code visibility when no code crosses", "DDR timing/board behavior"],
    }
    (output / "expectations.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    return summary
