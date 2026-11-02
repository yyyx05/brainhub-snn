#!/usr/bin/env python3
"""Read-only cost/crossing comparison using actual current-integer features.

Alternative B1 protocols below are proposals, not an approved B4T4 replacement,
not RTL evidence, and not B1-normalized SGD. Every proposal retains multiplier
640, cold codes/zero residuals, natural TP recurrence and fresh raw-input query.
"""
from __future__ import annotations

import argparse
import json

from supplement_scnn_graph_reference import (
    FEATURES, CLASSES, PARAMS, WEIGHTS, LABELS, forward, apply_update,
    read_signed_hex, softmax_error_q016, saturate16,
)
from analyze_scnn_b4t4_rtl_update_contract import round_nearest_even_ratio
from supplement_scnn_same_dut_b4t4_reference import INPUT, FRAME_SECONDS, BUILD_SECONDS, changed_rows


def sequential_sample(features, label, window_lengths, epochs=1, tail_steps=0):
    """Carry sample membrane (in features), cumulative logits and TP over time.

    Each update uses only its new time window and the latest classifier codes.
    This models the B1 trace-preserving public configuration, not repeated
    frozen classifier logits. A separate fresh T4 query is evaluated each time.
    """
    weights = read_signed_hex(WEIGHTS, PARAMS)
    initial = weights[:]
    residuals, trace, cumulative = [0] * PARAMS, [0] * FEATURES, [0] * CLASSES
    baseline = forward(features[:4], initial)
    if sum(window_lengths) > len(features) or any(n <= 0 for n in window_lengths):
        raise ValueError("invalid sequential windows")
    if epochs < 1 or not 0 <= tail_steps <= len(features):
        raise ValueError("invalid epoch or tail bound")
    offset, rows, total_frames, segments = 0, [], 0, []
    for epoch in range(epochs):
        for length in window_lengths:
            segments.append((epoch, offset, length))
            offset += length
        offset = 0
    if tail_steps:
        segments.append((epochs, 0, tail_steps))
    for epoch, offset, length in segments:
        if offset == 0:
            offset, trace, cumulative = 0, [0] * FEATURES, [0] * CLASSES
        gradients = [0] * PARAMS
        for frame in features[offset:offset+length]:
            current = forward([frame], weights)[0][0]
            cumulative = [a + b for a, b in zip(cumulative, current)]
            logits = [saturate16(x - max(cumulative)) for x in cumulative]
            trace = [(old * 250 >> 8) + value for old, value in zip(trace, frame)]
            if max(trace) > 32767:
                raise ValueError("trace exceeds production range")
            errors = softmax_error_q016(logits, label)
            for c, error in enumerate(errors):
                for i, value in enumerate(trace):
                    gradients[c*FEATURES+i] += error * (value*2)
        next_codes, next_residuals = apply_update(weights, residuals, gradients)
        max_delta = max(abs(round_nearest_even_ratio(-g*640, 1 << 26)) for g in gradients)
        query = forward(features[:4], next_codes)
        offset += length
        total_frames += length
        rows.append(dict(through_t=offset-1, committed_windows=len(rows)+1,
                         epoch=epoch + 1,
                         training_frames=total_frames, code_writes=sum(a != b for a, b in zip(weights, next_codes)),
                         different_codes=sum(a != b for a, b in zip(initial, next_codes)),
                         max_abs_delta=max_delta, max_abs_residual=max(map(abs, next_residuals)),
                         query_current_changes=changed_rows(baseline[0], query[0]),
                         query_logit_changes=changed_rows(baseline[1], query[1]),
                         query_T1_current_changes=changed_rows(baseline[0][:1], query[0][:1]),
                         query_T1_logit_changes=changed_rows(baseline[1][:1], query[1][:1]),
                         graph_frames_with_T4_query=total_frames+4,
                         estimated_simulation_hours=(total_frames+4)*FRAME_SECONDS/3600,
                         estimated_T1_query_simulation_hours=(total_frames+1)*FRAME_SECONDS/3600,
                         estimated_T1_query_build_plus_simulation_hours=((total_frames+1)*FRAME_SECONDS+BUILD_SECONDS)/3600))
        weights, residuals = next_codes, next_residuals
    return rows


def compare(samples):
    import torch
    from evaluate_scnn_full_integer import IntegerNetwork, check_prefix, read_hex

    torch.set_num_threads(1)
    network = IntegerNetwork("cpu")
    prefix = check_prefix(network)
    raw = read_hex(INPUT).reshape(10, 20, 2, 32, 32)
    labels = read_signed_hex(LABELS, 10)
    result = dict(status="software_assessment_only_not_RTL_PASS", multiplier=640,
                  numeric_prefix=prefix, samples=[],
                  note="B1 is underfilled at unchanged B4 scaling, requires protocol approval; runtime extrapolates failed r1's two completed graph frames (timing only) and excludes learning/check overhead; build measured separately")
    for sample in samples:
        features = []

        def capture(t, layer, values):
            if layer == 7:
                features.append(values["pooled"].flatten().tolist())

        network.forward(torch.tensor(raw[sample:sample+1]), sample_ids=[sample],
                        verify_ids=[sample], capture=capture)
        sequential = sequential_sample(features, labels[sample], [4]*5)
        singles = [sequential_sample(features, labels[sample], [n])[0] for n in (8, 12, 16, 20)]
        repeated = sequential_sample(features, labels[sample], [4]*5, epochs=4)
        repeat_first = next((r for r in repeated if r["different_codes"] and
                             r["query_current_changes"] and r["query_logit_changes"]), None)
        short_tail = [sequential_sample(features, labels[sample], [4]*5, tail_steps=n)[-1]
                      for n in (1, 2, 3, 4)]
        # Search the minimum natural continuous prefix, without changing its
        # real label, raw frames, scale, or initial state.
        minimal = None
        for n in range(1, 21):
            row = sequential_sample(features, labels[sample], [n])[0]
            if row["different_codes"] and row["query_current_changes"] and row["query_logit_changes"]:
                minimal = row
                break
        result["samples"].append(dict(sample=sample, label=labels[sample],
                                      continuous_B1T4_windows=sequential,
                                      repeated_20_frame_sample_first_visible=repeat_first,
                                      after_T20_fresh_short_tail=short_tail,
                                      single_B1_window=singles, minimal_visible_single_window=minimal))
        print(f"SCNN_CROSSING_OPTION sample={sample} minimal={minimal}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", choices=range(10), default=[0, 1, 2, 3])
    args = parser.parse_args()
    print(json.dumps(compare(args.samples), indent=2), flush=True)


if __name__ == "__main__":
    main()
