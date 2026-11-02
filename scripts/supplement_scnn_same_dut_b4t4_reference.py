#!/usr/bin/env python3
"""Cold, real-input B4T4 successor expectations; never invokes RTL or SSH.

Conv1..8 use the current full-integer reference and hardware banks. Classifier
logits and gradients are recomputed after every window. Repeated windows reset
sample membrane and TP trace, but retain naturally accumulated codes/residuals.
The older float-derived classifier-boundary fixture is deliberately not used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from supplement_scnn_graph_reference import (
    ROOT, FEATURES, CLASSES, PARAMS, WEIGHTS, LABELS, REFERENCE,
    forward, gradients_for_window, apply_update, read_signed_hex, write_signed_hex,
)
from analyze_scnn_b4t4_rtl_update_contract import round_nearest_even_ratio

BATCH, STEPS, FRAMES = 4, 4, 16
INPUT = ROOT / "imports/scnn/datasets/DVSGesture/hardware/test_10x20_inputs_q8_8.hex"
BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_same_dut_b4t4"
# r1 failed before its premature commit assertion, after both full graph frames.
# These are timing observations only, not successful learning evidence.
FRAME_SECONDS = 901.4026291780174 / 2
BUILD_SECONDS = 527.8376518646255


def flatten(rows):
    return [value for row in rows for value in row]


def window_update(features, labels, weights, residuals):
    if len(features) != BATCH or len(labels) != BATCH:
        raise ValueError("four actual samples are required")
    gradients = [0] * PARAMS
    currents, logits, traces = [], [], []
    for frames, label in zip(features, labels):
        if len(frames) != STEPS:
            raise ValueError("four time steps per sample are required")
        current, logit = forward(frames, weights)
        gradient = gradients_for_window(frames, logit, label)
        gradients = [a + b for a, b in zip(gradients, gradient)]
        currents.extend(current)
        logits.extend(logit)
        trace = [0] * FEATURES
        for frame in frames:
            trace = [(old * 250 >> 8) + x for old, x in zip(trace, frame)]
            traces.extend(trace)
    codes, next_residuals = apply_update(weights, residuals, gradients)
    deltas = [round_nearest_even_ratio(-g * 640, 1 << 26) for g in gradients]
    return dict(currents=currents, logits=logits, traces=traces, codes=codes,
                residuals=next_residuals, rmw_mask=[int(d != 0) for d in deltas],
                max_abs_delta=max(map(abs, deltas)),
                counts=[sum(a != b for a, b in zip(codes, weights)),
                        sum(a != b for a, b in zip(next_residuals, residuals)),
                        sum(d != 0 for d in deltas)])


def changed_rows(before, after):
    return sum(a != b for a, b in zip(flatten(before), flatten(after)))


def trajectory(features, labels, windows, output=None):
    if not 1 <= windows <= 16:
        raise ValueError("bounded successor supports 1..16 windows")
    initial = read_signed_hex(WEIGHTS, PARAMS)
    weights, residuals = initial[:], [0] * PARAMS
    baseline = [forward(frames, initial) for frames in features]
    rows = []
    witness = None
    if output is not None:
        write_signed_hex(output / "labels.hex", labels)
        write_signed_hex(output / "expected_features.hex", flatten(flatten(features)))
    for window in range(windows):
        result = window_update(features, labels, weights, residuals)
        if window == 0:
            address = max(range(PARAMS), key=lambda i: abs(result["residuals"][i]))
            delta = result["residuals"][address]
            witness = dict(address=address, class_id=address // FEATURES, feature=address % FEATURES,
                           initial_code=initial[address], first_delta=delta, residual_threshold=32768,
                           identical_cold_window_crossing=(32768+abs(delta)-1)//abs(delta) if delta else None)
        next_outputs = [forward(frames, result["codes"]) for frames in features]
        # Query the first affected sample, but always replay its entire T4 raw
        # sequence. This is a directed visibility witness, not accuracy sampling.
        affected = [sample for sample in range(BATCH)
                    if baseline[sample] != next_outputs[sample]]
        row = dict(window=window + 1, code_writes=result["counts"][0],
                   residual_writes=result["counts"][1], rmw=result["counts"][2],
                   codes_different_from_pretrained=sum(a != b for a, b in zip(initial, result["codes"])),
                   max_abs_delta=result["max_abs_delta"],
                   current_changes_from_pretrained=sum(changed_rows(a[0], b[0]) for a, b in zip(baseline, next_outputs)),
                   logit_changes_from_pretrained=sum(changed_rows(a[1], b[1]) for a, b in zip(baseline, next_outputs)),
                   affected_samples=affected)
        rows.append(row)
        if output is not None:
            directory = output / f"window{window:02d}"
            directory.mkdir()
            for name, values in (
                ("currents", flatten(result["currents"])), ("logits", flatten(result["logits"])),
                ("traces", result["traces"]), ("codes", result["codes"]),
                ("residuals", result["residuals"]), ("rmw_mask", result["rmw_mask"]),
                ("counts", result["counts"]), ("before_codes", weights),
                ("before_residuals", residuals),
            ):
                write_signed_hex(directory / (name + ".hex"), values)
        weights, residuals = result["codes"], result["residuals"]
    query = rows[-1]["affected_samples"][0] if rows[-1]["affected_samples"] else 0
    query_current, query_logits = forward(features[query], weights)
    if output is not None:
        write_signed_hex(output / "query_sample.hex", [query])
        write_signed_hex(output / "query_currents.hex", flatten(query_current))
        write_signed_hex(output / "query_logits.hex", flatten(query_logits))
        write_signed_hex(output / "query_baseline_currents.hex", flatten(baseline[query][0]))
        write_signed_hex(output / "query_baseline_logits.hex", flatten(baseline[query][1]))
        write_signed_hex(output / "query_counts.hex", [
            changed_rows(baseline[query][0], query_current), changed_rows(baseline[query][1], query_logits)])
    return dict(windows=rows, threshold_witness=witness, query_sample=query,
                query_current_changes=changed_rows(baseline[query][0], query_current),
                query_logit_changes=changed_rows(baseline[query][1], query_logits),
                first_code_crossing=next((r["window"] for r in rows if r["code_writes"]), None),
                first_forward_change=next((r["window"] for r in rows if r["affected_samples"]), None))


def integer_features(output=None):
    # Lazy imports keep ordinary arithmetic tests independent of Torch.
    import numpy as np
    import torch
    from evaluate_scnn_full_integer import IntegerNetwork, check_prefix, read_hex

    torch.set_num_threads(1)
    network = IntegerNetwork("cpu")
    prefix = check_prefix(network)
    raw = read_hex(INPUT).reshape(10, 20, 2, 32, 32)
    original = read_hex(REFERENCE / "input_sample0_t0_t1_q8_8.hex").reshape(2, 2, 32, 32)
    if not np.array_equal(raw[0, :2], original):
        raise ValueError("raw fixture no longer matches the independently checked T2 prefix")
    samples = []
    for sample in range(BATCH):
        features = []
        directory = None if output is None else output / f"sample{sample}"
        if directory is not None:
            directory.mkdir()
            for t in range(STEPS):
                write_signed_hex(directory / f"input_sample0_t{t}_q8_8.hex", raw[sample, t].reshape(-1).tolist())

        def capture(t, layer, values):
            if layer == 7:
                features.append(values["pooled"].flatten().tolist())
            if directory is None or layer == 8:
                return
            for field, suffix in (("pre", "pre_q8_8"), ("next", "next_q8_8"),
                                  ("spike", "spike"), ("range", "range"),
                                  ("pooled", "pooled_q8_8")):
                codes = values[field].flatten().tolist()
                write_signed_hex(directory / f"conv{layer+1}_t{t}_expected_{suffix}.hex", codes,
                                 bits=1 if field in ("spike", "range") else 16)
            occupancy = (values["pooled"] != 0).flatten().tolist()
            write_signed_hex(directory / f"conv{layer+1}_t{t}_expected_pooled_occupancy.hex", occupancy, bits=1)

        network.forward(torch.tensor(raw[sample:sample+1, :STEPS]), sample_ids=[sample],
                        verify_ids=[sample], capture=capture)
        samples.append(features)
        print(f"SCNN_B4T4_INTEGER_SAMPLE sample={sample} frames={STEPS}", flush=True)
    return samples, dict(prefix=prefix, independent_cpu_checks=network.checks)


def generate(windows=12, output=None):
    start = time.perf_counter()
    if output is not None:
        output = Path(output).resolve()
        if not output.is_relative_to(BASE):
            raise ValueError("use a new directory under scnn_same_dut_b4t4")
        output.mkdir(parents=True, exist_ok=False)
    features, numeric = integer_features(output)
    labels = read_signed_hex(LABELS, 10)[:BATCH]
    result = trajectory(features, labels, windows, output)
    frames = windows * FRAMES + STEPS
    result.update(status="independent_integer_expectations_only_not_RTL_PASS",
                  samples=list(range(BATCH)), labels=labels, timesteps=STEPS, multiplier=640,
                  protocol="repeated cold B4T4 windows; fresh raw forward each window; unchanged fixed banks except selected classifier; natural residual persistence; TP/sample states reset each window",
                  numeric=numeric, graph_frames=frames, query_frames=STEPS,
                  estimated_simulation_hours=frames * FRAME_SECONDS / 3600,
                  observed_B1T2_build_seconds=BUILD_SECONDS,
                  estimate_basis="same_dut_r1 run_on.json: 901.402629 s through two full graph frames before premature TB commit assertion; build.json: 527.837652 s at -j2; learning/checking overhead additional; failed r1 is timing only, not PASS evidence",
                  final_composition_top=False, whole_dataset=False,
                  reference_wall_seconds=time.perf_counter() - start)
    if output is not None:
        (output / "expectations.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, choices=range(1, 17), default=12)
    parser.add_argument("--output", type=Path, help="omit for read-only numeric assessment")
    args = parser.parse_args()
    print(json.dumps(generate(args.windows, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
