#!/usr/bin/env python3
"""Small H160 numeric fixture, independent of DUT load/decode and RTL runners.

Equations come from the imported P3A H160 RTL, not the FP32 report runner.
The shared-FC comparison is a source-derived hypothesis, not RTL simulation.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "imports/fc/p3a_h160/data/weights/p3a_nmnist_h160_k8_int5_w2_int8_hash8"
MODEL = MODEL_DIR / "weights_h160_int5_w2_int8.npz"
EVENTS = ROOT / "imports/fc/p3a_h160/data/eval_1024_t100/eval1024_events.hex"
EXPORTED = ROOT / "data/weights/p3a_nmnist_h160_h480_active4_input2312"


def signed_wrap(value: int, bits: int) -> int:
    return (int(value) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def lif18(current: int, previous: int = 0, *, leak: int = 32113,
          threshold: int = 1024) -> tuple[int, int, int]:
    """Return pre-state, spike, post-state; round ties towards +infinity."""
    decayed = signed_wrap((signed_wrap(previous, 18) * signed_wrap(leak, 16)
                           + 16384) >> 15, 18)
    pre = signed_wrap(decayed + signed_wrap(current, 18), 18)
    threshold = signed_wrap(threshold, 18)
    spike = int(pre >= threshold)
    return pre, spike, signed_wrap(pre - spike * threshold, 18)


def trace16(previous: int, spike: int, *, fractional_bits: int = 0,
            decay: int = 64225) -> int:
    if not 0 <= fractional_bits < 16 or spike not in (0, 1):
        raise ValueError("invalid trace format or spike")
    if not 0 <= previous <= 65535 or not 0 <= decay <= 65535:
        raise ValueError("trace state and decay must be unsigned 16-bit")
    decayed = ((previous * decay + 32768) >> 16) & 65535
    return min(65535, decayed + (spike << fractional_bits))


def w2_dot(weights: list[list[int]], spikes: list[int]) -> list[int]:
    if len(weights) != len(spikes) or not weights or not weights[0]:
        raise ValueError("weights and spikes must have matching nonzero size")
    classes = len(weights[0])
    if any(len(row) != classes for row in weights):
        raise ValueError("ragged weights")
    if any(s not in (0, 1) for s in spikes):
        raise ValueError("W2 reads spikes, not trace or spike counts")
    if any(not -128 <= w <= 127 for row in weights for w in row):
        raise ValueError("W2 codes must fit signed INT8")
    return [signed_wrap(sum(int(s) * int(row[c])
                            for s, row in zip(spikes, weights)), 24)
            for c in range(classes)]


def accumulate_logits(previous: list[int], dot: list[int], *,
                      sample_start: bool = False) -> list[int]:
    if len(previous) != len(dot):
        raise ValueError("logit dimensions differ")
    return [signed_wrap((0 if sample_start else old) + new, 24)
            for old, new in zip(previous, dot)]


def scale_raw_to_q6(raw: int, scale_q16: int) -> int:
    """Existing numeric-adapter RNE/sat16 rule, not the final replay router."""
    product = signed_wrap(raw, 24) * scale_q16
    quotient, remainder = divmod(abs(product), 65536)
    quotient += int(remainder > 32768 or (remainder == 32768 and quotient % 2))
    rounded = -quotient if product < 0 else quotient
    return min(32767, max(-32768, rounded))


def load_model() -> dict:
    with np.load(MODEL, allow_pickle=False) as packed:
        model = {key: packed[key].copy() for key in packed.files}
    if model["W0"].shape != (2312, 160) or model["W2"].shape != (160, 10):
        raise ValueError("not the H160 model")
    if model["W2"].dtype != np.int8 or model["W0"].dtype != np.int8:
        raise ValueError("expected integer code arrays")
    if int(model["W0_bits"]) != 5 or int(model["W2_bits"]) != 8:
        raise ValueError("unexpected weight precision")
    if (int(model["input_shift"]) != 6 or float(model["W0_scale"]) != 0.0625
            or int(model["lif_threshold_fixed"]) != 1024):
        raise ValueError("unexpected H160 scaling")
    if np.any((model["W0"] < -16) | (model["W0"] > 15)):
        raise ValueError("W0 outside INT5")
    # Independently decode both existing H40 exports, class 0 in low byte.
    for directory in (MODEL_DIR, EXPORTED):
        for tile in range(4):
            rows = (directory / f"w2_cluster_{tile:02d}.hex").read_text(
                encoding="ascii").splitlines()
            if len(rows) != 40:
                raise ValueError("expected 40 rows per W2 tile")
            for row_index, row in enumerate(rows):
                if len(row) != 20:
                    raise ValueError("expected 80-bit W2 row")
                decoded = [signed_wrap(int(row, 16) >> (8 * c), 8)
                           for c in range(10)]
                if decoded != model["W2"][tile * 40 + row_index].tolist():
                    raise ValueError("NPZ and packed W2 disagree")
    return model


def sample0_prefix(steps: int = 16) -> list[list[int]]:
    """Read only a bounded prefix, using the imported TB's 49-slot layout."""
    if not 1 <= steps <= 16:
        raise ValueError("this sidecar only reads 1..16 timesteps of sample 0")
    with EVENTS.open(encoding="ascii") as stream:
        words = [int(line.strip(), 16) for line in islice(stream, steps * 49)]
    if len(words) != steps * 49:
        raise ValueError("short event fixture")
    frames = []
    for start in range(0, len(words), 49):
        features = []
        for word in words[start:start + 49]:
            feature = word >> 8
            if feature == 4095:
                continue
            if feature >= 2312 or (word & 255) != 1:
                raise ValueError("expected binary H160 input event")
            features.append(feature)
        frames.append(features)
    return frames


def build_fixture(steps: int = 16) -> dict:
    model = load_model()
    weights = model["W2"].tolist()
    w0 = model["W0"].tolist()
    minimum = np.unravel_index(model["W2"].argmin(), model["W2"].shape)
    maximum = np.unravel_index(model["W2"].argmax(), model["W2"].shape)
    min_h, min_c = map(int, minimum)
    max_h, max_c = map(int, maximum)
    positive_class1 = [h for h in range(160) if weights[h][1] > 0]
    scale = float(model["W2_scale"])
    scale_q16 = round(scale * 64 * 65536)
    selections = [("zero", [])]
    selections += [(f"onehot_{h}", [h])
                   for h in sorted({0, 39, 40, 79, 80, 119, 120, 159, min_h, max_h})]
    selections += [("even", list(range(0, 160, 2))),
                   ("odd", list(range(1, 160, 2))),
                   ("all", list(range(160))),
                   ("positive_class1", positive_class1)]
    cases = []
    for name, active in selections:
        spikes = [int(h in active) for h in range(160)]
        dot = w2_dot(weights, spikes)
        cases.append({"name": name, "active_hidden": active,
                      "spikes_hex_lsb_hidden0": f"{sum(s << h for h, s in enumerate(spikes)):040x}",
                      "dot_raw24": dot,
                      "q6_rne_using_model_scale": [scale_raw_to_q6(x, scale_q16) for x in dot]})

    sequence = [(True, [min_h]), (False, []), (False, [max_h]),
                (False, list(range(160))), (False, []), (True, []),
                (False, [min_h]), (True, positive_class1),
                (False, positive_class1), (False, []), (True, [])]
    temporal = []
    logits, shared_state, shared_trace = [0] * 10, [0] * 10, [0] * 10
    for start, active in sequence:
        if start:
            shared_state, shared_trace = [0] * 10, [0] * 10
        dot = w2_dot(weights, [int(h in active) for h in range(160)])
        logits = accumulate_logits(logits, dot, sample_start=start)
        current_fc = [lif18(d, v) for d, v in zip(dot, shared_state)]
        pre, spike, shared_state = map(list, zip(*current_fc))
        shared_trace = [trace16(old, s, fractional_bits=10)
                        for old, s in zip(shared_trace, spike)]
        temporal.append({"sample_start": start, "active_hidden": active,
                         "dot_raw24": dot, "cumulative_raw24": logits,
                         "shared_fc_if_hidden_parameters_reused": {
                             "pre18": pre, "spike": spike, "post18": shared_state,
                             "trace_uq6_10": shared_trace,
                             "router_q6": [p >> 4 for p in pre]}})

    prefix = []
    voltage, trace_original, trace_q10 = [0] * 160, [0] * 160, [0] * 160
    unshifted_voltage, logits = [0] * 160, [0] * 10
    for timestep, features in enumerate(sample0_prefix(steps)):
        raw = [sum(w0[fi][h] for fi in features) for h in range(160)]
        current = [signed_wrap(x << 6, 18) for x in raw]
        pre, spikes, voltage = map(list, zip(*[lif18(i, v)
                                            for i, v in zip(current, voltage)]))
        unshifted = [lif18(i, v) for i, v in zip(raw, unshifted_voltage)]
        _, unshifted_spikes, unshifted_voltage = map(list, zip(*unshifted))
        trace_original = [trace16(e, s) for e, s in zip(trace_original, spikes)]
        trace_q10 = [trace16(e, s, fractional_bits=10)
                     for e, s in zip(trace_q10, spikes)]
        dot = w2_dot(weights, spikes)
        logits = accumulate_logits(logits, dot)
        prefix.append({"timestep": timestep, "input_features": features,
                       "w0_current_shift6_wrap18": current, "pre18": pre,
                       "active_hidden": [h for h, s in enumerate(spikes) if s],
                       "post18": voltage, "trace_original_u16": trace_original,
                       "trace_same_spikes_uq6_10": trace_q10,
                       "shared_unshifted_hidden_if_original_parameters":
                           [h for h, s in enumerate(unshifted_spikes) if s],
                       "dot_raw24": dot, "cumulative_raw24": logits})

    return {
        "scope": "Independent source-derived Python fixture, not DUT simulation or accuracy",
        "sources": {"model": MODEL.relative_to(ROOT).as_posix(),
                    "events": EVENTS.relative_to(ROOT).as_posix()},
        "model": {"w0_scale": float(model["W0_scale"]), "w2_scale": scale,
                  "w2_scale_times64_q16_rne": scale_q16,
                  "w2_min": {"hidden": min_h, "class": min_c, "code": weights[min_h][min_c]},
                  "w2_max": {"hidden": max_h, "class": max_c, "code": weights[max_h][max_c]},
                  "w2_outside_int5": int(np.count_nonzero((model["W2"] < -16) | (model["W2"] > 15))),
                  "npz_imported_exported_w2_equal": True},
        "numeric": {"w0_shift": 6, "membrane_bits": 18, "membrane_fractional_bits": 10,
                    "leak_q15": 32113, "threshold": 1024, "trace_decay_q16": 64225,
                    "original_trace_fractional_bits": 0, "current_trace_fractional_bits": 10,
                    "w2_dot_and_temporal_bits": 24, "output_lif_in_reference": False,
                    "q6_scale_status": "model-derived proposal using existing adapter arithmetic; not final-top binding"},
        "w2_codes_source_major": weights,
        "dot_cases": cases,
        "temporal_cases": temporal,
        "sample0_prefix": prefix,
        "summary": {"dot_cases": len(cases), "temporal_steps": len(temporal),
                    "prefix_steps": steps, "prefix_events": sum(len(p["input_features"]) for p in prefix),
                    "prefix_hidden_spikes": sum(len(p["active_hidden"]) for p in prefix),
                    "prefix_unshifted_hidden_spikes": sum(len(p["shared_unshifted_hidden_if_original_parameters"]) for p in prefix),
                    "prefix_final_logits_raw24": logits,
                    "final_top_simulated": False, "accuracy_measured": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--output", type=Path, help="optional JSON output inside this workspace")
    args = parser.parse_args()
    fixture = build_fixture(args.steps)
    if args.output is None:
        print(json.dumps(fixture, indent=2))
    else:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            parser.error("output must remain inside this workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="ascii") as stream:
            json.dump(fixture, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"output": path.relative_to(ROOT).as_posix(), **fixture["summary"]}))


if __name__ == "__main__":
    main()
