#!/usr/bin/env python3
"""Audit the SCNN B4xT4 fixed-SGD scale against the RTL integer datapath.

The production bridge represents a nonnegative Q8.8 trace as ``code * 2`` at
the shared 16-bit trace port.  This is the real trace divided by 128 in Q0.16.
Consequently the runtime magnitude multiplier must restore that factor while
also implementing the software B4 normalization and learning rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = (
    ROOT / "results" / "paper_evidence" / "rtl" / "scnn_q015_dataset_parity"
)
DEFAULT_LABELS = (
    ROOT / "imports" / "scnn" / "datasets" / "DVSGesture" / "hardware"
    / "test_10_labels.hex"
)
DEFAULT_OUTPUT = DEFAULT_FIXTURE_ROOT / "b4t4_update_contract.json"
DEFAULT_EMIT_DIR = DEFAULT_FIXTURE_ROOT / "b4t4_update_fixture"

BATCH = 4
TIMESTEPS = 4
FEATURES = 2048
CLASSES = 11
TRACE_LEAK_Q8_8 = 250
SOFTWARE_LR_NUMERATOR = 5
SOFTWARE_LR_SHIFT = 22
WEIGHT_FRACTIONAL_BITS = 15
RESIDUAL_FRACTIONAL_BITS = 15
RTL_SCALE_SHIFT = 26
OLD_MULTIPLIER = 5
CORRECTED_MULTIPLIER = 640
REPEATED_WINDOWS = 16

EXP_LUT = (
    65536, 57835, 51039, 45042, 39750, 35079, 30957, 27319,
    24109, 21276, 18776, 16570, 14623, 12905, 11388, 10050,
    8869, 7827, 6907, 6096, 5380, 4747, 4190, 3697, 3263,
    2879, 2541, 2243, 1979, 1746, 1541, 1360, 1200, 1059,
    935, 825, 728, 642, 567, 500, 442, 390, 344, 303, 268,
    236, 209, 184, 162, 143, 127, 112, 99, 87, 77, 68, 60,
    53, 47, 41, 36, 32, 28, 25, 22,
)


def read_signed_hex(path: Path, expected: int) -> list[int]:
    values: list[int] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        value = int(line, 16)
        if value >= 0x8000:
            value -= 0x10000
        values.append(value)
    if len(values) != expected:
        raise ValueError(f"{path}: read {len(values)} values, expected {expected}")
    return values


def round_nearest_even_ratio(numerator: int, denominator: int) -> int:
    """Round a signed rational to nearest, ties to even."""
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (
        twice_remainder == denominator and (quotient & 1)
    ):
        quotient += 1
    return sign * quotient


def q8_8_to_q6(value: int) -> int:
    magnitude = abs(value)
    rounded = (magnitude + 2) >> 2
    return -rounded if value < 0 else rounded


def softmax_error_q016(centered_logits_q8_8: list[int], label: int) -> list[int]:
    logits_q6 = [q8_8_to_q6(value) for value in centered_logits_q8_8]
    maximum = max(logits_q6)
    exponentials = [
        EXP_LUT[min(64, (maximum - value + 4) >> 3)] for value in logits_q6
    ]
    denominator = sum(exponentials)
    probabilities = [
        ((value << 16) + (denominator >> 1)) // denominator
        for value in exponentials
    ]
    probabilities[label] -= 1 << 16
    return probabilities


def saturate16(value: int) -> int:
    return max(-32768, min(32767, value))


def write_signed_hex(path: Path, values: list[int], bits: int = 16) -> None:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def load_window(
    fixture_root: Path, labels_path: Path
) -> tuple[list[list[list[int]]], list[list[list[int]]], list[int]]:
    labels = [
        int(line, 16)
        for line in labels_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ][:BATCH]
    features: list[list[list[int]]] = []
    currents: list[list[list[int]]] = []
    for sample in range(BATCH):
        sample_dir = fixture_root / f"sample{sample}"
        feature_flat = read_signed_hex(
            sample_dir / "conv8_features_q8_8.hex", 20 * FEATURES
        )
        current_flat = read_signed_hex(
            sample_dir / "classifier_current_q8_8.hex", 20 * CLASSES
        )
        features.append([
            feature_flat[t * FEATURES:(t + 1) * FEATURES]
            for t in range(TIMESTEPS)
        ])
        currents.append([
            current_flat[t * CLASSES:(t + 1) * CLASSES]
            for t in range(TIMESTEPS)
        ])
    return features, currents, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--emit-dir", type=Path, default=DEFAULT_EMIT_DIR)
    args = parser.parse_args()

    features, currents, labels = load_window(args.fixture_root, args.labels)
    gradients = [[0] * FEATURES for _ in range(CLASSES)]
    temporal_logits: list[list[list[int]]] = []
    trace_max = 0
    centered_logit_min = 0

    for sample in range(BATCH):
        trace = [0] * FEATURES
        cumulative_logits = [0] * CLASSES
        sample_temporal_logits: list[list[int]] = []
        for timestep in range(TIMESTEPS):
            trace = [
                ((old * TRACE_LEAK_Q8_8) >> 8) + spike
                for old, spike in zip(trace, features[sample][timestep])
            ]
            trace_max = max(trace_max, max(trace))
            cumulative_logits = [
                old + current
                for old, current in zip(cumulative_logits, currents[sample][timestep])
            ]
            row_max = max(cumulative_logits)
            centered = [saturate16(value - row_max) for value in cumulative_logits]
            sample_temporal_logits.append(centered)
            centered_logit_min = min(centered_logit_min, min(centered))
            errors = softmax_error_q016(centered, labels[sample])
            for class_index, error in enumerate(errors):
                row = gradients[class_index]
                for feature_index, trace_code in enumerate(trace):
                    # Production trace adapter emits Q8.8 code * 2.
                    row[feature_index] += error * (trace_code * 2)
        temporal_logits.append(sample_temporal_logits)

    software_denominator = (
        BATCH
        * (1 << 16)
        * (1 << 8)
        * (1 << SOFTWARE_LR_SHIFT)
    )
    software_numerator_scale = (
        SOFTWARE_LR_NUMERATOR
        * (1 << WEIGHT_FRACTIONAL_BITS)
        * (1 << RESIDUAL_FRACTIONAL_BITS)
    )
    rtl_denominator = 1 << RTL_SCALE_SHIFT

    old_mismatches = 0
    corrected_mismatches = 0
    expected_nonzero = 0
    old_nonzero = 0
    corrected_nonzero = 0
    max_abs_gradient = 0
    max_abs_expected_delta = 0
    expected_residuals: list[int] = []
    examples: list[dict[str, int]] = []

    for class_index, row in enumerate(gradients):
        for feature_index, gradient_q25 in enumerate(row):
            # Recover sum(error_q016 * trace_q8.8_code) from the adapter's x2 form.
            gradient_base = gradient_q25 // 2
            if gradient_q25 != gradient_base * 2:
                raise AssertionError("trace adapter gradient must be even")
            expected = round_nearest_even_ratio(
                -gradient_base * software_numerator_scale,
                software_denominator,
            )
            old_delta = round_nearest_even_ratio(
                -gradient_q25 * OLD_MULTIPLIER, rtl_denominator
            )
            corrected_delta = round_nearest_even_ratio(
                -gradient_q25 * CORRECTED_MULTIPLIER, rtl_denominator
            )
            max_abs_gradient = max(max_abs_gradient, abs(gradient_q25))
            max_abs_expected_delta = max(max_abs_expected_delta, abs(expected))
            expected_nonzero += expected != 0
            old_nonzero += old_delta != 0
            corrected_nonzero += corrected_delta != 0
            old_mismatches += old_delta != expected
            corrected_mismatches += corrected_delta != expected
            expected_residuals.append(expected)
            if old_delta != expected and len(examples) < 16:
                examples.append({
                    "class": class_index,
                    "feature": feature_index,
                    "gradient_q25": gradient_q25,
                    "expected_residual_delta": expected,
                    "old_multiplier_5_delta": old_delta,
                    "corrected_multiplier_640_delta": corrected_delta,
                })

    if corrected_mismatches:
        raise AssertionError(
            f"corrected multiplier mismatch at {corrected_mismatches} addresses"
        )
    if not old_mismatches:
        raise AssertionError("audit stimulus did not distinguish multiplier 5 from 640")

    repeated_codes = [0] * len(expected_residuals)
    repeated_residuals = [0] * len(expected_residuals)
    repeated_code_write_events = 0
    repeated_residual_write_events = 0
    first_code_crossing_window: int | None = None
    for window_index in range(REPEATED_WINDOWS):
        for address, delta in enumerate(expected_residuals):
            if delta == 0:
                continue
            total = repeated_residuals[address] + delta
            code_delta_magnitude = abs(total) >> RESIDUAL_FRACTIONAL_BITS
            code_delta = (
                -code_delta_magnitude if total < 0 else code_delta_magnitude
            )
            residual_magnitude = abs(total) & (
                (1 << RESIDUAL_FRACTIONAL_BITS) - 1
            )
            next_residual = -residual_magnitude if total < 0 else residual_magnitude
            next_code = repeated_codes[address] + code_delta
            if next_code != repeated_codes[address]:
                repeated_code_write_events += 1
                if first_code_crossing_window is None:
                    first_code_crossing_window = window_index + 1
            if next_residual != repeated_residuals[address]:
                repeated_residual_write_events += 1
            repeated_codes[address] = next_code
            repeated_residuals[address] = next_residual

    if first_code_crossing_window is None:
        raise AssertionError("repeated-window fixture did not cross a Q0.15 code LSB")

    args.emit_dir.mkdir(parents=True, exist_ok=True)
    write_signed_hex(
        args.emit_dir / "features_q8_8.hex",
        [
            value
            for sample in features
            for timestep in sample
            for value in timestep
        ],
    )
    write_signed_hex(
        args.emit_dir / "temporal_logits_q8_8.hex",
        [
            value
            for sample in temporal_logits
            for timestep in sample
            for value in timestep
        ],
    )
    write_signed_hex(args.emit_dir / "expected_residual_q15.hex", expected_residuals)
    write_signed_hex(
        args.emit_dir / f"expected_code_after_{REPEATED_WINDOWS}_windows.hex",
        repeated_codes,
    )
    write_signed_hex(
        args.emit_dir / f"expected_residual_after_{REPEATED_WINDOWS}_windows.hex",
        repeated_residuals,
    )
    (args.emit_dir / "labels.hex").write_text(
        "".join(f"{label:02x}\n" for label in labels), encoding="ascii"
    )

    result = {
        "contract": {
            "batch": BATCH,
            "timesteps": TIMESTEPS,
            "labels": labels,
            "trace_recurrence": "floor(old_code*250/256)+spike_q8_8_code",
            "trace_adapter": "q8_8_code*2 (real trace divided by 128 in Q0.16)",
            "software_learning_rate": "5*2^-22",
            "software_batch_normalization": "/4 per timestep",
            "weight_fractional_bits": WEIGHT_FRACTIONAL_BITS,
            "residual_fractional_bits": RESIDUAL_FRACTIONAL_BITS,
            "rtl_scale_shift": RTL_SCALE_SHIFT,
            "derived_rtl_multiplier": CORRECTED_MULTIPLIER,
            "derivation": "640 = 5 * 128; B4 normalization is already represented by the fixed-SGD 5*2^-22 scale relation",
        },
        "fixture": {
            "samples": [str(args.fixture_root / f"sample{i}") for i in range(BATCH)],
            "trace_code_max": trace_max,
            "centered_logit_code_min": centered_logit_min,
            "parameter_count": CLASSES * FEATURES,
            "max_abs_gradient_q25": max_abs_gradient,
            "emitted_directory": str(args.emit_dir),
            "layout": {
                "features_q8_8.hex": "sample-major, timestep-major, feature-major",
                "temporal_logits_q8_8.hex": "sample-major, timestep-major, class-major",
                "expected_residual_q15.hex": "class-major, feature-major",
                f"expected_code_after_{REPEATED_WINDOWS}_windows.hex":
                    "class-major, feature-major",
                f"expected_residual_after_{REPEATED_WINDOWS}_windows.hex":
                    "class-major, feature-major",
                "labels.hex": "sample-major",
            },
        },
        "comparison": {
            "expected_nonzero_deltas": expected_nonzero,
            "old_multiplier_5_nonzero_deltas": old_nonzero,
            "corrected_multiplier_640_nonzero_deltas": corrected_nonzero,
            "old_multiplier_5_mismatches": old_mismatches,
            "corrected_multiplier_640_mismatches": corrected_mismatches,
            "max_abs_expected_residual_delta": max_abs_expected_delta,
            "examples": examples,
        },
        "repeated_window_contract": {
            "windows": REPEATED_WINDOWS,
            "stimulus_policy": "repeat the same frozen real B4xT4 observation window to isolate persistent residual normalization and committed-code visibility",
            "first_code_crossing_window": first_code_crossing_window,
            "total_code_write_events": repeated_code_write_events,
            "total_residual_write_events": repeated_residual_write_events,
            "final_nonzero_codes": sum(value != 0 for value in repeated_codes),
            "final_nonzero_residuals": sum(
                value != 0 for value in repeated_residuals
            ),
            "max_abs_final_code": max(abs(value) for value in repeated_codes),
            "max_abs_final_residual": max(
                abs(value) for value in repeated_residuals
            ),
        },
        "verdict": "PASS: multiplier 640 is bit-exact to the quantized B4xT4 fixed-SGD scale; multiplier 5 is not",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))
    print(result["verdict"])
    print(f"wrote {args.output}")
    print(f"wrote RTL fixture under {args.emit_dir}")


if __name__ == "__main__":
    main()
