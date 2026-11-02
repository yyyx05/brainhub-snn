#!/usr/bin/env python3
"""Build an exact two-timestep fixed-point VGG-9 SCNN reference.

The reference keeps one Q8.8 membrane state per logical convolution neuron,
applies the production leak/soft-reset arithmetic, and emits timestep-major
fixtures for the shared-array state-virtualization regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT
    / "imports/scnn/datasets/DVSGesture/hardware/"
    "test_10x20_inputs_q8_8.hex"
)
MODEL_DIR = ROOT / "models/scnn_dvsgesture_vgg9_q0_15"
OUTPUT_DIR = (
    ROOT
    / "results/paper_evidence/rtl/"
    "scnn_full_graph_same_array_20260927/t2_reference"
)

SAMPLES = 10
DATASET_TIMESTEPS = 20
TEST_TIMESTEPS = 2
INPUT_SHAPE = (2, 32, 32)
WEIGHT_FRACTIONAL_BITS = 15
UNIT_SPIKE_Q8_8 = 256
THRESHOLD_Q8_8 = 256
LEAK_Q8_8 = 135
CLASSES = 11

LAYER_CHANNELS = [
    (2, 64),
    (64, 128),
    (128, 256),
    (256, 256),
    (256, 512),
    (512, 512),
    (512, 512),
    (512, 512),
]
POOL_MODES = [
    "IDENTITY",
    "MAX2",
    "IDENTITY",
    "MAX2",
    "IDENTITY",
    "MAX2",
    "IDENTITY",
    "AAVG2",
]


def read_i16_hex(path: Path) -> np.ndarray:
    words = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()),
        dtype=np.uint16,
    )
    return words.view(np.int16).astype(np.int64)


def write_i16_hex(path: Path, values: np.ndarray) -> None:
    words = values.astype(np.int16, copy=False).view(np.uint16).reshape(-1)
    path.write_text(
        "".join(f"{int(word):04x}\n" for word in words), encoding="ascii"
    )


def write_i48_hex(path: Path, values: np.ndarray) -> None:
    mask = (1 << 48) - 1
    path.write_text(
        "".join(f"{int(value) & mask:012x}\n" for value in values.reshape(-1)),
        encoding="ascii",
    )


def write_bits(path: Path, values: np.ndarray) -> None:
    path.write_text(
        "".join(f"{int(value)}\n" for value in values.reshape(-1)),
        encoding="ascii",
    )


def signed_low_bits(values: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    encoded = values.astype(np.int64, copy=False) & mask
    return ((encoded ^ sign) - sign).astype(np.int64, copy=False)


def explicit_integer_dot(
    features: np.ndarray,
    weights: np.ndarray,
    out_channel: int,
    out_row: int,
    out_col: int,
    feature_factor: int,
) -> int:
    total = 0
    for in_channel in range(features.shape[0]):
        for kernel_row in range(3):
            input_row = out_row + kernel_row - 1
            if input_row < 0 or input_row >= features.shape[1]:
                continue
            for kernel_col in range(3):
                input_col = out_col + kernel_col - 1
                if input_col < 0 or input_col >= features.shape[2]:
                    continue
                total += (
                    int(features[in_channel, input_row, input_col])
                    * feature_factor
                    * int(weights[out_channel, in_channel, kernel_row, kernel_col])
                )
    return total


def checked_conv(
    features: np.ndarray,
    weights: np.ndarray,
    feature_factor: int,
    device: torch.device,
    layer_index: int,
    timestep: int,
) -> np.ndarray:
    input_tensor = torch.from_numpy(features.astype(np.float64, copy=False))
    input_tensor = input_tensor.unsqueeze(0).to(device)
    if feature_factor != 1:
        input_tensor = input_tensor * float(feature_factor)
    weight_tensor = torch.from_numpy(weights.astype(np.float64, copy=False)).to(device)
    raw_float = functional.conv2d(input_tensor, weight_tensor, padding=1)
    rounded = torch.round(raw_float)
    if not torch.equal(raw_float, rounded):
        maximum_error = torch.max(torch.abs(raw_float - rounded)).item()
        raise RuntimeError(
            f"timestep {timestep} layer {layer_index}: non-integral float64 "
            f"convolution, max error {maximum_error}"
        )
    raw = rounded.to(torch.int64).squeeze(0).cpu().numpy()

    rng = np.random.default_rng(
        0x54320000 + timestep * 0x100 + layer_index
    )
    points = {
        (0, 0, 0),
        (weights.shape[0] - 1, raw.shape[1] - 1, raw.shape[2] - 1),
        (weights.shape[0] // 2, raw.shape[1] // 2, raw.shape[2] // 2),
    }
    while len(points) < 19:
        points.add(
            (
                int(rng.integers(weights.shape[0])),
                int(rng.integers(raw.shape[1])),
                int(rng.integers(raw.shape[2])),
            )
        )
    for out_channel, out_row, out_col in sorted(points):
        expected = explicit_integer_dot(
            features,
            weights,
            out_channel,
            out_row,
            out_col,
            feature_factor,
        )
        actual = int(raw[out_channel, out_row, out_col])
        if actual != expected:
            raise RuntimeError(
                f"timestep {timestep} layer {layer_index}: integer spot-check "
                f"mismatch at ({out_channel}, {out_row}, {out_col}): "
                f"got {actual}, expected {expected}"
            )
    return raw


def max_pool2(spikes: np.ndarray) -> np.ndarray:
    channels, height, width = spikes.shape
    if height % 2 or width % 2:
        raise ValueError("MAX2 requires even spatial dimensions")
    return spikes.reshape(channels, height // 2, 2, width // 2, 2).max(
        axis=(2, 4)
    )


def average_pool2_q8_8(spikes: np.ndarray) -> np.ndarray:
    channels, height, width = spikes.shape
    if height % 2 or width % 2:
        raise ValueError("AAVG2 requires even spatial dimensions")
    counts = spikes.reshape(
        channels, height // 2, 2, width // 2, 2
    ).sum(axis=(2, 4), dtype=np.int64)
    return (counts * 64).astype(np.int16)


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = read_i16_hex(INPUT_PATH).reshape(
        SAMPLES, DATASET_TIMESTEPS, *INPUT_SHAPE
    )
    test_frames = frames[0, :TEST_TIMESTEPS]
    if np.any(test_frames < 0) or np.any(test_frames >= (1 << 15)):
        raise ValueError("current value-input contract requires nonnegative Q8.8")

    weights_by_layer: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for layer_index, (in_channels, out_channels) in enumerate(
        LAYER_CHANNELS, start=1
    ):
        weight_path = MODEL_DIR / f"bank_{layer_index:02d}.hex"
        weights = read_i16_hex(weight_path).reshape(
            out_channels, in_channels, 3, 3
        )
        weights_by_layer.append(weights)
        spatial = 32 // (2 ** ((layer_index - 1) // 2))
        states.append(np.zeros((out_channels, spatial, spatial), dtype=np.int64))

    classifier_weights = read_i16_hex(MODEL_DIR / "bank_12.hex").reshape(
        CLASSES, -1
    )
    layer_records: list[list[dict[str, object]]] = [[] for _ in range(8)]
    pre_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    next_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    spike_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    range_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    pooled_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    pooled_q8_8_by_layer: list[list[np.ndarray]] = [[] for _ in range(8)]
    classifier_raw_frames: list[np.ndarray] = []
    classifier_current_frames: list[np.ndarray] = []
    cumulative = np.zeros(CLASSES, dtype=np.int64)
    centered_frames: list[np.ndarray] = []

    for timestep in range(TEST_TIMESTEPS):
        features = test_frames[timestep]
        for layer_offset, ((in_channels, out_channels), pool_mode) in enumerate(
            zip(LAYER_CHANNELS, POOL_MODES)
        ):
            layer_index = layer_offset + 1
            if features.shape[0] != in_channels:
                raise ValueError(
                    f"timestep {timestep} layer {layer_index}: input channel "
                    f"mismatch {features.shape[0]} != {in_channels}"
                )
            feature_factor = 1 if layer_index == 1 else UNIT_SPIKE_Q8_8
            raw = checked_conv(
                features,
                weights_by_layer[layer_offset],
                feature_factor,
                device,
                layer_index,
                timestep,
            )
            shifted = raw >> WEIGHT_FRACTIONAL_BITS
            if shifted.min() < -32768 or shifted.max() > 32767:
                raise RuntimeError(
                    f"timestep {timestep} layer {layer_index}: convolution "
                    "normalizer exceeds signed 16-bit contract"
                )
            current = shifted.astype(np.int64, copy=False)
            leaked = (states[layer_offset] * LEAK_Q8_8) >> 8
            pre17 = signed_low_bits(current + leaked, 17)
            range_mask = (pre17 > 32767) | (pre17 < -32768)
            reported_pre = signed_low_bits(pre17, 16).astype(np.int16)
            spike = pre17 >= THRESHOLD_Q8_8
            post_threshold = np.clip(
                pre17 - THRESHOLD_Q8_8, -32768, 32767
            )
            next_state = np.where(
                spike, post_threshold, reported_pre.astype(np.int64)
            ).astype(np.int16)
            states[layer_offset] = next_state.astype(np.int64)

            if pool_mode == "MAX2":
                pooled = max_pool2(spike)
                pooled_q8_8 = pooled.astype(np.int16) * UNIT_SPIKE_Q8_8
                next_features = pooled
            elif pool_mode == "AAVG2":
                pooled_q8_8 = average_pool2_q8_8(spike)
                pooled = pooled_q8_8 != 0
                next_features = pooled
            else:
                pooled = spike
                pooled_q8_8 = spike.astype(np.int16) * UNIT_SPIKE_Q8_8
                next_features = spike

            pre_by_layer[layer_offset].append(reported_pre)
            next_by_layer[layer_offset].append(next_state)
            spike_by_layer[layer_offset].append(spike)
            range_by_layer[layer_offset].append(range_mask)
            pooled_by_layer[layer_offset].append(pooled)
            pooled_q8_8_by_layer[layer_offset].append(pooled_q8_8)
            layer_records[layer_offset].append(
                {
                    "timestep": timestep,
                    "input_shape_chw": list(features.shape),
                    "conv_shape_chw": list(reported_pre.shape),
                    "pool_mode": pool_mode,
                    "pooled_shape_chw": list(pooled.shape),
                    "pre_min_q8_8": int(reported_pre.min()),
                    "pre_max_q8_8": int(reported_pre.max()),
                    "state_min_q8_8": int(next_state.min()),
                    "state_max_q8_8": int(next_state.max()),
                    "spike_count": int(np.count_nonzero(spike)),
                    "pooled_occupancy_count": int(np.count_nonzero(pooled)),
                    "range_count": int(np.count_nonzero(range_mask)),
                    "integer_spot_checks": 19,
                }
            )
            features = next_features

        classifier_features = pooled_q8_8_by_layer[-1][-1].reshape(-1).astype(
            np.int64
        )
        raw_classifier = classifier_weights @ classifier_features
        current = np.clip(
            raw_classifier >> WEIGHT_FRACTIONAL_BITS, -32768, 32767
        ).astype(np.int16)
        cumulative += current.astype(np.int64)
        centered = cumulative - int(cumulative.max())
        if centered.min() < -32768 or centered.max() > 32767:
            raise RuntimeError(
                f"timestep {timestep}: centered cumulative logit exceeds Q8.8"
            )
        classifier_raw_frames.append(raw_classifier)
        classifier_current_frames.append(current)
        centered_frames.append(centered.astype(np.int16))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_i16_hex(OUTPUT_DIR / "input_sample0_t0_t1_q8_8.hex", test_frames)
    for timestep in range(TEST_TIMESTEPS):
        write_i16_hex(
            OUTPUT_DIR / f"input_sample0_t{timestep}_q8_8.hex",
            test_frames[timestep],
        )
    for layer_offset in range(8):
        prefix = f"conv{layer_offset + 1}"
        write_i16_hex(
            OUTPUT_DIR / f"{prefix}_expected_pre_q8_8.hex",
            np.stack(pre_by_layer[layer_offset]),
        )
        write_i16_hex(
            OUTPUT_DIR / f"{prefix}_expected_next_q8_8.hex",
            np.stack(next_by_layer[layer_offset]),
        )
        write_bits(
            OUTPUT_DIR / f"{prefix}_expected_spike.hex",
            np.stack(spike_by_layer[layer_offset]),
        )
        write_bits(
            OUTPUT_DIR / f"{prefix}_expected_range.hex",
            np.stack(range_by_layer[layer_offset]),
        )
        write_bits(
            OUTPUT_DIR / f"{prefix}_expected_pooled_occupancy.hex",
            np.stack(pooled_by_layer[layer_offset]),
        )
        if POOL_MODES[layer_offset] == "AAVG2":
            write_i16_hex(
                OUTPUT_DIR / f"{prefix}_expected_pooled_q8_8.hex",
                np.stack(pooled_q8_8_by_layer[layer_offset]),
            )
        for timestep in range(TEST_TIMESTEPS):
            write_i16_hex(
                OUTPUT_DIR / f"{prefix}_t{timestep}_expected_pre_q8_8.hex",
                pre_by_layer[layer_offset][timestep],
            )
            write_i16_hex(
                OUTPUT_DIR / f"{prefix}_t{timestep}_expected_next_q8_8.hex",
                next_by_layer[layer_offset][timestep],
            )
            write_bits(
                OUTPUT_DIR / f"{prefix}_t{timestep}_expected_spike.hex",
                spike_by_layer[layer_offset][timestep],
            )
            write_bits(
                OUTPUT_DIR / f"{prefix}_t{timestep}_expected_range.hex",
                range_by_layer[layer_offset][timestep],
            )
            write_bits(
                OUTPUT_DIR
                / f"{prefix}_t{timestep}_expected_pooled_occupancy.hex",
                pooled_by_layer[layer_offset][timestep],
            )
            if POOL_MODES[layer_offset] == "AAVG2":
                write_i16_hex(
                    OUTPUT_DIR
                    / f"{prefix}_t{timestep}_expected_pooled_q8_8.hex",
                    pooled_q8_8_by_layer[layer_offset][timestep],
                )

    write_i48_hex(
        OUTPUT_DIR / "classifier_expected_raw_q15.hex",
        np.stack(classifier_raw_frames),
    )
    write_i16_hex(
        OUTPUT_DIR / "classifier_expected_current_q8_8.hex",
        np.stack(classifier_current_frames),
    )
    write_i16_hex(
        OUTPUT_DIR / "classifier_expected_centered_cumulative_q8_8.hex",
        np.stack(centered_frames),
    )
    for timestep in range(TEST_TIMESTEPS):
        write_i48_hex(
            OUTPUT_DIR / f"classifier_t{timestep}_expected_raw_q15.hex",
            classifier_raw_frames[timestep],
        )
        write_i16_hex(
            OUTPUT_DIR / f"classifier_t{timestep}_expected_current_q8_8.hex",
            classifier_current_frames[timestep],
        )
        write_i16_hex(
            OUTPUT_DIR
            / f"classifier_t{timestep}_expected_centered_cumulative_q8_8.hex",
            centered_frames[timestep],
        )

    summary = {
        "source_input": INPUT_PATH.relative_to(ROOT).as_posix(),
        "sample": 0,
        "timesteps": TEST_TIMESTEPS,
        "device": str(device),
        "numeric_contract": {
            "conv1_input": "nonnegative signed Q8.8 value input",
            "later_conv_inputs": "binary spike multiplied by Q8.8 unit 256",
            "weights": "signed Q0.15",
            "padding": 1,
            "normalization": "signed arithmetic shift right by 15",
            "lif": (
                "SCNN state*135 >>> 8, signed-17 compare, threshold 256, "
                "soft reset and signed-16 stored state"
            ),
            "conv8_pool": "AAVG2 exact count*64 Q8.8 plus binary occupancy",
            "classifier": (
                "signed 2048x11 Q8.8*Q0.15 dot, per-frame current, then "
                "32-bit temporal accumulation and row-max centered Q8.8 wave"
            ),
        },
        "layers": layer_records,
        "classifier_current_q8_8": [
            frame.astype(int).tolist() for frame in classifier_current_frames
        ],
        "centered_cumulative_logit_q8_8": [
            frame.astype(int).tolist() for frame in centered_frames
        ],
        "scope": (
            "Independent fixed-point T=2 reference for current-source RTL "
            "state virtualization, complete Conv1-to-classifier execution and "
            "temporal-logit parity; not RTL evidence by itself"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
