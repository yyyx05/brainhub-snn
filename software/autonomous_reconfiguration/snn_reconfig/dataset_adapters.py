"""Thin adapters from project datasets into the shared fingerprint schema."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from .fingerprint import fingerprint_histogram_samples


def _histogram(values: Iterable[float], bins: int, lower: float, upper: float) -> list[float]:
    result = [0.0] * bins
    width = (upper - lower) / bins
    for raw_value in values:
        value = float(raw_value)
        if value < lower or value > upper:
            continue
        index = bins - 1 if value == upper else int((value - lower) / width)
        if 0 <= index < bins:
            result[index] += 1.0
    return result


def fingerprint_shd_h5(
    path: str | Path,
    sample_indices: Iterable[int],
    *,
    temporal_bins: int = 20,
    temporal_upper_seconds: float = 1.4,
    channel_bins: int = 70,
) -> dict:
    """Fingerprint selected SHD samples from the project's native HDF5 file."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("SHD HDF5 input requires the optional h5py package") from exc

    indices = [int(value) for value in sample_indices]
    if not indices:
        raise ValueError("at least one SHD sample index is required")
    event_counts: list[float] = []
    durations: list[float] = []
    temporal_hists: list[list[float]] = []
    channel_hists: list[list[float]] = []
    with h5py.File(Path(path), "r") as handle:
        total_samples = len(handle["labels"])
        for index in indices:
            if not 0 <= index < total_samples:
                raise IndexError(f"SHD sample index {index} is outside 0..{total_samples - 1}")
            times = handle["spikes/times"][index]
            units = handle["spikes/units"][index]
            if len(times) != len(units):
                raise ValueError(f"SHD sample {index} has mismatched time/unit arrays")
            event_counts.append(float(len(times)))
            durations.append(float(max(times)) if len(times) else 0.0)
            temporal_hists.append(
                _histogram(times, temporal_bins, 0.0, temporal_upper_seconds)
            )
            channel_hists.append(_histogram(units, channel_bins, 0.0, 700.0))

    return fingerprint_histogram_samples(
        event_counts,
        durations,
        temporal_hists,
        channel_hists,
        {
            "feature_rank": 1,
            "ordered_channel_count": 700,
            "spatial_axes": 0,
            "polarity_channels": 1,
            "class_count": 20,
        },
    )


def fingerprint_dvsgesture_q8_8_hex(
    path: str | Path,
    sample_count: int,
    *,
    timesteps: int = 20,
    polarity_channels: int = 2,
    height: int = 32,
    width: int = 32,
    duration_seconds: float = 1.5,
) -> dict:
    """Fingerprint the existing dense DVS-Gesture Q8.8 frame export.

    The accumulation matches the teammate implementation: total frame mass is
    the event count proxy, and per-timestep/per-polarity sums form histograms.
    """

    if min(sample_count, timesteps, polarity_channels, height, width) <= 0:
        raise ValueError("DVS dimensions and sample_count must be positive")
    values_per_sample = timesteps * polarity_channels * height * width
    expected_words = sample_count * values_per_sample
    event_counts = [0.0] * sample_count
    temporal_hists = [[0.0] * timesteps for _ in range(sample_count)]
    channel_hists = [[0.0] * polarity_channels for _ in range(sample_count)]

    word_count = 0
    with Path(path).open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            token = line.strip()
            if not token:
                continue
            if word_count >= expected_words:
                raise ValueError(f"DVS input has more than {expected_words} words")
            raw = int(token, 16)
            signed = raw if raw < 0x8000 else raw - 0x10000
            value = signed / 256.0
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"DVS word on line {line_number} is not a non-negative Q8.8 count")
            sample = word_count // values_per_sample
            within_sample = word_count % values_per_sample
            spatial_area = height * width
            timestep = within_sample // (polarity_channels * spatial_area)
            channel = (within_sample // spatial_area) % polarity_channels
            event_counts[sample] += value
            temporal_hists[sample][timestep] += value
            channel_hists[sample][channel] += value
            word_count += 1

    if word_count != expected_words:
        raise ValueError(f"DVS input has {word_count} words; expected {expected_words}")
    return fingerprint_histogram_samples(
        event_counts,
        [duration_seconds] * sample_count,
        temporal_hists,
        channel_hists,
        {
            "feature_rank": 3,
            "ordered_channel_count": polarity_channels * height * width,
            "spatial_axes": 2,
            "spatial_shape": [height, width],
            "polarity_channels": polarity_channels,
            "class_count": 11,
        },
    )
