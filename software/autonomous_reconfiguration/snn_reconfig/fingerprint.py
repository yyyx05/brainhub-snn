"""Dependency-free event-sample fingerprint extraction.

Dataset adapters should decode their native files into the normalized sample
format consumed here. This keeps the profile selector independent of tonic,
PyTorch, HDF5, and legacy absolute paths.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Any


def _normalized_entropy(histogram: list[float]) -> float:
    total = sum(histogram)
    if total <= 0 or len(histogram) <= 1:
        return 0.0
    result = 0.0
    for count in histogram:
        if count > 0:
            probability = count / total
            result -= probability * math.log(probability)
    return result / math.log(len(histogram))


def _histogram(values: list[float], bins: int, lower: float, upper: float) -> list[int]:
    if bins <= 0 or not upper > lower:
        raise ValueError("histogram requires positive bins and upper > lower")
    result = [0] * bins
    width = (upper - lower) / bins
    for value in values:
        if value < lower or value > upper:
            continue
        index = bins - 1 if value == upper else int((value - lower) / width)
        if 0 <= index < bins:
            result[index] += 1
    return result


def _coefficient_of_variation(values: list[float]) -> float:
    mean = fmean(values)
    return pstdev(values) / max(abs(mean), 1e-8)


def fingerprint_histogram_samples(
    event_counts: list[float],
    durations: list[float],
    temporal_hists: list[list[float]],
    channel_hists: list[list[float]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a fingerprint from per-sample event totals and histograms.

    Dataset adapters can use weighted histograms for already-binned fixed-point
    frames without expanding every bin value into synthetic event records.
    """

    sample_count = len(event_counts)
    if sample_count <= 0:
        raise ValueError("at least one sample is required")
    if not (
        len(durations) == sample_count
        and len(temporal_hists) == sample_count
        and len(channel_hists) == sample_count
    ):
        raise ValueError("fingerprint sample arrays must have equal lengths")
    required_metadata = (
        "feature_rank",
        "ordered_channel_count",
        "spatial_axes",
        "polarity_channels",
        "class_count",
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(f"metadata is missing required keys: {', '.join(missing)}")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in event_counts):
        raise ValueError("event counts must be finite and non-negative")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in durations):
        raise ValueError("durations must be finite and non-negative")
    if any(
        not row
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in row)
        for row in temporal_hists + channel_hists
    ):
        raise ValueError("histograms must be non-empty, finite, and non-negative")

    temporal_bins = len(temporal_hists[0])
    channel_bins = len(channel_hists[0])
    if any(len(row) != temporal_bins for row in temporal_hists):
        raise ValueError("temporal histogram widths differ")
    if any(len(row) != channel_bins for row in channel_hists):
        raise ValueError("channel histogram widths differ")

    temporal_occupied = sum(value > 0 for row in temporal_hists for value in row)
    channel_occupied = sum(value > 0 for row in channel_hists for value in row)
    result: dict[str, Any] = {
        "event_count_log_mean": fmean(math.log1p(float(value)) for value in event_counts),
        "event_count_cv": _coefficient_of_variation([float(value) for value in event_counts]),
        "duration_mean": fmean(float(value) for value in durations),
        "duration_cv": _coefficient_of_variation([float(value) for value in durations]),
        "temporal_entropy": fmean(_normalized_entropy(row) for row in temporal_hists),
        "channel_entropy": fmean(_normalized_entropy(row) for row in channel_hists),
        "temporal_occupancy": temporal_occupied / (sample_count * temporal_bins),
        "channel_occupancy": channel_occupied / (sample_count * channel_bins),
        "sample_count": sample_count,
        "feature_rank": int(metadata["feature_rank"]),
        "ordered_channel_count": int(metadata["ordered_channel_count"]),
        "spatial_axes": int(metadata["spatial_axes"]),
        "polarity_channels": int(metadata["polarity_channels"]),
        "class_count": int(metadata["class_count"]),
    }
    if result["spatial_axes"] == 2:
        shape = metadata.get("spatial_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("2D metadata requires a two-element spatial_shape")
        result["spatial_shape"] = [int(value) for value in shape]
    return result


def fingerprint_event_samples(
    samples: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build the teammate-compatible fingerprint from normalized event samples.

    Each sample must contain `timestamps_seconds`, `channel_ids`, and
    `duration_seconds`. Metadata defines the dataset protocol and histogram
    ranges; no dataset name is used in selection.
    """

    if not samples:
        raise ValueError("at least one sample is required")
    required_metadata = (
        "feature_rank",
        "ordered_channel_count",
        "spatial_axes",
        "polarity_channels",
        "class_count",
        "temporal_bins",
        "temporal_range_seconds",
        "channel_histogram_bins",
        "channel_id_upper_bound",
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(f"metadata is missing required keys: {', '.join(missing)}")

    temporal_bins = int(metadata["temporal_bins"])
    temporal_upper = float(metadata["temporal_range_seconds"])
    channel_bins = int(metadata["channel_histogram_bins"])
    channel_upper = float(metadata["channel_id_upper_bound"])

    event_counts: list[float] = []
    durations: list[float] = []
    temporal_hists: list[list[int]] = []
    channel_hists: list[list[int]] = []
    for index, sample in enumerate(samples):
        timestamps = [float(value) for value in sample.get("timestamps_seconds", [])]
        channels = [float(value) for value in sample.get("channel_ids", [])]
        if len(timestamps) != len(channels):
            raise ValueError(f"sample {index} timestamp/channel lengths differ")
        duration = float(sample["duration_seconds"])
        if duration < 0.0 or not math.isfinite(duration):
            raise ValueError(f"sample {index} has invalid duration")
        if any(not math.isfinite(value) for value in timestamps + channels):
            raise ValueError(f"sample {index} contains a non-finite event")
        event_counts.append(float(len(timestamps)))
        durations.append(duration)
        temporal_hists.append(_histogram(timestamps, temporal_bins, 0.0, temporal_upper))
        channel_hists.append(_histogram(channels, channel_bins, 0.0, channel_upper))

    return fingerprint_histogram_samples(
        event_counts, durations, temporal_hists, channel_hists, metadata
    )
