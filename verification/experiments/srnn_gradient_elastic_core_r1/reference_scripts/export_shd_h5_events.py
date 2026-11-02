#!/usr/bin/env python3
"""Export deterministic SHD test streams without a runtime Tonic dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


TIME_END = 0x40000
SAMPLE_END = 0x80000
INPUT_DIM = 700


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tonic_slice_indices(times_us: np.ndarray, time_window_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Match Tonic SliceByTime(include_incomplete=False) exactly."""
    if times_us.ndim != 1 or times_us.size == 0:
        raise ValueError("SHD sample must contain at least one timestamp")
    start_time = int(times_us[0])
    end_time = int(times_us[-1])
    duration = end_time - start_time
    slices = max(int(np.floor((duration - time_window_us) / time_window_us) + 1), 1)
    starts = np.arange(slices, dtype=np.int64) * time_window_us + start_time
    ends = starts + time_window_us
    return np.searchsorted(times_us, starts), np.searchsorted(times_us, ends)


def encode_sample(times_s: np.ndarray, units: np.ndarray, time_window_us: int) -> tuple[bytes, dict]:
    # Tonic's SHD loader multiplies seconds by 1e6 and stores into an integer field.
    times_us = (np.asarray(times_s) * 1_000_000).astype(np.int64)
    units = np.asarray(units, dtype=np.int64)
    if times_us.shape != units.shape:
        raise ValueError("timestamp and unit arrays have different lengths")
    if np.any((units < 0) | (units >= INPUT_DIM)):
        raise ValueError("SHD unit index is outside 0..699")

    starts, ends = tonic_slice_indices(times_us, time_window_us)
    words: list[int] = []
    clipped = 0
    for frame_index, (begin, end) in enumerate(zip(starts, ends)):
        counts = np.bincount(units[begin:end], minlength=INPUT_DIM)
        for feature in np.flatnonzero(counts):
            value = int(counts[feature])
            if value > 255:
                value = 255
                clipped += 1
            words.append((int(feature) << 8) | value)
        words.append(SAMPLE_END if frame_index == len(starts) - 1 else TIME_END)

    encoded = "".join(f"{word:05x}\n" for word in words).encode("ascii")
    return encoded, {
        "events": int(times_us.size),
        "time_steps": int(len(starts)),
        "records": len(words),
        "clipped_feature_counts": clipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--time-window-us", type=int, default=10_000)
    args = parser.parse_args()

    root = args.root.resolve()
    h5_path = (args.h5 or root / "imports" / "srnn" / "dataset_cache" / "SHD" / "shd_test.h5").resolve()
    output = (args.output or root / "data" / "shd" / "official_test").resolve()
    if args.start_index < 0 or args.count <= 0:
        raise ValueError("start-index must be non-negative and count must be positive")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    labels = []
    with h5py.File(h5_path, "r") as dataset:
        total = len(dataset["labels"])
        stop = args.start_index + args.count
        if stop > total:
            raise ValueError(f"requested samples [{args.start_index},{stop}) exceed dataset size {total}")
        for local_index, dataset_index in enumerate(range(args.start_index, stop)):
            encoded, stats = encode_sample(
                dataset["spikes/times"][dataset_index],
                dataset["spikes/units"][dataset_index],
                args.time_window_us,
            )
            path = output / f"sample_{local_index:04d}.hex"
            path.write_bytes(encoded)
            label = int(dataset["labels"][dataset_index])
            labels.append(label)
            records.append({
                "local_index": local_index,
                "dataset_index": dataset_index,
                "label": label,
                "sha256": sha256(path),
                **stats,
            })

    labels_path = output / "labels.hex"
    labels_path.write_text("".join(f"{label:02x}\n" for label in labels), encoding="ascii")
    manifest = {
        "schema_version": 1,
        "dataset": "SHD official test",
        "source_h5": h5_path.relative_to(root).as_posix(),
        "source_h5_sha256": sha256(h5_path),
        "time_window_us": args.time_window_us,
        "include_incomplete": False,
        "start_index": args.start_index,
        "count": args.count,
        "labels_sha256": sha256(labels_path),
        "samples": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": "pass", "output": str(output), "count": args.count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
