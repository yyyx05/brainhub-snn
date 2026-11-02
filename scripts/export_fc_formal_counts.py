#!/usr/bin/env python3
"""Export bounded, count-preserving FC inputs without constructing a model."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

# Limit native pools before importing numpy or the formal loader's torch imports.
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _name in THREAD_ENV:
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_formal_counts"
EPISODE = ROOT / "results/paper_evidence/supplement_20261004/algorithm/a2/fc/seed38472_residual_on.json"
UINT32_MAX = np.iinfo(np.uint32).max
ARRAY_KEYS = ("shape", "lengths", "frame_offsets", "feature_ids", "counts")


def _integers(value, name, *, maximum=UINT32_MAX):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real integer-valued numbers")
    if (not np.all(np.isfinite(array)) or np.any(array < 0)
            or np.any(array > maximum) or np.any(array != np.floor(array))):
        raise ValueError(f"{name} must contain finite integers in [0, {maximum}]")
    return array


def encode_counts(frames, lengths=None):
    """Encode [B,T,F] as sample-major CSR rows, one row per timestep."""
    frames = _integers(frames, "frames")
    if (frames.ndim != 3 or any(size == 0 for size in frames.shape)
            or frames.shape[2] > 65536):
        raise ValueError("expected nonempty [B,T,F] with F <= 65536")
    batch, steps, features = frames.shape
    lengths = _integers([steps] * batch if lengths is None else lengths,
                        "lengths", maximum=steps)
    if lengths.shape != (batch,):
        raise ValueError("lengths must have one entry per sample")
    for sample, length in enumerate(lengths):
        if np.any(frames[sample, int(length):]):
            raise ValueError("nonzero data in padding")
    rows = frames.reshape(batch * steps, features)
    row_ids, feature_ids = np.nonzero(rows)
    offsets = np.zeros(batch * steps + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(np.bincount(row_ids, minlength=batch * steps), dtype=np.uint64)
    return {
        "shape": np.array(frames.shape, dtype=np.uint32),
        "lengths": lengths.astype(np.uint32),
        "frame_offsets": offsets,
        "feature_ids": feature_ids.astype(np.uint16),
        "counts": rows[row_ids, feature_ids].astype(np.uint32),
    }


def validate_counts(bundle):
    """Reject malformed files before indexing, including duplicate CSR features."""
    for name in ARRAY_KEYS:
        if name not in bundle or np.asarray(bundle[name]).dtype.kind not in "iu":
            raise ValueError(f"missing or noninteger {name}")
    shape = _integers(bundle["shape"], "shape")
    if shape.shape != (3,) or np.any(shape == 0) or shape[2] > 65536:
        raise ValueError("invalid [B,T,F] shape")
    batch, steps, features = map(int, shape)
    lengths = _integers(bundle["lengths"], "lengths", maximum=steps)
    offsets = _integers(bundle["frame_offsets"], "frame_offsets", maximum=np.iinfo(np.uint64).max)
    ids = _integers(bundle["feature_ids"], "feature_ids", maximum=features - 1)
    counts = _integers(bundle["counts"], "counts")
    if lengths.shape != (batch,) or offsets.shape != (batch * steps + 1,):
        raise ValueError("invalid lengths or frame_offsets size")
    if ids.ndim != 1 or counts.shape != ids.shape or np.any(counts == 0):
        raise ValueError("entries require 1D features and strictly positive counts")
    if (offsets[0] != 0 or offsets[-1] != ids.size
            or np.any(offsets[1:] < offsets[:-1]) or np.any(offsets > ids.size)):
        raise ValueError("invalid frame_offsets boundaries/order")
    for row in range(batch * steps):
        start, end = int(offsets[row]), int(offsets[row + 1])
        row_ids = ids[start:end]
        if np.any(row_ids[1:] <= row_ids[:-1]):
            raise ValueError("features must be strictly increasing in each frame")
        if row % steps >= lengths[row // steps] and end != start:
            raise ValueError("nonempty padding frame")
    return batch, steps, features


def decode_counts(bundle):
    batch, steps, features = validate_counts(bundle)
    output = np.zeros((batch * steps, features), dtype=np.uint32)
    for row in range(batch * steps):
        start, end = map(int, bundle["frame_offsets"][row:row + 2])
        output[row, bundle["feature_ids"][start:end]] = bundle["counts"][start:end]
    return output.reshape(batch, steps, features)


def iter_frame_events(bundle, sample, timestep):
    """Yield one feature per event, including repetitions; validate at file load."""
    batch, steps, _ = map(int, bundle["shape"])
    if (not isinstance(sample, (int, np.integer)) or isinstance(sample, (bool, np.bool_))
            or not isinstance(timestep, (int, np.integer)) or isinstance(timestep, (bool, np.bool_))
            or not 0 <= sample < batch or not 0 <= timestep < steps):
        raise ValueError("sample/timestep outside exported batch")
    row = int(sample) * steps + int(timestep)
    start, end = map(int, bundle["frame_offsets"][row:row + 2])
    for index in range(start, end):
        yield from itertools.repeat(int(bundle["feature_ids"][index]), int(bundle["counts"][index]))


def read_counts(path):
    with np.load(path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    validate_counts(bundle)
    return bundle


def verify_roundtrip(bundle, frames):
    """Check both CSR decoding and independent repeated-event histograms."""
    decoded = decode_counts(bundle)
    mismatches = int(np.count_nonzero(decoded != frames))
    if mismatches:
        raise ValueError(f"CSR roundtrip mismatch: {mismatches}")
    batch, steps, features = decoded.shape
    event_mismatches = 0
    events_total = 0
    max_events = 0
    over49 = empty = 0
    for sample in range(batch):
        for timestep in range(steps):
            events = np.fromiter(iter_frame_events(bundle, sample, timestep), dtype=np.uint16)
            histogram = np.bincount(events, minlength=features)
            event_mismatches += int(np.count_nonzero(histogram != frames[sample, timestep]))
            event_count = int(events.size)
            events_total += event_count
            max_events = max(max_events, event_count)
            over49 += event_count > 49
            empty += event_count == 0
    if event_mismatches:
        raise ValueError(f"repeated-event roundtrip mismatch: {event_mismatches}")
    return {
        "shape": list(decoded.shape), "compared_cells_per_decoder": int(decoded.size),
        "csr_mismatches": mismatches, "repeated_event_mismatches": event_mismatches,
        "events": events_total, "nonzero_feature_counts": int(bundle["counts"].size),
        "feature_counts_gt1": int(np.count_nonzero(bundle["counts"] > 1)),
        "max_feature_count": int(bundle["counts"].max(initial=0)),
        "max_frame_events": max_events, "frames_gt49_events": int(over49),
        "empty_frames_including_padding": int(empty),
        "padding_frames": int(batch * steps - np.sum(bundle["lengths"])),
        "dense_fp32_bytes": int(decoded.size * 4),
        "csr_array_bytes": sum(int(bundle[key].nbytes) for key in ARRAY_KEYS),
    }


def select_saved_indices(episode):
    if (episode.get("episode_seed") != 38472 or episode.get("network") != "2312-160-10"
            or episode.get("query_split") != "train"):
        raise ValueError("expected saved A2 seed38472 H160 train-query episode")
    selected = {}
    for role, expected_size in (("support", 160), ("query", 100)):
        ids = episode.get(f"{role}_indices", [])
        if (len(ids) != expected_size or any(type(index) is not int or index < 0 for index in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"invalid saved {role} indices")
        selected[role] = ids[:16]
    if set(episode["support_indices"]) & set(episode["query_indices"]):
        raise ValueError("saved support/query overlap")
    if episode.get("shift") != {"dx_pixels": 2, "dy_pixels": 0, "zero_fill": True}:
        raise ValueError("expected the saved +2/0 zero-fill shift")
    return selected


def load_formal_batch(*, traces_root, data_root, role, indices):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic
    from scripts.run_fc_tp_supervised_shift_adaptation import load_dataset_batches

    records = []
    original_getitem = tonic.datasets.NMNIST.__getitem__

    def observed_getitem(dataset, index):
        frames, label = original_getitem(dataset, index)
        _integers(frames, "raw ToFrame counts")
        rawfile = Path(dataset.data[index]).resolve()
        records.append({
            "role": role, "split": "train", "position": len(records),
            "index": int(index), "label": int(label),
            "rawfilename": rawfile.relative_to(data_root.resolve()).as_posix(),
            "raw_path": str(rawfile), "valid_timesteps": int(frames.shape[0]),
            "dataset_size": len(dataset),
        })
        return frames, label

    # The formal loader constructs NMNIST internally. Block its only download/
    # extraction entry, and observe __getitem__ without changing its output.
    with patch.object(tonic.datasets.NMNIST, "download", side_effect=RuntimeError(
            "External input is read-only; download/extraction is disabled")), patch.object(
                tonic.datasets.NMNIST, "__getitem__", observed_getitem):
        batches = load_dataset_batches(
            traces_root=traces_root, data_root=data_root, train=True,
            indices=indices, batch_size=16)
    if len(batches) != 1 or [record["index"] for record in records] != indices:
        raise ValueError("formal loader changed selected sample order/batch count")
    frames, labels = batches[0]
    if frames.device.type != "cpu" or tuple(frames.shape)[::2] != (16, 2312):
        raise ValueError("expected CPU [16,T,2312] formal inputs")
    if labels.tolist() != [record["label"] for record in records]:
        raise ValueError("formal loader label mismatch")
    for record in records:
        record["padded_timesteps"] = int(frames.shape[1])
    return frames, labels, records


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--episode", type=Path, default=EPISODE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(OUTPUT.resolve()):
        parser.error(f"output must stay inside {OUTPUT}")
    output.mkdir(parents=True, exist_ok=True)
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    selected = select_saved_indices(episode)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from scripts.run_fc_tp_supervised_shift_adaptation import shift_nmnist_frames
    from scripts.run_fc_tp_h160_mask_ablation import DEFAULT_WEIGHTS

    if not DEFAULT_WEIGHTS.is_file():
        raise FileNotFoundError(DEFAULT_WEIGHTS)
    generated = [output / f"{role}_{view}.npz"
                 for role in selected for view in ("clean", "shifted")]
    generated += [output / "samples.csv", output / "summary.json"]
    if any(path.exists() for path in generated):
        raise FileExistsError("choose a new output subdirectory; existing evidence is not overwritten")
    all_records = []
    results = {}
    for role, indices in selected.items():
        role_started = time.perf_counter()
        frames, labels, records = load_formal_batch(
            traces_root=args.traces_root, data_root=args.data_root, role=role, indices=indices)
        all_records.extend(records)
        lengths = [record["valid_timesteps"] for record in records]
        for view in ("clean", "shifted"):
            current = frames if view == "clean" else shift_nmnist_frames(frames, dx=2, dy=0)
            bundle = encode_counts(current.numpy(), lengths)
            bundle["dense_counts"] = current.numpy().astype(np.uint32)
            bundle["labels"] = labels.numpy()
            bundle["dataset_indices"] = np.array(indices, dtype=np.uint32)
            filename = output / f"{role}_{view}.npz"
            np.savez_compressed(filename, **bundle)
            # Validate the actual saved/reloaded bytes, not just in-memory arrays.
            loaded = read_counts(filename)
            if (not np.array_equal(loaded["labels"], labels.numpy())
                    or not np.array_equal(loaded["dataset_indices"], indices)
                    or not np.array_equal(loaded["dense_counts"], current.numpy())):
                raise ValueError("saved identity mismatch")
            result = verify_roundtrip(loaded, current.numpy())
            result.update(file=filename.name, file_bytes=filename.stat().st_size,
                          dx_pixels=0 if view == "clean" else 2, dy_pixels=0)
            results[f"{role}_{view}"] = result
            print(json.dumps({"view": f"{role}_{view}", **result}), flush=True)
            del current, bundle, loaded
        print(f"{role}: {time.perf_counter() - role_started:.3f} seconds", flush=True)
        del frames, labels
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    import tonic
    summary = {
        "status": "pass", "scope": "bounded input export only; no model forward, training or RTL",
        "episode": str(args.episode.resolve()), "episode_seed": 38472,
        "network": "2312-160-10", "weights_reference": str(DEFAULT_WEIGHTS.relative_to(ROOT)),
        "weights_action": "not loaded or modified; existing H160 path only",
        "selection": "first 16 saved IDs per role, unchanged order, both Train",
        "selected_indices": selected, "samples": all_records,
        "data_root": str(args.data_root.resolve()), "traces_root": str(args.traces_root.resolve()),
        "preprocessing": {
            "loader": "scripts/run_fc_tp_supervised_shift_adaptation.py:load_dataset_batches",
            "first_saccade_only": True, "time_window_us": 1000,
            "tonic_defaults": "start at first event; include_incomplete=False; no forced T100",
            "flatten": "feature = polarity*1156 + y*34 + x",
            "padding": "PadTensors(batch_first=True), zero tail to each B16 maximum",
            "shifted": "existing shift_nmnist_frames(dx=2,dy=0), zero fill; no wrap",
        },
        "device": "cpu", "batch_size": 16, "num_workers": 0,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "torch": torch.__version__, "tonic": tonic.__version__},
        "command": subprocess.list2cmdline([sys.executable, "-B", *sys.argv]),
        "wall_seconds_after_numpy_import": time.perf_counter() - started,
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: 32 real samples; evidence: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
