"""Focused counts/export tests; no model construction or external data access."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.export_fc_formal_counts import (
    EPISODE, OUTPUT, decode_counts, encode_counts, iter_frame_events,
    load_formal_batch, read_counts, select_saved_indices, validate_counts,
    verify_roundtrip,
)


class CountsTests(unittest.TestCase):
    def test_repeated_features_are_not_bitmasks(self):
        frames = np.array([[[0, 3, 0, 2], [0, 0, 0, 0]]], dtype=np.float32)
        bundle = encode_counts(frames)
        self.assertEqual(list(iter_frame_events(bundle, 0, 0)), [1, 1, 1, 3, 3])
        self.assertEqual(list(iter_frame_events(bundle, 0, 1)), [])
        np.testing.assert_array_equal(decode_counts(bundle), frames)
        result = verify_roundtrip(bundle, frames)
        self.assertEqual(result["feature_counts_gt1"], 2)
        self.assertEqual(result["events"], 5)

    def test_above49_unique_and_repeated_events(self):
        frames = np.zeros((1, 2, 2312), dtype=np.uint32)
        frames[0, 0, :70] = 1
        frames[0, 1, 2311] = 257
        result = verify_roundtrip(encode_counts(frames), frames)
        self.assertEqual(result["events"], 327)
        self.assertEqual(result["max_frame_events"], 257)
        self.assertEqual(result["frames_gt49_events"], 2)

    def test_all_empty_and_padding(self):
        frames = np.zeros((2, 4, 8), dtype=np.float32)
        bundle = encode_counts(frames, [0, 3])
        self.assertEqual(bundle["frame_offsets"].tolist(), [0] * 9)
        result = verify_roundtrip(bundle, frames)
        self.assertEqual(result["empty_frames_including_padding"], 8)
        self.assertEqual(result["padding_frames"], 5)

    def test_padding_and_internal_empty_frame_keep_positions(self):
        frames = np.zeros((2, 4, 8), dtype=np.int16)
        frames[0, 0, 1] = 2
        frames[1, 3, 7] = 5
        bundle = encode_counts(frames, [2, 4])
        self.assertEqual(bundle["frame_offsets"].tolist(), [0, 1, 1, 1, 1, 1, 1, 1, 2])
        np.testing.assert_array_equal(decode_counts(bundle), frames)
        with self.assertRaises(ValueError):
            encode_counts(frames, [2, 3])

    def test_invalid_dense_values(self):
        for value in (-1, .5, np.nan, np.inf, -np.inf, 2**32, 1 + 2j, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_counts(np.full((1, 1, 1), value))

    def test_invalid_shapes_and_lengths(self):
        for frames in (np.zeros((1, 2)), np.zeros((1, 0, 2)), np.zeros((1, 1, 65537))):
            with self.subTest(shape=frames.shape), self.assertRaises(ValueError):
                encode_counts(frames)
        for lengths in ([3], [-1], [.5], [1, 2], [np.nan]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                encode_counts(np.zeros((1, 2, 3)), lengths)

    def test_uint32_max_does_not_overflow(self):
        frames = np.array([[[2**32 - 1]]], dtype=np.uint64)
        np.testing.assert_array_equal(decode_counts(encode_counts(frames)), frames)

    def test_timesteps_above100_are_preserved(self):
        frames = np.zeros((2, 137, 2312), dtype=np.uint32)
        frames[0, 120, 0] = 3
        frames[1, 136, 2311] = 5
        bundle = encode_counts(frames, [121, 137])
        result = verify_roundtrip(bundle, frames)
        self.assertEqual(result["shape"], [2, 137, 2312])
        self.assertEqual(list(iter_frame_events(bundle, 1, 136)), [2311] * 5)

    @unittest.skipUnless((OUTPUT / "summary.json").is_file(), "real export not yet generated")
    def test_real_saved_dense_sparse_and_metadata_agree(self):
        summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        for name in summary["results"]:
            with self.subTest(view=name):
                bundle = read_counts(OUTPUT / f"{name}.npz")
                self.assertEqual(bundle["dense_counts"].dtype, np.dtype("uint32"))
                result = verify_roundtrip(bundle, bundle["dense_counts"])
                self.assertEqual(result["events"], summary["results"][name]["events"])
                role = name.split("_")[0]
                records = [item for item in summary["samples"] if item["role"] == role]
                np.testing.assert_array_equal(bundle["dataset_indices"], [r["index"] for r in records])
                np.testing.assert_array_equal(bundle["labels"], [r["label"] for r in records])
                np.testing.assert_array_equal(bundle["lengths"], [r["valid_timesteps"] for r in records])

    def test_invalid_sparse_files(self):
        good = encode_counts(np.array([[[1, 2, 0], [0, 0, 1]]]))
        corruptions = (
            ("frame_offsets", [1, 2, 3]), ("frame_offsets", [0, 3, 2]),
            ("frame_offsets", [0, 2, 4]), ("frame_offsets", [0, -1, 3]),
            ("feature_ids", [0, 0, 2]), ("feature_ids", [1, 0, 2]),
            ("feature_ids", [0, 3, 2]), ("feature_ids", [0, -1, 2]),
            ("counts", [0, 2, 1]), ("counts", [1, -2, 1]),
            ("counts", [1, 2**32, 1]), ("counts", [1.5, 2, 1]),
            ("lengths", [1]), ("shape", [1, 0, 3]),
        )
        for name, value in corruptions:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                bundle = dict(good)
                bundle[name] = np.asarray(value)
                validate_counts(bundle)

    def test_invalid_event_coordinates(self):
        bundle = encode_counts(np.zeros((1, 1, 3)))
        for sample, timestep in ((-1, 0), (1, 0), (0, -1), (0, 1), (.5, 0), (True, 0)):
            with self.subTest(sample=sample, timestep=timestep), self.assertRaises(ValueError):
                list(iter_frame_events(bundle, sample, timestep))

    def test_npz_roundtrip_in_project(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        frames = np.zeros((2, 3, 2312), dtype=np.float32)
        frames[0, 0, 2311] = 70
        frames[1, 2, 5] = 3
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="test_") as directory:
            path = Path(directory) / "counts.npz"
            np.savez_compressed(path, **encode_counts(frames, [1, 3]))
            result = verify_roundtrip(read_counts(path), frames)
            self.assertEqual(result["repeated_event_mismatches"], 0)

    def test_saved_selection_preserves_first16_order(self):
        episode = json.loads(EPISODE.read_text(encoding="utf-8"))
        selection = select_saved_indices(episode)
        self.assertEqual(selection["support"], episode["support_indices"][:16])
        self.assertEqual(selection["query"], episode["query_indices"][:16])
        self.assertEqual(selection["support"][:3], [1941, 7719, 13784])
        self.assertEqual(selection["query"][:3], [2883, 8656, 12740])
        for change in ("overlap", "negative", "float", "split", "network"):
            bad = deepcopy(episode)
            if change == "overlap":
                bad["query_indices"][0] = bad["support_indices"][0]
            elif change == "negative":
                bad["support_indices"][0] = -1
            elif change == "float":
                bad["support_indices"][0] = 1.5
            elif change == "split":
                bad["query_split"] = "test"
            else:
                bad["network"] = "2312-400-10"
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_saved_indices(bad)

    def test_missing_dataset_never_downloads_or_extracts(self):
        import tonic.dataset
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="test_missing_") as directory:
            with patch.object(tonic.dataset, "download_and_extract_archive") as download:
                with self.assertRaisesRegex(RuntimeError, "download/extraction is disabled"):
                    load_formal_batch(traces_root=Path(directory), data_root=Path(directory),
                                      role="support", indices=list(range(16)))
                download.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
