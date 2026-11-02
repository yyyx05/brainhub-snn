"""Tests only the new zero-fill transform, without running accepted workloads."""
import unittest
import numpy as np
from run_srnn_shift_transfer import shift_counts


class ShiftTests(unittest.TestCase):
    def test_both_directions_match_independent_channel_mapping(self):
        raw = (np.arange(2100).reshape(3, 700) % 256).astype(np.int64)
        for shift in (-16, 32):
            expected = np.zeros_like(raw)
            for channel in range(700):
                destination = channel+shift
                if 0 <= destination < 700:
                    expected[:, destination] = raw[:, channel]
            actual = shift_counts(raw, shift)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(raw, (np.arange(2100).reshape(3, 700) % 256))

    def test_edges_drop_without_wrap_or_binarization(self):
        raw = np.zeros((1, 700), dtype=np.int64)
        raw[0, [0, 15, 16, 667, 668, 699]] = [1, 2, 3, 4, 5, 255]
        negative = shift_counts(raw, -16)
        positive = shift_counts(raw, 32)
        self.assertEqual(int(negative.sum()), 267)
        self.assertEqual(int(positive.sum()), 10)
        self.assertEqual(negative[0, 683], 255)
        self.assertEqual(positive[0, 48], 3)

    def test_unplanned_condition_and_invalid_counts_rejected(self):
        for value, shift in ((np.zeros((2, 700), dtype=np.int64), 16),
                             (np.full((2, 700), 256, dtype=np.int64), 32),
                             (np.zeros((2, 700), dtype=float), -16)):
            with self.assertRaises(ValueError):
                shift_counts(value, shift)


if __name__ == "__main__":
    unittest.main(verbosity=2)
