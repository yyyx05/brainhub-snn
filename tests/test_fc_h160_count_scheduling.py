import unittest

import numpy as np

from scripts.run_fc_h160_shared_numeric import count_groups, validate_counts


class CountSchedulingTests(unittest.TestCase):
    def test_repeated_features_same_bank_and_empty_frame(self):
        frame = np.zeros(2312, dtype=np.uint16)
        frame[[0, 1, 4, 2311]] = [2, 1, 3, 1]
        groups = list(count_groups(frame))
        self.assertEqual(groups, [(0, 3), (0, 1), (1, 1), (1, 1), (1, 1), (577, 8)])
        restored = np.zeros_like(frame)
        for row, mask in groups:
            for lane in range(4):
                restored[row*4+lane] += (mask >> lane) & 1
        np.testing.assert_array_equal(frame, restored)
        self.assertEqual(list(count_groups(np.zeros(2312, dtype=np.uint16))), [])

    def test_more_than_49_events_are_not_truncated(self):
        frame = np.zeros(2312, dtype=np.uint16)
        frame[:70] = 1
        self.assertEqual(sum(mask.bit_count() for _, mask in count_groups(frame)), 70)

    def test_capacity_and_type_are_explicit(self):
        self.assertEqual(validate_counts(np.zeros((99, 2312), dtype=np.uint32)).dtype, np.uint16)
        for bad in (np.zeros((101, 2312), dtype=np.uint32),
                    np.zeros((2, 700), dtype=np.uint32),
                    np.zeros((2, 2312), dtype=np.float32),
                    np.full((1, 2312), -1), np.full((1, 2312), 65536)):
            with self.assertRaises(ValueError):
                validate_counts(bad)


if __name__ == "__main__":
    unittest.main()
