"""Focused checks of the new test-only continuous-parameter oracle."""
import unittest

import numpy as np

from scripts.experiments.build_srnn_resident_multisequence import reference, m2


class ContinuousReferenceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = np.zeros((2, 1, 450), dtype=np.int64)
        self.inputs[:, 0, 0] = 16384
        self.wh1 = np.zeros((256, 450), dtype=np.int64)
        self.wh1[0, 0] = 16384
        self.code = np.zeros((20, 256), dtype=np.int64)

    def test_zero_residual_matches_existing_reference(self):
        expected, _ = m2.reference(self.inputs, np.array([2]), self.wh1, self.code, learning=True)
        actual, _ = reference(self.inputs, 2, self.wh1, self.code, np.zeros_like(self.code), True)
        for field in actual:
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)

    def test_nonzero_residual_is_retained_before_the_first_update(self):
        residual = np.full_like(self.code, 34)
        actual, _ = reference(self.inputs[:1], 2, self.wh1, self.code, residual, True)
        self.assertEqual(actual["wout_after"][0, 2, 0], 1)
        self.assertEqual(actual["residual"][0, 2, 0], 0)
        self.assertTrue(np.all(residual == 34))
        self.assertTrue(np.all(self.code == 0))

    def test_learning_off_does_not_zero_existing_residuals(self):
        residual = np.full_like(self.code, 34)
        actual, traffic = reference(self.inputs, 2, self.wh1, self.code, residual, False)
        for row in actual["residual"]:
            np.testing.assert_array_equal(row, residual)
        self.assertEqual(sum(row["rmw"] for row in traffic), 0)


if __name__ == "__main__":
    unittest.main()
