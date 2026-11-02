"""Calibration isolation and unchanged new-rate arithmetic; no old workload rerun."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
import run_srnn_sgd_motion_control as subject
from supplement_srnn_magnitude_sgd import magnitude_update_tensor


def fixture():
    return [dict(method="direct_response", counters=dict(code_lsb_steps=451401)),
            dict(method="magnitude_sgd", learning_rate=0.01, counters=dict(code_lsb_steps=334578)),
            dict(method="magnitude_sgd", learning_rate=0.03, counters=dict(code_lsb_steps=1027258))]


class CalibrationTests(unittest.TestCase):
    def test_exact_interpolation(self):
        result = subject.calibrate(fixture())
        expected = Fraction(1, 100) + Fraction(2, 100)*Fraction(451401-334578, 1027258-334578)
        self.assertEqual(result["rate_fraction"], str(expected))
        self.assertEqual(result["rate"], float(expected))

    def test_query_outcomes_do_not_affect_rate(self):
        rows = fixture()
        before = subject.calibrate(rows)
        for r in rows:
            r.update(clean_accuracy_change=999999, shifted_accuracy_improvement=-999999,
                     predictions=object(), update_regime=object())
        with patch("builtins.open", side_effect=AssertionError("calibration must not open query files")):
            self.assertEqual(subject.calibrate(rows), before)

    def test_nearest_bracket_only_and_no_mutation(self):
        rows = fixture()+[dict(method="magnitude_sgd", learning_rate=0.001, counters=dict(code_lsb_steps=27259))]
        original = copy.deepcopy(rows)
        self.assertEqual(subject.calibrate(rows), subject.calibrate(fixture()))
        self.assertEqual(rows, original)

    def test_missing_duplicate_or_reversed_bracket_rejected(self):
        bad = [fixture()[:2], fixture()+fixture()[1:2], fixture()[1:]]
        reversed_rows = fixture()
        reversed_rows[1]["learning_rate"] = 0.1
        bad.append(reversed_rows)
        for rows in bad:
            with self.assertRaises(ValueError):
                subject.calibrate(rows)

    def test_new_rate_matches_independent_scalar_state_update(self):
        rate = subject.calibrate(fixture())["rate"]
        scale = 1 << 24
        codes = np.asarray([0, 0, 123, -9, 32767, -32768, 5], dtype=np.int64)
        residual = np.asarray([scale-1, 1-scale, 321, -654, scale-1, 1-scale, 0], dtype=np.int64)
        gradients = np.asarray([-800, 800, 0, 1 << 28, -(1 << 40), 1 << 40, -1], dtype=np.int64)
        expected_code, expected_residual = [], []
        for c, r, g in zip(codes.tolist(), residual.tolist(), gradients.tolist()):
            delta = round(float(g)*(-rate/4))
            value = r+delta
            change = (1 if value >= 0 else -1)*(abs(value)//scale)
            unclipped = c+change
            expected_code.append(max(-32768, min(32767, unclipped)))
            expected_residual.append(0 if unclipped != expected_code[-1] else value-change*scale)
        code, rest, stats = magnitude_update_tensor(codes, residual, gradients, rate)
        np.testing.assert_array_equal(code, expected_code)
        np.testing.assert_array_equal(rest, expected_residual)
        self.assertEqual(stats["zero_skips"], 1)
        self.assertEqual(stats["saturations"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
