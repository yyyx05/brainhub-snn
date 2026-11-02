"""Focused selection and unchanged SGD arithmetic checks, no dataset rerun."""
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import run_srnn_sgd_expanded as subject
from supplement_srnn_magnitude_sgd import magnitude_update_tensor


def rows():
    return [dict(method="magnitude_sgd", learning_rate=rate,
                 shifted_accuracy_improvement=0.0, clean_accuracy_change=0.0,
                 update_regime=dict(zero_code_updates=False, all_saturated=False))
            for rate in subject.ALL_RATES]


class SelectionTests(unittest.TestCase):
    def test_ties_choose_smallest(self):
        self.assertEqual(subject.select_expanded(rows())["selected_rate"], subject.ALL_RATES[0])

    def test_clean_constraint_and_ineligible_updates(self):
        data = rows()
        for r in data:
            r["shifted_accuracy_improvement"] = r["learning_rate"]
        data[-1]["clean_accuracy_change"] = -0.0101
        data[-2]["update_regime"]["all_saturated"] = True
        data[-3]["update_regime"]["zero_code_updates"] = True
        self.assertEqual(subject.select_expanded(data)["selected_rate"], subject.ALL_RATES[-4])

    def test_missing_or_duplicate_rate_rejected(self):
        for data in (rows()[:-1], rows()+rows()[:1]):
            with self.assertRaises(ValueError):
                subject.select_expanded(data)

    def test_no_candidate_and_upper_endpoint(self):
        data = rows()
        for r in data:
            r["update_regime"]["zero_code_updates"] = True
        self.assertEqual(subject.select_expanded(data)["status"], "inconclusive")
        data[-1]["update_regime"]["zero_code_updates"] = False
        self.assertTrue(subject.select_expanded(data)["upper_grid_endpoint_selected"])

    def test_selection_never_reads_files(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected query I/O")):
            subject.select_expanded(rows())

    def test_already_measured_winner_never_reexecutes_report(self):
        candidates = rows()
        dev = dict(status="PASS_expanded_integer_SGD_dev", episodes=candidates,
                   selection=subject.select_expanded(candidates))
        previous = dict(episodes=[dict(seed=seed, method=method,
            learning_rate=subject.ALL_RATES[0] if method == "magnitude_sgd" else None)
            for seed in subject.REPORT_SEEDS for method in ("direct_response", "magnitude_sgd")])
        with patch.object(subject, "Inputs") as inputs, \
                patch.object(subject, "read_signed_hex", return_value=np.zeros(5120, dtype=np.int64)), \
                patch.object(subject, "write_json"), patch.object(subject, "write_tables"), \
                patch.object(subject, "query", side_effect=AssertionError("query rerun")), \
                patch.object(subject, "episode", side_effect=AssertionError("adaptation rerun")):
            inputs.return_value.json.side_effect = [dev, previous]
            result = subject.run("report", Path("unused/report"))
        self.assertEqual(result["new_adaptation_episodes"], 0)
        self.assertEqual(len(result["episodes"]), 6)


class ArithmeticTests(unittest.TestCase):
    def test_independent_scalar_round_carry_clip_and_zero(self):
        scale = 1 << 24
        codes = np.asarray([0, 0, 0, 0, 32767, -32768, 123], dtype=np.int64)
        residual = np.asarray([0, 1, scale-1, 1-scale, scale-1, 1-scale, 321], dtype=np.int64)
        gradients = np.asarray([4, 12, -16, 16, -(1 << 32), 1 << 32, 0], dtype=np.int64)
        for rate in (0.5, 1.0, 10.0):
            expected_code, expected_residual = [], []
            for code, rest, grad in zip(codes.tolist(), residual.tolist(), gradients.tolist()):
                delta = round(-Fraction(rate) * grad / 4)
                total = rest + delta
                change = (1 if total >= 0 else -1) * (abs(total) // scale)
                candidate = code + change
                new_code = max(-32768, min(32767, candidate))
                expected_code.append(new_code)
                expected_residual.append(0 if new_code != candidate else total-change*scale)
            actual_code, actual_residual, _ = magnitude_update_tensor(codes, residual, gradients, rate)
            np.testing.assert_array_equal(actual_code, expected_code)
            np.testing.assert_array_equal(actual_residual, expected_residual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
