"""Focused numeric and protocol tests, no model training or EDA."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
import run_srnn_integer_optimizer_comparison as ex
from supplement_srnn_magnitude_sgd import magnitude_update_tensor


class ComparisonTest(unittest.TestCase):
    def test_frozen_sum_equals_timestep_replay(self):
        rng = np.random.default_rng(10)
        spikes = rng.integers(0, 2, (7, 3, 256), dtype=np.uint8)
        codes = rng.integers(-32768, 32768, (20, 256), dtype=np.int64)
        actual = ex.frozen_logits(spikes.sum(axis=0), codes)
        expected = ex.integer_head.replay(spikes, codes)["logits"]
        np.testing.assert_array_equal(actual, expected)

    def test_direct_two_batches_exact_existing_reference(self):
        rng = np.random.default_rng(11)
        codes = rng.integers(-50, 51, (20, 256), dtype=np.int64)
        residual = np.zeros_like(codes)
        for _ in range(2):
            spikes = rng.integers(0, 2, (4, 3, 256), dtype=np.uint8)
            labels = np.array([1, 5, 11], dtype=np.int64)
            actual = ex.adapt(spikes, labels, codes, residual, "direct_response", verify=True)
            expected = ex.integer_head.replay(spikes, codes, labels=labels, residual=residual)
            for key in ("codes", "residual", "logits"):
                np.testing.assert_array_equal(actual[key], expected[key])
            codes, residual = actual["codes"], actual["residual"]

    def test_zero_gradient_preserves_both_states(self):
        code = np.array([-32768, -1, 0, 32767], dtype=np.int64)
        remainder = np.array([-50000, 6000, 0, 20], dtype=np.int64)
        updated, residual, stats = magnitude_update_tensor(code, remainder, np.zeros(4, dtype=np.int64), 1e-4)
        np.testing.assert_array_equal(updated, code)
        np.testing.assert_array_equal(residual, remainder)
        self.assertEqual(stats["code_writes"], 0)

    def test_sgd_matches_scalar_rounding_crossing_and_saturation(self):
        scale = 1 << 24
        code = np.array([32767, -32768, 0, 20, 0, 0], dtype=np.int64)
        residual = np.array([0, 0, scale-1, -scale+1, 0, 0], dtype=np.int64)
        grad = np.array([-(1<<50), 1<<50, -1000000, 1000000, -20000, -60000], dtype=np.int64)
        rate = 1e-4
        updated, remainder, stats = magnitude_update_tensor(code, residual, grad, rate)
        for i in range(len(code)):
            delta = round(int(grad[i]) * (-rate/4))
            total = int(residual[i]) + delta
            step = (1 if total >= 0 else -1) * (abs(total)//scale)
            proposed = int(code[i]) + step
            clipped = not -32768 <= proposed <= 32767
            self.assertEqual(updated[i], min(32767, max(-32768, proposed)))
            self.assertEqual(remainder[i], 0 if clipped else total-step*scale)
        self.assertEqual(stats["saturations"], 2)
        self.assertEqual(remainder[4], 0)  # RNE(0.5)
        self.assertEqual(remainder[5], 2)  # RNE(1.5)

    def test_sgd_keeps_sub_lsb_remainder_across_batches(self):
        spikes = np.ones((2, 2, 256), dtype=np.uint8)
        codes = np.zeros((20, 256), dtype=np.int64)
        first = ex.adapt(spikes, np.array([0,1]), codes, codes.copy(), "magnitude_sgd", 1e-4)
        self.assertGreater(np.abs(first["residual"]).max(), 65)
        second = ex.adapt(spikes, np.array([0,1]), first["codes"], first["residual"], "magnitude_sgd", 1e-4)
        self.assertTrue(np.all(np.abs(second["residual"]) < 1 << 24))
        code, residual = first["codes"].copy(), first["residual"].copy()
        trace = np.zeros((2,256), dtype=np.int64)
        logit = np.zeros((2,20), dtype=np.int64)
        lut = ex.reference.read_exp_lut()
        for spike in spikes.astype(np.int64):
            logit += ex.reference.sat32(spike @ code.T) * 16384
            trace = ex.reference.sat32(ex.round_shift_even(trace * 13926,14) + spike * 16384)
            _, _, gradient = ex.reference.wout_gradient(logit, trace, np.array([0,1]), lut)
            code, residual, _ = magnitude_update_tensor(code, residual, gradient, 1e-4)
        np.testing.assert_array_equal(second["codes"], code)
        np.testing.assert_array_equal(second["residual"], residual)
        fresh = ex.adapt(spikes, np.array([0,1]), first["codes"], np.zeros_like(code), "magnitude_sgd", 1e-4)
        self.assertFalse(np.array_equal(second["residual"], fresh["residual"]))

    def test_evaluation_never_mutates_parameters(self):
        counts = np.ones((2,256), dtype=np.int64)
        codes = np.zeros((20,256), dtype=np.int64)
        codes.setflags(write=False)
        self.assertEqual(ex.evaluate(counts, np.array([0,1]), codes)["correct"], 1)
        self.assertEqual(np.count_nonzero(codes), 0)

    def test_validation_split_is_disjoint(self):
        ex.validate_dev_split(list(range(128)), list(range(128,816)))
        with self.assertRaises(ValueError):
            ex.validate_dev_split(list(range(128)), list(range(688)))

    def test_frozen_bounds_reject_unproved_cases(self):
        with self.assertRaises(ValueError):
            ex.frozen_logits(np.full((1,256),65536), np.zeros((20,256), dtype=np.int64))
        with self.assertRaises(ValueError):
            ex.frozen_logits(np.zeros((1,256)), np.zeros((20,256), dtype=np.int64))

    def test_selection_preserves_negative_results_and_excludes_invalid_arms(self):
        rows = [dict(method="magnitude_sgd", learning_rate=rate,
                     shifted_accuracy_improvement=-.01, clean_accuracy_change=0,
                     update_regime=dict(zero_code_updates=False, all_saturated=False)) for rate in ex.RATES]
        selected = ex.select_rate(rows)
        self.assertEqual(selected["selected_rate"], 3e-5)
        self.assertFalse(selected["selected_gain_positive"])
        for row in rows:
            row["update_regime"]["zero_code_updates"] = True
        self.assertEqual(ex.select_rate(rows)["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
