"""Small CPU update parity tests; no checkpoint or dataset is loaded."""

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from scripts.supplement_algorithm_residual import Counters, commands, parse_args, update_adapter
from scripts import run_fc_tp_h160_mask_ablation as fc
from scripts import run_scnn_tp_selected_layer_shift_adaptation as scnn
import compare_srnn_shd_shifted_optimizers as srnn


class ResidualTests(unittest.TestCase):
    def test_scnn_retained_matches_original_and_drop_has_dead_zone(self):
        original = scnn.apply_fixed_sgd_residual
        kwargs = dict(learning_rate=0.375 / 32768, weight_fractional_bits=15,
                      residual_fractional_bits=15)
        outcomes = {}
        for arm in ("original", "on", "drop"):
            weight = torch.nn.Parameter(torch.zeros(1))
            residual = torch.zeros(1, dtype=torch.int32)
            counter = Counters()
            if arm == "original":
                for _ in range(4):
                    original(weight, torch.ones(1), residual, **kwargs)
            else:
                with update_adapter("scnn", arm, counter):
                    for _ in range(4):
                        scnn.apply_fixed_sgd_residual(weight, torch.ones(1), residual, **kwargs)
            outcomes[arm] = (weight.detach().clone(), residual.clone())
        for index in (0, 1):
            self.assertTrue(torch.equal(outcomes["original"][index], outcomes["on"][index]))
        self.assertNotEqual(float(outcomes["on"][0]), 0)
        self.assertEqual(float(outcomes["drop"][0]), 0)
        self.assertEqual(int(outcomes["drop"][1]), 0)
        self.assertIs(scnn.apply_fixed_sgd_residual, original)

    def test_scnn_saturation_and_multicode_survive_drop(self):
        weight = torch.nn.Parameter(torch.tensor([32767 / 32768, 0.0]))
        residual = torch.zeros(2, dtype=torch.int32)
        counter = Counters()
        with update_adapter("scnn", "drop", counter):
            scnn.apply_fixed_sgd_residual(weight, torch.tensor([-1.0, 1.0]), residual,
                learning_rate=2.5 / 32768, weight_fractional_bits=15, residual_fractional_bits=15)
        self.assertEqual(counter.values["saturations"], 1)
        self.assertEqual(counter.values["code_lsb_steps"], 2)
        self.assertEqual(int(torch.count_nonzero(residual)), 0)

    def test_srnn_on_parity_drop_every_step_and_restore(self):
        original = srnn.direct_update_tensor
        outcomes = {}
        for arm in ("original", "on", "drop"):
            code = np.zeros(3, dtype=np.int64)
            residual = code.copy()
            gradient = np.array([1, -1, 0])
            if arm == "original":
                for _ in range(3):
                    code, residual, _ = original(code, residual, gradient, 31, 65)
            else:
                with update_adapter("srnn", arm, Counters()):
                    for _ in range(3):
                        code, residual, _ = srnn.direct_update_tensor(code, residual, gradient, 31, 65)
            outcomes[arm] = (code, residual)
        np.testing.assert_array_equal(outcomes["original"], outcomes["on"])
        np.testing.assert_array_equal(outcomes["on"][0], [-1, 1, 0])
        np.testing.assert_array_equal(outcomes["drop"], np.zeros((2, 3)))
        self.assertIs(srnn.direct_update_tensor, original)

    def test_fc_on_parity_and_drop(self):
        original = fc.Int8MagnitudeResidualOptimizer
        kwargs = dict(weight_scale=1.0, learning_rate=0.25, batch_count=1,
                      raw_gradient_fractional_bits=2)
        def exercise(cls):
            weight = torch.nn.Parameter(torch.zeros((1, 2)))
            optimizer = cls([weight], **kwargs)
            for _ in range(6):
                weight.grad = torch.ones_like(weight)
                optimizer.step(torch.tensor([[4, -4]]))
            return optimizer.codes[0].clone(), optimizer.residuals[0].clone()
        expected = exercise(original)
        with update_adapter("fc", "on", Counters()):
            actual = exercise(fc.Int8MagnitudeResidualOptimizer)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))
        with update_adapter("fc", "drop", Counters()):
            dropped = exercise(fc.Int8MagnitudeResidualOptimizer)
        self.assertEqual(sum(torch.count_nonzero(t).item() for t in dropped), 0)
        self.assertIs(fc.Int8MagnitudeResidualOptimizer, original)

    def test_commands_are_bounded_and_development_only(self):
        args = parse_args(["--topology", "scnn"])
        jobs = list(commands(args))
        self.assertEqual(len(jobs), 6)
        self.assertEqual([int(job[job.index("--seed") + 1]) for job in jobs],
                         [38472, 38472, 1729, 1729, 65537, 65537])
        self.assertFalse(args.run)


if __name__ == "__main__":
    unittest.main()
