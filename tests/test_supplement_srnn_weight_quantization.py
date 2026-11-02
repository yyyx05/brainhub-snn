import unittest
from types import SimpleNamespace

import torch

from scripts.supplement_srnn_weight_quantization import apply_arm, compare_arms, q14_effective


class WeightQuantizationTests(unittest.TestCase):
    def test_round_to_even_and_saturate(self):
        value = torch.tensor([-40000., -2.5, -1.5, 0.5, 1.5, 2.5, 40000.]) / 16384
        quantized, stats = q14_effective(value)
        self.assertEqual((quantized * 16384).tolist(), [-32768., -2., -2., 0., 2., 2., 32767.])
        self.assertEqual(stats["clipped_elements"], 2)

    def test_only_m2_weights_change(self):
        parameters = {name: torch.tensor([0.123456]) for name in
                      ("network.0.fc.weight", "network.0.rc.weight", "network.1.fc.weight", "network.2.fc.weight")}
        original = {name: value.clone() for name, value in parameters.items()}
        model = SimpleNamespace(named_parameters=lambda: parameters.items())
        apply_arm(model, "fp32")
        self.assertTrue(all(torch.equal(value, original[name]) for name, value in parameters.items()))
        apply_arm(model, "m2_q2_14")
        for name in ("network.0.fc.weight", "network.0.rc.weight"):
            self.assertTrue(torch.equal(parameters[name], original[name]))
        self.assertFalse(torch.equal(parameters["network.2.fc.weight"], original["network.2.fc.weight"]))

    def test_comparison_checks_identity_and_direction(self):
        fp = dict(labels=[0, 1, 2], predictions=[0, 0, 2], samples=3,
                  padded_timesteps_per_batch=[2], accuracy_percent=200/3)
        quant = dict(fp, predictions=[1, 1, 2])
        result = compare_arms(fp, quant)
        self.assertEqual(result["predictions_changed"], 2)
        self.assertEqual(result["correct_to_wrong"], 1)
        self.assertEqual(result["wrong_to_correct"], 1)
        with self.assertRaises(ValueError):
            compare_arms(fp, dict(quant, labels=[1, 0, 2]))
        with self.assertRaises(ValueError):
            compare_arms(fp, dict(quant, padded_timesteps_per_batch=[3]))


if __name__ == "__main__":
    unittest.main()
