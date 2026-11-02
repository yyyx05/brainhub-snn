"""Small CPU tests for the weight-only switch and streamed fixture decoding."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_scnn_tp_selected_layer_shift_adaptation as api
from scripts.supplement_scnn_weight_quantization import build_weight_arm, input_batches


class WeightOnlyTests(unittest.TestCase):
    def test_streamed_signed_decode_matches_original_reader(self):
        words = "0000\n0100\nffff\n8000\n7fff\n\n0080\n"
        path = SimpleNamespace(open=lambda *args, **kwargs: io.StringIO(words))
        original = api.read_signed_q8_8(path, 6)
        batches = list(input_batches(io.StringIO(words), 3, 2, (2,)))
        self.assertEqual([start for start, _ in batches], [0, 2])
        np.testing.assert_array_equal(np.concatenate([x for _, x in batches]).flatten(), original)

    def test_missing_and_extra_words_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            list(input_batches(io.StringIO("0001\n"), 1, 4, (2,)))
        with self.assertRaisesRegex(ValueError, "extra"):
            list(input_batches(io.StringIO("0001\n0002\n0003\n"), 1, 4, (2,)))

    def test_only_quantizer_changes_and_is_restored(self):
        effective = torch.tensor([-1.01, -0.000031, 0.000048, 0.4, 1.1])
        fake = SimpleNamespace(quantize_weight=api.quantize_weight)
        calls = []

        def build(*args, **kwargs):
            calls.append((args, kwargs))
            return fake.quantize_weight(effective, fractional_bits=kwargs["weight_fractional_bits"])

        fake.build_model = build
        fp = build_weight_arm(fake, Path("traces"), Path("original.pt"), arm="fp32", device="cpu")
        quantized = build_weight_arm(fake, Path("traces"), Path("original.pt"), arm="q015", device="cpu")
        self.assertTrue(torch.equal(fp, effective))
        self.assertTrue(torch.equal(quantized, api.quantize_weight(effective, fractional_bits=15)))
        self.assertEqual(calls[0], calls[1])
        self.assertIs(fake.quantize_weight, api.quantize_weight)


if __name__ == "__main__":
    unittest.main()
