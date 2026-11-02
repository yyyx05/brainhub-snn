"""Focused tests for the full formal Test numerical evaluation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_fc_formal_test_numeric import (
    C, F, H, OUTPUT, check_scalar, estimate_full_seconds, integer_step, inventory,
    metrics_for, run_batch, to_q6, validate_counts,
)
from scripts.build_fc_w2_numeric_reference import accumulate_logits, lif18, load_model, scale_raw_to_q6, trace16, w2_dot


class FormalTestNumericTests(unittest.TestCase):
    def test_counts_repeated_over49_and_empty_frames_against_scalar(self):
        model = load_model()
        frames = np.zeros((3, 7, F), dtype=np.uint32)
        frames[0, 0, 17] = 3
        frames[0, 2, :70] = 1
        frames[1, 1, 2311] = 70
        frames[1, 4, 500:570] = 2
        result = run_batch(frames, model, capture=range(3))
        for sample in range(3):
            self.assertEqual(check_scalar(frames[sample], model, result['history'][sample])['mismatches'], 0)
        self.assertTrue(np.all(result['logits'][0][2] == 0))

    def test_integer_boundaries_wrap_rounding_trace_and_cumulative(self):
        model = load_model()
        raw = np.resize(np.array([-8193, -2049, -2048, -1, 0, 16, 2048, 8193], dtype=np.int32), (2, H))
        voltage = np.resize(np.array([-131072, -1, 0, 1024, 131071], dtype=np.int32), (2, H))
        trace = np.full((2, H), 65535, dtype=np.int64)
        logits = np.resize(np.array([-(1 << 23), (1 << 23) - 1], dtype=np.int32), (2, C))
        result = integer_step(raw, model['W2'].astype(np.int64), voltage, trace, logits, 47303)
        for s in range(2):
            spikes = []
            for h in range(H):
                pre, spike, post = lif18(int(raw[s, h]) * 64, int(voltage[s, h]))
                self.assertEqual(result['pre'][s, h], pre)
                self.assertEqual(result['voltage'][s, h], post)
                self.assertEqual(result['trace'][s, h], trace16(int(trace[s, h]), spike, fractional_bits=10))
                spikes.append(spike)
            dot = w2_dot(model['W2'].tolist(), spikes)
            cumulative = accumulate_logits(logits[s].tolist(), dot)
            np.testing.assert_array_equal(result['logits'][s], cumulative)
        self.assertGreater(result['overflow']['current_wrap18'].sum(), 0)
        # At decay64225, even the largest legal trace plus one spike stays below 65535.
        self.assertEqual(result['overflow']['trace_saturation16'].sum(), 0)
        self.assertLessEqual(int(result['trace'].max()), 65248)
        self.assertGreater(result['overflow']['q6_saturation16'].sum(), 0)

    def test_q6_rne_signed_ties_and_saturation(self):
        raw = np.array([[-32768, -5, -3, -1, 0, 1, 3, 5, 32767, (1 << 23) - 1]], dtype=np.int32)
        for scale in (32768, 47303, 65536):
            actual, _ = to_q6(raw, scale)
            expected = [[scale_raw_to_q6(int(v), scale) for v in raw[0]]]
            np.testing.assert_array_equal(actual, expected)

    def test_padding_is_processed_and_samples_are_independent(self):
        model = load_model()
        frames = np.zeros((2, 5, F), dtype=np.uint32)
        frames[0, 0, :100] = 1
        frames[1, 3, :60] = 2
        batch = run_batch(frames, model)
        for s in range(2):
            single = run_batch(frames[s:s + 1], model)
            for arm in range(4):
                np.testing.assert_allclose(batch['logits'][arm][s], single['logits'][arm][0], rtol=1e-5, atol=1e-5)

    def test_invalid_counts(self):
        for value in (-1, .5, np.nan, np.inf, 32768, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_counts(np.full((1, 1, F), value))
        for shape in ((1, F), (1, 0, F), (1, 1, 10)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                validate_counts(np.zeros(shape))

    def test_metrics_correct_class_order_and_ties(self):
        logits = np.zeros((3, 10))
        logits[1, 9] = 1
        logits[2, 2:4] = 1
        result = metrics_for(np.array([0, 9, 3]), logits)
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['argmax_ties'], 2)
        self.assertEqual(result['confusion_true_rows'][3][2], 1)

    def test_two_hour_estimate_uses_first64_and_margin(self):
        self.assertEqual(estimate_full_seconds(2), 628)
        self.assertGreater(estimate_full_seconds(30), 7200)

    def test_missing_data_does_not_download(self):
        import tonic.dataset
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix='test_missing_') as directory:
            with patch.object(tonic.dataset, 'download_and_extract_archive') as download:
                with self.assertRaisesRegex(RuntimeError, 'download/extraction disabled'):
                    inventory(Path(directory), Path(directory))
                download.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
