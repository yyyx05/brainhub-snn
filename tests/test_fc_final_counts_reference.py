"""Integer boundary tests and independent scalar B16 fixture verification."""

from fractions import Fraction
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from scripts.build_fc_final_counts_fixture import (
    BATCH, CLASSES, DEFAULT_COUNTS, DELTA_MAX, HIDDEN, INPUTS, MEM_LAYOUT, OUTPUT,
    W2_BASE, apply_residual_update, batch_gradient, build_fixture, forward_step,
    load_counts, quantized_error, scale_gradient,
)
from scripts.build_fc_tp_b16_softmax_rtl_fixture import _exp_table, _quantized_row
from scripts.build_fc_w2_numeric_reference import (
    accumulate_logits, lif18, load_model, scale_raw_to_q6, signed_wrap, trace16, w2_dot,
)


def read_mem(path, bits, signed=False):
    with path.open(encoding="utf-8") as stream:
        values = np.fromiter((int(line.strip(), 16) for line in stream), dtype=np.int64)
    if signed:
        values = np.where(values >= (1 << (bits - 1)), values - (1 << bits), values)
    return values


class IntegerTests(unittest.TestCase):
    def test_scaler_exact_ties_signs_and_symmetric_saturation(self):
        values = np.array([0, 1, -1, (1 << 25) - 1, 1 << 25, (1 << 25) + 1,
                           3 << 25, -(1 << 25), -(3 << 25), (1 << 47) - 1, -(1 << 47)], dtype=np.int64)
        actual, saturated = scale_gradient(values)
        raw = [round(Fraction(-int(g) * 11, 1 << 26)) for g in values]
        expected = [min(DELTA_MAX, max(-DELTA_MAX, d)) for d in raw]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(saturated, [abs(d) > DELTA_MAX for d in raw])
        self.assertEqual(actual[4], -6)
        self.assertEqual(actual[6], -16)

    def test_residual_truncation_multiple_crossings_and_saturation(self):
        code = np.array([0, 0, 0, 0, 127, -127, 127, 126])
        residual = np.array([-1, 1, 0, 0, 32767, -32767, 3, 32767])
        delta = np.array([0, 0, -32769, 131071, 1, -1, 7, 2])
        next_code, next_residual, saturated = apply_residual_update(code, residual, delta)
        np.testing.assert_array_equal(next_code, [0, 0, -1, 3, 127, -127, 127, 127])
        np.testing.assert_array_equal(next_residual, [-1, 1, -1, 32767, 0, 0, 10, 1])
        np.testing.assert_array_equal(saturated, [False, False, False, False, True, True, False, False])

    def test_invalid_numeric_inputs(self):
        for value in (1.5, np.nan, 1 << 47, -(1 << 47) - 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scale_gradient(np.array([value]))
        for code, residual, delta in ((128, 0, 0), (-128, 0, 0), (0, -32768, 0),
                                       (0, 32768, 0), (0, 0, -131072)):
            with self.subTest(code=code, residual=residual, delta=delta), self.assertRaises(ValueError):
                apply_residual_update(np.array([code]), np.array([residual]), np.array([delta]))

    def test_softmax_lut_probability_error_and_no_row_renormalization(self):
        q6 = np.zeros((BATCH, CLASSES), dtype=np.int64)
        q6[1] = np.array([-32768, 32767, 0, 4, 8, 12, -4, -8, 32000, -32000])
        labels = np.arange(BATCH) % CLASSES
        probability, error = quantized_error(q6, labels)
        self.assertEqual(probability[0].tolist(), [6554] * CLASSES)
        self.assertEqual(int(probability[0].sum()), 65540)
        for sample in range(BATCH):
            _, expected = _quantized_row(q6[sample] / 64)
            np.testing.assert_array_equal(probability[sample], expected)
            expected[labels[sample]] -= 65536
            np.testing.assert_array_equal(error[sample], expected)

    def test_gradient_is_batch_sum_without_average(self):
        trace = np.full((BATCH, HIDDEN), 65535, dtype=np.int64)
        error = np.full((BATCH, CLASSES), 65536, dtype=np.int64)
        error[:, 0] *= -1
        gradient = batch_gradient(trace, error)
        self.assertEqual(int(gradient[0, 1]), 16 * 65535 * 65536)
        self.assertEqual(int(gradient[0, 0]), -16 * 65535 * 65536)

    def test_reused_softmax_lut_matches_production_constants(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'rtl/learning/unified_tp_row_softmax_q6.sv').read_text(encoding='utf-8')
        table = text.split('EXP_LUT [0:64]', 1)[1].split('};', 1)[0]
        values = [int(value) for value in re.findall(r"17'd(\d+)", table)]
        np.testing.assert_array_equal(_exp_table(), values)

    def test_updated_code_is_used_only_in_next_forward(self):
        # Synthetic boundary state only; real fixture initialization is zero.
        w0 = np.zeros((INPUTS, HIDDEN), dtype=np.int64)
        w0[0] = 1
        w2 = np.zeros((HIDDEN, CLASSES), dtype=np.int64)
        counts = np.zeros((BATCH, INPUTS), dtype=np.int64)
        counts[:, 0] = 16
        state = np.zeros((BATCH, HIDDEN), dtype=np.int64)
        logits = np.zeros((BATCH, CLASSES), dtype=np.int64)
        first = forward_step(counts, w0, w2, state, state, logits, 65536)
        self.assertTrue(np.all(first['dot'] == 0))
        _, error = quantized_error(first['q6'], np.zeros(BATCH, dtype=np.int64))
        delta, _ = scale_gradient(batch_gradient(first['trace'], error))
        residual = np.zeros_like(w2)
        residual[0, 0] = 32767
        updated, _, _ = apply_residual_update(w2, residual, delta)
        self.assertEqual(updated[0, 0], 1)
        second = forward_step(counts, w0, updated, first['voltage'], first['trace'], first['cumulative'], 65536)
        self.assertTrue(np.all(second['dot'][:, 0] == 1))
        self.assertTrue(np.all(second['cumulative'][:, 0] == 1))
        self.assertTrue(np.all(w2 == 0))

    def test_forward_counts_wrap_empty_frames_and_trace_match_scalar(self):
        model = load_model()
        counts = np.zeros((BATCH, INPUTS), dtype=np.int64)
        counts[0, 0] = 2
        counts[1, :70] = 1
        counts[2, 2311] = 70
        voltage = np.tile(np.array([-131072, -1, 0, 1024, 131071]), (BATCH, HIDDEN // 5))
        trace = np.tile(np.array([0, 1, 1024, 32768, 65535]), (BATCH, HIDDEN // 5))
        cumulative = np.full((BATCH, CLASSES), (1 << 23) - 1, dtype=np.int64)
        scale = round(float(model['W2_scale']) * 64 * 65536)
        result = forward_step(counts, model['W0'], model['W2'], voltage, trace, cumulative, scale)
        for sample in range(BATCH):
            active = np.flatnonzero(counts[sample])
            spikes = []
            for h in range(HIDDEN):
                raw = sum(int(counts[sample, f]) * int(model['W0'][f, h]) for f in active)
                pre, spike, post = lif18(signed_wrap(raw << 6, 18), int(voltage[sample, h]))
                self.assertEqual(result['pre'][sample, h], pre)
                self.assertEqual(result['voltage'][sample, h], post)
                self.assertEqual(result['trace'][sample, h], trace16(int(trace[sample, h]), spike, fractional_bits=10))
                spikes.append(spike)
            dot = w2_dot(model['W2'].tolist(), spikes)
            total = accumulate_logits(cumulative[sample].tolist(), dot)
            np.testing.assert_array_equal(result['dot'][sample], dot)
            np.testing.assert_array_equal(result['cumulative'][sample], total)
            np.testing.assert_array_equal(result['q6'][sample], [scale_raw_to_q6(v, scale) for v in total])

    def test_reject_long_source_even_for_short_debug_prefix(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix='test_input_') as directory:
            path = Path(directory) / 'input.npz'
            np.savez_compressed(path, dense_counts=np.zeros((16, 101, 2312), dtype=np.uint32),
                                labels=np.arange(16) % 10, lengths=np.full(16, 101), dataset_indices=np.arange(16))
            with self.assertRaisesRegex(ValueError, 'source T'):
                load_counts(path, 2)

    def test_reject_malformed_counts_and_padding(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix='test_input_') as directory:
            path = Path(directory) / 'input.npz'
            good = dict(dense_counts=np.zeros((16, 2, 2312), dtype=np.uint32),
                        labels=np.arange(16) % 10, lengths=np.full(16, 2), dataset_indices=np.arange(16))
            for field, value in (('dense_counts', np.zeros((15, 2, 2312), dtype=np.uint32)),
                                 ('dense_counts', np.full((16, 2, 2312), -1)),
                                 ('dense_counts', np.full((16, 2, 2312), 0.5)),
                                 ('labels', np.full(16, 10)), ('lengths', np.full(16, 3)),
                                 ('dataset_indices', np.zeros(16, dtype=np.int64))):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    np.savez_compressed(path, **{**good, field: value})
                    load_counts(path, 2)
            counts = good['dense_counts'].copy()
            counts[0, 1, 0] = 1
            np.savez_compressed(path, **{**good, 'dense_counts': counts, 'lengths': np.ones(16, dtype=np.int64)})
            with self.assertRaisesRegex(ValueError, 'padding'):
                load_counts(path, 1)
            np.savez_compressed(path, **good)
            for steps in (0, 3, True, 1.5):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    load_counts(path, steps)


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=OUTPUT, prefix='test_fixture_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.results = {arm: build_fixture(cls.base / arm, steps=2, learning=arm == 'on') for arm in ('on', 'off')}

    def test_independent_initial_parameters_and_off_zero_updates(self):
        model = load_model()
        expected = np.concatenate([model['W0'].ravel(), model['W2'].ravel()])
        for arm in ('on', 'off'):
            np.testing.assert_array_equal(read_mem(self.base / arm / 'parameters.mem', 16, True), expected)
            self.assertEqual(self.results[arm]['w2_parameter_base'], W2_BASE)
        off = self.results['off']
        self.assertEqual(off['learning_windows'], 0)
        self.assertEqual(off['totals']['update_requests'], 0)
        self.assertEqual(off['final_changed_w2'], 0)
        self.assertEqual(off['final_nonzero_residual'], 0)
        np.testing.assert_array_equal(read_mem(self.base / 'off/w2_after.mem', 16, True),
                                      np.tile(model['W2'].ravel(), 2))

    def test_time_major_counts_metadata_and_file_lengths(self):
        counts, labels, lengths, indices = load_counts(DEFAULT_COUNTS, 2)
        expected = counts[:, :2].transpose(1, 0, 2).reshape(-1)
        for arm in ('on', 'off'):
            directory = self.base / arm
            np.testing.assert_array_equal(read_mem(directory / 'input_counts.mem', 32), expected)
            np.testing.assert_array_equal(read_mem(directory / 'labels.mem', 8), labels)
            for name, spec in self.results[arm]['files'].items():
                self.assertEqual(read_mem(directory / name, spec['bits'], spec['signed']).size, spec['words'])
            for s, sample in enumerate(self.results[arm]['samples']):
                self.assertEqual(sample['dataset_index'], indices[s])
                self.assertEqual(sample['valid_timesteps'], lengths[s])
            self.assertEqual(json.loads((directory / 'reference.json').read_text(encoding='utf-8')), self.results[arm])

    def test_two_steps_against_independent_scalar_learning_reference(self):
        model = load_model()
        counts, labels, _, _ = load_counts(DEFAULT_COUNTS, 2)
        scale = round(float(model['W2_scale']) * 64 * 65536)
        for arm in ('on', 'off'):
            code = model['W2'].astype(np.int64).copy()
            residual = np.zeros((HIDDEN, CLASSES), dtype=np.int64)
            voltage = [[0] * HIDDEN for _ in range(BATCH)]
            trace = [[0] * HIDDEN for _ in range(BATCH)]
            cumulative = [[0] * CLASSES for _ in range(BATCH)]
            saved = {key: read_mem(self.base / arm / (key + '.mem'), spec[0], spec[1]).reshape(2, -1)
                     for key, spec in MEM_LAYOUT.items() if key != 'input_counts'}
            for t in range(2):
                expected = {key: [] for key in ('pre', 'trace', 'spike', 'dot', 'q6', 'cumulative')}
                error = []
                for s in range(BATCH):
                    active = np.flatnonzero(counts[s, t])
                    spikes = []
                    for h in range(HIDDEN):
                        current = sum(int(counts[s, t, f]) * int(model['W0'][f, h]) for f in active) << 6
                        pre, spike, voltage[s][h] = lif18(signed_wrap(current, 18), voltage[s][h])
                        trace[s][h] = trace16(trace[s][h], spike, fractional_bits=10)
                        spikes.append(spike)
                        expected['pre'].append(pre)
                    dot = w2_dot(code.tolist(), spikes)
                    cumulative[s] = accumulate_logits(cumulative[s], dot)
                    q6 = [scale_raw_to_q6(value, scale) for value in cumulative[s]]
                    _, probability = _quantized_row(np.array(q6) / 64)
                    probability[labels[s]] -= 65536
                    error.append(probability.tolist())
                    for key, values in (('trace', trace[s]), ('spike', spikes), ('dot', dot),
                                         ('q6', q6), ('cumulative', cumulative[s])):
                        expected[key].extend(values)
                gradient_nonzero = 0
                for h in range(HIDDEN):
                    for c in range(CLASSES):
                        g = sum(trace[s][h] * error[s][c] for s in range(BATCH))
                        gradient_nonzero += g != 0
                        if arm == 'on':
                            delta = min(DELTA_MAX, max(-DELTA_MAX, round(Fraction(-g * 11, 1 << 26))))
                            total = int(residual[h, c]) + delta
                            movement = (1 if total >= 0 else -1) * (abs(total) // 32768)
                            candidate = int(code[h, c]) + movement
                            code[h, c] = min(127, max(-127, candidate))
                            residual[h, c] = 0 if code[h, c] != candidate else total - movement * 32768
                expected['w2_after'] = code.ravel()
                expected['residual_after'] = residual.ravel()
                for key, values in expected.items():
                    np.testing.assert_array_equal(saved[key][t], values, err_msg=f'{arm} t={t} {key}')
                self.assertEqual(self.results[arm]['records'][t]['gradient_nonzero'], gradient_nonzero)

    def test_protect_existing_outputs_and_output_scope(self):
        with self.assertRaises(FileExistsError):
            build_fixture(self.base / 'on', 2, True)
        with self.assertRaises(ValueError):
            build_fixture(OUTPUT.parent / 'not_this_task', 2, True)


@unittest.skipUnless((OUTPUT / 't99_on/reference.json').is_file() and
                     (OUTPUT / 't99_off/reference.json').is_file(), 'full real fixtures not generated')
class SavedFullFixtureTests(unittest.TestCase):
    def test_full_time_major_inputs_and_independent_initial_parameters(self):
        counts, labels, _, _ = load_counts(DEFAULT_COUNTS, 99)
        expected = counts.transpose(1, 0, 2).reshape(-1)
        for arm in ('on', 'off'):
            directory = OUTPUT / f't99_{arm}'
            np.testing.assert_array_equal(read_mem(directory / 'input_counts.mem', 32), expected)
            np.testing.assert_array_equal(read_mem(directory / 'labels.mem', 8), labels)
        self.assertEqual((OUTPUT / 't99_on/parameters.mem').read_bytes(),
                         (OUTPUT / 't99_off/parameters.mem').read_bytes())

    def test_full_saved_gradients_and_residuals_with_scalar_normalization(self):
        model = load_model()
        _, labels, _, _ = load_counts(DEFAULT_COUNTS, 99)
        for arm in ('on', 'off'):
            directory = OUTPUT / f't99_{arm}'
            summary = json.loads((directory / 'reference.json').read_text(encoding='utf-8'))
            traces = read_mem(directory / 'trace.mem', 32).reshape(99, BATCH, HIDDEN)
            q6 = read_mem(directory / 'q6.mem', 16, True).reshape(99, BATCH, CLASSES)
            codes = read_mem(directory / 'w2_after.mem', 16, True).reshape(99, HIDDEN, CLASSES)
            residuals = read_mem(directory / 'residual_after.mem', 16, True).reshape(99, HIDDEN, CLASSES)
            code = model['W2'].astype(np.int64).copy()
            residual = np.zeros_like(code)
            for t in range(99):
                errors = []
                for s in range(BATCH):
                    _, probability = _quantized_row(q6[t, s] / 64)
                    probability[labels[s]] -= 65536
                    errors.append(probability)
                gradient = np.sum(traces[t, :, :, None] * np.array(errors)[:, None, :], axis=0)
                record = summary['records'][t]
                self.assertEqual(record['gradient_nonzero'], np.count_nonzero(gradient))
                self.assertEqual(record['gradient_abs_max'], np.abs(gradient).max())
                before_code, before_residual = code.copy(), residual.copy()
                for h in range(HIDDEN):
                    for c in range(CLASSES):
                        if arm == 'on':
                            delta = min(DELTA_MAX, max(-DELTA_MAX,
                                        round(Fraction(-int(gradient[h, c]) * 11, 1 << 26))))
                            total = int(residual[h, c]) + delta
                            movement = (1 if total >= 0 else -1) * (abs(total) // 32768)
                            candidate = int(code[h, c]) + movement
                            code[h, c] = min(127, max(-127, candidate))
                            residual[h, c] = 0 if code[h, c] != candidate else total - movement * 32768
                np.testing.assert_array_equal(codes[t], code)
                np.testing.assert_array_equal(residuals[t], residual)
                self.assertEqual(record['code_changes'], np.count_nonzero(code != before_code))
                self.assertEqual(record['residual_changes'], np.count_nonzero(residual != before_residual))
            self.assertGreater(summary['totals']['gradient_nonzero'], 0)


if __name__ == '__main__':
    unittest.main()
