"""Small CPU tensors and mocked data only; never open the formal package."""

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_srnn_magnitude_sgd as a3
import compare_srnn_shd_shifted_optimizers as srnn

torch.set_num_threads(1)


def q40(values):
    return np.rint(np.asarray(values) * a3.GRADIENT_SCALE).astype(np.int64)


def selection_rows():
    return [dict(method="magnitude_sgd", learning_rate=r,
                 clean_accuracy_change=0.0, shifted_accuracy_improvement=0.01,
                 update_regime=dict(zero_code_updates=False, all_saturated=False))
            for r in a3.RATES]


class TinyModel(torch.nn.Module):
    """A test double for the existing adaptation loop, not a model implementation."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.ModuleList([
            torch.nn.Module(), torch.nn.Module(), torch.nn.Module()])
        for layer in self.network:
            layer.fc = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                layer.fc.weight.copy_(torch.eye(2))
        self.resets = 0
        self.detaches = 0
        self.seen = []
        self.trace = []
        self.mem = None

    def reset_potential(self):
        self.resets += 1
        self.trace = []
        self.mem = None

    def forward(self, x, target):
        previous = self.trace[-1] if self.trace else torch.zeros_like(x)
        self.trace = [previous * 0.5 + x]
        current = x @ self.network[-1].fc.weight.T
        self.mem = current if self.mem is None else self.mem + current
        self.e = torch.softmax(self.mem, dim=1) - target
        self.seen.append((self.trace[-1].detach().clone(), self.network[-1].fc.weight.detach().clone()))
        return self.mem, self.mem

    def detach_membrane_states(self):
        self.detaches += 1
        self.trace = [self.trace[-1].detach()]
        self.mem = self.mem.detach()


class MagnitudeTests(unittest.TestCase):
    def test_amplitude_not_vote_rate(self):
        zeros = np.zeros(2, dtype=np.int64)
        gradient = q40([1.0, 4.0])
        code, residual, stats = a3.magnitude_update_tensor(zeros, zeros, gradient, 1 / a3.CODE_SCALE)
        np.testing.assert_array_equal(code, [-1, -4])
        np.testing.assert_array_equal(residual, [0, 0])
        self.assertEqual(stats["code_writes"], 2)
        direct, votes, _ = srnn.direct_update_tensor(zeros, zeros, gradient, 31, 65)
        np.testing.assert_array_equal(direct, [0, 0])
        np.testing.assert_array_equal(votes, [-31, -31])

    def test_fraction_accumulates_and_negative_crossings_truncate(self):
        code = np.zeros(2, dtype=np.int64)
        residual = code.copy()
        for _ in range(3):
            code, residual, _ = a3.magnitude_update_tensor(
                code, residual, q40([1.0, -1.0]), 0.375 / a3.CODE_SCALE)
        np.testing.assert_array_equal(code, [-1, 1])
        np.testing.assert_array_equal(residual, [-a3.RESIDUAL_SCALE // 8, a3.RESIDUAL_SCALE // 8])
        code2, residual2, _ = a3.magnitude_update_tensor(
            code, residual, q40([-1.0, 1.0]), 0.125 / a3.CODE_SCALE)
        np.testing.assert_array_equal(code2, code)
        np.testing.assert_array_equal(residual2, [0, 0])

    def test_round_nearest_even_at_sub_lsb_quantization(self):
        zeros = np.zeros(4, dtype=np.int64)
        _, residual, _ = a3.magnitude_update_tensor(zeros, zeros, q40([0.5, 1.5, -0.5, -1.5]),
                                                  1 / (a3.CODE_SCALE * a3.RESIDUAL_SCALE))
        np.testing.assert_array_equal(residual, [0, -2, 0, 2])

    def test_zero_gradient_preserves_state_and_rails(self):
        code = np.array([-32768, 32767, 2])
        residual = np.array([-100, 200, 5])
        actual, remainder, stats = a3.magnitude_update_tensor(code, residual, q40([0, 0, 0]), 1e-4)
        np.testing.assert_array_equal(actual, code)
        np.testing.assert_array_equal(remainder, residual)
        self.assertEqual(stats["zero_skips"], 3)
        self.assertEqual(stats["saturations"], 0)

    def test_saturation_clears_remainder_and_multi_lsb_survives(self):
        code = np.array([32767, -32768, 0])
        updated, residual, stats = a3.magnitude_update_tensor(
            code, np.zeros(3, dtype=np.int64), q40([-1, 1, 1]), 2.5 / a3.CODE_SCALE)
        np.testing.assert_array_equal(updated, [32767, -32768, -2])
        np.testing.assert_array_equal(residual, [0, 0, -a3.RESIDUAL_SCALE // 2])
        self.assertEqual(stats["saturations"], 2)
        self.assertEqual(stats["proposed_code_changes"], 3)

    def test_sub_lsb_integer_conservation_across_updates(self):
        rng = np.random.default_rng(32)
        code = np.zeros((2, 3), dtype=np.int64)
        residual, total = code.copy(), code.copy()
        for _ in range(100):
            gradient = rng.integers(-1000000000000, 1000000000000, size=code.shape)
            total += np.rint(-gradient.astype(np.float64) * 1e-4 / 4).astype(np.int64)
            code, residual, _ = a3.magnitude_update_tensor(code, residual, gradient, 1e-4)
            np.testing.assert_array_equal(code * a3.RESIDUAL_SCALE + residual, total)
            self.assertTrue(np.all(np.abs(residual) < a3.RESIDUAL_SCALE))

    def test_invalid_and_overflow_inputs(self):
        z = np.zeros(1, dtype=np.int64)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                a3.magnitude_update_tensor(z, z, z, rate)
        with self.assertRaises(ValueError):
            a3.magnitude_update_tensor(z, z, np.zeros(1), 1e-4)
        with self.assertRaises(ValueError):
            a3.magnitude_update_tensor(z, np.array([a3.RESIDUAL_SCALE]), z, 1e-4)
        with self.assertRaises(OverflowError):
            a3.magnitude_update_tensor(z, z, q40([1]), 1e20)

    def test_vote_observer_exact_parity_and_patch_restoration(self):
        original = srnn.direct_update_tensor
        rng = np.random.default_rng(12)
        code = np.array([0, 32767, -32768, 100])
        residual = np.zeros_like(code)
        expected_code, expected_residual = code.copy(), residual.copy()
        with a3.update_adapter(srnn, "direct_response") as observer:
            for _ in range(100):
                g = rng.integers(-100, 100, size=4)
                expected_code, expected_residual, expected_stats = original(
                    expected_code, expected_residual, g, 31, 65)
                code, residual, stats = srnn.direct_update_tensor(code, residual, g, 31, 65)
                np.testing.assert_array_equal(code, expected_code)
                np.testing.assert_array_equal(residual, expected_residual)
                self.assertEqual(stats, expected_stats)
        self.assertIs(srnn.direct_update_tensor, original)
        self.assertEqual(observer.totals["update_calls"], 100)
        with self.assertRaisesRegex(RuntimeError, "test"):
            with a3.update_adapter(srnn, "magnitude_sgd", 1e-4):
                raise RuntimeError("test")
        self.assertIs(srnn.direct_update_tensor, original)

    def test_existing_loop_gradient_batch_division_state_and_freeze(self):
        model = TinyModel()
        state = srnn.direct_initial_state(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        data = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]],
                             [[0., 1.], [1., 0.], [1., 1.]]])
        labels = torch.tensor([1, 0])
        gradients = []
        residual_inputs = []
        original = a3.magnitude_update_tensor

        def observed(code, residual, gradient, rate):
            expected = torch.round((model.e.T @ model.trace[-1] / 2) * a3.GRADIENT_SCALE).long().detach().numpy()
            np.testing.assert_array_equal(gradient, expected)
            gradients.append(gradient.copy())
            residual_inputs.append(residual.copy())
            return original(code, residual, gradient, rate)

        arch = SimpleNamespace(full_local_loss=lambda model, _: model.trace[-1].sum() * 0)
        with patch.dict(sys.modules, architecture_search=arch), patch.object(a3, "magnitude_update_tensor", observed):
            with a3.update_adapter(srnn, "magnitude_sgd", 4 / a3.CODE_SCALE) as counter:
                for _ in range(2):
                    srnn.direct_adapt_batch(model, data, labels, {"output_size": 2},
                                            state, a3.CANDIDATE, "wout")
        self.assertEqual(model.resets, 2)
        self.assertEqual(model.detaches, 6)
        self.assertEqual(len(gradients), 6)
        self.assertEqual(counter.totals["update_calls"], 6)
        self.assertGreater(counter.totals["code_changes"], 0)
        self.assertGreater(np.count_nonzero(residual_inputs[3]), 0)
        torch.testing.assert_close(model.seen[1][0], data[:, 0] * 0.5 + data[:, 1])
        torch.testing.assert_close(model.seen[3][0], data[:, 0])
        self.assertFalse(torch.equal(model.seen[0][1], model.seen[1][1]))
        changes = a3.parameter_changes(model, before)
        self.assertEqual(changes["network.0.fc.weight"]["nonzero"], 0)
        self.assertEqual(changes["network.1.fc.weight"]["nonzero"], 0)
        self.assertGreater(changes["network.2.fc.weight"]["nonzero"], 0)


class ProtocolTests(unittest.TestCase):
    def test_validation_split_matches_existing_shuffle_and_report_uses_all_test(self):
        indices = list(range(1000, 1816))
        order = np.asarray(indices)
        np.random.default_rng(a3.DEV_SEED).shuffle(order)
        support, query = a3.episode_indices(indices, a3.DEV_SEED, "pilot")
        self.assertEqual(support, order[:128].tolist())
        self.assertEqual(query, order[128:].tolist())
        self.assertFalse(set(support) & set(query))
        support2, query2 = a3.episode_indices(indices, a3.DEV_SEED, "report")
        self.assertEqual(support2, support)
        self.assertEqual(query2, list(range(2264)))
        with self.assertRaises(ValueError):
            a3.episode_indices(indices, a3.DEV_SEED, "report", 64)

    def test_selection_ties_clean_budget_and_negative_results(self):
        rows = selection_rows()
        self.assertEqual(a3.select_rate(rows)["selected_rate"], 3e-5)
        rows[1]["clean_accuracy_change"] = 0.001
        self.assertEqual(a3.select_rate(rows)["selected_rate"], 1e-4)
        rows[2]["shifted_accuracy_improvement"] = 0.05
        rows[2]["clean_accuracy_change"] = -0.011
        self.assertEqual(a3.select_rate(rows)["selected_rate"], 1e-4)
        for row in rows:
            row["shifted_accuracy_improvement"] = -0.01
        self.assertFalse(a3.select_rate(rows)["selected_gain_positive"])

    def test_no_selection_when_all_zero_saturated_or_clean_fails(self):
        for reason in ("zero_code_updates", "all_saturated", "clean_loss"):
            rows = selection_rows()
            for row in rows:
                if reason == "clean_loss":
                    row["clean_accuracy_change"] = -0.02
                else:
                    row["update_regime"][reason] = True
            self.assertEqual(a3.select_rate(rows)["status"], "inconclusive")
            self.assertIsNone(a3.select_rate(rows)["selected_rate"])
        rows = selection_rows()
        rows[0]["update_regime"]["zero_code_updates"] = True
        rows[1]["update_regime"]["all_saturated"] = True
        self.assertEqual(a3.select_rate(rows)["selected_rate"], 3e-4)

    def test_saturation_regime_counts_attempts_not_gradient_zeros(self):
        regime = a3.update_regime(dict(proposed_code_changes=3, saturations=3, code_changes=1), np.array([100]))
        self.assertTrue(regime["all_saturated"])
        self.assertFalse(regime["zero_code_updates"])
        regime = a3.update_regime(dict(proposed_code_changes=3, saturations=2, code_changes=1), np.array([100]))
        self.assertFalse(regime["all_saturated"])

    def test_report_requires_validation_selection_and_identical_device(self):
        args = a3.parse_args(["--phase", "report", "--pilot-result", "unused.json"])
        rows = selection_rows()
        pilot = dict(status="completed", phase="pilot", episodes=rows, selection=a3.select_rate(rows),
                     protocol=dict(dataset_partition="validation", device="cpu", threads=1),
                     inputs=dict(handoff=str(args.handoff), dataset_root=str(args.dataset_root)))
        self.assertEqual(a3.report_rate(pilot, args), 3e-5)
        for key, value in (("dataset_partition", "official_test"), ("device", "cuda"), ("threads", 2)):
            modified = copy.deepcopy(pilot)
            modified["protocol"][key] = value
            with self.assertRaises(ValueError):
                a3.report_rate(modified, args)
        modified = copy.deepcopy(pilot)
        modified["selection"]["selected_rate"] = 1e-4
        with self.assertRaises(ValueError):
            a3.report_rate(modified, args)

    def test_default_dry_run_never_loads_data_or_model(self):
        stream = io.StringIO()
        with patch.object(a3, "load_context", side_effect=AssertionError("must not load")), redirect_stdout(stream):
            self.assertEqual(a3.main([]), 0)
        output = json.loads(stream.getvalue())
        self.assertEqual(output["status"], "dry_run")
        self.assertEqual(len(output["jobs"]), 4)
        self.assertEqual(output["protocol"]["state_bytes"]["sgd_total"], 30720)
        self.assertFalse(Path(output["command"].split("--output-dir", 1)[1].split("--run")[0].strip()).exists())

    def test_report_dry_run_only_pairs_selected_rate_and_fixed_seeds(self):
        args = a3.parse_args(["--phase", "report", "--pilot-result", "unused.json"])
        rows = selection_rows()
        pilot = dict(status="completed", phase="pilot", episodes=rows, selection=a3.select_rate(rows),
                     protocol=dict(dataset_partition="validation", device="cpu", threads=1),
                     inputs=dict(handoff=str(args.handoff), dataset_root=str(args.dataset_root)))
        stream = io.StringIO()
        with patch.object(Path, "read_text", return_value=json.dumps(pilot)), redirect_stdout(stream), \
                patch.object(a3, "load_context", side_effect=AssertionError("dry run must not load")):
            a3.main(["--phase", "report", "--pilot-result", "unused.json"])
        output = json.loads(stream.getvalue())
        self.assertEqual(output["jobs"], [[seed, method, 3e-5 if method == "magnitude_sgd" else None]
                                         for seed in a3.REPORT_SEEDS for method in a3.METHODS])
        self.assertEqual(output["protocol"]["query_samples"], 2264)
        self.assertEqual(output["protocol"]["dataset_partition"], "train_support_official_test")

    def test_paired_report_rejects_mismatched_inputs_and_initialization(self):
        rows = [dict(seed=seed, method=method, support_indices=[seed], query_indices=[0, 1], device="cpu",
                     evaluations=[dict(clean=1, shifted=0)],
                     shifted_accuracy_improvement=0.01, clean_accuracy_change=0.0)
                for seed in a3.REPORT_SEEDS for method in a3.METHODS]
        self.assertEqual(len(a3.paired_summary(rows)["pairs"]), 3)
        for key, value in (("support_indices", [0]), ("evaluations", [dict(clean=0, shifted=0)])):
            modified = copy.deepcopy(rows)
            modified[1][key] = value
            with self.assertRaises(ValueError):
                a3.paired_summary(modified)

    def test_rate_screen_cannot_expand_silently(self):
        rows = selection_rows()
        rows.append(dict(rows[-1], learning_rate=1e-3))
        with self.assertRaises(ValueError):
            a3.select_rate(rows)


if __name__ == "__main__":
    unittest.main()
