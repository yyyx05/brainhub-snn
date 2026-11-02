"""Bounded local reference/generator tests, never compile RTL or launch SSH."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scnn_same_dut_b4t4 as runner
import supplement_scnn_same_dut_b4t4_reference as reference
from assess_scnn_same_dut_b4t4_options import sequential_sample


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.features, cls.numeric = reference.integer_features()
        cls.labels = reference.read_signed_hex(reference.LABELS, 10)[:4]
        cls.initial = reference.read_signed_hex(reference.WEIGHTS, reference.PARAMS)
        cls.cold = reference.window_update(cls.features, cls.labels, cls.initial, [0]*reference.PARAMS)
        cls.path = reference.trajectory(cls.features, cls.labels, 13)

    def test_current_integer_prefix(self):
        self.assertEqual(self.numeric["prefix"]["compared_values"], 3743810)
        self.assertEqual(self.labels, [8, 5, 3, 6])
        self.assertEqual([len(s) for s in self.features], [4]*4)

    def test_cold_window_does_not_cross(self):
        self.assertEqual(self.cold["counts"], [0, 20195, 20195])
        self.assertEqual(self.cold["max_abs_delta"], 2942)
        self.assertEqual(self.cold["codes"], self.initial)
        self.assertEqual(max(map(abs, self.cold["residuals"])), 2942)

    def test_real_natural_crossing_and_fresh_query(self):
        path = self.path
        self.assertEqual(path["first_code_crossing"], 12)
        self.assertEqual(path["first_forward_change"], 12)
        self.assertEqual([r["code_writes"] for r in path["windows"][:11]], [0]*11)
        row = path["windows"][11]
        self.assertEqual(row["code_writes"], 288)
        self.assertEqual(row["current_changes_from_pretrained"], 16)
        self.assertEqual(row["logit_changes_from_pretrained"], 16)
        path12 = reference.trajectory(self.features, self.labels, 12)
        self.assertEqual((path12["query_sample"], path12["query_current_changes"], path12["query_logit_changes"]), (0, 4, 4))

    def test_logits_are_recomputed_after_crossing(self):
        codes = self.initial[:]
        residuals = [0]*reference.PARAMS
        for _ in range(12):
            result = reference.window_update(self.features, self.labels, codes, residuals)
            codes, residuals = result["codes"], result["residuals"]
        updated = reference.window_update(self.features, self.labels, codes, residuals)
        self.assertNotEqual(updated["logits"], self.cold["logits"])
        self.assertEqual(updated["currents"][:4], reference.forward(self.features[0], codes)[0])

    def test_gradient_reduction_rounds_once_after_all_samples(self):
        gradients = [0]*reference.PARAMS
        for frames, label in zip(self.features, self.labels):
            logits = reference.forward(frames, self.initial)[1]
            part = reference.gradients_for_window(frames, logits, label)
            gradients = [a+b for a, b in zip(gradients, part)]
        expected = [reference.round_nearest_even_ratio(-g*640, 1 << 26) for g in gradients]
        self.assertEqual(expected, self.cold["residuals"])
        self.assertEqual(self.cold["rmw_mask"], [int(d != 0) for d in expected])

    def test_trace_resets_per_actual_sample(self):
        for sample in range(4):
            trace = [0]*reference.FEATURES
            for t, frame in enumerate(self.features[sample]):
                trace = [(old*250 >> 8)+value for old, value in zip(trace, frame)]
                start = (sample*4+t)*reference.FEATURES
                self.assertEqual(self.cold["traces"][start:start+reference.FEATURES], trace)

    def test_shape_and_bound_errors(self):
        with self.assertRaises(ValueError):
            reference.window_update(self.features[:3], self.labels, self.initial, [0]*reference.PARAMS)
        for windows in (0, 17):
            with self.assertRaises(ValueError):
                reference.trajectory(self.features, self.labels, windows)
        with self.assertRaises(ValueError):
            sequential_sample(self.features[0], self.labels[0], [5])


class GeneratorTests(unittest.TestCase):
    def test_b4t4_shape_and_single_dut(self):
        source = runner.testbench_source(12)
        self.assertEqual(source.count(") dut ("), 1)
        for expected in (".MAX_BATCH(16)", ".CLASSIFIER_CAPTURE_MAX_BATCH(4)",
                         ".i_learning_batch_count(5'd4)", ".i_learning_timestep_count(3'd4)",
                         "longint unsigned cycle_count", "B4_WINDOWS = 12;",
                         ".i_learning_vote_increment(16'd0)", ".i_learning_vote_threshold(16'h8000)",
                         ".i_parameter_load_residual(16'sd0)", "b4_check_store();"):
            self.assertIn(expected, source)
        self.assertNotIn("force ", source)
        self.assertNotIn("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS", source)

    def test_all_original_layer_checks_remain(self):
        source = runner.testbench_source(1)
        for layer in range(1, 9):
            self.assertIn(f"conv{layer}_context_count != (timestep_phase+1)*CONV{layer}_CONTEXTS", source)
            self.assertIn(f"conv{layer}_expected_pre", source)
            self.assertIn(f"conv{layer}_state_backing", source)
        self.assertIn("if (b4_t != 0) begin", source)
        self.assertIn("b4_query ? b4_query_currents", source)
        self.assertIn("cfg_initial_active_pair = !classifier_pair;", source)

    def test_no_dependency_on_running_b1t2_successor(self):
        paths = runner.dependencies()
        self.assertTrue(all(p.is_file() for p in paths))
        for name in ("run_scnn_same_dut.py", "supplement_scnn_same_dut_reference.py", "scnn_same_dut_checks.svh"):
            self.assertNotIn(name, [p.name for p in paths])
        self.assertIn("unified_scnn_tp_surrogate_type1_q8_8.mem", [p.name for p in paths])

    def test_build_only_low_parallelism_explicit_command(self):
        command = runner.build_command(Path("out"), "verilator", 2)
        self.assertEqual(command[:4], ["verilator", "--binary", "-j", "2"])
        self.assertNotIn("--trace", command)
        self.assertNotIn("ssh", command)

    def test_commit_wait_precedes_store_and_release(self):
        task = runner.testbench_source(12).split("task automatic b4_check_frame;", 1)[1].split("endtask", 1)[0]
        self.assertLess(task.index("while ((b4_commits != b4_window+1"), task.index("b4_check_store();"))
        self.assertLess(task.index("b4_check_store();"), task.index("b4_release = 1;"))
        self.assertIn("watchdog < 8000000", task)

    def test_missing_pass_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix="scnn_b4t4_test_") as tmp:
            output = Path(tmp)
            (output / "run.json").write_text(json.dumps(dict(status="fail", exit_code=1)), encoding="ascii")
            with self.assertRaises(ValueError):
                runner.summarize(output)

    def test_summary_does_not_upgrade_residual_only_to_crossing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix="scnn_b4t4_test_") as tmp:
            output = Path(tmp)
            (output / "fixture").mkdir()
            expected = dict(windows=[dict(rmw=20195, code_writes=0, residual_writes=20195)],
                            graph_frames=20, query_current_changes=0, query_logit_changes=0)
            (output / "fixture/expectations.json").write_text(json.dumps(expected), encoding="ascii")
            (output / "run.json").write_text(json.dumps(dict(status="pass", exit_code=0, wall_seconds=1)), encoding="ascii")
            metrics = "windows=1 frames=20 features=40960 currents=220 logits=220 traces=32768 rmw=20195 code_writes=0 residual_writes=20195 commits=1 publisher_words=3784 query_current_changes=0 query_logit_changes=0 cycles=100"
            (output / "run.log").write_text(runner.PASS + " " + metrics, encoding="ascii")
            summary = runner.summarize(output)
            self.assertFalse(summary["code_crossing_visible"])
            self.assertFalse(summary["final_composition_top"])
            self.assertIn("residual_only", summary["status"])


if __name__ == "__main__":
    unittest.main()
