"""Small Python checks only. No RTL compilation or emulated RTL evidence."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_scnn_graph_closure as runner
import supplement_scnn_graph_reference as ref


class GraphClosureTests(unittest.TestCase):
    def test_real_t2_reference_currents_and_logits(self):
        features = [ref.read_signed_hex(ref.REFERENCE / f"conv8_t{t}_expected_pooled_q8_8.hex", ref.FEATURES) for t in range(2)]
        currents, logits = ref.forward(features, ref.read_signed_hex(ref.WEIGHTS, ref.PARAMS))
        for t in range(2):
            self.assertEqual(currents[t], ref.read_signed_hex(ref.REFERENCE / f"classifier_t{t}_expected_current_q8_8.hex", ref.CLASSES))
            self.assertEqual(logits[t], ref.read_signed_hex(ref.REFERENCE / f"classifier_t{t}_expected_centered_cumulative_q8_8.hex", ref.CLASSES))
        label = int(ref.LABELS.read_text(encoding="ascii").splitlines()[0], 16)
        self.assertEqual(label, 8)
        gradients = ref.gradients_for_window(features, logits, label)
        weights = ref.read_signed_hex(ref.WEIGHTS, ref.PARAMS)
        codes, residuals = ref.apply_update(weights, [0] * ref.PARAMS, gradients)
        self.assertEqual(codes, weights)  # A real cold window need not cross a code LSB.
        self.assertGreater(sum(x != 0 for x in residuals), 0)

    def test_reference_reproduces_existing_b4t4_residuals(self):
        fixture = ROOT / "results/paper_evidence/rtl/scnn_q015_dataset_parity/b4t4_update_fixture"
        values = ref.read_signed_hex(fixture / "features_q8_8.hex", 4 * 4 * ref.FEATURES)
        logits = ref.read_signed_hex(fixture / "temporal_logits_q8_8.hex", 4 * 4 * ref.CLASSES)
        labels = ref.read_signed_hex(fixture / "labels.hex", 4)
        gradients = [0] * ref.PARAMS
        for sample in range(4):
            frames = [values[(sample * 4 + t) * ref.FEATURES:(sample * 4 + t + 1) * ref.FEATURES] for t in range(4)]
            rows = [logits[(sample * 4 + t) * ref.CLASSES:(sample * 4 + t + 1) * ref.CLASSES] for t in range(4)]
            partial = ref.gradients_for_window(frames, rows, labels[sample])
            gradients = [a + b for a, b in zip(gradients, partial)]
        codes, residuals = ref.apply_update([0] * ref.PARAMS, [0] * ref.PARAMS, gradients)
        self.assertEqual(codes, [0] * ref.PARAMS)
        self.assertEqual(residuals, ref.read_signed_hex(fixture / "expected_residual_q15.hex", ref.PARAMS))

    def test_signed_rounding_and_residual_normalization(self):
        self.assertEqual([ref.round_nearest_even_ratio(x, 2) for x in (1, 3, 5, -1, -3, -5)], [0, 2, 2, 0, -2, -2])
        # Each gradient maps to delta +/-5, exactly, with multiplier640.
        codes, residuals = ref.apply_update([3, -3, 32767], [32765, -32765, 32765], [-524288, 524288, -524288])
        self.assertEqual(codes, [4, -4, 32767])
        self.assertEqual(residuals, [2, -2, 0])

    def test_capture_parser_accepts_complete_reordered_coordinates(self):
        # A unit-test stream only; never written as a result or an RTL fixture.
        lines = ["kind,timestep,index,value\n"]
        for t in (1, 0):
            lines.extend(f"feature,{t},{i},{64 * (i % 5)}\n" for i in reversed(range(ref.FEATURES)))
            lines.extend(f"logit,{t},{i},{-i}\n" for i in range(ref.CLASSES))
        with patch.object(Path, "open", return_value=io.StringIO("".join(lines))):
            features, logits = ref.read_capture(Path("unit-test-only.csv"))
        self.assertEqual(features[0][:5], [0, 64, 128, 192, 256])
        self.assertEqual(logits[1], list(range(0, -11, -1)))

    def test_capture_rejects_missing_duplicate_and_invalid_values(self):
        for rows in ("feature,0,0,0\n", "feature,0,0,0\nfeature,0,0,64\n",
                     "feature,2,0,0\n", "feature,0,2048,0\n", "feature,0,0,1\n",
                     "logit,0,0,-32769\n", "other,0,0,0\n"):
            with self.subTest(rows=rows), patch.object(Path, "open", return_value=io.StringIO("kind,timestep,index,value\n" + rows)):
                with self.assertRaises(ValueError):
                    ref.read_capture(Path("unit-test-only.csv"))

    def test_missing_graph_pass_does_not_emit_fixture(self):
        with patch.object(Path, "read_text", return_value="build-only log"):
            with self.assertRaisesRegex(ValueError, "no terminal"):
                ref.make_fixture(Path("missing"), Path("build.log"), Path("unused"))

    def test_generated_learner_retains_numeric_and_transaction_assertions(self):
        source = runner.learner_source()
        self.assertIn("localparam int BATCH = 1;", source)
        self.assertIn("localparam int TIMESTEPS = 2;", source)
        self.assertIn(".i_cfg_vote_threshold(16'h8000)", source)
        self.assertIn("initial_residual_q15.hex", source)
        self.assertNotIn("after_11_windows", source)
        self.assertNotIn("b4t4_update_fixture/", source)
        for assertion in ("committed code mismatch", "committed residual mismatch",
                          "classifier current mismatch", "temporal logit mismatch",
                          "frame incomplete", "prime fill mismatch", "cold residual writes mismatch"):
            self.assertIn(assertion, source)
        self.assertIn("commit_sequence != 1", source)
        self.assertIn("perf[4].residual_writes != 0", source)
        self.assertIn(runner.LEARN_PASS, source)

    def test_dependencies_include_current_v3_and_both_reference_trees(self):
        paths = runner.dependencies()
        self.assertTrue(all(path.is_file() for path in paths))
        relative = [path.relative_to(ROOT).as_posix() for path in paths]
        self.assertIn("rtl/tiles/unified_dual_role_lif10_arithmetic_candidate_v3.sv", relative)
        self.assertTrue(any("full_frame_t1_reference/" in path for path in relative))
        self.assertTrue(any("t2_reference/" in path for path in relative))
        wrapper = (ROOT / "tb/tb_supplement_scnn_graph_capture.sv").read_text()
        self.assertIn("u_classifier_tail.u_scheduler.i_feature_wave_valid", wrapper)
        self.assertNotIn("if (graph.dut.o_feature_wave_fire)", wrapper)
        self.assertNotIn("$readmemh", wrapper)

    def test_failed_process_cannot_be_reported_as_pass(self):
        runner.BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="unit_", dir=runner.BASE) as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                runner.run_logged([sys.executable, "-c", "raise SystemExit(3)"], output, "failure")
            record = json.loads((output / "failure.json").read_text())
            self.assertEqual(record["exit_code"], 3)
            self.assertEqual(record["status"], "failed_or_interrupted")

    def test_optional_activity_source_reaches_both_build_commands(self):
        monitor = ROOT / "tb/tb_supplement_scnn_graph_capture.sv"
        for stage in ("graph", "learner"):
            with patch.object(runner, "run_logged") as run:
                runner.build(runner.BASE / "unit_not_built", stage, "verilator", 4, monitor)
                command = run.call_args.args[0]
                self.assertEqual(command[-1], str(monitor.resolve()))
                self.assertEqual(command[command.index("-j") + 1], "4")


if __name__ == "__main__":
    unittest.main()
