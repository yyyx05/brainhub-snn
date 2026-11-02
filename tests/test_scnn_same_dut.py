"""Lightweight preparation/reference tests. These never invoke an RTL tool."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scnn_same_dut as runner
import supplement_scnn_same_dut_reference as reference


class ScnnSameDutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = reference.expectations(True)
        cls.source = runner.testbench_source()

    def temporary(self):
        # Test-generated files remain in this workspace and are automatically
        # removed. No accepted evidence or production files are touched.
        return tempfile.TemporaryDirectory(prefix="scnn_same_dut_unit_", dir=ROOT / "work")

    def test_cold_reference_matches_existing_independent_graph(self):
        self.assertEqual(self.expected["label"], 8)
        self.assertEqual(self.expected["counts"], [0, 16391, 16391])
        self.assertEqual(self.expected["codes"], self.expected["weights"])
        self.assertEqual(self.expected["currents"][0], self.expected["currents"][2])
        self.assertEqual(self.expected["logits"][0], self.expected["logits"][2])
        self.assertNotEqual(self.expected["features"][0], self.expected["features"][1])
        self.assertEqual(len(self.expected["codes"]), 2048 * 11)

    def test_trace_is_temporal_and_not_raw_feature_replay(self):
        e = self.expected
        self.assertEqual(e["traces"][:2048], e["features"][0])
        self.assertEqual(e["traces"][2048:], [(a * 250 >> 8) + b for a, b in zip(*e["features"][:2])])
        self.assertTrue(any(x > 256 for x in e["traces"][2048:]))

    def test_off_control_has_identical_input_but_zero_learning(self):
        off = reference.expectations(False)
        self.assertEqual(off["features"], self.expected["features"])
        self.assertEqual(off["weights"], self.expected["weights"])
        self.assertEqual(off["counts"], [0, 0, 0])
        self.assertEqual(off["residuals"], [0] * 22528)

    def test_fixture_is_expected_data_not_synthetic_DUT_capture(self):
        with self.temporary() as directory:
            fixture = Path(directory) / "expected"
            report = reference.write_fixture(fixture, True)
            self.assertEqual(report["status"], "independent_expectations_only_not_RTL_PASS")
            self.assertEqual(report["multiplier"], 640)
            self.assertEqual(reference.read_signed_hex(fixture / "expected_residuals.hex", 22528), self.expected["residuals"])
            self.assertFalse((fixture / "graph_capture.csv").exists())

    def test_single_DUT_no_force_or_second_learner(self):
        self.assertEqual(self.source.count("unified_scnn_h40_array16_graph_online_learning_system_v1 #("), 1)
        self.assertNotRegex(self.source, r"\bforce\s+")
        self.assertNotIn("unified_scnn_classifier_online_cache_learning_client_v1 #(", self.source)
        self.assertEqual(self.source.count("rst_n = 1'b1;"), 1)
        self.assertEqual(self.source.count("load_all_classifier_parameters();"), 1)
        self.assertEqual(self.source.count("prime_classifier_cache();"), 1)

    def test_retains_every_original_fatal_check(self):
        original = runner.GRAPH_TB.read_text(encoding="utf-8-sig")
        for message in re.findall(r'\$fatal\(1,\s*"([^"]+)"', original):
            self.assertIn(message, self.source)
        for layer in range(1, 9):
            self.assertIn(f"conv{layer}_request_count != (timestep_phase+1)*CONV{layer}_REQUESTS", self.source)
        self.assertIn("else if (!dut.prescan_busy)", self.source)
        self.assertIn("sd_prescan_requests != sd_learn_on*4096", self.source)

    def test_public_controls_and_replay_version_are_dynamic(self):
        for fragment in (".i_learning_enable(sd_learning)", ".i_learning_vote_threshold(16'h8000)",
                         ".i_learning_vote_increment(16'd0)", "cfg_initial_active_pair = !classifier_pair;",
                         "context_active_pair != classifier_pair", "classifier_pair != !(sd_learn_on && timestep_phase >= 2)",
                         "load_reference_frame(timestep_phase == 2 ? 0 : timestep_phase)",
                         "if (timestep_phase == 1) begin", ".i_sample_reset(timestep_phase != 1)",
                         "timestep_phase < 3;", "[0:3*CONV1_SCALARS-1]"):
            self.assertIn(fragment, self.source)
        self.assertNotIn("cfg_initial_active_pair = 1'b0;", self.source)
        self.assertNotIn("learning_off=1", self.source)
        self.assertNotIn("T2_SAME_ARRAY_PASS", self.source)
        self.assertIn(".g_bank[2].memory[0] :", self.source)

    def test_full_numeric_and_commit_checks_present(self):
        for fragment in ("same-DUT classifier current mismatch", "same-DUT feature mismatch", "classifier cumulative logit mismatch",
                         "same-DUT trace mismatch", "same-DUT RMW mismatch", "same-DUT premature/non-atomic commit",
                         "same-DUT committed store mismatch", "same-DUT publication value", "same-DUT learner did not detach",
                         "same-DUT next forward reloaded", "sd_check_store();", "sd_rmw_seen[address] != (sd_rmw_mask[address] != 0)"):
            self.assertIn(fragment, self.source)

    def test_required_files_exist_and_are_in_workspace(self):
        paths = runner.dependencies()
        self.assertTrue(all(path.is_file() and path.is_relative_to(ROOT) for path in paths))
        self.assertIn(ROOT / "rtl/top/unified_scnn_h40_array16_graph_online_learning_system_v1.sv", paths)
        self.assertIn(ROOT / "tests/scnn_same_dut_checks.svh", paths)
        self.assertIn(ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem", paths)

    def test_r2_graph_done_does_not_assume_last_frame_commit(self):
        begin = self.source.index("        if (conv8_context_count !=")
        end = self.source.index("        sd_check_frame();", begin)
        check = self.source[begin:end]
        self.assertIn("(timestep_phase != 1 || !sd_learn_on)", check)
        self.assertIn("classifier_pair != !(sd_learn_on && timestep_phase >= 2)", check)
        self.assertIn("classifier_version != CLASSIFIER_VERSION + 16'(sd_learn_on && timestep_phase >= 2)", check)
        self.assertNotIn("sd_learn_on && timestep_phase >= 1", check)
        # Phase1 may be in the idle-before-hold gap or already committed.
        # Its strict publication/version check is the commit monitor instead.
        for learning in (False, True):
            for phase in range(3):
                enabled = phase != 1 or not learning
                expected_pair = not (learning and phase >= 2)
                expected_version_delta = int(learning and phase >= 2)
                if phase == 1 and learning:
                    self.assertFalse(enabled)
                else:
                    self.assertEqual(expected_pair, expected_version_delta == 0)

    def test_r2_waits_for_commit_before_store_check_and_detach(self):
        task = self.source.split("task automatic sd_check_frame;", 1)[1].split("endtask", 1)[0]
        wait = task.index("while ((sd_commits != 1 || dut.o_learning_commit_sequence != 1")
        self.assertLess(wait, task.index("sd_check_store();"))
        self.assertLess(task.index("sd_check_store();"), task.index("sd_release = 1;"))
        self.assertIn("if (timestep_phase == 1 && sd_learn_on)", task)
        self.assertIn("watchdog < 4000000", task)
        self.assertIn("dut.cache_publish_busy || dut.learning_inference_hold || !idle", task)
        self.assertIn("same-DUT commit wait timeout", task)
        # The wait observes the DUT; it cannot manufacture a learning commit.
        wait_body = task[wait:task.index("            if (timestep_phase >= 1)")]
        self.assertNotIn("sd_release =", wait_body)
        self.assertNotIn("parameter_load_valid =", wait_body)

    def test_prepare_has_no_subprocess_or_compile(self):
        with self.temporary() as directory, patch.object(runner, "run_logged") as execute:
            output = Path(directory) / "prepared"
            report = runner.prepare(output)
            self.assertEqual(report["status"], "prepared_not_compiled_or_simulated")
            self.assertEqual((output / (runner.TOP + ".sv")).read_text(), self.source)
            self.assertFalse(execute.called)
            self.assertFalse((output / "summary_on.json").exists())
            with self.assertRaises(FileExistsError):
                runner.prepare(output)

    def test_server_compile_command_uses_bounded_parallelism_and_host_makefile(self):
        command = runner.build_command(Path("unit_not_run"), "/server/verilator", 2)
        self.assertEqual(command[command.index("-j") + 1], "2")
        self.assertEqual(command[command.index("--top-module") + 1], runner.TOP)
        self.assertEqual(command[command.index("--prefix") + 1], "Vscgraph")
        self.assertIn("supplement_scnn_graph_host.mk", command[command.index("-MAKEFLAGS") + 1])

    def test_summary_rejects_missing_failed_or_wrong_counts(self):
        with self.temporary() as directory:
            output = Path(directory)
            reference.write_fixture(output / "fixture_on", True)
            record = output / "run_on.json"
            log = output / "run_on.log"
            record.write_text(json.dumps(dict(status="failed_or_interrupted", exit_code=1)))
            with self.assertRaises(ValueError):
                runner.summarize(output, "on")
            record.write_text(json.dumps(dict(status="pass", exit_code=0, wall_seconds=1)))
            log.write_text("build completed\n")
            with self.assertRaises(ValueError):
                runner.summarize(output, "on")
            # Deliberately fabricated unit input, isolated and auto-deleted.
            line = (runner.PASS + " learning=1 frames=3 features=6144 currents=33 logits=33 traces=4096 "
                    "rmw=16391 code_writes=0 residual_writes=16391 commits=1 publisher_words=3784 cycles=36000000\n")
            log.write_text(line.replace("commits=1", "commits=0"))
            with self.assertRaises(ValueError):
                runner.summarize(output, "on")
            log.write_text(line)
            result = runner.summarize(output, "on")
            self.assertTrue(result["same_DUT_graph_learning"])
            self.assertFalse(result["final_composition_top"])
            log.write_text(line + line)
            with self.assertRaises(ValueError):
                runner.summarize(output, "on")


if __name__ == "__main__":
    unittest.main()
