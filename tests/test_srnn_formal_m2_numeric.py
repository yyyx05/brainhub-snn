"""Directed integer checks without formal-model execution or Verilator."""
import sys
from pathlib import Path
import unittest
from contextlib import redirect_stdout
import io
import json
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_srnn_formal_m2_fixture as ref
import run_srnn_formal_m2_numeric as runner


class IntegerReferenceTests(unittest.TestCase):
    def test_rne_signed_ties(self):
        np.testing.assert_array_equal(ref.round_shift_even(np.array([1, 3, 5, -1, -3, -5]), 1),
                                      [0, 2, 2, 0, -2, -2])

    def test_membrane_leak_discrepancy_minimal_example(self):
        args = (np.zeros((1, 1), dtype=np.int64), np.ones((1, 1), dtype=np.int64),
                np.ones((1, 1), dtype=np.int64), np.array([[16384]]),
                np.array([[16384]]), np.zeros((1, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64))
        production = ref.forward_step(*args, arithmetic="legacy_h40")
        formal = ref.forward_step(*args, arithmetic="formal_q14")
        corrected = ref.forward_step(*args)
        for key in formal:
            np.testing.assert_array_equal(corrected[key], formal[key])
        self.assertEqual(int(production["pre"][0, 0]), 6963)
        self.assertEqual(int(formal["pre"][0, 0]), 13926)
        self.assertEqual(int(production["spike"][0, 0]), 0)
        self.assertEqual(int(formal["spike"][0, 0]), 1)
        self.assertEqual(int(production["trace"][0, 0]), 13926)
        self.assertEqual(int(formal["trace"][0, 0]), 30310)

    def test_negative_membrane_uses_arithmetic_shift_not_truncation(self):
        z = np.zeros((1, 1), dtype=np.int64)
        out = ref.forward_step(z, z, z, np.array([[-1]]), z, z, z)
        self.assertEqual(int(out["pre"][0, 0]), -1)

    def test_threshold_inclusive_and_soft_reset(self):
        inp = np.array([[16384]])
        z = np.zeros((1, 1), dtype=np.int64)
        out = ref.forward_step(inp, np.array([[8192]]), np.array([[100]]), z, z, z, z)
        self.assertEqual(int(out["spike"][0, 0]), 1)
        self.assertEqual(int(out["membrane"][0, 0]), 0)
        self.assertEqual(int(out["trace"][0, 0]), 16384)
        self.assertEqual(int(out["feature"][0, 0]), 16384)
        self.assertEqual(int(out["logit"][0, 0]), 100 << 14)

    def test_trace_and_feature_are_distinct_and_logits_use_new_increment(self):
        z = np.zeros((1, 1), dtype=np.int64)
        one = np.array([[16384]])
        a = ref.forward_step(one, np.array([[8192]]), np.array([[100]]), z, z, z, z)
        b = ref.forward_step(one, np.array([[8192]]), np.array([[200]]),
                             a["membrane"], a["trace"], a["feature"], a["logit"])
        self.assertEqual(int(b["trace"][0, 0]), 30310)
        self.assertEqual(int(b["feature"][0, 0]), 32768)
        self.assertEqual(int(b["logit"][0, 0]), 300 << 14)
        self.assertNotEqual(int(b["logit"][0, 0]), 400 << 14)

    def test_softmax_uses_production_lut_and_rne_division(self):
        lut = ref.read_exp_lut()
        self.assertEqual(len(lut), 129)
        self.assertEqual(lut[0], 1 << 20)
        self.assertEqual(lut[-1], 352)
        np.testing.assert_array_equal(ref.softmax_q24(np.zeros((1, 2), dtype=np.int64)), [[1 << 23, 1 << 23]])
        np.testing.assert_array_equal(ref.softmax_q24(np.zeros((1, 3), dtype=np.int64)), [[5592405] * 3])

    def test_b2_reducer_fixed_div64_not_div2(self):
        probability, error, gradient = ref.wout_gradient(
            np.zeros((2, 2), dtype=np.int64), np.full((2, 1), 16384), np.array([0, 0]))
        np.testing.assert_array_equal(error, [[-(1 << 39), 1 << 39]] * 2)
        np.testing.assert_array_equal(gradient, [[-(1 << 34)], [1 << 34]])

    def test_real_vote_threshold_prevents_b2t2_code_crossing(self):
        inputs = np.full((4, 2, 1), 16384, dtype=np.int64)
        arrays, stats = ref.reference(inputs, np.array([0, 0]), np.array([[8192]]),
                                      np.zeros((2, 1), dtype=np.int64), learning=True)
        self.assertEqual(stats[0]["code_writes"], 0)
        self.assertEqual(stats[1]["code_writes"], 0)
        self.assertEqual(stats[2]["code_writes"], 2)
        np.testing.assert_array_equal(arrays["wout_after"][1], [[0], [0]])
        np.testing.assert_array_equal(arrays["wout_after"][2], [[1], [-1]])
        np.testing.assert_array_equal(arrays["residual"][2], [[28], [-28]])

    def test_learning_off_never_writes(self):
        inputs = np.full((4, 2, 1), 16384, dtype=np.int64)
        arrays, stats = ref.reference(inputs, np.array([0, 0]), np.array([[8192]]),
                                      np.array([[10], [-10]]), learning=False)
        self.assertEqual(sum(row["rmw"] for row in stats), 0)
        self.assertEqual(np.count_nonzero(arrays["residual"]), 0)
        for value in arrays["wout_after"]:
            np.testing.assert_array_equal(value, [[10], [-10]])

    def test_rejects_nonbinary_m2_input(self):
        z = np.zeros((1, 1), dtype=np.int64)
        with self.assertRaises(ValueError):
            ref.forward_step(np.array([[1]]), z, z, z, z, z, z)

    def test_saturating_vote_clears_residual(self):
        code, residual, counters = ref.direct_update_tensor(
            np.array([[32767, -32768]]), np.array([[62, -62]]),
            np.array([[-1, 1]]), 31, 65)
        np.testing.assert_array_equal(code, [[32767, -32768]])
        np.testing.assert_array_equal(residual, [[0, 0]])
        self.assertEqual(counters["saturations"], 2)
        self.assertEqual(counters["code_writes"], 0)

    def test_current_saturation_and_trace_wide_intermediate(self):
        inp = np.full((1, 3), 16384, dtype=np.int64)
        z = np.zeros((1, 1), dtype=np.int64)
        out = ref.forward_step(inp, np.full((1, 3), 1 << 30), z,
                               z, np.array([[(1 << 31) - 1]]), z, z)
        self.assertEqual(out["current"][0, 0], (1 << 31) - 1)
        self.assertEqual(out["membrane"][0, 0], (1 << 31) - 1 - 8192)
        self.assertEqual(out["trace"][0, 0],
                         ref.round_shift_even(np.array([((1 << 31) - 1) * ref.LEAK]), 14)[0] + 16384)

    def test_corrected_pre_compare_and_post_use_wide_sum(self):
        z = np.zeros((1, 1), dtype=np.int64)
        inp = np.array([[16384]])
        maximum = (1 << 31) - 1
        out = ref.forward_step(inp, np.array([[maximum]]), z, np.array([[16384]]), z, z, z)
        self.assertEqual(out["pre"][0, 0], maximum)
        self.assertEqual(out["spike"][0, 0], 1)
        self.assertEqual(out["membrane"][0, 0], maximum)
        self.assertNotEqual(out["membrane"][0, 0], out["pre"][0, 0] - 8192)


class RunnerTests(unittest.TestCase):
    def test_dry_run_has_no_build_or_filesystem_side_effect(self):
        output = runner.EVIDENCE / "unittest_dry_run_not_created"
        self.assertFalse(output.exists())
        stdout = io.StringIO()
        with patch.object(runner, "execute", side_effect=AssertionError("dry run executed")), redirect_stdout(stdout):
            self.assertEqual(runner.main(["--mode", "lif-probe", "--output-dir", str(output)]), 0)
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["status"], "dry_run")
        self.assertFalse(data["final_top_tested"])
        self.assertFalse(output.exists())
        command = data["compile_command"]
        self.assertEqual(command[command.index("-j") + 1], "1")
        self.assertEqual(command[command.index("--output-split") + 1], "3000")

    def test_formal_gap_is_not_reported_as_formal_or_finaltop_pass(self):
        status, values = runner.decode_marker("forward", "SRNN_FORMAL_M2_FORWARD_PASS steps=2 formal_pre_differences=256\n")
        self.assertEqual(status, "production_reference_pass_formal_q14_gap")
        self.assertEqual(values["formal_pre_differences"], 256)

    def test_missing_or_duplicate_marker_rejected(self):
        with self.assertRaises(ValueError):
            runner.decode_marker("forward", "all good")
        with self.assertRaises(ValueError):
            runner.decode_marker("forward", "SRNN_FORMAL_M2_FORWARD_PASS steps=2\n" * 2)

    def test_output_stays_in_owned_task_directory(self):
        with self.assertRaises(ValueError):
            runner.owned_output(ROOT / "rtl", runner.WORK)
        with self.assertRaises(ValueError):
            runner.owned_output(runner.WORK, runner.WORK)

    def test_online_build_uses_production_coordinator_parameter(self):
        command = runner.build_command("online", runner.WORK / "unittest_not_created", Path("verilator"))
        self.assertIn("-GWITH_COORDINATOR=1", command)
        self.assertIn("+define+SRNN_FORMAL_M2_ONLINE", command)
        self.assertEqual(command[command.index("-j") + 1], "1")
        self.assertNotIn("-GWITH_COORDINATOR=1", runner.build_command("forward", runner.WORK / "unittest_not_created", Path("verilator")))

    def test_online_formal_gap_remains_explicit(self):
        status, values = runner.decode_marker("online", "SRNN_FORMAL_M2_ONLINE_PASS learning=1 formal_pre_differences=3071\n")
        self.assertEqual(status, "production_online_reference_pass_formal_q14_gap")
        self.assertEqual(values["learning"], 1)

    def test_watchdog_report_retains_successful_first_window(self):
        progress = runner.partial_progress("SRNN_FORMAL_M2_ONLINE_STEP t=0 commits=1 detach=1\n"
            "M2_LEARNER_READY reducer_start=1 update_active=0\n"
            "%Fatal: M2_NO_PROGRESS t=1 coordinator=5\n")
        self.assertEqual(progress["completed_windows"], [{"t": 0, "commits": 1, "detach": 1}])
        self.assertEqual(progress["learner_ready"]["update_active"], 0)
        self.assertIn("not_a_DUT_numeric_assertion", progress["failure_kind"])

    def test_context_probe_has_explicit_leaf_boundary_and_coverage(self):
        marker = ("SRNN_M2_CONTEXT_PROBE_PASS windows=2 held_start_cycles=64 held_start_accepts=0 "
                  "configs=2 starts=2 gradients=10240 commits=2 contributions=20480 store_checks=10240")
        status, values = runner.decode_marker("context-probe", marker)
        self.assertIn("not_coordinator_pass", status)
        self.assertEqual(values["held_start_accepts"], 0)
        with self.assertRaises(ValueError):
            runner.decode_marker("context-probe", marker.replace("commits=2", "commits=1"))
        with self.assertRaises(ValueError):
            runner.decode_marker("context-probe", marker.replace("held_start_accepts=0", "held_start_accepts=1"))
        self.assertEqual(runner.sources_for("context-probe")[-1].name, "srnn_formal_m2_context_probe.svh")
        self.assertNotIn(runner.TB, runner.sources_for("context-probe"))

    def test_rearm_probe_requires_cfg_off_abort_and_restore_coverage(self):
        marker = ("SRNN_M2_REARM_PROBE_PASS windows=6 commits=6 detach=6 aborts=3 auto_configs=9 "
                  "external_configs=6 invalid_cfg=1 cfg_competition=1 cfg_pending_during_rearm=1 "
                  "initial_cfg_off_cfg=1 learning_off_cfg=1 clean_abort=1 dirty_abort=1 restore=1 "
                  "first_read_fault=1 legal_cfg_error_clear=1 recovery_blocked_cycles=64 idle_no_rearm_cycles=64")
        status, _ = runner.decode_marker("rearm-probe", marker)
        self.assertIn("not_forward", status)
        for field in ("restore=1", "learning_off_cfg=1", "initial_cfg_off_cfg=1", "cfg_pending_during_rearm=1", "legal_cfg_error_clear=1"):
            with self.assertRaises(ValueError):
                runner.decode_marker("rearm-probe", marker.replace(field, field[:-1] + "0"))

    def test_done_id_fault_requires_actual_done_busy_zero_and_active_phase(self):
        text = ("M2_ID_INJECTION kind=2 busy=0 done=1\n"
                "%Fatal M2_COORDINATOR_ERROR state=6 timestep_id=1 engine=0")
        self.assertTrue(runner.expected_id_failure(text, "done"))
        self.assertFalse(runner.expected_id_failure(text.replace("state=6", "state=3"), "done"))
        self.assertFalse(runner.expected_id_failure(text.replace("busy=0", "busy=1"), "done"))
        self.assertFalse(runner.expected_id_failure(text.replace("timestep_id=1", "timestep_id=0"), "done"))


if __name__ == "__main__":
    unittest.main()
