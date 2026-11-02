from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TimingClosureLaunchersTest(unittest.TestCase):
    def setUp(self):
        self.tcl = (ROOT / "vivado/run_unified_h40_timing_closure_impl.tcl").read_text()
        self.launcher = (ROOT / "server/launch_timing_closure_20261010.sh").read_text()
        self.runner = (ROOT / "server/run_unified_h40_timing_closure_impl.sh").read_text()

    def test_two_explicit_checkpoint_arms(self):
        self.assertIn("RUN138_SCNN_FRAME_SERVICE_LEARNING_OFF_SYNTH_S1", self.launcher)
        self.assertIn("RUN141_SCNN_FRAME_SERVICE_PRIVATE_OUTPUT_LEARNERS_SYNTH_S1", self.launcher)
        self.assertIn("recipe=wire_delay", self.launcher)
        self.assertIn("recipe=matched", self.launcher)
        self.assertIn('test ! -e "$target"', self.launcher)
        self.assertIn("an EDA process is already active", self.launcher)
        self.assertNotIn("rm -", self.launcher)

    def test_conservative_recipes_keep_prior_congestion_check(self):
        self.assertIn("set place_directive AltSpreadLogic_high", self.tcl)
        self.assertIn("set place_directive ExtraNetDelay_high", self.tcl)
        self.assertIn("phys_opt_design -directive AggressiveFanoutOpt", self.tcl)
        self.assertIn("phys_opt_design -directive ExploreWithHoldFix", self.tcl)
        self.assertIn("route_design -directive AggressiveExplore", self.tcl)
        self.assertIn("if {$post_place_level6}", self.tcl)
        self.assertIn('puts $placement_gate "place_directive=$place_directive"', self.tcl)

    def test_no_timing_or_functional_shortcuts(self):
        self.assertIn("get_property PERIOD $core_clock] - 15.0", self.tcl)
        self.assertIn("implementation changed protected primitive", self.tcl)
        self.assertIn("trace_payload_nonconstant_control_count != 0", self.tcl)
        self.assertIn("$setup_slack >= 0.0 && $hold_slack >= 0.0", self.tcl)
        for command in ("synth_design", "write_bitstream", "set_false_path", "set_multicycle_path",
                        "set_clock_uncertainty", "set_property DONT_TOUCH", "reset_property DONT_TOUCH"):
            self.assertNotIn(command, self.tcl)
        self.assertIn("open_checkpoint $source_dcp", self.tcl)

    def test_real_exit_and_recipe_propagation(self):
        self.assertIn('-tclargs "$root" "$source_dcp" "$recipe" "$source_stage"', self.runner)
        self.assertIn('rc=$?', self.runner)
        self.assertIn('exit "$rc"', self.runner)
        self.assertIn('grep -qx \'status=pass\' "$out/vivado_status.txt"', self.runner)
        self.assertIn('"$rc" -ne 0 || "$error_count" -ne 0', self.runner)


@unittest.skipUnless(shutil.which("tclsh"), "plain Tcl interpreter unavailable")
class TimingClosureTclControlFlowTest(unittest.TestCase):
    def run_stub(self, recipe="wire_delay", stage="post_synth", scenario="normal"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dcp = root / "source.dcp"
            dcp.touch()
            if stage == "post_place":
                original_name = "different_abort_reg" if scenario == "changed_source_protection" else "protected_abort_reg"
                (root / "protected_cells_pre_impl.rpt").write_text(
                    f"protected_primitive_count=1\n{original_name}\n"
                )
            result = subprocess.run(
                [shutil.which("tclsh"), str(ROOT / "tests/fixtures/timing_closure_stub.tcl"),
                 str(ROOT / "vivado/run_unified_h40_timing_closure_impl.tcl"),
                 str(root), str(dcp), recipe, stage, scenario],
                text=True, capture_output=True, timeout=10,
            )
            status = root / "server_results/final_composition_impl/vivado_status.txt"
            return result, status.read_text() if status.exists() else ""

    def test_both_recipes_execute_through_gate(self):
        for recipe in ("wire_delay", "matched"):
            with self.subTest(recipe=recipe):
                result, status = self.run_stub(recipe=recipe)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("MOCK_ROUTE", result.stdout)
                self.assertIn("status=pass", status)
                self.assertIn(f"recipe={recipe}", status)

    def test_post_place_continuation_does_not_repeat_placement(self):
        result, status = self.run_stub(stage="post_place")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source_stage=post_place", status)
        self.assertNotIn("MOCK_OPT", result.stdout)
        self.assertNotIn("MOCK_PLACE", result.stdout)
        self.assertNotIn("AggressiveFanoutOpt", result.stdout)
        self.assertIn("MOCK_ROUTE", result.stdout)

    def test_level6_stops_before_routing(self):
        result, status = self.run_stub(scenario="level6")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("post-place congestion remains level 6", result.stderr)
        self.assertNotIn("MOCK_ROUTE", result.stdout)
        self.assertFalse(status)

    def test_protection_and_final_trace_are_checked(self):
        for scenario, expected in (
            ("lost_protection", "implementation changed protected primitive"),
            ("live_trace", "post-route trace payload reset/set audit failed"),
        ):
            with self.subTest(scenario=scenario):
                result, status = self.run_stub(scenario=scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertFalse(status)

    def test_clock_and_continuation_placement_are_checked(self):
        for scenario, expected in (
            ("wrong_clock", "source checkpoint must retain the existing 15 ns"),
            ("unplaced", "post-place continuation contains unplaced primitives"),
            ("changed_source_protection", "saved placement changed the original protected primitive set"),
        ):
            with self.subTest(scenario=scenario):
                result, status = self.run_stub(stage="post_place", scenario=scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertFalse(status)

    def test_negative_timing_route_or_drc_not_promoted(self):
        for scenario in ("negative_setup", "incomplete_route", "drc_error"):
            with self.subTest(scenario=scenario):
                result, status = self.run_stub(scenario=scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("status=diagnostic_complete", status)
                self.assertNotIn("status=pass", status)


if __name__ == "__main__":
    unittest.main()
