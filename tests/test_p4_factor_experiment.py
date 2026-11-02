"""Host arithmetic, source wiring, report parser, and real Tcl with mocked EDA.

These checks neither compile/simulate SystemVerilog nor run Vivado.
All transient files stay inside verification/experiments/p4_factor.
"""
import contextlib
import io
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/experiments/p4_factor"
sys.path.insert(0, str(SCRIPTS))
import run as experiment

UTIL = """MOCK REPORT, NOT MEASUREMENT
| Slice LUTs* | 12 | 0 | 433200 | 0.00 |
| Slice Registers | 172 | 0 | 866400 | 0.00 |
| RAMB36/FIFO* | 0 | 0 | 1470 | 0.00 |
| RAMB18 | 0 | 0 | 2940 | 0.00 |
| DSPs | 2 | 0 | 3600 | 0.00 |
"""
TIMING = """MOCK REPORT, NOT MEASUREMENT
WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns) THS Failing Endpoints THS Total Endpoints
------- ------- --------------------- ------------------- ------- ------- --------------------- -------------------
1.250 0.000 0 172 0.050 0.000 0 172
"""


def signed(value, width):
    return ((value + (1 << (width - 1))) % (1 << width)) - (1 << (width - 1))


TCL_MOCK = r"""
namespace eval mock { variable calls {}; variable active 0; variable synth_count 0 }
proc record {name args} { lappend ::mock::calls [linsert $args 0 $name] }
proc create_project {args} {
    if {$::mock::active} { error "Concurrent projects" }
    set ::mock::active 1
    record create_project {*}$args
}
proc close_project {} { set ::mock::active 0; record close_project }
proc read_verilog {args} {
    foreach path [lindex $args end] { if {![file readable $path]} { error "Missing source $path" } }
    record read_verilog {*}$args
}
proc read_xdc {path} { record read_xdc $path; source $path }
proc get_ports {args} { return [lindex $args end] }
proc all_outputs {} { return o_result }
proc set_param {args} { record set_param {*}$args }
foreach command {create_clock set_input_delay set_output_delay} {
    proc $command {args} "record $command {*}\$args"
}
proc synth_design {args} {
    incr ::mock::synth_count
    record synth_design {*}$args
    if {$::mock::synth_count == $::fail_arm} { error "Injected mock synthesis failure" }
}
proc mock_report {command args} {
    record $command {*}$args
    set at [lsearch -exact $args -file]
    if {$at < 0} { error "Missing report path" }
    set f [open [lindex $args [expr {$at + 1}]] w]
    if {$command == "report_utilization"} {
        puts $f $::mock_util
    } elseif {$command == "report_timing_summary"} {
        puts $f $::mock_timing
    } else { puts $f "MOCK NOT MEASUREMENT" }
    close $f
}
foreach command {report_utilization report_timing_summary report_timing report_clocks check_timing} {
    proc $command {args} "mock_report $command {*}\$args"
}
proc write_checkpoint {args} { record write_checkpoint {*}$args }
proc write_verilog {args} { record write_verilog {*}$args }
proc version {args} { return MOCK_NOT_VIVADO }
proc get_cells {args} { return {} }
proc exit {code} { return -code error "mock_exit:$code" }
"""


class P4ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production_before = experiment.rtl_snapshot()

    @classmethod
    def tearDownClass(cls):
        if cls.production_before != experiment.rtl_snapshot():
            raise AssertionError("Production RTL changed during host tests")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".host_test_", dir=experiment.EXPERIMENT)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def info(self, arm):
        return {"arm": arm, "sop_control": "0" if arm == "factor" else "1",
                "stage": "post_synthesis", "part": experiment.PART, "period_ns": "15.000",
                "top": "p4_factor_top", "mode": "out_of_context", "flatten_hierarchy": "rebuilt",
                "directive": "Default", "production_source": experiment.PRODUCTION_REL,
                "tool_version": "MOCK_NOT_VIVADO", "blackboxes": "0"}

    def records(self, path, values):
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")

    def reports(self):
        for arm in ("factor", "sop"):
            folder = self.work / arm
            folder.mkdir()
            self.records(folder / "run_info.txt", self.info(arm))
            self.records(folder / "status.txt", {"status": "complete"})
            (folder / "utilization.rpt").write_text(UTIL)
            (folder / "timing_summary.rpt").write_text(TIMING)
        self.records(self.work / "flow.txt", {"status": "complete"})

    def test_source_reference_is_live_not_experiment_copy(self):
        self.assertEqual(experiment.check_reference(), ROOT / experiment.PRODUCTION_REL)
        source = experiment.check_reference().read_text()
        self.assertIn(".PREG(0)", source)
        self.assertIn(".PREG(1)", source)
        self.assertIn("pair01[25] == pair01[24]", source)
        self.assertNotIn("i_rst", source)
        for p in experiment.EXPERIMENT.glob("*.sv"):
            self.assertNotRegex(p.read_text(), r"\bmodule\s+" + experiment.PRODUCTION_MODULE + r"\b")

    def test_missing_production_reference_rejected(self):
        (self.work / "rtl").mkdir()
        (self.work / "rtl/final_sources.f").write_text("rtl/unrelated.sv\n")
        with self.assertRaises(ValueError):
            experiment.check_reference(self.work)

    def test_sop_preserves_widths_no_reset_or_forced_mapping(self):
        sop = (experiment.EXPERIMENT / "p4_sum_products.sv").read_text()
        top = (experiment.EXPERIMENT / "p4_factor_top.sv").read_text()
        self.assertEqual(len(re.findall(r"assign product[0-3] = i_term[0-3] \* i_factor;", sop)), 4)
        self.assertIn("[42:0] product0", sop)
        self.assertIn("[44:0] contribution", sop)
        self.assertIn("[47:0] o_result", sop)
        self.assertIn("if (i_fire)", sop)
        self.assertIn("o_result <= i_first ?", sop)
        for text in (sop, top):
            self.assertNotRegex(text, r"(?i)DONT_TOUCH|KEEP_HIERARCHY|USE_DSP|\(\*|i_rst|initial\s+begin")
        self.assertEqual(top.count("always_ff"), 1)
        self.assertEqual(top.count(".i_fire(fire_q)"), 2)
        self.assertEqual(top.count(".o_result(lane_result)"), 2)

    def test_mathematical_equivalence_in_production_domain(self):
        rng = random.Random(20261012)
        checked = 0
        for mask in range(16):
            for _ in range(800):
                terms = [rng.randrange(-(1 << 24), 1 << 24) for _ in range(4)]
                factor = rng.randrange(-(1 << 17), 1 << 17)
                enabled = [t if mask & (1 << index) else 0 for index, t in enumerate(terms)]
                pairs = [sum(enabled[:2]), sum(enabled[2:])]
                if any(not -(1 << 24) <= pair < (1 << 24) for pair in pairs):
                    continue
                physical = signed(sum(signed(pair, 25) * factor for pair in pairs), 48)
                sop = signed(sum(signed(term * factor, 43) for term in enabled), 48)
                self.assertEqual(physical, sop)
                checked += 1
        self.assertGreater(checked, 9000)

    def test_out_of_domain_cannot_be_claimed_equivalent(self):
        pair = 16777215 + 1
        self.assertNotEqual(signed(pair, 25), pair)
        self.assertIn("legal_pairs", (experiment.EXPERIMENT / "tb_p4_factor.sv").read_text())

    def test_host_state_model_wrap_bubbles_first_and_zero_mask(self):
        state = 0
        for _ in range(33):
            state = signed(state + (-16777216 * -131072) * 2, 48)
        self.assertLess(state, 0)
        held = state
        for fire, first, contribution in ((False, True, 123), (False, False, -123)):
            state = signed((0 if first else state) + contribution, 48) if fire else state
        self.assertEqual(state, held)
        self.assertEqual(signed(0 + 0, 48), 0)

    def test_missing_simulator_is_nonzero_and_not_pass(self):
        with patch.object(experiment.shutil, "which", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            code = experiment.simulate(self.work / "missing_sim", "iverilog")
        self.assertEqual(code, 2)
        import json
        data = json.loads((self.work / "missing_sim/simulation.json").read_text())
        self.assertEqual(data["status"], "missing_tool")
        self.assertTrue(data["production_unchanged"])
        self.assertNotIn("build_exit", data)

    def test_paths_cannot_escape_experiment(self):
        for name in ("..", "../rtl", "x/y", "C:\\rtl", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                experiment.run_path(name)

    def test_report_parsers_accept_vivado_format_and_keep_null(self):
        self.assertEqual(experiment.utilization(UTIL)["lut"], 12)
        self.assertEqual(experiment.utilization(UTIL)["bram36eq"], 0)
        self.assertIsNone(experiment.utilization("")["dsp"])
        self.assertIsNone(experiment.timing(TIMING.replace("1.250", "N/A"))["wns_ns"])
        self.assertIsNone(experiment.number("nan"))
        self.assertEqual(experiment.timing(TIMING)["setup_endpoints"], 172)

    def test_existing_report_format_read_only_not_new_experiment(self):
        # Parser compatibility only: no runner, no new use of historical cost.
        base = ROOT / "results/paper_evidence/physical/run136_scnn_frame_service_synth"
        self.assertEqual(experiment.utilization((base / "utilization_post_synth.rpt").read_text())["lut"], 180969)
        self.assertEqual(experiment.timing((base / "timing_summary_post_synth.rpt").read_text())["wns_ns"], 0.006)

    def test_complete_mock_reports_are_not_route_or_equivalence(self):
        self.reports()
        data = experiment.summarize(self.work)
        self.assertEqual(data["comparison_status"], "matched_reports")
        self.assertEqual(data["arms"]["factor"]["status"], "post_synth_estimate_met")
        self.assertEqual(data["sop_minus_factor"]["dsp"], 0)
        self.assertEqual(data["routed_sta"], "not_run")
        self.assertEqual(data["physical_equivalence"], "not_checked")
        self.assertIsNone(data["whole_core_cost"])

    def test_negative_timing_retained_not_hidden_by_exit_zero(self):
        self.reports()
        (self.work / "sop/timing_summary.rpt").write_text(TIMING.replace("1.250 0.000 0", "-0.250 -1.000 4"))
        data = experiment.summarize(self.work)
        self.assertEqual(data["comparison_status"], "matched_reports")
        self.assertEqual(data["arms"]["sop"]["status"], "post_synth_estimate_violated")
        self.assertEqual(data["arms"]["sop"]["metrics"]["tns_ns"], -1)

    def test_missing_reports_and_incomplete_arm_fail_closed(self):
        self.reports()
        (self.work / "sop/utilization.rpt").unlink()
        data = experiment.summarize(self.work)
        self.assertEqual(data["comparison_status"], "incomplete")
        self.assertTrue(all(x is None for x in data["sop_minus_factor"].values()))
        self.assertIsNone(data["arms"]["sop"]["metrics"]["lut"])

    def test_mismatched_recipe_or_version_is_not_matched(self):
        self.reports()
        for key, value in (("part", "xc7other"), ("period_ns", "10"),
                           ("tool_version", "OTHER"), ("directive", "AreaOptimized_high"),
                           ("production_source", "copied.sv"), ("blackboxes", "1")):
            values = self.info("sop")
            values[key] = value
            self.records(self.work / "sop/run_info.txt", values)
            self.assertEqual(experiment.summarize(self.work)["comparison_status"], "incomplete", key)

    def test_no_paths_no_completion_not_pass(self):
        self.reports()
        (self.work / "sop/timing_summary.rpt").write_text(TIMING.replace("172", "0"))
        self.assertEqual(experiment.summarize(self.work)["comparison_status"], "incomplete")
        (self.work / "sop/timing_summary.rpt").write_text(TIMING)
        self.records(self.work / "sop/status.txt", {"status": "failed"})
        self.assertEqual(experiment.summarize(self.work)["comparison_status"], "incomplete")

    def mock_tcl(self, fail_arm=0):
        import tkinter
        interp = tkinter.Tcl()
        scripts = self.work / "scripts/experiments/p4_factor"
        scripts.mkdir(parents=True)
        for name in ("synth_pair.tcl", "p4_factor.xdc"):
            (scripts / name).write_bytes((SCRIPTS / name).read_bytes())
        rtl = self.work / experiment.PRODUCTION_REL
        rtl.parent.mkdir(parents=True)
        rtl.write_text("// Mock file, not compiled\n")
        (self.work / "rtl/final_sources.f").write_text(experiment.PRODUCTION_REL + "\n")
        verif = self.work / "verification/experiments/p4_factor"
        verif.mkdir(parents=True)
        for name in ("p4_sum_products.sv", "p4_factor_top.sv"):
            (verif / name).write_text("// Mock file, not compiled\n")
        interp.eval(TCL_MOCK)
        interp.setvar("argv", ("mock_run",))
        interp.setvar("fail_arm", fail_arm)
        interp.setvar("mock_util", UTIL)
        interp.setvar("mock_timing", TIMING)
        previous = Path.cwd()
        try:
            interp.call("source", str(scripts / "synth_pair.tcl"))
            error = None
        except tkinter.TclError as exc:
            error = str(exc)
        finally:
            # Tcl cd changes the host process directory too.
            import os
            os.chdir(previous)
        calls = [tuple(map(str, interp.splitlist(call))) for call in interp.splitlist(interp.getvar("::mock::calls"))]
        return calls, error, verif / "runs/mock_run"

    def test_tcl_runs_two_sequential_identical_recipes(self):
        calls, error, out = self.mock_tcl()
        self.assertIsNone(error)
        sequence = [c[0] for c in calls if c[0] in ("create_project", "synth_design", "close_project")]
        self.assertEqual(sequence, ["create_project", "synth_design", "close_project"] * 2)
        self.assertEqual([c for c in calls if c[0] == "set_param"], [("set_param", "general.maxThreads", "1")])
        synths = [c for c in calls if c[0] == "synth_design"]
        self.assertEqual(synths[0][:-1], synths[1][:-1])
        self.assertEqual([c[-1] for c in synths], ["SOP_CONTROL=0", "SOP_CONTROL=1"])
        clocks = [c for c in calls if c[0] == "create_clock"]
        self.assertEqual(clocks[0], clocks[1])
        self.assertIn("15.000", clocks[0])
        self.assertEqual(len([c for c in calls if c[0] == "set_input_delay"]), 4)
        self.assertEqual(len([c for c in calls if c[0] == "set_output_delay"]), 4)
        self.assertEqual(experiment.summarize(out)["comparison_status"], "matched_reports")

    def test_tcl_failure_stops_second_arm_without_completion(self):
        calls, error, out = self.mock_tcl(fail_arm=1)
        self.assertEqual(error, "mock_exit:1")
        self.assertEqual(len([c for c in calls if c[0] == "synth_design"]), 1)
        self.assertFalse((out / "sop").exists())
        self.assertFalse((out / "flow.txt").exists())
        self.assertEqual(experiment.key_values(out / "factor/status.txt")["status"], "failed")

    def test_tcl_has_no_implementation_power_or_forced_mapping(self):
        script = (SCRIPTS / "synth_pair.tcl").read_text()
        self.assertNotRegex(script, r"(?i)DONT_TOUCH|KEEP_HIERARCHY|USE_DSP|route_design|place_design|report_power|write_bitstream")
        self.assertIn(experiment.PRODUCTION_REL, script)
        self.assertEqual(script.count("synth_design -top"), 1)


if __name__ == "__main__":
    unittest.main()
