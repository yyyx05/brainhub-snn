"""Lightweight generator/protocol checks; not an RTL simulation substitute."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_scnn_real_final_graph import (  # noqa: E402
    CLASSIFIER_BASE, CONV_BASES, CONV_WORDS, FINAL, PARAM_DEPTH,
    SOURCE, ansi_header, connections, replace_once, testbench_source,
)
from run_scnn_real_final_graph import build_command, dependencies, parse_result  # noqa: E402
from supplement_scnn_same_dut_reference import expectations  # noqa: E402


class FinalGraphGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = testbench_source()

    def test_real_top_and_single_dut(self):
        self.assertEqual(self.source.count("unified_h40_final_composition_top_v1 #("), 1)
        self.assertNotIn("unified_scnn_h40_array16_graph_online_learning_system_v1 #(", self.source)
        self.assertNotRegex(self.source, r"\bforce\b")
        self.assertNotRegex(self.source, r"dut\.[\w.\[\]]+\s*(?:<=|=(?!=))")
        self.assertIn("dut.u_system.u_array.g_scnn_graph", self.source)
        self.assertNotIn("dut.u_core", self.source)

    def test_every_public_port_connected_once(self):
        _, parameters, ports = ansi_header(FINAL)
        instance = self.source.split(") dut (", 1)[1].split("    );", 1)[0]
        linked = connections(instance)
        self.assertEqual(set(linked), {name for _, _, name in ports})
        for _, name, _ in parameters:
            self.assertEqual(len(re.findall(r"localparam\s+[^;]+\b" + name + r"\s*=", self.source)), 1, name)
        for direction, _, name in ports:
            self.assertEqual(linked[name], "ft_" + name)
            if direction == "input":
                self.assertEqual(self.source.count("assign ft_" + name + " = "), 1, name)

    def test_external_window_and_nonoverlapping_model_banks(self):
        self.assertEqual(CLASSIFIER_BASE, sum(CONV_WORDS))
        self.assertGreater(CLASSIFIER_BASE, 4194304)
        self.assertLess(CLASSIFIER_BASE+22528, PARAM_DEPTH)
        self.assertEqual(CONV_BASES, tuple(sum(CONV_WORDS[:i]) for i in range(8)))
        self.assertIn("localparam int PARAM_DEPTH = 16777216;", self.source)
        self.assertIn("localparam int PARAM_ADDR_WIDTH = 24;", self.source)
        self.assertIn("assign ft_i_store_forward_response_code = fg_forward_code;", self.source)
        self.assertNotIn("task automatic fill_weight", self.source)
        self.assertNotIn("load_conv1_context()", self.source)
        self.assertNotIn("i_external_weight_fill_valid", self.source)
        self.assertIn("fg_forward_code <= fg_ddr_code(address)", self.source)

    def test_window_descriptor_and_global_classes(self):
        self.assertIn("assign ft_i_classifier_class_count = CLASS_COUNT_WIDTH'(CLASSES);", self.source)
        self.assertIn("assign ft_i_graph_descriptor_last = layer_phase == 8;", self.source)
        self.assertIn("assign ft_i_timestep = '0;", self.source)
        self.assertIn("assign ft_i_timestep_count = TIMESTEP_COUNT_WIDTH'(timestep_phase == 2 ? 1 : 2);", self.source)
        self.assertEqual(self.source.count("fg_start_window();"), 2)
        self.assertIn("fg_descriptors != 24", self.source)
        self.assertIn("frame/sample/timestep moved before descriptor completion", self.source)

    def test_all_layer_numeric_checks_retained(self):
        for layer in range(1, 9):
            for check in ("parity mismatch", "incomplete"):
                message = f"Conv{layer} {check}"
                if layer == 8 and check == "incomplete":
                    message = "Conv8/classifier incomplete"
                self.assertIn(message, self.source)
            for name in ("expected_pre", "expected_spike", "expected_next"):
                self.assertIn(f"conv{layer}_{name}", self.source)
        self.assertIn("state spill parity mismatch", self.source)
        self.assertIn("classifier cumulative logit mismatch", self.source)
        self.assertIn("classifier current mismatch", self.source)
        self.assertIn("Conv8 feature-commit parity mismatch", self.source)
        self.assertIn("fg_classifier_restore();", self.source)
        self.assertIn("fg_class_state[tile][row] = state_data", self.source)
        self.assertIn("timestep_phase == 1 ? fg_class_state[tile][row] : 320'd0", self.source)
        self.assertNotRegex(self.source, r"(?:state_write_data|trace_write_data)\s*=\s*\w*expected")

    def test_fresh_frame_and_learning_scoreboard(self):
        self.assertIn("load_reference_frame(timestep_phase == 2 ? 0 : timestep_phase);", self.source)
        self.assertIn("timestep_phase < 3;", self.source)
        self.assertIn("fg_codes[index] != fg_initial_codes[index]", self.source)
        self.assertIn("fg_store_residual[index] !== fg_residuals[index]", self.source)
        self.assertIn("unexpected Conv8 learning/target path", self.source)
        self.assertIn("fresh T0 changed parameters or reinitialized cache", self.source)
        self.assertIn("fg_fresh_reads == 0", self.source)

    def test_no_obsolete_wrapper_hierarchy(self):
        for name in ("core_start_ready", "learning_inference_ready", "tx_state_q", "u_learning_client", "cache_publish_busy"):
            self.assertNotIn("dut." + name, self.source)

    def test_edit_failure_is_explicit(self):
        with self.assertRaises(ValueError):
            replace_once("already changed", "old", "new")


class FinalGraphRunnerTests(unittest.TestCase):
    def test_dependency_inventory_has_assets_and_full_top(self):
        paths = dependencies()
        for path in (FINAL, SOURCE, ROOT / "rtl/control/unified_scnn_frame_descriptor_sequencer_v1.sv",
                     ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem",
                     ROOT / "models/scnn_dvsgesture_vgg9_q0_15/bank_08.hex"):
            self.assertIn(path, paths)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(p.is_file() and p.is_relative_to(ROOT) for p in paths))

    def test_build_command_keeps_structural_warnings(self):
        command = build_command(ROOT / "work/test_only", jobs=2)
        self.assertIn("--assert", command)
        self.assertIn("--timing", command)
        self.assertIn("Vscgraph", command)
        for warning in ("MULTIDRIVEN", "UNOPTFLAT", "LATCH"):
            self.assertNotIn("-Wno-" + warning, command)
        with self.assertRaises(ValueError):
            build_command(ROOT, jobs=0)

    def test_independent_cold_reference_keeps_zero_crossing(self):
        on, off = expectations(True), expectations(False)
        self.assertEqual(on["counts"][0], 0)
        self.assertGreater(on["counts"][1], 0)
        self.assertEqual(off["counts"], [0, 0, 0])
        self.assertEqual(on["features"], off["features"])
        self.assertEqual(on["currents"], off["currents"])

    def test_summary_requires_full_numeric_counts(self):
        expected = dict(expected_rmw_transactions=9, expected_code_changes=0, expected_residual_changes=9)
        log = "\n".join(["SCNN_REAL_FINAL_FRAME_PASS phase=0"]*3 +
                        [f"SCNN_REAL_FINAL_DDR bank={i} reads=1" for i in range(9)] +
                        ["SCNN_REAL_FINAL_GRAPH_PASS learning=1 frames=3 descriptors=24 requests=2 features=6144 currents=33 logits=33 traces=4096 rmw=9 code_changes=0 residual_changes=9 publications=3784 commits=1 fresh_reads=8"])
        self.assertEqual(parse_result(log, expected, True)["rmw"], 9)
        for bad in (log.replace("descriptors=24", "descriptors=16"), log.replace("fresh_reads=8", "fresh_reads=0"),
                    log.replace("bank=8 reads=1", "bank=8 reads=0"), log + "\n" + log):
            with self.assertRaises(ValueError):
                parse_result(bad, expected, True)


if __name__ == "__main__":
    unittest.main()
