import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final_scnn_functional as runner


class FinalSCNNTests(unittest.TestCase):
    def test_wrapper_preserves_assertions_inputs_and_cold_protocol(self):
        root, _, _ = runner.roots()
        original = (root / runner.PRIOR_PREPARED / runner.TB).read_text()
        changed = runner.bind_wrapper(original)
        self.assertEqual(re.findall(r'\$fatal\(1,("[^"]*")', original),
                         re.findall(r'\$fatal\(1,("[^"]*")', changed))
        self.assertEqual(original.count("$readmemh"), changed.count("$readmemh"))
        self.assertNotIn("dut.u_core.u_core.", changed)
        self.assertIn(".i_raw_source_active(1'b0)", changed)
        self.assertIn("st_report();", changed)
        self.assertEqual(changed.count("FINAL_SCNN_WRAPPER_PASS"), 1)

    def test_transport_isolated_from_fc_identity(self):
        import run_final_fc_functional as fc
        module = runner.transport()
        self.assertEqual(module.NAME, runner.NAME)
        self.assertEqual(fc.NAME, "final_fc_functional_r1")
        self.assertEqual(module.PRIOR_PREPARED, runner.PRIOR_PREPARED)

    def test_prepared_parser_requires_marker_and_original_nonzero_checks(self):
        _, prepared, archive = runner.roots()
        log = (archive.parent / runner.PRIOR / "run.log").read_text()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            runner.parse_log(log, prepared)
        synthetic = log + "\nFINAL_SCNN_WRAPPER_PASS raw_active=0\n"
        result = runner.parse_log(synthetic, prepared)
        self.assertEqual(result["final"]["code_changes"], 662)
        with self.assertRaises(ValueError):
            runner.parse_log(synthetic.replace("code_changes=662", "code_changes=661", 1), prepared)


if __name__ == "__main__":
    unittest.main()
