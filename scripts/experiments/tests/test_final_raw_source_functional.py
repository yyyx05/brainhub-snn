"""Checks for the wrapper-only testbench transition."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final_raw_source_functional as runner


class WrapperBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, _, archive = runner.roots()
        cls.prior = root / "verification/experiments" / runner.PRIOR
        cls.original = (cls.prior / runner.TB).read_text()
        cls.changed = runner.bind_wrapper(cls.original)
        cls.archive = archive.parent / runner.PRIOR

    def test_no_old_producer_or_mux_and_manual_inputs_forwarded(self):
        self.assertNotIn("source_scan_experiment raw_source_i", self.changed)
        self.assertNotIn("i_fc_srnn_group_valid=feed_active", self.changed)
        for suffix in runner.GROUP:
            self.assertIn(f".i_fc_srnn_group_{suffix}(manual_group_{suffix})", self.changed)
            self.assertIn(f"assign i_fc_srnn_group_{suffix} = u_final.raw_group_{suffix};", self.changed)
        self.assertIn(".i_fc_srnn_group_feedforward_end(1'b0)", self.changed)

    def test_numerical_assertions_unchanged(self):
        import re
        original = re.findall(r'\$fatal\(1,("[^"]*")', self.original)
        changed = re.findall(r'\$fatal\(1,("[^"]*")', self.changed)
        self.assertEqual([s for s in original if s != '"source policy range"'],
                         [s for s in changed if s != '"static wrapper requires streaming=1"'])
        self.assertEqual(self.original.count("$readmemh"), self.changed.count("$readmemh"))
        self.assertNotIn("u_final.u_core.u_core.", self.changed)

    def test_parser_rejects_missing_wrapper_and_cycle_drift(self):
        import json
        log = (self.archive / "run_stream_on.log").read_text()
        info = json.loads((self.archive / "multi_fixture/reference.json").read_text())
        expected = json.loads((self.archive / "checks.json").read_text())["stream_on"]
        with self.assertRaisesRegex(ValueError, "wrapper completion"):
            runner.parse_log(log, info, expected, self.prior.parent / runner.NAME)
        valid = log + "\nFINAL_RAW_WRAPPER_PASS policy=1 sequences=8 steps=588\n"
        self.assertEqual(runner.parse_log(valid, info, expected, self.prior.parent / runner.NAME), expected)
        expected["total"]["cycles"] += 1
        with self.assertRaisesRegex(ValueError, "records"):
            runner.parse_log(valid, info, expected, self.prior.parent / runner.NAME)


if __name__ == "__main__":
    unittest.main()
