"""Generator/summary tests; these do not constitute RTL numeric PASS."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_scnn_real_final_b4t12 as gen
import run_scnn_real_final_b4t12 as runner


class TemporalFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = gen.testbench_source()

    def test_one_real_dut_and_public_data(self):
        self.assertEqual(self.source.count("unified_h40_final_composition_top_v1 #("), 1)
        self.assertNotRegex(self.source, r"dut\.[\w.\[\]]+\s*(?:<=|=(?!=))")
        self.assertNotRegex(self.source, r"\bforce\b")
        self.assertNotIn("task automatic fill_weight", self.source)
        self.assertNotIn("full_frame_t1_reference", self.source)

    def test_temporal_bank_and_raw_stimulus(self):
        self.assertIn("timestep_phase < 52", self.source)
        self.assertIn("tv_query ? 0 : tv_window*4", self.source)
        self.assertIn("input_patch[index] = tv_raw[", self.source)
        for n in range(1, 9):
            self.assertIn(f"conv{n}_state_backing[tv_sample*CONV{n}_SCALARS+index]", self.source)
        self.assertIn("continuation cleared TP state", self.source)
        self.assertIn("fg_store_residual[index] = 0", self.source)
        self.assertNotRegex(self.source, r"fg_store_residual\[.*\]\s*=\s*fg_residuals")

    def test_watchdog_exceeds_signed_int_without_overflow(self):
        self.assertIn("repeat (3) repeat (1000000000) @(posedge clk);", self.source)
        self.assertNotIn("repeat (3000000000)", self.source)
        probe = (ROOT / "tests/tb_scnn_long_watchdog.sv").read_text(encoding="ascii")
        self.assertIn("repeat (3) repeat (1000000000) @(posedge clk);", probe)

    def test_independent_observation_checks(self):
        for marker in ("temporal feature", "temporal current", "classifier cumulative logit mismatch",
                       "temporal trace", "reduced gradient mismatch", "scaled delta mismatch",
                       "temporal publication code mismatch", "full state spill consistency",
                       "independent membrane probe", "natural crossing witness mismatch"):
            self.assertIn(marker, self.source)
        self.assertIn("tv_contributions[index] > 16", self.source)

    def test_fixture_and_dependencies(self):
        manifest = json.loads((runner.FIXTURE / "manifest.json").read_text())
        self.assertEqual([w["code_changes"] for w in manifest["commits"]], [0, 0, 662])
        self.assertEqual(sum(w["nonzero_delta_rmw"] for w in manifest["commits"]), 60273)
        self.assertTrue(all(p.is_file() for p in runner.dependencies()))

    def test_summary_rejects_missing_or_mutated_evidence(self):
        log = "\n".join([f"SCNN_TEMPORAL_FRAME_STATUS frame={i} cycles=1" for i in range(52)] +
            [f"SCNN_TEMPORAL_COMMIT window={i} code_changes={c} residual_changes={r} rmw={r} version=1"
             for i, c, r in [(1, 0, 20041), (2, 0, 20011), (3, 662, 20221)]] +
            ["SCNN_TEMPORAL_CROSSING window=3 index=18433 code_after=869",
             "SCNN_TEMPORAL_FINAL_PASS frames=52 descriptors=416 features=106496 currents=572 logits=572 traces=98304 probes=1248 rmw=60273 commits=3 publications=7568 code_changes=662 query_currents=4 query_logits=4 fresh_reads=1 cycles=123"])
        self.assertEqual(runner.parse_result(log)["commits"], 3)
        for bad in (log.replace("descriptors=416", "descriptors=408"), log.replace("frame=27 ", "frame=26 "),
                    log.replace("query_logits=4", "query_logits=0"), log+"\n%Fatal: failed", log+"\n"+log):
            with self.assertRaises(ValueError):
                runner.parse_result(bad)


if __name__ == "__main__":
    unittest.main()
