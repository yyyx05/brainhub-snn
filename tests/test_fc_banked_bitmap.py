import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from prepare_fc_banked_bitmap import candidate, testbench
from prepare_fc_contribution_pipeline import ROOT, SOURCE, EXPERIMENT
from run_fc_bitmap_mapping import vivado_environment


class BankedBitmapTest(unittest.TestCase):
    def test_vivado_compatibility_environment(self):
        env = vivado_environment('/tools/Vivado/2022.2/bin/vivado',
                                 {'PATH': '/usr/bin', 'LD_LIBRARY_PATH': '/extra'})
        self.assertEqual(env['XILINX_VIVADO'], '/tools/Vivado/2022.2')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/tools/Vivado/2022.2/lib/lnx64.o/SuSE:/extra')
        self.assertTrue(env['PATH'].endswith(':/usr/bin'))

    def test_only_bank_storage_changes(self):
        source = (ROOT / EXPERIMENT / "candidate_sparse_r2" / SOURCE).read_text(encoding="utf-8")
        result = candidate(source)
        self.assertNotIn("trace_nonzero_mem", result)
        self.assertIn('(* ram_style = "distributed" *) logic nonzero_mem [0:MAX_HIDDEN-1];', result)
        self.assertEqual(source[source.index("    logic hidden_input_matches;"):].replace(
            "trace_nonzero_mem[s][emit_hidden_q]", "trace_nonzero_read[s]"),
            result[result.index("    logic hidden_input_matches;"):])

    def test_full_capacity_cycle_comparison(self):
        source = (ROOT / EXPERIMENT / "tb_fc_contribution_pipeline.sv").read_text(encoding="utf-8")
        result = testbench(source)
        self.assertIn("H = 2048", result)
        self.assertEqual(result.count("run_case(16, 2048, 10"), 2)
        self.assertNotIn("cycles + 7*d", result)
        self.assertNotIn("no scan improvement", result)
        self.assertIn("bitmap layout payload cycle mismatch", result)
        self.assertIn("bitmap layout control cycle mismatch", result)


if __name__ == "__main__":
    unittest.main()
