import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final_fc_functional as runner


class FinalFCTests(unittest.TestCase):
    def test_wrapper_changes_only_boundary_and_observer_paths(self):
        root, _, _ = runner.roots()
        original = (root / runner.PRIOR_PREPARED / runner.TB).read_text()
        changed = runner.bind_wrapper(original)
        self.assertEqual(re.findall(r'\$fatal\(1,("[^"]*")', original),
                         re.findall(r'\$fatal\(1,("[^"]*")', changed))
        self.assertEqual(original.count("$readmemh"), changed.count("$readmemh"))
        self.assertIn(".i_raw_source_active(1'b0)", changed)
        self.assertIn("unified_h40_raw_source_top_v1 #(", changed)
        self.assertNotIn("u_final.u_core.u_core.", changed)
        self.assertEqual(changed.count("FINAL_FC_WRAPPER_PASS"), 1)

    def test_missing_anchor_rejected(self):
        with self.assertRaises(ValueError):
            runner.bind_wrapper("module dummy; endmodule")


if __name__ == "__main__":
    unittest.main()
