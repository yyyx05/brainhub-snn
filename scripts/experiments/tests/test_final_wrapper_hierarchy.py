"""Check the narrow offline hierarchy correction without rerunning RTL."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_final_raw_source_functional import compare_wrapper_hierarchy


class HierarchyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.old = {"total": {"cycles": 123}, "tiles": [
            f"TOP.tb.u_final.array.g_region[{r}].g_tile[{t}].monitor"
            for r in range(3) for t in range(4)]}
        self.new = copy.deepcopy(self.old)
        self.new["tiles"] = [p.replace(".u_final.", ".u_final.u_core.") for p in self.old["tiles"]]
        self.old_log = "\n".join("H40_ACTIVITY tile=" + p + " cycles=123 useful=100"
                                 for p in self.old["tiles"])
        self.new_log = self.old_log.replace(".u_final.", ".u_final.u_core.")

    def test_only_one_expected_hierarchy_insertion(self):
        self.assertEqual(compare_wrapper_hierarchy(self.new, self.old, self.new_log, self.old_log), self.new)

    def test_cycle_and_raw_counter_drift_rejected(self):
        self.new["total"]["cycles"] += 1
        with self.assertRaisesRegex(ValueError, "non-hierarchy"):
            compare_wrapper_hierarchy(self.new, self.old, self.new_log, self.old_log)
        self.new["total"]["cycles"] -= 1
        with self.assertRaisesRegex(ValueError, "raw tile"):
            compare_wrapper_hierarchy(self.new, self.old, self.new_log.replace("useful=100", "useful=99", 1), self.old_log)

    def test_wrong_or_missing_tile_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw tile"):
            compare_wrapper_hierarchy(self.new, self.old, self.new_log.replace("g_tile[0]", "g_tile[9]", 1), self.old_log)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare_wrapper_hierarchy(self.new, self.old, "\n".join(self.new_log.splitlines()[:-1]), self.old_log)


if __name__ == "__main__":
    unittest.main()
