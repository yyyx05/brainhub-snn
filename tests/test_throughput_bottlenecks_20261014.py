import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_throughput_bottlenecks_20261014.py"
SPEC = importlib.util.spec_from_file_location("throughput_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ThroughputAnalysisTest(unittest.TestCase):
    def test_rate_formula(self):
        self.assertAlmostEqual(MODULE.rate(1_013_344, 16), 1052.620499, places=6)
        self.assertAlmostEqual(MODULE.rate(1_013_344, 16 * 99), 104209.429374, places=6)

    def test_current_evidence_closes(self):
        baseline, phases, candidates = MODULE.collect_current(ROOT)
        self.assertEqual(len(baseline), 6)
        self.assertEqual(len(phases), 30)
        self.assertEqual(len(candidates), 6)
        srnn_off = next(row for row in baseline if row["profile"] == "SRNN" and row["arm"] == "off")
        self.assertEqual(srnn_off["cycles"], 110_452_102)
        self.assertEqual(srnn_off["complete_samples_per_s"], "0.603580")

    def test_srnn_model_is_explicitly_unmeasured(self):
        _, _, candidates = MODULE.collect_current(ROOT)
        first_sample = next(
            row
            for row in candidates
            if row["arm"] == "off" and row["candidate"] == "resident_backbone_and_m2_first_sample"
        )
        self.assertEqual(first_sample["cycles"], "1903606.000")
        self.assertEqual(first_sample["projected_complete_samples_per_s"], "35.021253")
        self.assertIn("not yet RTL-verified", first_sample["evidence_status"])

    def test_resource_model_stays_below_half_device_as_capacity_estimate(self):
        rows = MODULE.resource_rows()
        total = next(row for row in rows if row["item"] == "current_plus_planning_ceiling")
        self.assertLess(float(total["device_share"]), 0.5)
        self.assertIn("unverified", total["status"])


if __name__ == "__main__":
    unittest.main()
