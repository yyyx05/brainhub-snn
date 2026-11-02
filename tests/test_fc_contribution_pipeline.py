import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from prepare_fc_contribution_pipeline import candidate, SOURCE
from run_srnn_resident_remote import compatibility


class ScannerPreparationTests(unittest.TestCase):
    def test_interface_and_observation_unchanged(self):
        source = (ROOT / SOURCE).read_text(encoding="utf-8-sig")
        changed = candidate(source)
        self.assertEqual(source[source.index("module "):source.index("    localparam")],
                         changed[changed.index("module "):changed.index("    localparam")])
        self.assertEqual(source[source.index("                STATE_IDLE: begin"):source.index("                STATE_SEALED: begin")],
                         changed[changed.index("                STATE_IDLE: begin"):changed.index("                STATE_SEALED: begin")])

    def test_preparation_matches_executed_candidate(self):
        source = (ROOT / SOURCE).read_text(encoding="utf-8-sig")
        saved = ROOT / "verification/experiments/fc_contribution_pipeline/candidate_r1" / SOURCE
        self.assertEqual(candidate(source), saved.read_text(encoding="utf-8"))

    def test_overlay_rejects_wrong_workload_before_remote_access(self):
        with self.assertRaises(ValueError):
            compatibility(None, "unit", "srnn_specialized_r1", "frontend", "candidate_r1")
        with self.assertRaises(ValueError):
            compatibility(None, "unit", "srnn_specialized_r1", "fc", "../candidate_r1")


if __name__ == "__main__":
    unittest.main()
