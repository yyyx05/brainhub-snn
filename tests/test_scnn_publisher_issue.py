import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from prepare_scnn_publisher_issue import ROOT, SOURCE, candidate
from run_scnn_publisher_issue_unit import parse_log
from run_srnn_resident_remote import compatibility


class PublisherIssueTests(unittest.TestCase):
    def test_narrow_transform(self):
        original = (ROOT / SOURCE).read_text(encoding="utf-8")
        changed = candidate(original)
        self.assertIn("o_forward_read_valid = state_q == STATE_GATHER", changed)
        self.assertNotIn("state_q <= STATE_READ_REQUEST;", changed)
        self.assertEqual(original.count("i_forward_response_error"), changed.count("i_forward_response_error"))
        self.assertEqual(original[original.index("                STATE_READ_RESPONSE: begin"):],
                         changed[changed.index("                STATE_READ_RESPONSE: begin"):])

    def test_source_anchor_required(self):
        with self.assertRaises(ValueError):
            candidate("not a publisher")

    def test_strict_completion(self):
        with self.assertRaises(ValueError):
            parse_log("SCNN_PUBLISHER_UNIT_PASS variants=2 cases_per_variant=14\n")
        with self.assertRaises(ValueError):
            parse_log("SCNN_PUBLISHER_UNIT_PASS variants=2 cases_per_variant=14\n%Fatal late failure")

    def test_combined_scnn_overlay_dispatch(self):
        with patch("run_srnn_resident_remote.scnn_compatibility") as launch:
            compatibility(None, "new_run", "srnn_specialized_r1", "scnn", "candidate_sparse_r2", "candidate_r1")
            launch.assert_called_once_with(None, "new_run", "srnn_specialized_r1", "scnn", "candidate_sparse_r2", "candidate_r1")

    def test_no_optimization_in_passive_probe(self):
        for workload, candidate_name in (("scnn_probe", "candidate_r1"), ("scnn", "../candidate_r1")):
            with self.assertRaises(ValueError):
                compatibility(None, "new_run", "srnn_specialized_r1", workload, None, candidate_name)


if __name__ == "__main__":
    unittest.main()
