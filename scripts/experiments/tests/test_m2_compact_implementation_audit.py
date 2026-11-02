import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_m2_compact_implementation import checkpoint, resources, violations, report_checks_pass
from audit_m2_compact_synthesis import timing
from run_srnn_resident_remote import ROOT, RESULT, local_path

REPORT = Path(local_path(ROOT / RESULT / "m2_compact_impl_r1/server_results/final_composition_impl"))


class ImplementationAuditTests(unittest.TestCase):
    def test_actual_report_resources(self):
        parsed = resources((REPORT / "utilization_post_route.rpt").read_text())
        self.assertEqual(parsed, dict(lut=192859, lut_logic=190977, lut_memory=1882,
                                     ff=80374, bram36eq=594.5, dsp=1196))

    def test_absent_final_checkpoint_is_unknown_crc_not_pass(self):
        self.assertEqual(checkpoint(REPORT / "post_route_physopt.dcp"),
                         dict(present=False, bytes=None, crc_pass=None))

    def test_report_zero_slack_does_not_hide_endpoint_or_route_failures(self):
        parsed = timing((REPORT / "timing_summary_post_route.rpt").read_text())
        route = dict(routable=353696, fully_routed=353696, errors=0)
        self.assertTrue(report_checks_pass(parsed, route))
        self.assertFalse(report_checks_pass(dict(parsed, hold_failing=1), route))
        self.assertFalse(report_checks_pass(parsed, dict(route, fully_routed=353695)))

    def test_actual_drc_limit_and_incomplete_rule_table(self):
        text = (REPORT / "drc_post_route.rpt").read_text()
        parsed = violations(text)
        self.assertEqual(parsed["total"], 3727)
        self.assertIn(dict(rule="CHECK-3", severity="Warning", count=2), parsed["rules"])
        with self.assertRaises(ValueError):
            violations(text.replace("Violations found: 3727", "Violations found: 3728"))


if __name__ == "__main__":
    unittest.main()
