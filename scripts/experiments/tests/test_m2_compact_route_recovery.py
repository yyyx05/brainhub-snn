import re
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_m2_compact_route_recovery import route_tcl, implementation_status, REOPEN, ORIGINAL_TCL
from run_srnn_resident_remote import ROOT, RESULT, local_path


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(local_path(ROOT / RESULT / "m2_compact_impl_r1" / ORIGINAL_TCL)).read_text()
        cls.generated = route_tcl(cls.original)

    def test_resume_omits_completed_stages(self):
        commands = re.findall(r"^([a-z_]+)\b", self.generated, re.M)
        self.assertFalse(set(commands) & {"synth_design", "opt_design", "place_design", "write_bitstream", "report_power"})
        self.assertEqual(commands.count("route_design"), 1)
        self.assertEqual(commands.count("phys_opt_design"), 1)
        self.assertIn("route_design -directive AggressiveExplore\nphys_opt_design -directive Explore", self.generated)
        self.assertIn("set_param general.maxThreads 8", self.generated)

    def test_route_reports_and_acceptance_block_unchanged(self):
        start = "# RUN115 and its route-only RUN116"
        before = self.original[self.original.index(start):]
        after = self.generated[self.generated.index(start):]
        self.assertEqual(after.replace(REOPEN.strip(), "write_checkpoint -force [file join $out_dir post_route_physopt.dcp]"), before)

    def test_pass_requires_exit_crc_and_reopen(self):
        args = (0, "", {"status": "pass"}, {"status": "pass"}, True)
        self.assertEqual(implementation_status(*args), "PASS_route_recovery_checkpoint_reopened")
        self.assertEqual(implementation_status(1, *args[1:]), "FAIL")
        self.assertEqual(implementation_status(*args[:4], None), "FAIL")
        self.assertEqual(implementation_status(*args[:3], {}, True), "FAIL")
        self.assertEqual(implementation_status(0, "ERROR: failure\n", *args[2:]), "FAIL")
        self.assertTrue(implementation_status(0, "", {"status": "diagnostic_complete"}, args[3], True).startswith("DIAGNOSTIC"))


if __name__ == "__main__":
    unittest.main()
