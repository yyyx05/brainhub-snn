import sys
from pathlib import Path
import re
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_m2_checkpoint_roundtrip import completed, TCL


class CheckpointProbeTests(unittest.TestCase):
    def test_marker_and_exit_required(self):
        marker = "CHECKPOINT_ROUNDTRIP_COMPLETE\n"
        self.assertTrue(completed(0, marker))
        self.assertFalse(completed(1, marker))
        self.assertFalse(completed(0, "# puts CHECKPOINT_ROUNDTRIP_COMPLETE\n"))
        self.assertFalse(completed(0, marker + marker))
        self.assertFalse(completed(0, marker + "ERROR: write failed\n"))

    def test_no_physical_reexecution(self):
        root = Path(__file__).resolve().parents[3]
        text = (root / TCL).read_text()
        commands = re.findall(r"^\s*([a-z_]+)\b", text, re.M)
        self.assertFalse(set(commands) & {"synth_design", "opt_design", "place_design", "route_design",
                                          "phys_opt_design", "write_bitstream", "report_power"})
        self.assertEqual(commands.count("open_checkpoint"), 2)
        self.assertEqual(commands.count("write_checkpoint"), 1)


if __name__ == "__main__":
    unittest.main()
