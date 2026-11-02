import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from run_srnn_resident_remote import select_placement


class PlacementSelectionTest(unittest.TestCase):
    SCRIPT = """read_checkpoint $source_dcp
opt_design
place_design -directive AltSpreadLogic_high
phys_opt_design -directive ExploreWithHoldFix
route_design -directive AggressiveExplore
phys_opt_design -directive Explore
"""

    def test_default_is_unchanged(self):
        self.assertEqual(select_placement(self.SCRIPT, "AltSpreadLogic_high"), self.SCRIPT)

    def test_net_delay_changes_only_placement(self):
        changed = select_placement(self.SCRIPT, "ExtraNetDelay_high")
        self.assertEqual(changed.replace("ExtraNetDelay_high", "AltSpreadLogic_high"), self.SCRIPT)
        self.assertEqual(changed.count("place_design -directive ExtraNetDelay_high"), 1)

    def test_rejects_unknown_or_ambiguous_input(self):
        with self.assertRaises(ValueError):
            select_placement(self.SCRIPT, "Quick")
        for source in ("opt_design\n", self.SCRIPT + self.SCRIPT):
            with self.assertRaises(ValueError):
                select_placement(source, "ExtraNetDelay_high")

    def test_future_report_label_matches_actual_command(self):
        original = self.SCRIPT + 'puts $placement_gate "place_directive=AltSpreadLogic_high"\n'
        changed = select_placement(original, "ExtraNetDelay_high")
        self.assertIn('place_directive=ExtraNetDelay_high', changed)
        self.assertNotIn('AltSpreadLogic_high', changed)


if __name__ == "__main__":
    unittest.main()
