import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from run_m2_storage_shape import parse_unit
from audit_m2_storage_shape import utilization, timing
from run_srnn_resident_remote import ROOT, RESULT, local_path


class ParserTests(unittest.TestCase):
    text = ("M2_SHAPE_PASS depth=512 reads=9000 writes=8000 stalls=3000 conflicts=4000 replacements=3000\n"
            "M2_SHAPE_PASS depth=578 reads=9000 writes=8000 stalls=3000 conflicts=4000 replacements=3000\n"
            "M2_STORAGE_SHAPE_COMPLETE\n")

    def test_both_domains(self):
        self.assertEqual([r["depth"] for r in parse_unit(self.text)], [512, 578])

    def test_missing_duplicate_and_early(self):
        for text in (self.text.splitlines(True)[0]+self.text.splitlines(True)[2],
                     self.text+self.text, "M2_STORAGE_SHAPE_COMPLETE\n"+self.text.replace("M2_STORAGE_SHAPE_COMPLETE\n", "")):
            with self.assertRaises(ValueError):
                parse_unit(text)

    def test_fatal_and_uncovered(self):
        for text in (self.text+"%Error: failure", self.text.replace("stalls=3000", "stalls=0")):
            with self.assertRaises(ValueError):
                parse_unit(text)

    def test_actual_archived_unit_log(self):
        root = Path(local_path(ROOT/RESULT/"m2_storage_shape_r1"))
        rows = parse_unit((root/"run.log").read_text())
        self.assertEqual(sum(row["reads"] for row in rows), 14295)

    def test_actual_mapping_reports(self):
        root = Path(local_path(ROOT/RESULT/"m2_storage_shape_r1"))
        for variant, expected in (("baseline",20.0),("full578",18.5),("selected512",10.0)):
            usage = utilization((root/(variant+"_utilization.rpt")).read_text())
            self.assertEqual(usage["bram36eq"], expected)
            self.assertEqual(usage["bram36eq"],usage["bram36"]+usage["bram18"]/2)
            measured = timing((root/(variant+"_timing.rpt")).read_text())
            self.assertEqual(measured["setup_fail"], 0)

    def test_report_parser_rejects_missing_or_duplicate(self):
        root = Path(local_path(ROOT/RESULT/"m2_storage_shape_r1"))
        report = (root/"baseline_utilization.rpt").read_text()
        for text in (report.replace("Block RAM Tile","missing"),report+report):
            with self.assertRaises(ValueError):
                utilization(text)


if __name__ == "__main__":
    unittest.main()
