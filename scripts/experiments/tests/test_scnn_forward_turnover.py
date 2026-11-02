import re
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_scnn_forward_turnover import SOURCES,publisher,backend,router
from run_srnn_resident_remote import ROOT,RESULT,local_path
from run_scnn_forward_turnover_unit import parse_log


class TurnoverTests(unittest.TestCase):
    def test_no_added_register_declarations(self):
        source=Path(local_path(ROOT/RESULT/"m2_compact_synth_r2"))
        for relative,change in zip(SOURCES,(publisher,backend,router)):
            before=(source/relative).read_text()
            after=change(before)
            registers=lambda text:[line.strip() for line in text.splitlines()
                                   if line.lstrip().startswith("logic ") and re.search(r"_q\s*;",line)]
            self.assertEqual(registers(before),registers(after))

    def test_pending_priority_and_owner_errors_unchanged(self):
        source=Path(local_path(ROOT/RESULT/"m2_compact_synth_r2"))
        for relative,change in zip(SOURCES[1:],(backend,router)):
            before=(source/relative).read_text()
            self.assertEqual(before[before.index("    always_ff"):],change(before)[change(before).index("    always_ff"):])
        text=router((source/SOURCES[2]).read_text())
        self.assertEqual(text.count("forward_response_fire && i_active_owner == forward_owner_q"),2)

    def test_reparse_actual_crossed_services(self):
        text=Path(local_path(ROOT/RESULT/"scnn_forward_turnover_unit_r1/run.log")).read_text()
        checked=parse_log(text)
        large=[r for r in checked["comparison"] if r["case_id"]==5]
        self.assertEqual([(r["service"],r["saved_cycles"]) for r in large],[(0,0),(1,20636)])

    def test_reject_legacy_service_false_gain(self):
        text=Path(local_path(ROOT/RESULT/"scnn_forward_turnover_unit_r1/run.log")).read_text()
        line=next(line for line in text.splitlines() if line.startswith("SCNN_TURNOVER_CASE service=0 opt=1 case_id=5 "))
        with self.assertRaises(ValueError):
            parse_log(text.replace(line,line.replace("cycles=47169","cycles=47168")))

    def test_reject_incomplete_log(self):
        text=Path(local_path(ROOT/RESULT/"scnn_forward_turnover_unit_r1/run.log")).read_text()
        with self.assertRaises(ValueError):
            parse_log(text.replace("SCNN_TURNOVER_UNIT_PASS","REMOVED"))

    def test_geometry_keeps_same_candidate_rtl(self):
        root=Path(local_path(ROOT/"verification/experiments/scnn_forward_turnover"))
        for relative in SOURCES:
            self.assertEqual((root/"candidate_r1"/relative).read_bytes(),
                             (root/"candidate_geometry_r2"/relative).read_bytes())
        tb=(root/"candidate_geometry_r2/tb_scnn_forward_turnover.sv").read_text()
        self.assertIn("run_case(4608, 16, 0)",tb)
        self.assertIn("MIN_RESPONSE_LATENCY = SERVICE == 2 ? 4 : SERVICE == 3 ? 12 : 1",tb)
        self.assertIn("wire [7:0] finished",tb)
        self.assertIn("logic signed [15:0] resp_code",tb)

    def test_actual_geometry_latency_profiles(self):
        text=Path(local_path(ROOT/RESULT/"scnn_forward_turnover_geometry_r2/run.log")).read_text()
        rows=[r for r in parse_log(text)["comparison"] if r["case_id"]==5]
        self.assertEqual([r["features"] for r in rows],[4608]*4)
        self.assertEqual([r["saved_cycles"] for r in rows],[0,67584,67584,67584])
        self.assertEqual([r["candidate_cycles"] for r in rows],[153665,86081,307265,897089])


if __name__=="__main__":
    unittest.main()
