import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_srnn_resident_remote import ROOT, RESULT, local_path
from run_scnn_turnover_frame import BANK_READS, parse_log, transform


class FrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        accepted=Path(local_path(ROOT/RESULT/"scnn_schedule_probe_r2"))
        cls.tb=(accepted/"tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
        cls.reference=json.loads((accepted/"reference.json").read_text())["expected"]["off"]
        original=(accepted/"run_off.log").read_text()
        banks="".join(f"SCNN_TURNOVER_BANK bank={i} reads={n} responses={n}\n" for i,n in enumerate(BANK_READS))
        cls.log=original+banks+"SCNN_TURNOVER_SERVICE_PASS reads=9239680 responses=9239680 replacements=0 slots=1 code_bits=16 address_bits=24 tag_bits=16 minimum_latency=1 replaceable=1 m2_cache_depth=512 end_cycle=39154300\n"

    def test_original_checks_retained_with_service_scoreboard(self):
        result=parse_log(self.log,self.reference)
        self.assertEqual(result["frame_cycles"],39082705)
        self.assertEqual(result["initialization_cycles"],71591)
        self.assertEqual(result["service"]["replacements"],0)

    def test_numeric_mutation_rejected(self):
        with self.assertRaises(ValueError):
            parse_log(self.log.replace("pooled_q8_8_values=2048","pooled_q8_8_values=2047"),self.reference)

    def test_missing_or_duplicate_service_rejected(self):
        for log in (self.log[:self.log.index("SCNN_TURNOVER_SERVICE_PASS")],self.log+self.log[self.log.index("SCNN_TURNOVER_SERVICE_PASS"):]):
            with self.assertRaises(ValueError):parse_log(log,self.reference)

    def test_bank_work_and_width_rejected(self):
        for old,new in (("bank=0 reads=1152","bank=0 reads=1151"),("address_bits=24","address_bits=17"),("responses=9239680 replacements","responses=9239679 replacements")):
            with self.assertRaises(ValueError):parse_log(self.log.replace(old,new),self.reference)

    def test_transform_preserves_request_priority_and_native_width(self):
        out=transform(self.tb)
        start=out.index("always @(posedge clk) begin : temporal_external_ddr")
        self.assertLess(out.index("fg_forward_valid <= 0;",start),out.index("fg_forward_valid <= 1;",start))
        self.assertIn("pm_response_bank <= bank;",out)
        self.assertIn(".M2_CACHE_DEPTH(512)",out)
        self.assertEqual(out.count(".g_context[0].g_enabled.g_original.u_store.g_bank["),2)
        self.assertNotIn(".g_context[0].g_enabled.u_store",out)
        self.assertIn("localparam int PARAM_ADDR_WIDTH = 24;",out)
        self.assertIn("break; // First-frame diagnostic only.",out)
        self.assertEqual(out.count("st_report();"),1)
        self.assertIn("turnover held response changed",out)


if __name__=="__main__":unittest.main()
