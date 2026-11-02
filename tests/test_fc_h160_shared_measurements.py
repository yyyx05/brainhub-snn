import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fc_h160", ROOT / "scripts/run_fc_h160_shared_numeric.py")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LOG = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_repair/h160/run_20261006_111424_191777/run.log"


class Measurements(unittest.TestCase):
    def setUp(self):
        self.text = LOG.read_text(encoding="utf-8")

    def test_accepted_real_log(self):
        result = MODULE.parse_measurements(self.text)
        self.assertEqual(result["load"]["parameter_bytes"], 743040)
        self.assertEqual(result["passes"][0]["cold_cycles_including_refill_and_init"], 808056)
        self.assertEqual(result["passes"][0]["state_trace_interface_bytes"], 272000)
        self.assertEqual(result["passes"][0]["dense_model_sops"], 37152000)
        self.assertFalse(result["final_top_simulated"])
        self.assertFalse(result["physical_timing_applies"])

    def test_missing_pass(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("FC_H160_SHARED_NUMERIC_PASS", "INCOMPLETE"))

    def test_incomplete_refill(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("parameter_responses=371520", "parameter_responses=371519"))

    def test_check_count_mismatch(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("q6_checks=2000", "q6_checks=1999"))

    def test_phase_mismatch(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("cycles=17930", "cycles=17931", 1))

    def test_traffic_mismatch(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("weight_read_bytes=613500", "weight_read_bytes=24540", 1))

    def test_backpressure_work_mismatch(self):
        with self.assertRaises(ValueError):
            MODULE.parse_measurements(self.text.replace("commands=1866", "commands=1867", 1))


if __name__ == "__main__":
    unittest.main()
