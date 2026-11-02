import ast
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_scnn_row_turnover_core as runner
from run_srnn_resident_remote import ROOT, RESULT, local_path


class RowTurnoverCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(runner.__file__).with_name("run_final_scnn_functional.py").read_text()
        cls.checks = json.loads(Path(local_path(ROOT / RESULT / runner.PRIOR / "checks.json")).read_text())

    def test_only_latency_and_overlap_excluded(self):
        changed = copy.deepcopy(self.checks)
        changed["final"]["cycles"] += 23
        changed["windows"][0]["cycle"] += 7
        changed["layers"][0]["cycles"] += 11
        changed["frames"][0]["cycles"] += 11
        changed["native_service"]["replacements"] += 1
        self.assertEqual(runner.numerical_records(changed), runner.numerical_records(self.checks))

    def test_numeric_or_credit_change_remains_visible(self):
        for section, field in (("final", "code_changes"), ("native_service", "slots"),
                               ("native_service", "reads")):
            changed = copy.deepcopy(self.checks)
            changed[section][field] += 1
            self.assertNotEqual(runner.numerical_records(changed), runner.numerical_records(self.checks))

    def test_actual_worker_and_limits_preserved(self):
        text = runner.worker_source(self.original)
        ast.parse(text)
        self.assertIn("non-latency numerical/traffic/commit/version records changed", text)
        self.assertIn('process.wait(timeout=timeout)', text)
        self.assertNotIn("physical_rtl_files_byte_equal=113", text)
        self.assertIn("Unchanged shared owner/config/RMW/publication", text)
        self.assertNotIn("process.kill(", text)


if __name__ == "__main__":
    unittest.main()
