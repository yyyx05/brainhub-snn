import ast
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_scnn_row_reuse_core as runner
from run_srnn_resident_remote import ROOT, RESULT, local_path


class RowReuseCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(runner.__file__).with_name("run_final_scnn_functional.py").read_text()
        cls.checks = json.loads(Path(local_path(ROOT / RESULT / runner.REFERENCE / "checks.json")).read_text())

    def test_only_descriptive_labels_excluded(self):
        changed = copy.deepcopy(self.checks)
        changed.update(status="new_label", scope="new_description")
        self.assertEqual(runner.numerical_records(changed), runner.numerical_records(self.checks))

    def test_cycles_and_traffic_are_not_excluded(self):
        for section, field in (("final", "cycles"), ("final", "code_changes"),
                               ("native_service", "end_cycle"), ("native_service", "reads"),
                               ("native_service", "replacements")):
            changed = copy.deepcopy(self.checks)
            changed[section][field] += 1
            self.assertNotEqual(runner.numerical_records(changed), runner.numerical_records(self.checks))
        for section, field in (("windows", "cycle"), ("layers", "cycles"), ("frames", "cycles")):
            changed = copy.deepcopy(self.checks)
            changed[section][0][field] += 1
            self.assertNotEqual(runner.numerical_records(changed), runner.numerical_records(self.checks))

    def test_execution_and_archive_functions_preserved(self):
        text = runner.worker_source(self.original)
        actual, prior = ast.parse(text), ast.parse(self.original)
        for name in ("worker", "launch", "transport"):
            new = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == name)
            old = next(n for n in prior.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(new), ast.dump(old))
        self.assertIn("cycle/numerical/traffic/commit/version records changed", text)
        self.assertIn("process.wait(timeout=timeout)", text)
        self.assertNotIn("process.kill(", text)


if __name__ == "__main__":
    unittest.main()
