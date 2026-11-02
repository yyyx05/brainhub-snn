import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_scnn_row_turnover_implementation as runner


class CandidateImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(runner.__file__).parent
        cls.original = (scripts / "run_final_raw_source_implementation.py").read_text()
        cls.generated = runner.worker_source(cls.original)
        cls.audit_original = (scripts / "audit_final_raw_source_implementation.py").read_text()

    def test_one_arm_and_new_identity(self):
        values = {}
        for n in ast.parse(self.generated).body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                try:
                    values[n.targets[0].id] = ast.literal_eval(n.value)
                except ValueError:
                    pass
        self.assertEqual(values["ARMS"], ("candidate",))
        self.assertEqual(values["NAME"], runner.NAME)
        self.assertEqual(values["SYNTHESIS"], runner.SYNTHESIS)

    def test_unchanged_physical_recipe_and_classifier(self):
        def functions(text):
            return {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(self.original), functions(self.generated)
        for name in ("implementation_tcl", "classify", "fields"):
            self.assertEqual(a[name], b[name])

    def test_audit_checks_not_weakened(self):
        generated = runner.audit_source(self.audit_original)
        def checks(text):
            return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "checks"]
        self.assertEqual(checks(self.audit_original), checks(generated))
        self.assertNotIn("both_internal_implementations_accepted", generated)
        self.assertIn("candidate_internal_implementation_accepted", generated)

    def test_timeout_reserves_and_negative_sta_kept(self):
        for text in ("process.wait(timeout=7200)", "768 * 1024**2", "6 * 1024**3",
                     "DIAGNOSTIC_implementation", "TIMEOUT_PROCESS_RETAINED"):
            self.assertIn(text, self.generated)


if __name__ == "__main__":
    unittest.main()
