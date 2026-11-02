import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_scnn_row_turnover_synthesis as runner


class CandidateSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(runner.__file__).with_name("run_final_raw_source_synthesis.py").read_text()
        cls.generated = runner.worker_source(cls.original)

    def test_candidate_only_worker_loop(self):
        tree = ast.parse(self.generated)
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        arms = [n for n in ast.walk(worker) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "arm"]
        self.assertEqual(len(arms), 1)
        self.assertEqual(ast.literal_eval(arms[0].iter), ("candidate",))

    def test_existing_classification_and_archival_unchanged(self):
        def functions(source):
            return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        before, after = functions(self.original), functions(self.generated)
        for name in ("classify", "archive_files", "active_eda"):
            self.assertEqual(before[name], after[name])

    def test_timeout_and_reserves_retained(self):
        for text in ("process.wait(timeout=7200)", "TIMEOUT_PROCESS_RETAINED",
                     "512*1024**2", "4*1024**3", "128*1024**2"):
            self.assertIn(text, self.generated)
        self.assertNotIn("process.kill(", self.generated)
        self.assertNotIn("process.terminate(", self.generated)

    def test_unknown_template_rejected(self):
        with self.assertRaises(ValueError):
            runner.worker_source(self.original.replace("NAME = 'final_raw_source_synth_r1'", "NAME = 'other'"))


if __name__ == "__main__":
    unittest.main()
