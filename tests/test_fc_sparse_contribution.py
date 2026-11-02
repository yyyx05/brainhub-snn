import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from prepare_fc_sparse_contribution import candidate, ROOT, EXPERIMENT, SOURCE, trace_profile


class SparsePreparationTests(unittest.TestCase):
    def test_edit_scope(self):
        source = (ROOT / EXPERIMENT / "candidate_r1" / SOURCE).read_text(encoding="utf-8")
        changed = candidate(source)
        self.assertEqual(source[:source.index("    localparam")],
                         changed[:changed.index("    localparam")])
        self.assertIn("batch_count_q > BATCH_COUNT_WIDTH'(2)", changed)
        self.assertIn("(s < int'(batch_count_q))", changed)
        self.assertIn("if (emit_advance)", changed)
        self.assertNotIn("emit_stream_last", changed)

    def test_fixture_accounting(self):
        result = trace_profile()
        self.assertEqual(result["traces"], 253440)
        self.assertEqual(result["nonzero_traces"], 64630)
        self.assertEqual(sum(result["nonzero_per_window"]), result["nonzero_traces"])
        self.assertEqual(result["unstalled_issue_plus_mask_cycles_model"], 841720)
        self.assertEqual(sum(result["active_samples_per_hidden_histogram"].values()), 99 * 160)


if __name__ == "__main__":
    unittest.main()
