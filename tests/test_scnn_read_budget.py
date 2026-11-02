import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from summarize_scnn_read_budget import numeric_constant, reconcile


class ReadBudgetTests(unittest.TestCase):
    def fixture(self):
        source = "localparam int CLASSES = 2;\n" + "\n".join(
            f"localparam int CONV{i}_CHANNELS = 4; localparam int CONV{i}_CONTRIBUTIONS = 18;" for i in range(1,9))
        result = {}
        for arm, commits in (("off",0),("on",1)):
            banks = [dict(bank=str(i),forward_reads="288",forward_responses="288") for i in range(8)]
            banks.append(dict(bank="8",forward_reads=str(8*(1+commits)),forward_responses=str(8*(1+commits))))
            result[arm] = dict(banks=banks,final=dict(features=16,frames=4,commits=commits),
                               traffic=dict(forward_reads=2304+8*(1+commits)))
        return source,result

    def test_one_matrix_per_frame(self):
        rows, head = reconcile(*self.fixture())
        self.assertEqual(head,8)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(r["on_reads_per_unique_weight_per_frame"] == 1 for r in rows))

    def test_extra_reload_is_not_hidden(self):
        source, result = self.fixture()
        result["on"]["banks"][2].update(forward_reads="576",forward_responses="576")
        with self.assertRaises(ValueError):
            reconcile(source,result)

    def test_nonliteral_geometry_not_guessed(self):
        with self.assertRaises(ValueError):
            numeric_constant("localparam int F = 2*4;", "F")


if __name__ == "__main__":
    unittest.main()
