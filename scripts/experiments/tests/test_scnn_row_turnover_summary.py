from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_scnn_row_turnover_core import intervals, paired_rows


class PairedLatencyTests(unittest.TestCase):
    def test_signed_reductions_retain_regressions(self):
        rows = paired_rows([dict(frame=i, cycles=100) for i in range(3)],
                           [dict(frame=i, cycles=c) for i, c in enumerate((80, 100, 120))])
        self.assertEqual([r["saved_cycles"] for r in rows], [20, 0, -20])

    def test_pair_identity_and_count(self):
        with self.assertRaises(ValueError):
            paired_rows([dict(frame=0, cycles=10)], [dict(frame=1, cycles=10)])
        with self.assertRaises(ValueError):
            paired_rows([dict(frame=0, cycles=10)], [])

    def test_intervals_partition_total(self):
        data = dict(windows=[dict(window=1, cycle=5), dict(window=2, cycle=17)],
                    final=dict(cycles=30))
        self.assertEqual([r["cycles"] for r in intervals(data)], [5, 12, 13])
        self.assertEqual(sum(r["cycles"] for r in intervals(data)), 30)


if __name__ == "__main__":
    unittest.main()
