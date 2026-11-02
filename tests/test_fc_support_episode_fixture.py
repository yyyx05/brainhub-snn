import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.build_fc_support_episode_fixture import ROOT, run_batch


class ContinuousEpisodeTest(unittest.TestCase):
    def setUp(self):
        base = ROOT / "work/supplement_20261004/fc_episode_unit"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.out = Path(self.temp.name)
        self.counts = np.zeros((16, 1, 2312), dtype=np.int64)
        self.counts[:, 0, 0] = 100
        self.labels = np.zeros(16, dtype=np.int64)
        self.w0 = np.zeros((2312, 160), dtype=np.int64)
        self.w0[0] = 15
        self.w2 = np.zeros((160, 10), dtype=np.int64)
        self.residual = np.zeros_like(self.w2)
        self.residual[:, 0] = 32767

    def tearDown(self):
        self.temp.cleanup()

    def run_one(self, name, w2, residual, learning=True):
        return run_batch(self.out / name, self.counts, self.labels, self.w0,
                         w2, residual, self.w2, 47303, learning)

    def test_parameter_and_residual_persist_but_sample_state_resets(self):
        w2, residual, first = self.run_one("first", self.w2, self.residual)
        self.assertEqual(first[0]["code_changes"], 160)
        self.assertTrue(np.all(w2[:, 0] == 1))
        _, continued_residual, _ = self.run_one("second", w2, residual)
        _, cleared_residual, _ = self.run_one("cleared_control", w2, np.zeros_like(residual))
        self.assertFalse(np.array_equal(continued_residual, cleared_residual))
        for name in ("first", "second"):
            pre = np.array([int(x, 16) for x in (self.out/name/"pre.mem").read_text().split()])
            np.testing.assert_array_equal(pre, np.full(16*160, 96000))
        q6 = np.array([int(x, 16) for x in (self.out/"second/q6.mem").read_text().split()]).reshape(16, 10)
        # 160*47303 = 115*65536 + 31840, below the half-LSB boundary.
        np.testing.assert_array_equal(q6[:, 0], np.full(16, 115))

    def test_learning_off_preserves_both_parameter_fields(self):
        w2, residual, records = self.run_one("off", self.w2, self.residual, False)
        np.testing.assert_array_equal(w2, self.w2)
        np.testing.assert_array_equal(residual, self.residual)
        self.assertEqual(records[0]["update_requests"], 0)
        self.assertEqual(records[0]["residual_changes"], 0)

    def test_caller_initial_state_is_not_modified(self):
        initial_w2, initial_residual = self.w2.copy(), self.residual.copy()
        self.run_one("copy", self.w2, self.residual)
        np.testing.assert_array_equal(self.w2, initial_w2)
        np.testing.assert_array_equal(self.residual, initial_residual)


if __name__ == "__main__":
    unittest.main()
