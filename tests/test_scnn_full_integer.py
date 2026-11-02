import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_scnn_full_integer import equal, lif, pool_q8, wrap


class SCNNFullIntegerTest(unittest.TestCase):
    def test_signed_wrap(self):
        x = torch.tensor([-65537, -65536, -32769, -32768, 32767, 32768, 65536])
        self.assertEqual(wrap(x, 16).tolist(), [-1, 0, 32767, -32768, 32767, -32768, 0])

    def test_lif_boundary_and_negative_floor(self):
        currents = [-32768, -1, 0, 255, 256, 32767]
        states = [-32768, -1, 0, 1, 32767]
        for current in currents:
            for state in states:
                result = lif(torch.tensor([current]), torch.tensor([state]))
                pre = ((current + state * 135 // 256 + 65536) % 131072) - 65536
                pre16 = (pre + 32768) % 65536 - 32768
                after = (pre - 256 + 65536) % 131072 - 65536
                post = min(32767, max(-32768, after)) if pre >= 256 else pre16
                self.assertEqual([int(x[0]) for x in result[:4]],
                                 [pre16, post, int(pre >= 256), int(pre < -32768 or pre > 32767)])

    def test_pool_uses_magnitude_not_occupancy(self):
        x = torch.tensor([[[[1, 1, 0, 0], [1, 0, 0, 0],
                            [1, 1, 1, 0], [1, 1, 0, 0]]]], dtype=torch.bool)
        self.assertEqual(pool_q8(x, "AAVG2").flatten().tolist(), [192, 0, 256, 64])
        self.assertEqual(pool_q8(x, "MAX2").flatten().tolist(), [256, 0, 256, 256])
        self.assertTrue(torch.equal(pool_q8(x, "IDENTITY"), x.to(torch.int64) * 256))

    def test_equal_rejects_shape_and_value(self):
        self.assertEqual(equal([1, 2], np.array([1, 2]), "test"), 2)
        with self.assertRaises(AssertionError):
            equal([1], [1, 2], "shape")
        with self.assertRaises(AssertionError):
            equal([1, 3], [1, 2], "value")


if __name__ == "__main__":
    unittest.main()
