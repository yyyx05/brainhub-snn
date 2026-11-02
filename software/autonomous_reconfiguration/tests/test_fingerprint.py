from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snn_reconfig.fingerprint import fingerprint_event_samples


class FingerprintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(
            (ROOT / "examples" / "normalized_event_samples_smoke.json").read_text(
                encoding="utf-8"
            )
        )

    def test_normalized_event_fingerprint(self) -> None:
        result = fingerprint_event_samples(
            self.payload["samples"], self.payload["metadata"]
        )
        expected_log_mean = (math.log1p(3) + math.log1p(2)) / 2
        self.assertAlmostEqual(result["event_count_log_mean"], expected_log_mean)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["ordered_channel_count"], 4)
        self.assertGreater(result["temporal_entropy"], 0.0)
        self.assertGreater(result["channel_entropy"], 0.0)

    def test_mismatched_event_arrays_are_rejected(self) -> None:
        samples = [
            {
                "duration_seconds": 1.0,
                "timestamps_seconds": [0.1],
                "channel_ids": [],
            }
        ]
        with self.assertRaises(ValueError):
            fingerprint_event_samples(samples, self.payload["metadata"])


if __name__ == "__main__":
    unittest.main()
