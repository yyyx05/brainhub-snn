from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snn_reconfig.dataset_adapters import fingerprint_dvsgesture_q8_8_hex


class DatasetAdapterTest(unittest.TestCase):
    def test_small_dvs_q8_8_export(self) -> None:
        # Two samples, two timesteps, two channels, one pixel. Values are Q8.8.
        words = [0x0100, 0x0000, 0x0200, 0x0100, 0x0000, 0x0100, 0x0000, 0x0300]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.hex"
            path.write_text("".join(f"{word:04x}\n" for word in words), encoding="ascii")
            result = fingerprint_dvsgesture_q8_8_hex(
                path,
                2,
                timesteps=2,
                polarity_channels=2,
                height=1,
                width=1,
            )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["ordered_channel_count"], 2)
        self.assertEqual(result["spatial_shape"], [1, 1])
        self.assertGreater(result["event_count_log_mean"], 0.0)

    def test_dvs_word_count_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.hex"
            path.write_text("0100\n", encoding="ascii")
            with self.assertRaises(ValueError):
                fingerprint_dvsgesture_q8_8_hex(
                    path,
                    1,
                    timesteps=1,
                    polarity_channels=2,
                    height=1,
                    width=1,
                )


if __name__ == "__main__":
    unittest.main()
