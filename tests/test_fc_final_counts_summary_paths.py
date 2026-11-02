"""Remote result retrieval must not require changing the original commands.json."""
from pathlib import Path
import unittest

from scripts.summarize_fc_final_counts import fixture_path


class RetrievedPathTests(unittest.TestCase):
    def test_linux_record_rebased_to_local_project(self):
        self.assertEqual(fixture_path({"cwd": "/remote/project",
            "fixture": "/remote/project/results/fc/on"}, root=Path("local")),
            Path("local/results/fc/on"))

    def test_windows_record_rebased_without_drive_assumptions(self):
        self.assertEqual(fixture_path({"cwd": "D:\\my project", "fixture": "D:\\my project\\results\\fc"},
                                      root=Path("local")), Path("local/results/fc"))

    def test_outside_recorded_root_rejected(self):
        for fixture in ("/other/results/fc", "/remote/project/../elsewhere"):
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                fixture_path({"cwd": "/remote/project", "fixture": fixture})


if __name__ == "__main__":
    unittest.main()
