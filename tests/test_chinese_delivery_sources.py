"""Focused source-selection checks; no package or experiment is modified."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_chinese_delivery import expected_rtl_files, local_link_target


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("rtl/top/final.sv", "rtl/core/pe.sv", "rtl/history/old.sv"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("module example; endmodule\n", encoding="ascii")
        self.source_list = self.root / "sources.f"

    def test_legacy_default_keeps_full_tree(self):
        self.assertEqual(len(expected_rtl_files(self.root)), 3)

    def test_existing_reachable_list_excludes_history(self):
        self.source_list.write_text("# final top\nrtl/top/final.sv\nrtl\\core\\pe.sv\n", encoding="ascii")
        self.assertEqual(expected_rtl_files(self.root, self.source_list),
                         {"rtl/top/final.sv", "rtl/core/pe.sv"})

    def test_empty_duplicate_missing_and_escaping_lists_rejected(self):
        for content in ("", "rtl/top/final.sv\nrtl/top/final.sv", "rtl/missing.sv",
                        "rtl/../core.sv", "D:/other.sv", "/rtl/top/final.sv",
                        "rtl//top/final.sv", "rtl/top/final.txt"):
            with self.subTest(content=content):
                self.source_list.write_text(content, encoding="ascii")
                with self.assertRaises(ValueError):
                    expected_rtl_files(self.root, self.source_list)

    def test_local_link_may_traverse_inside_delivery(self):
        document = self.root / "docs/guide.md"
        self.assertEqual(local_link_target(document, self.root, "../rtl/top/final.sv"),
                         self.root / "rtl/top/final.sv")

    def test_link_cannot_succeed_by_reading_original_host_folder(self):
        document = self.root / "docs/guide.md"
        with self.assertRaises(ValueError):
            local_link_target(document, self.root, "../../original-project/README.md")


if __name__ == "__main__":
    unittest.main()
