"""Focused parser/accounting checks, including the existing report pair."""

import csv
import json
import re
import unittest

import summarize_supplement_learning_memory as target


def report(rows):
    lines = ["|" + "|".join(target.HEADER) + "|"]
    for depth, name, b36, b18 in rows:
        fields = [" " * (1 + 2 * depth) + name, "module_v1_12"]
        fields += [str(v) for v in (0, 0, 0, 0, 0, b36, b18, 0)]
        lines.append("|" + "|".join(fields) + "|")
    return "\n".join(lines)


class LearningMemoryTests(unittest.TestCase):
    def test_vivado_timing_device_spelling(self):
        self.assertEqual(target.device({"Device": "7vx690t-ffg1761", "Speed File": "-3 PRODUCTION 1.11"}),
                         target.device({"Device": "xc7vx690tffg1761-3"}))

    def test_self_and_children_are_not_double_counted(self):
        text = report([(0, "top", 12, 3), (1, "a", 9, 3),
                       (2, "(a)", 1, 1), (2, "b", 8, 2), (1, "other", 3, 0)])
        rows, _ = target.parse_hierarchy(text)
        values = target.partition(rows, "top/a", {"top/a"})
        self.assertEqual(values[("top/a", "self_minus_children")], (1, 1))
        self.assertEqual(target.add_all(list(values.values())), (9, 3))
        self.assertEqual(target.eq((9, 3)), 10.5)
        self.assertNotIn("top/a/(a)", rows)

    def test_missing_branch_and_unattributed_difference_remain_visible(self):
        on, al = target.parse_hierarchy(report([(0, "top", 12, 3), (1, "a", 9, 3)]))
        off, bl = target.parse_hierarchy(report([(0, "top", 2, 0)]))
        result = target.decompose(on, off, al, bl, {"a": "top/a"}, set())
        self.assertIsNone(result["coarse"][0]["off_line"])
        self.assertEqual(result["total"]["delta_bram36eq"], 11.5)
        self.assertEqual(result["unattributed_delta_bram36eq"], 1.0)

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            target.check_disjoint(["top/a", "top/a/b"])
        target.check_disjoint(["top/a", "top/ab"])

    def test_malformed_counts_columns_and_duplicates_are_rejected(self):
        text = report([(0, "top", 12, 3)])
        for broken in (text.replace("|12|", "|NA|"), text.replace("RAMB36", "RAM36")):
            with self.assertRaises(ValueError):
                target.parse_hierarchy(broken)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            target.parse_hierarchy(report([(0, "top", 1, 0), (1, "a", 1, 0), (1, "a", 1, 0)]))

    def test_negative_exclusive_memory_is_rejected(self):
        rows, _ = target.parse_hierarchy(report([(0, "top", 1, 0), (1, "a", 2, 0)]))
        with self.assertRaisesRegex(ValueError, "negative"):
            target.partition(rows, "top", {"top"})

    def test_real_reports_provenance_reconciliation_and_denominators(self):
        result = target.summarize()
        mem = result["memory"]
        self.assertEqual([r["delta_bram36eq"] for r in mem["coarse"]], [40, 16.5, 22.5, 0.5])
        self.assertEqual(mem["total"]["on_bram36eq"], 513.5)
        self.assertEqual(mem["total"]["off_bram36eq"], 434)
        self.assertEqual(mem["unattributed_delta_bram36eq"], 0)
        self.assertEqual(sum(r["delta_bram36eq"] for r in mem["details"]), 79.5)
        self.assertEqual(result["runs"]["learning_on"]["hierarchy_row_count"], 800)
        self.assertEqual(result["runs"]["learning_off"]["hierarchy_row_count"], 765)
        self.assertEqual(result["runs"]["learning_on"]["hierarchy_source"]["metadata"]["Date"],
                         "Mon Sep 28 14:44:17 2026")
        self.assertEqual(result["runs"]["learning_off"]["hierarchy_source"]["metadata"]["Date"],
                         "Mon Sep 28 15:54:39 2026")
        comparisons = {(r["comparison"], r["metric"]): r for r in result["comparisons"]}
        expected = {"unified_vs_sum": (1347.5, -834),
                    "unified_vs_maxsingle": (474.5, 39),
                    "learning_on_vs_off": (434, 79.5), "shared_vs_private": (517, -3.5)}
        for key, (reference, delta) in expected.items():
            row = comparisons[key, "bram36eq"]
            self.assertEqual(row["reference_value"], reference)
            self.assertEqual(row["delta_target_minus_reference"], delta)
            self.assertAlmostEqual(row["percent_of_reference"], delta / reference * 100)
        self.assertEqual(result["packing"]["incremental_site_count_bounds"], [79, 83])
        self.assertIsNone(result["packing"]["actual_placed_site_counts"])
        self.assertTrue(all(run["bram36eq_minus_reported_tiles"] == 0 for run in result["runs"].values()))

    def test_delivered_json_and_csv_equal_fresh_parse(self):
        summary = target.summarize()
        out = target.ROOT / target.OUT
        self.assertEqual(json.loads((out / "learning_memory_summary.json").read_text(encoding="ascii")), summary)
        mem = summary["memory"]
        for filename, rows in (
            ("learning_memory_hierarchy.csv", mem["coarse"] + [mem["remainder"]]),
            ("learning_memory_detail.csv", mem["details"]),
            ("resource_comparisons.csv", summary["comparisons"]),
        ):
            with (out / filename).open(encoding="utf-8", newline="") as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual(actual, [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows])

    def test_chinese_report_numbers_and_links(self):
        summary = target.summarize()
        report_path = target.ROOT / target.OUT / "LEARNING_MEMORY_COMPARISON_ZH.md"
        text = report_path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            self.assertTrue((report_path.parent / link).resolve().is_file(), link)
        labels = {"Unified": "Unified V2 mask7", "FC-only": "FC-only mask1",
                  "SRNN-only": "SRNN-only mask2", "SCNN-only": "SCNN-only mask4",
                  "private": "Private RUN123", "learning_on": "Learning-on RUN120 R2",
                  "learning_off": "Learning-off RUN122"}
        keys = ("slice_luts", "registers", "bram36eq", "dsp")
        for key, label in labels.items():
            run = summary["runs"][key]
            values = run["resources"]
            stage = "post-route" if run["stage"] == "post_route" else "post-synthesis"
            cells = [f"{values[k]:.1f}" if k == "bram36eq" else f"{values[k]:.0f}" for k in keys]
            line = f"| {label} | {stage} | " + " | ".join(cells)
            line += f" | {run['timing']['wns_ns']:+.3f} / {run['timing']['whs_ns']:+.3f} |"
            self.assertIn(line, text)
        for row in summary["comparisons"]:
            if row["metric"] not in keys:
                continue
            delta = abs(row["delta_target_minus_reference"])
            count = f"{delta:.1f}" if row["metric"] == "bram36eq" else f"{delta:.0f}"
            cell = f"{count}\uff08{abs(row['percent_of_reference']):.3f}%\uff09"
            self.assertIn(cell, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
