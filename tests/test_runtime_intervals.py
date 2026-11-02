"""Interval algebra and actual archived-profile crosschecks; no EDA imports."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_intervals", ROOT / "scripts/experiments/summarize_runtime_intervals.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class AlgebraTests(unittest.TestCase):
    def test_empty_zero_adjacent_duplicate_nested(self):
        self.assertEqual(runtime.union([]), [])
        self.assertEqual(runtime.union([(4, 4)]), [])
        self.assertEqual(runtime.union([(3, 8), (0, 3), (1, 2), (3, 8)]), [(0, 8)])

    def test_invalid(self):
        for spans in ([(-1, 2)], [(5, 4)]):
            with self.assertRaises(ValueError):
                runtime.union(spans)
        with self.assertRaises(ValueError):
            runtime.partition((1, 1), {})

    def test_intersection_touching_and_overlap(self):
        self.assertEqual(runtime.intersection([(0, 3)], [(3, 5)]), [])
        self.assertEqual(runtime.intersection([(0, 5), (7, 9)], [(2, 8)]), [(2, 5), (7, 8)])

    def test_partition_clipping_gaps_three_way_overlap(self):
        view, parts = runtime.coverage((10, 20), {"a": [(0, 14)], "b": [(12, 17)], "c": [(13, 15)]})
        self.assertEqual(sum(b-a for a, b, _ in parts), 10)
        self.assertEqual(view["union_cycles"], 7)
        self.assertEqual(view["unclassified_cycles"], 3)
        self.assertEqual(view["non_overlapping_phase_cycles"]["a&b&c"], 1)
        self.assertEqual(view["pairwise_intersection_cycles"]["a&b"], 2)

    def test_small_integer_oracle(self):
        groups = {"a": [(0, 5), (4, 6), (9, 11)], "b": [(2, 8)], "c": [(3, 3), (5, 10)]}
        parts = runtime.partition((1, 12), groups)
        for point in range(1, 12):
            observed = [labels for a, b, labels in parts if a <= point < b]
            expected = tuple(sorted(k for k, spans in groups.items() if any(a <= point < b for a, b in spans)))
            self.assertEqual(observed, [expected])


class ExistingLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result, cls.segments = runtime.collect(ROOT)
        cls.profiles = {(p["mode"], p["arm"]): p for p in result["profiles"]}

    def test_six_runtime_denominators(self):
        expected = {("FC", "off"): 1013344, ("FC", "on"): 8714352,
                    ("SRNN", "off"): 110452102, ("SRNN", "on"): 112087773,
                    ("SCNN", "off"): 781657379, ("SCNN", "on"): 784525456}
        self.assertEqual({k: p["runtime_cycles"] for k, p in self.profiles.items()}, expected)

    def test_fc_no_fabricated_endpoints_or_busy_partition(self):
        p = self.profiles["FC", "on"]
        self.assertIsNone(p["runtime_bounds"])
        self.assertIsNone(p["views"]["stage_counts"]["union_cycles"])
        self.assertEqual(p["views"]["stage_counts"]["non_overlapping_phase_cycles"],
                         {"runtime": 1013344, "learning_tail_wait": 7288871, "runtime_refill": 412137})
        self.assertEqual(p["learning_counts_not_additive"]["work_overlap_forward"], 612216)

    def test_srnn_exact_overlap(self):
        p = self.profiles["SRNN", "on"]
        v = p["views"]["hardware_detail"]
        self.assertEqual(v["pairwise_intersection_cycles"]["m2_forward&pure_learning_busy"], 73)
        self.assertEqual(p["views"]["stages"]["unclassified_cycles"], 0)
        self.assertEqual(p["setup_cycles"], 120426)
        overlaps = [r for r in self.segments if r[:3] == ("SRNN", "on", "hardware_detail")
                    and r[6] == "m2_forward&pure_learning_busy"]
        self.assertEqual(len(overlaps), 73)
        self.assertTrue(all(r[5] == 1 for r in overlaps))

    def test_scnn_residual_independent_timestamp_union(self):
        p = self.profiles["SCNN", "on"]
        v = p["views"]["service"]
        self.assertEqual(v["unclassified_cycles"], 1642320)
        self.assertEqual(v["union_cycles"], 782883136)
        self.assertEqual(v["phase_union_cycles"], {"descriptor": 781913760, "learning": 611436, "publication": 357940})
        self.assertEqual(set(v["pairwise_intersection_cycles"].values()), {0})
        self.assertEqual(p["setup_cycles"], 71594)

    def test_scnn_control_envelopes_are_not_busy_counts(self):
        p = self.profiles["SCNN", "on"]
        v = p["views"]["control_envelopes"]
        self.assertEqual(v["phase_union_cycles"]["pool_envelope"], 171068620)
        self.assertEqual(p["active_counts_not_additive"]["pool_control"], 28604080)
        self.assertEqual(v["pairwise_intersection_cycles"]["conv8_envelope&pool_envelope"], 171068620)
        self.assertEqual(v["pairwise_intersection_cycles"]["commit_pending_to_pulse&publication"], 357940)
        self.assertEqual(v["unclassified_cycles"], 1642300)

    def test_setup_monitor_and_runtime_are_separate(self):
        for mode in ("FC", "SRNN", "SCNN"):
            for arm in ("off", "on"):
                p = self.profiles[mode, arm]
                reset = 6 if mode == "SRNN" else 0
                self.assertEqual(p["monitor"]["cycles_per_tile"],
                                 p["setup_cycles"] + p["runtime_cycles"] - reset)
                self.assertIsNone(p["monitor"]["absolute_bounds"])

    def test_all_partitions_are_disjoint_and_close(self):
        for p in self.profiles.values():
            for name, view in p["views"].items():
                self.assertEqual(sum(view["non_overlapping_phase_cycles"].values()), p["runtime_cycles"])
                spans = [r for r in self.segments if r[:3] == (p["mode"], p["arm"], name)]
                if spans:
                    self.assertEqual(spans[0][3], p["runtime_bounds"][0])
                    self.assertEqual(spans[-1][4], p["runtime_bounds"][1])
                    self.assertTrue(all(a[4] == b[3] for a, b in zip(spans, spans[1:])))


if __name__ == "__main__":
    unittest.main()
