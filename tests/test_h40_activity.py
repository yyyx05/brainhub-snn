import unittest
from scripts.summarize_h40_activity import parse_activity


def log():
    text = "H40_ACTIVITY_UNIT_PASS\n"
    text += ("H40_ACTIVITY tile=one cycles=8 issue_fires=0 fill_fires=1 contributions=1 "
             "lane_fires=2 enabled_terms=3 abort=1 compute=1 issue_wait=1 region_wait=1 "
             "observation_wait=1 fill=1 internal=1 idle=1\n")
    for m in range(4):
        text += (f"H40_ACTIVITY_MODE tile=one mode={m} contributions={int(m == 0)} "
                 f"lane_fires={2*int(m == 0)} enabled_terms={3*int(m == 0)}\n")
    return text


class ActivityTests(unittest.TestCase):
    def test_denominators_include_idle_physical_capacity(self):
        result = parse_activity(log(), "H40_ACTIVITY_UNIT_PASS", 1)
        self.assertEqual(result["whole_array_destination_lane_occupancy"], 2/(8*40))
        self.assertEqual(result["whole_array_p4_term_occupancy"], 3/(8*160))
        self.assertEqual(result["accepted_lane_source_term_occupancy"], 3/8)
        self.assertEqual(sum(result["totals"][k] for k in ("abort", "compute", "issue_wait",
            "region_wait", "observation_wait", "fill", "internal", "idle")), 8)

    def test_missing_tile_cannot_claim_full_array(self):
        with self.assertRaises(ValueError):
            parse_activity(log(), "H40_ACTIVITY_UNIT_PASS")

    def test_failed_functional_run_overrides_counter_output(self):
        for text in (log().replace("H40_ACTIVITY_UNIT_PASS", ""), log()+"%Error: mismatch"):
            with self.assertRaises(ValueError):
                parse_activity(text, "H40_ACTIVITY_UNIT_PASS", 1)

    def test_bucket_and_mode_mismatches_rejected(self):
        for text in (log().replace("idle=1", "idle=2"), log().replace("mode=0 contributions=1", "mode=0 contributions=0")):
            with self.assertRaises(ValueError):
                parse_activity(text, "H40_ACTIVITY_UNIT_PASS", 1)


if __name__ == "__main__":
    unittest.main()
