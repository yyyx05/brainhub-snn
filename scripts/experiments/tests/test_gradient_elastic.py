import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gradient_elastic_local as local
import run_srnn_gradient_elastic_core as core


class ElasticTests(unittest.TestCase):
    def test_changed_source_keeps_interface_and_datapath_widths(self):
        source = local.BASE.read_text()
        new = local.candidate(source)
        self.assertEqual(source.split("    logic configured_q;")[0],
                         new.split("    logic configured_q;")[0])
        for name in ("accumulator_q", "gradient_q", "gradient_addr_q"):
            old_decl = [s for s in source.splitlines() if "logic" in s and name + ";" in s]
            new_decl = [s for s in new.splitlines() if "logic" in s and name + ";" in s]
            self.assertEqual(old_decl, new_decl)
        a = source[source.index("    function automatic"):source.index("    always_comb")]
        b = new[new.index("    function automatic"):new.index("    always_comb")]
        self.assertEqual(a, b)

    def test_source_rejects_unrecognized_baseline(self):
        with self.assertRaises(ValueError):
            local.candidate(local.BASE.read_text().replace("logic retire_and_accept;", "logic changed;"))

    def test_only_latency_is_excluded(self):
        base = dict(service_cycles=20, all_sequences_per_second_15ns_projection=1,
            warm_sequences_per_second_with_setup_15ns_projection=2,
            total=dict(cycles=30, parameter_loads=4), sequences=[dict(cycles=20, code=2)],
            source=dict(words=175), state_setup_cycles=3)
        changed = copy.deepcopy(base)
        changed["total"]["cycles"] -= 5
        changed["sequences"][0]["cycles"] -= 5
        changed["service_cycles"] -= 5
        self.assertEqual(core.numerical_records(base), core.numerical_records(changed))
        for change in ("code", "setup", "source", "loads"):
            bad = copy.deepcopy(changed)
            if change == "code":
                bad["sequences"][0]["code"] += 1
            elif change == "setup":
                bad["state_setup_cycles"] += 1
            elif change == "source":
                bad["source"]["words"] += 1
            else:
                bad["total"]["parameter_loads"] += 1
            self.assertNotEqual(core.numerical_records(base), core.numerical_records(bad))

    def test_partial_or_damaged_local_transcript_is_rejected(self):
        log = (local.OUT / "candidate_run.log").read_text()
        for bad in (log.replace("GRADIENT_ELASTIC_PASS", "MISSING", 1),
                    log.replace("GRADIENT_ELASTIC_TWO_SLOT_PASS", "MISSING", 1),
                    log + "\n%Error: synthetic\n"):
            with self.assertRaises(ValueError):
                local.parse(bad, 1)


if __name__ == "__main__":
    unittest.main()
