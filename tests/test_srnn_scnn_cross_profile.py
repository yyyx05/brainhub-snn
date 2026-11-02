import re
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
import prepare_srnn_scnn_cross_profile as prep
from prepare_scnn_resident_compatibility import transform as scnn_prefix
from test_scnn_resident_compatibility import transcript
import srnn_resident_server_worker as worker


def cross_transcript(learning):
    scnn, expected = transcript(learning)
    first = "CROSS_PROFILE_M2_PASS sequence=1 bone=450 hidden=256 logits=20 store=120320 backbone_loads=2 wh1_loads=1 wout_loads=1\n"
    last = "CROSS_PROFILE_M2_PASS sequence=2 bone=450 hidden=256 logits=20 store=120320 backbone_loads=4 wh1_loads=1 wout_loads=1\n"
    end = f"CROSS_PROFILE_PASS resets=1 m2_parameter_loads=120320 m2_sequences=2 scnn_frames=4 scnn_descriptors=32 scnn_learning={int(learning)}\n"
    return first+scnn+last+end, expected


class CrossProfilePreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scnn = (prep.BASE / "server_pair_r2/tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
        cls.srnn = (prep.RESULTS / "srnn_specialized_r1/tb_srnn_profile_measurement.sv").read_text(encoding="utf-8")
        cls.tb = prep.transform(cls.scnn, cls.srnn)

    def test_one_dut_one_reset_and_order(self):
        self.assertEqual(self.tb.count("unified_h40_final_composition_top_v1 #("), 1)
        self.assertEqual(self.tb.count("rst_n = 1'b1;"), 1)
        first = self.tb.index("g_cross_m2.run_first_step();")
        initialize = self.tb.index("        fg_initialize_top();", first)
        end_scnn = self.tb.index("        fg_finish();", initialize)
        second = self.tb.index("g_cross_m2.run_first_step();", first+1)
        self.assertLess(first, initialize)
        self.assertLess(initialize, end_scnn)
        self.assertLess(end_scnn, second)
        self.assertNotIn("force ", self.tb)

    def test_actual_numeric_and_backbone_restore_checks_remain(self):
        for text in ("BONE_MISMATCH", "M2_HIDDEN_MISMATCH", "M2_LOGIT_MISMATCH", "store mismatch",
                     "complete gradient/store/mask check", "timestep_phase < 4;", "timestep<1;",
                     "resident_backbone_loads!=2*sequence_runs", "resident_wh1_loads!=1", "resident_wout_loads!=1",
                     "cross_parameter_loads!=120320", "cross_reset_assertions!=1"):
            self.assertIn(text, self.tb)
        self.assertGreaterEqual(self.tb.count("$fatal"), scnn_prefix(self.scnn).count("$fatal"))

    def test_only_top_level_inputs_are_renamed(self):
        self.assertIn("`ENG.o_cache_valid", self.tb)
        self.assertNotIn("`ENG.ft_o_cache_valid", self.tb)
        self.assertIn("dut.u_parameter_plane.g_srnn_client.u_srnn_m2", self.tb)
        self.assertNotIn("u_final.", self.tb)

    def test_driven_inputs_have_no_duplicate_continuous_assignment(self):
        block = self.tb.split("begin : g_cross_m2", 1)[1]
        procedural = set(re.findall(r"(?<!assign )\b(ft_i_\w+)\s*(?:<=|=(?!=))", block))
        for name in procedural:
            if name == "ft_i_srnn_backbone_forward_read_ready":
                continue
            self.assertNotRegex(self.tb, r"assign " + name + r" =")
        self.assertEqual(self.tb.count("assign ft_i_srnn_backbone_forward_read_ready ="), 1)

    def test_requires_actual_two_sides_and_scnn_completion(self):
        for learning in (False, True):
            scnn, expected = transcript(learning)
            first = "CROSS_PROFILE_M2_PASS sequence=1 bone=450 hidden=256 logits=20 store=120320 backbone_loads=2 wh1_loads=1 wout_loads=1\n"
            last = "CROSS_PROFILE_M2_PASS sequence=2 bone=450 hidden=256 logits=20 store=120320 backbone_loads=4 wh1_loads=1 wout_loads=1\n"
            end = f"CROSS_PROFILE_PASS resets=1 m2_parameter_loads=120320 m2_sequences=2 scnn_frames=4 scnn_descriptors=32 scnn_learning={int(learning)}\n"
            log = first+scnn+last+end
            self.assertTrue(prep.parse_log(log, expected, learning)["status"].startswith("PASS_same_DUT"))
            for broken in (log.replace(first,""), log.replace("backbone_loads=4","backbone_loads=2"),
                           log.replace("wh1_loads=1","wh1_loads=2"), log.replace("resets=1","resets=2"),
                           log.replace("m2_parameter_loads=120320","m2_parameter_loads=240640"),
                           log.replace("sequence=2","sequence=1"), log+"%Error: late failure\n",
                           scnn+first+last+end):
                with self.assertRaises(ValueError):
                    prep.parse_log(broken, expected, learning)

    def run_worker(self, missing_return=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(scope="test", fixture="scnn-fixture", srnn_fixture="srnn-fixture",
                          expected={arm:transcript(arm=="on")[1] for arm in ("off","on")})
            (root/"reference.json").write_text(json.dumps(config))
            accepted = json.loads((prep.BASE/"server_pair_r2/build.status.json").read_text())
            (root/"scnn_compatibility_build.json").write_text(json.dumps(accepted))
            commands = []
            record = {}
            def fake_stage(record, name, command, timeout):
                commands.append((name, command))
                log = ""
                if name.startswith("run_"):
                    log = cross_transcript(name=="run_on")[0]
                    if missing_return:
                        log = "\n".join(line for line in log.splitlines()
                                        if not line.startswith("CROSS_PROFILE_M2_PASS sequence=2 "))
                (root/(name+".log")).write_text(log)
                record[name] = dict(exit=0)
            with patch.object(worker,"ROOT",root), patch.object(worker,"stage",side_effect=fake_stage), patch.object(worker,"save"):
                if missing_return:
                    with self.assertRaisesRegex(ValueError,"M2 numerical/residency"):
                        worker.compatibility(record,"cross_profile")
                    self.assertNotIn("finished_utc",record)
                    self.assertEqual([name for name,_ in commands],["build","run_off"])
                else:
                    worker.compatibility(record,"cross_profile")
                    self.assertTrue(record["status"].startswith("PASS_same_DUT"))
                    self.assertEqual([name for name,_ in commands],["build","run_off","run_on"])
                    for name, command in commands[1:]:
                        self.assertEqual(command,["obj/Vscgraph","+SCNN_PROFILE_FIXTURE=scnn-fixture",
                                                  "+SCNN_LEARNING="+str(int(name=="run_on")),
                                                  "+RAW=srnn-fixture","+FIXTURE=srnn-fixture"])

    def test_worker_passes_both_model_fixtures_to_same_executable(self):
        self.run_worker()

    def test_worker_rejects_scnn_pass_without_m2_return(self):
        self.run_worker(missing_return=True)


if __name__ == "__main__":
    unittest.main()
