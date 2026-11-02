import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
import prepare_srnn_resident_abort as preparation
from srnn_resident_server_worker import parse_recovery_log


class AbortRecoveryTest(unittest.TestCase):
    def test_transform_retries_without_reset_or_reload(self):
        source = (preparation.ACCEPTED / "tb_srnn_profile_measurement.sv").read_text(encoding="utf-8")
        tb = preparation.transform(source)
        self.assertIn("attempt<2", tb)
        self.assertIn("timestep<RECOVERY_STEPS", tb)
        self.assertIn("while (!o_learning_srnn_abort_ready)", tb)
        self.assertIn("RESIDENT_ABORT_RECOVERY_PASS", tb)
        self.assertEqual(tb.count("i_rst_n = 1"), source.count("i_rst_n = 1"))
        self.assertEqual(tb.count("i_learning_srnn_parameter_load_valid = 1"), 1)
        self.assertNotIn("force ", tb)

    def test_parser_requires_abort_and_exact_retry_counts(self):
        expected = dict(learning=1, resets=1, aborts=1, bone=450, hidden=256, logits=20,
            gradients=5120, store=120320, code=0, residual=0, commits=1, detaches=1,
            backbone_loads=2, wh1_loads=1, wout_loads=1, parameter_loads=120320, store_loads=120320)
        aborted = "RESIDENT_ABORT_ACCEPTED learning=1 aborts=1 bone=450 hidden=0 logits=0 gradients=0 code=0 residual=0 commits=0 unchanged_parameters=120320\n"
        complete = "RESIDENT_ABORT_RECOVERY_PASS " + " ".join(f"{k}={v}" for k,v in expected.items()) + "\n"
        self.assertEqual(parse_recovery_log(aborted+complete, "on", expected)["recovered"], expected)
        for log in (complete, aborted+complete.replace("parameter_loads=120320", "parameter_loads=240640"),
                    aborted.replace("commits=0", "commits=1")+complete):
            with self.assertRaises(ValueError):
                parse_recovery_log(log, "on", expected)

    def test_generator_rejects_wrong_source(self):
        with self.assertRaises(ValueError):
            preparation.transform("module some_other_test; endmodule")


if __name__ == "__main__":
    unittest.main()
