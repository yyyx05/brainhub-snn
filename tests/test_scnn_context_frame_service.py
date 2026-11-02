"""Runner checks only; actual protocol assertions execute in the Verilator TB."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_scnn_context_frame_service.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ContextFrameRunnerTests(unittest.TestCase):
    def test_two_source_leaf_build(self):
        for _, top in runner.CASES:
            command = runner.build_command(top, runner.BASE / "unit/obj", "verilator", False)
            self.assertEqual(command[command.index("-j") + 1], "1")
            self.assertEqual(command[-2:], [runner.RTL, f"tb/{top}.sv"])
            for source in command[-2:]:
                self.assertTrue((ROOT / source).is_file())

    def test_new_interface_is_not_waived_as_missing(self):
        command = runner.build_command(runner.CASES[0][1], runner.BASE / "unit", "verilator", False)
        self.assertNotIn("-Wno-PINMISSING", command)

    def test_incomplete_log_rejected(self):
        with self.assertRaises(ValueError):
            runner.check_log("frame_service", "SCNN_CONTEXT_FRAME_SERVICE_PASS cases=14 cache=1 service=2 load=1 release=1")

    def test_duplicate_cases_rejected(self):
        log = "SCNN_CONTEXT_FRAME_CASE_PASS duplicate\n" * 14
        log += "SCNN_CONTEXT_FRAME_SERVICE_PASS cases=14 cache=1 service=2 load=1 release=1"
        with self.assertRaises(ValueError):
            runner.check_log("frame_service", log)

    def test_complete_log(self):
        log = "\n".join(f"SCNN_CONTEXT_FRAME_CASE_PASS case{i}" for i in range(14))
        log += "\nSCNN_CONTEXT_FRAME_SERVICE_PASS cases=14 cache=12 service=24 load=12 release=12"
        result = runner.check_log("frame_service", log)
        self.assertEqual(result["handshakes"]["service"], 24)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
