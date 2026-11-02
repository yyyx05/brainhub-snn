"""Focused profiling preparation/parser tests: no Verilator or dataset/model work."""
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_fc_profile_measurement as runner


def fixture(directory, learning, steps=2):
    directory.mkdir()
    files = {}
    for name in ("input_counts", "parameters", "labels", "pre", "trace", "spike",
                 "dot", "q6", "w2_after", "residual_after"):
        path = directory / (name + ".mem")
        path.write_text("00\n", encoding="ascii")
        files[path.name] = {"bytes": path.stat().st_size}
    ref = {"status": "PASS", "learning": learning, "steps": steps, "source_steps": steps,
           "batch": 16, "hidden": 160, "classes": 10, "files": files,
           "numeric": {"scale_to_q6_q16": 47303}, "w2_parameter_base": 369920,
           "samples": [{"dataset_index": s, "valid_timesteps": steps, "label": s % 10}
                       for s in range(16)]}
    (directory / "reference.json").write_text(json.dumps(ref), encoding="utf-8")


def scope(arm="on", full=True):
    return {"steps": 2, "full_sample": full, "expected": {
        a: {"hidden_checks": 5120, "dot_checks": 320, "q6_checks": 320,
            "commits": 2 if a == "on" else 0} for a in ("on", "off")}}


def make_log(arm="on"):
    rows = []
    for st in range(8):
        row = {key: 0 for key in (*runner.BUS_FIELDS, "window_busy", "work_busy",
                "work_overlap_forward", "work_only", "work_other", "forward_scope",
                "refill_busy", "commit_present", "commit_stall", "state_write_mask_bytes",
                "trace_write_mask_bytes")}
        row.update(id=st, cycles=10 if st not in (4, 5, 6) else {4: 100, 5: 20, 6: 10}[st])
        if st == 1:
            row.update(parameter_input=371520, store_load=371520)
        if st == 2:
            row.update(forward_request=100, forward_response=100)
        if st == 3:
            row.update(state_write=320, trace_write=320, state_write_mask_bytes=12800,
                       trace_write_mask_bytes=12800)
        if st == 4 and arm == "on":
            row.update(work_busy=20, work_overlap_forward=6, work_only=14, window_busy=30,
                       learning_request=2, learning_response=2, learning_write=2, residual_write=2)
        rows.append("FC_PROFILE_STAGE " + " ".join(f"{k}={v}" for k, v in row.items()))
    windows = []
    if arm == "on":
        for index in range(2):
            windows.append(f"FC_PROFILE_WINDOW index={index} observation=30 update_start=40 done=50 commit_first=51 commit=52 work_first=10 last_forward=29 last_write=49 work_busy=5 work_overlap_forward=1 work_other=0")
    learning = int(arm == "on")
    count = 2 if learning else 0
    return "\n".join([*windows,
        f"FC_FINAL_COUNTS_PASS learning={learning} batch=16 batches=1 steps=2 cycles=130 hidden_checks=5120 dot_checks=320 q6_checks=320 commits={count} forward_reads=100 learning_reads={count} code_writes=0 residual_writes={count}",
        f"FC_PROFILE_MEASUREMENT_PASS learning={learning} steps=2", *rows])


class FCProfileMeasurementTests(unittest.TestCase):
    def test_matching_existing_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            on, off = Path(directory) / "on", Path(directory) / "off"
            fixture(on, True)
            fixture(off, False)
            result = runner.inspect_pair(on, off, 2)
        self.assertTrue(result["full_sample"])
        self.assertEqual(result["samples"][15]["dataset_index"], 15)

    def test_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            on, off = Path(directory) / "on", Path(directory) / "off"
            fixture(on, True)
            fixture(off, False)
            (off / "input_counts.mem").write_text("01\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                runner.inspect_pair(on, off, 2)

    def test_reject_truncated_existing_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            on, off = Path(directory) / "on", Path(directory) / "off"
            fixture(on, True)
            fixture(off, False)
            (on / "q6.mem").write_text("", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "truncated"):
                runner.inspect_pair(on, off, 2)

    def test_no_drive_or_timing_change_in_copied_body(self):
        body = (runner.ROOT / runner.BODY).read_text(encoding="utf-8")
        monitor = (runner.ROOT / runner.MONITOR).read_text(encoding="utf-8")
        result = runner.instrument_body(body, monitor)
        driven = r"\b[io]_\w+\s*(?:<=|=(?!=))[^;]*;"
        self.assertEqual(re.findall(driven, body), re.findall(driven, result))
        events = r"@\([^)]*\)|always\s+#\d+|repeat\s*\([^)]*\)"
        # Only the passive sampling always block adds one posedge sensitivity.
        self.assertEqual(re.findall(events, body), re.findall(events, result[len(monitor):]))
        self.assertEqual(re.findall(r"\$fatal\([^;]+;", body), re.findall(r"\$fatal\([^;]+;", result))
        self.assertNotRegex(monitor, driven)

    def test_reject_changed_shared_body_anchor(self):
        with self.assertRaisesRegex(ValueError, "marker changed"):
            runner.instrument_body("unknown replacement body", "")

    def test_real_105_closure_and_separate_store(self):
        with tempfile.TemporaryDirectory() as directory:
            tb = Path(directory) / "tb.sv"
            runner.generate_tb(tb, top=runner.TOP)
            files = runner.simulation_sources(runner.ROOT, runner.ROOT / runner.SOURCE_LIST, tb, True)
        self.assertEqual(len(files), 108)
        self.assertEqual(len(set(files)), 108)
        self.assertIn(runner.resolve_verification_store(), files[105:])
        self.assertIn(runner.ROOT / "tb/monitors/supplement_h40_activity_monitor.sv", files)

    def test_on_overlap_is_not_added_twice(self):
        result = runner.parse_log(make_log(), "on", scope())
        self.assertEqual(result["stages"]["runtime"]["work_only"], 14)
        self.assertEqual(result["windows"][0]["observation_to_commit_clocks"], 22)
        self.assertEqual(result["windows"][0]["commit_valid_to_accept_clocks"], 1)
        self.assertEqual(result["runtime_cycles"], 130)
        self.assertEqual(result["bus_payload_bits"]["learning_write"], 32)
        self.assertEqual(result["bus_payload_bits"]["state_write"], 102400)

    def test_off_has_no_learning_windows(self):
        result = runner.parse_log(make_log("off"), "off", scope())
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["bus_handshakes"]["learning_write"], 0)

    def test_debug_prefix_has_no_full_sample_projection(self):
        result = runner.parse_log(make_log(), "on", scope(full=False))
        self.assertIsNone(result["amortized_cycles_per_supplied_sample"])
        self.assertIsNone(result["projected_samples_per_second_at_15ns"])

    def test_reject_runtime_misalignment(self):
        with self.assertRaisesRegex(ValueError, "runtime interval"):
            runner.parse_log(make_log().replace("cycles=130 hidden", "cycles=131 hidden"), "on", scope())

    def test_reject_incomplete_commit_and_failures(self):
        with self.assertRaisesRegex(ValueError, "commit windows"):
            runner.parse_log(make_log().replace("FC_PROFILE_WINDOW index=1", "IGNORED index=1"), "on", scope())
        with self.assertRaisesRegex(ValueError, "simulation failed"):
            runner.parse_log(make_log() + "\n%Error: broken\n", "on", scope())

    def test_reject_bus_mismatch(self):
        with self.assertRaisesRegex(ValueError, "passive bus"):
            runner.parse_log(make_log().replace("store_load=371520", "store_load=371519"), "on", scope())

    def _mocked_main(self, *, prepare_only, warning=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            on, off = root / "on", root / "off"
            fixture(on, True)
            fixture(off, False)
            (root / runner.BODY).parent.mkdir(parents=True)
            (root / runner.BODY).write_text((runner.ROOT / runner.BODY).read_text(encoding="utf-8"), encoding="ascii")
            (root / runner.MONITOR).write_text((runner.ROOT / runner.MONITOR).read_text(encoding="utf-8"), encoding="ascii")
            rom = root / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
            rom.parent.mkdir(parents=True)
            rom.write_text("00\n", encoding="ascii")
            tb_deps = [root / "stub.sv"]
            tb_deps[0].write_text("", encoding="ascii")
            argv = ["runner", "--run-name", "prepare", "--steps", "2",
                    "--fixture-on", str(on), "--fixture-off", str(off), "--source-list", "sources.f"]
            if prepare_only:
                argv.append("--prepare-only")
            calls = []
            def process(command, **kwargs):
                calls.append(command)
                if not warning:
                    raise AssertionError("tool launched")
                kwargs["stdout"].write("%Warning-" + warning + ": test structural warning\n")
                return SimpleNamespace(returncode=0)
            # generate_tb still uses the real unchanged header/store; no simulator is invoked.
            with patch.object(runner, "ROOT", root), patch("sys.argv", argv), \
                    patch.object(runner, "project_path", side_effect=lambda p: (root / p).resolve()), \
                    patch.object(runner, "simulation_sources", return_value=tb_deps), \
                    patch.object(runner.subprocess, "run", side_effect=process):
                code = runner.main()
            record = json.loads((root / runner.OUTPUT / "prepare/commands.json").read_text())
            return code, record, calls

    def test_prepare_only_does_not_launch_tools(self):
        code, record, calls = self._mocked_main(prepare_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(record["status"], "PREPARED_NOT_SIMULATED")
        self.assertEqual(record["stages"], {})
        self.assertNotIn("reuse_executable", record)

    def test_structural_warning_names_and_ordinary_warnings(self):
        self.assertEqual(runner.structural_warnings("%Warning-WIDTH: ordinary\n"), [])
        self.assertEqual(runner.structural_warnings("%Warning-LATCH: a\n%Warning-UNOPTFLAT: b\n%Warning-MULTIDRIVEN: c"),
                         ["UNOPTFLAT", "LATCH", "MULTIDRIVEN"])

    def test_build_exit_zero_with_structure_warning_starts_neither_arm(self):
        for warning in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN"):
            with self.subTest(warning=warning):
                code, record, calls = self._mocked_main(prepare_only=False, warning=warning)
                self.assertEqual(code, 1)
                self.assertEqual(record["status"], "FAILED_STRUCTURAL_BUILD")
                self.assertEqual(list(record["stages"]), ["build"])
                self.assertEqual(record["stages"]["build"]["exit_code"], 0)
                self.assertEqual(record["stages"]["build"]["structural_warnings"], [warning])
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
