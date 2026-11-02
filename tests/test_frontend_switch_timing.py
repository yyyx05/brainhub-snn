"""SYNTHETIC_NOT_HDL_EVIDENCE: tests never compile or execute an HDL model."""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frontend_switch_timing", ROOT / "scripts/experiments/frontend_switch_timing.py")
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def synthetic_log(golden):
    """Invented parser-only timestamps; legacy rows come from archived r5."""
    result = [timing.SYNTHETIC]
    now = 1000
    switch_begin = now
    for line in golden.splitlines():
        if line.startswith("FT_ACK,"):
            _, index, operation, opcode, _ = line.split(",")
            index, operation, opcode = int(index), int(operation), int(opcode)
            start = now
            if operation == 1:
                switch_begin = start
            now += 0 if operation in (1, 5) else operation
            result.append(f"FT_INTERVAL,{index},{operation},{opcode},{start},{now},{now-start}")
            if operation in (5, 8):
                marker = "FT_SWITCH_READY" if operation == 5 else "FT_SWITCH_TOTAL"
                result.append(f"{marker},{index},{opcode},{switch_begin},{now},{now-switch_begin}")
        result.append(line)
    return "\n".join(result) + "\n"


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = (timing.R5 / "simulation.log").read_text(encoding="utf-8")
        cls.commands = timing.read_program(timing.R5 / "program.txt")
        cls.synthetic = synthetic_log(cls.golden)

    def parse(self, log):
        return timing.parse_timing(log, self.commands, self.golden)

    def edit_row(self, marker, column, value):
        lines = self.synthetic.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith(marker + ","))
        cells = lines[index].split(",")
        cells[column] = str(value)
        lines[index] = ",".join(cells)
        return "\n".join(lines)

    def test_synthetic_valid_zero_elapsed_and_boundaries(self):
        result = self.parse(self.synthetic)
        self.assertEqual(result["evidence_kind"], timing.SYNTHETIC)
        self.assertFalse(result["rtl_execution_verified"])
        self.assertFalse(result["hardware_timing_gap_closed"])
        self.assertEqual(result["acknowledged_commands"], 32)
        self.assertEqual(result["terminal_cycles"], 9040)
        self.assertEqual(len(result["switches"]), 4)
        self.assertEqual(result["operations"][0]["cycles"], 0)
        self.assertEqual(result["switches"][0]["stop_to_version_aligned"]["cycles"], 9)
        self.assertEqual(result["switches"][0]["full_transaction_including_workload"]["cycles"], 30)

    def test_unmarked_supplied_log_does_not_establish_hdl_execution(self):
        result = self.parse(self.synthetic.replace(timing.SYNTHETIC, ""))
        self.assertIn("PROVENANCE_NOT_ESTABLISHED", result["evidence_kind"])
        self.assertFalse(result["hardware_timing_gap_closed"])

    def test_raw_r5_cannot_fill_missing_timestamps(self):
        with self.assertRaises(ValueError):
            self.parse(self.golden)

    def test_missing_duplicate_and_reordered_intervals(self):
        lines = self.synthetic.splitlines()
        indices = [i for i, line in enumerate(lines) if line.startswith("FT_INTERVAL,")]
        variants = [lines[:indices[0]] + lines[indices[0]+1:],
                    lines[:indices[0]] + [lines[indices[0]]] + lines[indices[0]:]]
        reordered = lines.copy()
        reordered[indices[0]], reordered[indices[1]] = reordered[indices[1]], reordered[indices[0]]
        variants.append(reordered)
        for variant in variants:
            with self.subTest(variant=variant[0]):
                with self.assertRaises(ValueError):
                    self.parse("\n".join(variant))

    def test_ack_count_order_and_golden_counts(self):
        for old, new in (("FT_ACK,0,1,1,0", "FT_ACK,1,1,1,0"),
                         ("FT_ACK,0,1,1,0", ""),
                         ("FT_CYCLES,9040", "FT_CYCLES,9041"),
                         ("FT_COUNTS,1,3,3,6,32,0,0,0,0", "FT_COUNTS,1,3,3,6,31,0,0,0,0"),
                         ("FT_SCNN,32,4,68,68", "FT_SCNN,32,4,67,68")):
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    self.parse(self.synthetic.replace(old, new))

    def test_invalid_identity_endpoint_duration_and_fields(self):
        for column, value in ((2, 2), (3, 32), (4, -1), (4, 1001), (5, 999),
                              (5, 9041), (6, 1), (4, "missing")):
            with self.subTest(column=column, value=value):
                with self.assertRaises(ValueError):
                    self.parse(self.edit_row("FT_INTERVAL", column, value))
        with self.assertRaises(ValueError):
            self.parse(self.synthetic.replace("FT_INTERVAL,0,1,1,1000,1000,0", "FT_INTERVAL,0,1,1,1000,1000"))

    def test_temporal_overlap(self):
        with self.assertRaises(ValueError):
            self.parse(self.synthetic.replace("FT_INTERVAL,1,2,1,1000,1002,2", "FT_INTERVAL,1,2,1,999,1002,3"))

    def test_between_operation_gap_is_unclassified_not_a_stall(self):
        result = self.parse(self.synthetic.replace("FT_INTERVAL,1,2,1,1000,1002,2",
                                                   "FT_INTERVAL,1,2,1,1001,1002,1"))
        for name in ("stop_to_version_aligned", "full_transaction_including_workload"):
            self.assertEqual(result["switches"][0][name]["unclassified_between_operations_cycles"], 1)

    def test_ready_and_total_must_match_actual_operation_boundaries(self):
        for marker in ("FT_SWITCH_READY", "FT_SWITCH_TOTAL"):
            for column, value in ((2, 32), (3, 999), (4, 1000), (5, 1)):
                with self.subTest(marker=marker, column=column):
                    with self.assertRaises(ValueError):
                        self.parse(self.edit_row(marker, column, value))


class PreparationTests(unittest.TestCase):
    def test_actual_prepare_additive_only_and_package_local(self):
        timing.EXPERIMENT.mkdir(parents=True, exist_ok=True)
        body = ROOT / "tb/tb_frontend_final_top_smoke.sv"
        before = body.read_bytes()
        with tempfile.TemporaryDirectory(prefix="test_prepare_", dir=timing.EXPERIMENT) as directory:
            out = Path(directory)
            with patch.object(timing.base, "run_bounded", side_effect=AssertionError("HDL execution forbidden")):
                report = timing.prepare(out)
            plain = (out / "base_generated_tb.sv").read_bytes()
            modified = (out / (timing.base.TOP + ".sv")).read_bytes()
            self.assertEqual(timing.remove_observability(modified), plain)
            added = [line for line in modified.decode().splitlines() if timing.MARK in line]
            self.assertEqual(len(added), 6)
            for line in added:
                self.assertNotIn("@", line)
                self.assertNotIn("#", line)
                self.assertNotIn("wait", line)
                self.assertTrue("ft_timing_" in line)
            self.assertEqual(report["status"], "PREPARED_NOT_RUN")
            self.assertEqual(report["restored_source_files"], [])
            self.assertEqual(timing.read_program(out / "program.txt"), timing.read_program(timing.R5 / "program.txt"))
            manifest = (out / "sources.f").read_text().splitlines()
            self.assertEqual(sum(timing.base.TOP + ".sv" in p for p in manifest), 1)
            self.assertFalse(any("base_generated_tb" in p for p in manifest))
            self.assertTrue(all((ROOT / p.strip('"')).is_file() for p in manifest))
            self.assertEqual(report["source_size_bytes"],
                             sum((ROOT / p.strip('"')).stat().st_size for p in manifest))
            self.assertIsNone(report["measured_compile_peak_rss_bytes"])
            self.assertIsNone(report["measured_compile_seconds"])
            driver = out / "run_on_server.py"
            tree = ast.parse(driver.read_text())
            compile(tree, "server_driver", "exec")
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "ROOT" for target in node.targets))
            expression = compile(ast.Expression(assignment.value), "server_root_expression", "eval")
            self.assertEqual(eval(expression, {"Path": Path, "__file__": str(driver)}), ROOT)
        self.assertEqual(body.read_bytes(), before)

    def test_missing_anchor_duplicate_anchor_and_repeat_instrumentation(self):
        body = (ROOT / "tb/tb_frontend_final_top_smoke.sv").read_text()
        for variant in (body.replace("    longint cycles = 0;", ""),
                        body + "\n    longint cycles = 0;\n", timing.instrument(body)):
            with self.assertRaises(ValueError):
                timing.instrument(variant)

    def test_output_cannot_overwrite_original_results(self):
        with self.assertRaises(ValueError):
            timing.output_directory(timing.R5)


if __name__ == "__main__":
    unittest.main()
