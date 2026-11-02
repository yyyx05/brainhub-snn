"""Focused local parser tests; no simulator, model execution or shared outputs."""

import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_final_profile_metrics import (
    CURRENT_FC, CURRENT_SRNN, SRNN_RAW, CURRENT_SCNN, SCNN_RAW, SCNN_RAW_FILES,
    OUTPUT, PROFILE_CSV_FIELDS, activity_csv_rows, collect, filesystem_path, local_fixture, main, normal_path, read_json,
    parse_current_fc, parse_current_srnn, parse_current_scnn, parse_fc, parse_scnn, parse_srnn, parse_timing,
    profile_csv_rows, project_cycles, validate_current_srnn_commands, validate_current_scnn_records, write_report,
)


def srnn_log():
    return "\n".join([
        "SRNN_SAME_DUT_STEP t=0 cycles=10 learning=0 bone=450 hidden=256 logits=20 code=0 residual=0 commits=0",
        "SRNN_SAME_DUT_STEP t=1 cycles=25 learning=0 bone=900 hidden=512 logits=40 code=0 residual=0 commits=0",
        "SRNN_SAME_DUT_PASS batch=1 steps=2 cycles=25 learning=0 bone=900 hidden=512 logits=40 code=0 residual=0 commits=0",
    ])


def scnn_log():
    lines = ["SUPPLEMENT_FULLGRAPH_INITIALIZATION cycles=5 weight_fill_words=2",
             "SCNN_TEMPORAL_WINDOW_START window=1 query=0 base=0 batch=1 cycle=5"]
    for frame in range(2):
        for layer in range(1, 9):
            lines.append(f"SCNN_TEMPORAL_LAYER frame={frame} sample=0 timestep={frame} "
                         f"layer={layer} cycles=10 descriptors={frame*8+layer}")
        lines.append(f"SCNN_TEMPORAL_FRAME_STATUS frame={frame} sample=0 timestep={frame} cycles=85")
        vector = "+".join(str((frame + 1) * layer) for layer in range(1, 9))
        lines.append(f"SCNN_TEMPORAL_FRAME_PASS frame={frame} requests={vector} contexts={vector} scalars={vector}")
    lines.extend(["SCNN_TEMPORAL_COMMIT window=1 rmw=3 code_changes=1 residual_changes=3",
                  "SCNN_TEMPORAL_FINAL_PASS frames=2 descriptors=16 cycles=180 commits=1 rmw=3 code_changes=1"])
    lines.extend(f"SCNN_TEMPORAL_DDR bank={bank} reads=7" for bank in range(9))
    return "\n".join(lines)


class ParserTests(unittest.TestCase):
    def test_srnn_cumulative_cycles_become_nonoverlapping_deltas(self):
        final, steps = parse_srnn(srnn_log())
        self.assertEqual([s["interval_cycles"] for s in steps], [10, 15])
        self.assertEqual(final["cycles"], 25)

    def test_srnn_missing_step_or_wrong_terminal_rejected(self):
        for text in (srnn_log().replace("t=1", "t=0"),
                     srnn_log().replace("steps=2 cycles=25", "steps=2 cycles=26")):
            with self.assertRaises(ValueError):
                parse_srnn(text)

    def test_failure_overrides_terminal_pass(self):
        with self.assertRaises(ValueError):
            parse_srnn(srnn_log() + "\n%Error: mismatch")
        with self.assertRaises(ValueError):
            parse_scnn(scnn_log().replace("SCNN_TEMPORAL_FINAL_PASS", "RUNNING"))

    def test_scnn_parent_intervals_and_remainders_not_pure_learning(self):
        final, initial, stages, commits, ddr = parse_scnn(scnn_log())
        self.assertEqual(final["cycles"], 180)
        self.assertEqual(initial["cycles"], 5)
        self.assertEqual(len([r for r in stages if r["kind"] == "layer"]), 16)
        self.assertEqual(sum(r["cycles"] for r in stages if r["stage"] == "frame_minus_layers"), 10)
        self.assertEqual(next(r["cycles"] for r in stages if r["stage"] == "outside_frames"), 5)
        self.assertTrue(any(r["stage"] == "Conv8+pool+classifier" for r in stages))
        self.assertEqual([r["requests"] for r in stages if r["kind"] == "layer" and r["layer"] == 8], [8, 8])
        self.assertEqual(commits[0]["rmw"], 3)
        self.assertEqual(len(ddr), 9)

    def test_scnn_missing_duplicate_layer_and_bad_accounting_rejected(self):
        lines = scnn_log().splitlines()
        duplicate = next(line for line in lines if "layer=8 " in line)
        cases = ("\n".join(line for line in lines if "frame=0 " not in line or "layer=8 " not in line),
                 scnn_log() + "\n" + duplicate,
                 scnn_log().replace("cycles=85", "cycles=79"),
                 scnn_log().replace("descriptors=16 cycles=180", "descriptors=15 cycles=180"),
                 scnn_log().replace("requests=2+4+6+8+10+12+14+16", "requests=1+2+3"))
        for text in cases:
            with self.assertRaises(ValueError):
                parse_scnn(text)

    def test_fc_complete_batch_and_initialization_boundary(self):
        text = "\n".join([
            "FC_FINAL_COUNTS_LOAD cycles=11 init_cycles=2",
            "FC_FINAL_COUNTS_STEP batch_index=0 t=0 cycles=5",
            "FC_FINAL_COUNTS_STEP batch_index=0 t=1 cycles=12",
            "FC_FINAL_COUNTS_PASS batch=16 batches=1 steps=2 cycles=12",
        ])
        final, load = parse_fc(text)
        self.assertEqual(final["cycles"], 12)
        self.assertEqual(load["cycles"] + load["init_cycles"], 13)
        with self.assertRaises(ValueError):
            parse_fc(text.replace("t=1", "t=0"))

    def test_timing_projection_not_walltime_or_failed_setup(self):
        text = ("WNS(ns) TNS(ns) endpoints total WHS(ns) THS(ns)\n"
                "-------\n 0.102 0.000 0 12 0.078 0.000\n"
                "core_clk {0.000 7.500} 15.000 66.667\n"
                "All user specified timing constraints are met.\n")
        timing = parse_timing(text)
        self.assertAlmostEqual(project_cycles(100, timing), 0.0015)
        self.assertIsNone(project_cycles(100, {**timing, "wns_ns": -0.107}))
        with self.assertRaises(ValueError):
            parse_timing("running")

    def test_fixture_and_output_stay_scoped(self):
        self.assertEqual(local_fixture({"run": ["unused-exe", "+FIXTURE=results/example"]}), Path("results/example"))
        with self.assertRaises(ValueError):
            local_fixture({"run": ["unused-exe", "+FIXTURE=../other"]})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--output-dir", str(ROOT / "docs")])
        self.assertTrue((ROOT / OUTPUT).is_relative_to(ROOT))


class CsvViewTests(unittest.TestCase):
    def test_profile_fields_are_fixed_flat_and_unknown_values_stay_null(self):
        source = {"id": "example", "samples": 0, "timing": {"period_ns": 15},
                  "input": {"support_indices": [1, 2]}, "execution": {"command": ["example"]}}
        original = json.loads(json.dumps(source))
        row = profile_csv_rows([source])[0]
        self.assertEqual(tuple(row), PROFILE_CSV_FIELDS)
        self.assertEqual(len(row), 26)
        self.assertEqual(row["period_ns"], 15)
        self.assertEqual(row["samples"], 0)
        self.assertIsNone(row["initial_load_and_refill_cycles"])
        self.assertIsNone(row["full_sample_throughput_samples_s"])
        self.assertIsNone(profile_csv_rows([{"timing": None}])[0]["period_ns"])
        self.assertNotIn("input", row)
        self.assertNotIn("execution", row)
        self.assertEqual(source, original)

    def test_activity_totals_are_flat_without_removing_json_details(self):
        source = {"id": "example", "cycles": 10, "totals": {"cycles": 120, "lane_fires": 0},
                  "tiles": {"tile0": {"cycles": 10}}, "modes": {"0": {"lane_fires": 0}}}
        original = json.loads(json.dumps(source))
        row = activity_csv_rows([source])[0]
        self.assertEqual(row, {"id": "example", "cycles": 10, "total_cycles": 120, "total_lane_fires": 0})
        self.assertEqual(source, original)


@unittest.skipUnless(os.name == "nt", "Windows extended-length path regression")
class WindowsPathTests(unittest.TestCase):
    def test_windows_io_prefix_is_idempotent_and_unc_roundtrips(self):
        normal = normal_path(ROOT)
        extended = filesystem_path(normal)
        self.assertTrue(str(extended).startswith("\\\\?\\"))
        self.assertEqual(filesystem_path(extended), extended)
        self.assertEqual(normal_path(extended), normal)
        unc = Path("\\\\example-server\\share\\directory\\report.json")
        self.assertEqual(filesystem_path(unc), Path("\\\\?\\UNC\\example-server\\share\\directory\\report.json"))
        self.assertEqual(normal_path(filesystem_path(unc)), unc)

    def test_actual_light_package_normal_extended_collect_and_long_output_recompute(self):
        packages = sorted((ROOT / "handoff_packages").glob(
            "SNN_Accelerator_FC_SRNN_SCNN_frame_service_*_zh_CN"))
        self.assertLessEqual(len(packages), 1, "Multiple active Chinese deliveries")
        package = packages[0] if packages else ROOT
        package = normal_path(package)
        reference = package / SRNN_RAW / CURRENT_SRNN.parent / "fixture_sample0_t73/reference.json"
        self.assertGreater(len(str(reference)), 260)
        self.assertTrue(filesystem_path(reference).is_file())
        normal_report = collect(package)
        extended_report = collect(filesystem_path(package))
        self.assertEqual(normal_report, extended_report)
        self.assertEqual((len(normal_report["profiles"]), len(normal_report["stages"]),
                          len(normal_report["activity"]), len(normal_report["software_accuracy"])), (11, 2884, 13, 8))
        self.assertIn((SRNN_RAW / CURRENT_SRNN.parent / "fixture_sample0_t73/reference.json").as_posix(),
                      normal_report["sources"])
        self.assertTrue(all(not p.startswith(("//?/", "\\\\?\\")) for p in normal_report["sources"]))
        # Recompute from the real package, but write test outputs only inside the authorized generated tree.
        parent = filesystem_path(ROOT / OUTPUT)
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            output = normal_path(directory) / ("long_output_" + "x" * 150)
            self.assertGreater(len(str(output / "metrics.json")), 260)
            write_report(normal_report, output, root=package)
            names = ("metrics.json", "profiles.csv", "activity.csv", "stages.csv",
                     "software_accuracy.csv", "missing_counters.csv", "README.md")
            original = {name: (filesystem_path(output) / name).read_bytes() for name in names}
            write_report(extended_report, filesystem_path(output), root=filesystem_path(package))
            self.assertEqual(read_json(output / "metrics.json"), normal_report)
            self.assertEqual(original, {name: (filesystem_path(output) / name).read_bytes() for name in names})
            with filesystem_path(output / "profiles.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 11)
            self.assertEqual([r["full_sample_throughput_samples_s"] for r in rows[:5]], [""] * 5)
            self.assertAlmostEqual(float(rows[-1]["full_sample_throughput_samples_s"]), 0.08497705989882713, places=15)
            text = (filesystem_path(output) / "README.md").read_text(encoding="utf-8")
            self.assertNotIn("//?/", text)
            for link in re.findall(r"\]\(([^)]+)\)", text):
                self.assertTrue(filesystem_path(output / link).is_file(), link)


class ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = collect()

    def test_actual_profiles_and_scope(self):
        rows = self.report["profiles"]
        self.assertEqual([r["cycles"] for r in rows],
                         [10147245, 87013885, 12104533, 12250769, 2038360664, 1013344, 8714352,
                          110452102, 112087773, 781657379, 784525456])
        self.assertEqual(rows[0]["cycles_including_initialization"], 11690420)
        self.assertEqual(rows[1]["cycles_including_initialization"], 88557060)
        for row in rows[:5]:
            self.assertIsNone(row["full_sample_throughput_samples_s"])
        for row in rows:
            self.assertIsNone(row["rtl_dataset_accuracy_percent"])
            self.assertFalse(row["hardware_time_measured"])
        self.assertIsNone(rows[2]["initial_load_and_refill_cycles"])
        self.assertEqual(rows[4]["query_samples"], 1)
        self.assertEqual(rows[4]["query_timesteps"], 4)
        self.assertEqual([r["code_changes"] for r in rows[4]["commits"]], [0, 0, 662])

    def test_actual_activity_denominators_and_no_full_window_inheritance(self):
        activity = self.report["activity"]
        self.assertEqual(len(activity), 13)
        for row in activity:
            totals = row["totals"]
            self.assertEqual(sum(totals[k] for k in ("abort", "compute", "issue_wait", "region_wait",
                                                    "observation_wait", "fill", "internal", "idle")),
                             row["cycles"] * 12)
            self.assertAlmostEqual(row["whole_array_destination_lane_occupancy"],
                                   totals["lane_fires"] / (row["cycles"] * 480))
        self.assertLess(activity[0]["whole_array_destination_lane_occupancy"], 0.001)
        self.assertGreater(activity[0]["accepted_issue_destination_occupancy"], 0.95)

    def test_current_fc_full_supplied_samples_and_old_cache_rows_remain_distinct(self):
        rows = {r["id"]: r for r in self.report["profiles"]}
        for arm, cycles, rate, all_cycles in (("off", 1013344, 1052.6204987, 2556519),
                                              ("on", 8714352, 122.4034405, 10257529)):
            current = rows[f"FC_current_source_B16T99_{arm}"]
            self.assertEqual(current["source_family"], "frame_service")
            self.assertEqual(current["cycles"], cycles)
            self.assertAlmostEqual(current["full_sample_throughput_samples_s"], rate, places=7)
            self.assertTrue(current["full_sample_throughput_is_projection"])
            self.assertEqual(current["input"]["shape"], [16, 99, 2312])
            self.assertEqual((current["input"]["valid_timesteps_min"], current["input"]["valid_timesteps_max"]), (94, 99))
            self.assertEqual(current["samples"], 16)
            self.assertEqual(current["cycles_including_initialization"], all_cycles)
            self.assertIsNone(current["isolated_pure_learning_latency_cycles"])
            self.assertIsNone(current["platform_publication_latency_cycles"])
            self.assertEqual(rows[f"FC_support160_cache_r3_{arm}"]["source_family"], "cache_r3")
            self.assertEqual(rows[f"FC_support160_cache_r3_{arm}"]["samples"], 160)
        self.assertEqual(rows["FC_support160_cache_r3_on"]["counters"]["code_writes"], 30)
        current = rows["FC_current_source_B16T99_on"]
        self.assertEqual(current["counters"]["code_writes"], 0)
        self.assertEqual(len(current["commits"]), 99)
        self.assertIsNone(current["commits"][0]["last_write"])
        self.assertIsNone(current["commits"][0]["last_parameter_write_to_commit_clocks"])
        self.assertEqual(sum(s["work_overlap_forward"] for s in current["passive_stages"].values()), 612216)
        self.assertEqual(sum(s["work_only"] for s in current["passive_stages"].values()), 7288079)
        partition = [r for r in self.report["stages"] if r["profile"] == current["id"] and
                     r["kind"] == "passive_stage_partition"]
        self.assertEqual(sum(r["cycles"] for r in partition), current["cycles_including_initialization"])
        self.assertTrue(any(r["profile"] == current["id"] for r in self.report["missing_counters"]))

    def test_current_fc_activity_uses_own_all_reset_released_clocks(self):
        rows = {r["id"]: r for r in self.report["profiles"]}
        for arm, monitor_cycles in (("off", 2556519), ("on", 10257529)):
            current = rows[f"FC_current_source_B16T99_{arm}"]
            activity = next(r for r in self.report["activity"] if r["id"] == current["activity_id"])
            self.assertEqual(activity["cycles"], monitor_cycles)
            self.assertEqual(activity["profile_relation"], current["id"])
            self.assertEqual(activity["physical_tiles"], 12)
            self.assertIn("server_pair_r1", activity["source_log"])
            self.assertEqual(activity["physical_lane_cycle_denominator"], monitor_cycles * 480)
            self.assertNotEqual(activity["physical_lane_cycle_denominator"], current["cycles"] * 480)

    def test_current_fc_summary_or_activity_mismatch_is_rejected(self):
        text = (ROOT / CURRENT_FC / "off.log").read_text(encoding="utf-8")
        archived = json.loads((ROOT / CURRENT_FC / "off.json").read_text(encoding="utf-8"))
        for key in ("runtime_cycles", "h40_activity"):
            changed = json.loads(json.dumps(archived))
            if key == "runtime_cycles":
                changed[key] += 1
            else:
                changed[key]["totals"]["lane_fires"] += 1
            with self.assertRaises(ValueError):
                parse_current_fc(text, changed, "off", archived["scope"])

    def test_current_srnn_complete_native_sample_and_stage_handshake_boundaries(self):
        rows = {r["id"]: r for r in self.report["profiles"]}
        for arm, cycles, rate, all_cycles in (("off", 110452102, 0.6035798817723421, 110572528),
                                             ("on", 112087773, 0.5947719798721192, 112208199)):
            row = rows[f"SRNN_current_source_B1T73_native_{arm}"]
            definition = row["input"]["sample_definition"]
            self.assertEqual((row["batch"], row["samples"], row["timesteps_per_sample"]), (1, 1, 73))
            self.assertEqual((definition["native_steps"], definition["padded_steps"], definition["timestep_policy"]), (73, 0, "native"))
            self.assertEqual(definition["official_b64_first_batch_steps"], 89)
            self.assertEqual(row["input"]["gradient_divisor"], 64)
            self.assertEqual(row["input"]["shape"], [1, 73, 700])
            self.assertTrue(row["input"]["full_sample"])
            self.assertTrue(row["full_sample_throughput_is_projection"])
            self.assertEqual(row["source_family"], "frame_service")
            self.assertIn("SCNN current-source original B1T20 R2", row["timing"]["source_boundary"])
            self.assertEqual(row["cycles"], cycles)
            self.assertAlmostEqual(row["full_sample_throughput_samples_s"], rate, places=15)
            self.assertEqual(row["cycles_including_initialization"], all_cycles)
            self.assertEqual(row["initial_load_and_state_cycles"], 120426)
            self.assertEqual((row["initial_load_cycles"], row["initial_state_cycles"],
                              row["initial_reset_and_configuration_cycles"]), (120321, 97, 8))
            self.assertEqual(row["tb_store_check_cycles"], 0)
            for key in ("tb_store_check_host_seconds", "bus_payload_bytes", "commit_latency_cycles",
                        "isolated_pure_learning_latency_cycles", "platform_publication_latency_cycles",
                        "measured_hardware_throughput_samples_s"):
                self.assertIsNone(row[key])
            self.assertEqual(row["counters"]["store"], 8783360)
            self.assertEqual(row["bus_handshakes"]["store_forward_read"]["accepted"], 8783360)
            self.assertEqual(row["bus_handshakes"]["backbone_parameter_read"]["accepted"], 37777500)
            partition = [s for s in self.report["stages"] if s["profile"] == row["id"] and s["kind"] == "stage"]
            self.assertEqual(sum(s["cycles"] for s in partition), all_cycles)
            self.assertEqual(len([s for s in row["passive_intervals"] if s["phase"] == "m2_forward"]), 73)
            old = rows[f"SRNN_raw700_M2_cache_r3_{arm}"]
            self.assertEqual(old["timesteps_per_sample"], 8)
            self.assertEqual(old["source_family"], "cache_r3")
            self.assertIsNone(old["full_sample_throughput_samples_s"])
        on = rows["SRNN_current_source_B1T73_native_on"]
        self.assertEqual(on["passive_counters"]["learner_busy_cycles"], 1906488)
        self.assertEqual(on["passive_counters"]["pure_learning_busy_cycles"], 1635671)
        self.assertEqual(on["learner_busy_excluding_pure_predicate_cycles"], 270817)
        self.assertEqual((on["counters"]["code"], on["counters"]["residual"], len(on["commits"])), (68040, 145500, 73))
        forward = {s["timestep"]: s for s in on["passive_intervals"] if s["phase"] == "m2_forward"}
        busy = {s["timestep"]: s for s in on["passive_intervals"] if s["phase"] == "pure_learning_busy"}
        overlap = [max(0, min(forward[t]["end"], busy[t]["end"]) -
                       max(forward[t]["start"], busy[t]["start"])) for t in range(73)]
        self.assertEqual(overlap, [1] * 73)
        self.assertEqual(sum(overlap), 73)
        intervals = [s for s in self.report["stages"] if s["profile"] == on["id"] and s["kind"] == "hardware"]
        self.assertTrue(all("73 clocks total" in s["boundary"] for s in intervals))
        self.assertIn("overlap", on["learning_busy_boundary"])

    def test_current_srnn_own_activity_and_minimal_archived_source_reads(self):
        for arm, cycles in (("off", 110572522), ("on", 112208193)):
            profile = next(r for r in self.report["profiles"] if r["id"] == f"SRNN_current_source_B1T73_native_{arm}")
            activity = next(r for r in self.report["activity"] if r["id"] == profile["activity_id"])
            self.assertEqual(activity["profile_relation"], profile["id"])
            self.assertEqual(activity["physical_tiles"], 12)
            self.assertEqual(activity["cycles"], cycles)
            self.assertEqual(activity["tile_cycle_denominator"], cycles * 12)
            self.assertEqual(activity["physical_lane_cycle_denominator"], cycles * 480)
            self.assertNotEqual(activity["cycles"], profile["cycles"])
            self.assertEqual(activity["totals"]["lane_fires"], 14825060)
            self.assertEqual(activity["sample_definition"]["steps"], 73)
            self.assertEqual(activity["source_log"], (CURRENT_SRNN / arm / "run.log").as_posix())
        raw_reads = [p for p in self.report["sources"] if p.startswith(SRNN_RAW.as_posix() + "/")]
        fixture = next(r for r in self.report["profiles"] if r["id"] == "SRNN_current_source_B1T73_native_on")["input"]["fixture"]
        self.assertEqual(raw_reads, sorted([(Path(fixture) / "reference.json").as_posix(),
                                          (SRNN_RAW / "scripts/run_srnn_profile_measurement.py").as_posix()]))

    def test_current_srnn_summary_activity_and_incomplete_raw_log_rejected(self):
        commands = json.loads((ROOT / CURRENT_SRNN / "commands.json").read_text(encoding="utf-8"))
        metadata = commands["sample_definition"]
        directory = ROOT / CURRENT_SRNN / "on"
        text = (directory / "run.log").read_text(encoding="utf-8")
        archived = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
        activity = json.loads((directory / "activity.json").read_text(encoding="utf-8"))
        changed = json.loads(json.dumps(archived))
        changed["sample_service_cycles_including_refill_excluding_tb_checks"] += 1
        with self.assertRaises(ValueError):
            parse_current_srnn(text, changed, activity, commands, "on", metadata)
        changed_activity = json.loads(json.dumps(activity))
        changed_activity["totals"]["lane_fires"] += 1
        with self.assertRaises(ValueError):
            parse_current_srnn(text, archived, changed_activity, commands, "on", metadata)
        for broken in (text + "\n%Error: failure", text.replace("SRNN_SAME_DUT_STEP t=72 ", "REMOVED t=72 "),
                       text.replace("SRNN_PROFILE_COMPLETE ", "REMOVED "),
                       text.replace("kind=hardware phase=m2_forward timestep=72 ", "kind=hardware phase=REMOVED timestep=72 ")):
            with self.assertRaises(ValueError):
                parse_current_srnn(broken, archived, activity, commands, "on", metadata)

    def test_current_srnn_exit_scope_validation_preserves_server_path_provenance(self):
        commands = json.loads((ROOT / CURRENT_SRNN / "commands.json").read_text(encoding="utf-8"))
        metadata = commands["sample_definition"]
        commands["cwd"] = "/root/autodl-tmp/SNN_CURRENT_PROFILE_MEASUREMENTS_20261009_R1/srnn"
        validate_current_srnn_commands(commands, metadata)
        for stage in ("build", "off", "on"):
            changed = json.loads(json.dumps(commands))
            changed[stage + "_exit"] = 1
            with self.assertRaises(ValueError):
                validate_current_srnn_commands(changed, metadata)
        for key, value in (("steps", 89), ("timestep_policy", "official-b64"), ("complete_sample", False)):
            changed = json.loads(json.dumps(commands))
            changed["sample_definition"][key] = value
            with self.assertRaises(ValueError):
                validate_current_srnn_commands(changed, changed["sample_definition"])
        changed = json.loads(json.dumps(commands))
        changed["run"]["on"][0] += "_another_exe"
        with self.assertRaises(ValueError):
            validate_current_srnn_commands(changed, metadata)

    def test_current_scnn_complete_original_sample_counts_projection_and_unknowns(self):
        rows = {r["id"]: r for r in self.report["profiles"]}
        for arm, cycles, all_cycles, rate in (("off", 781657379, 781728973, 0.0852888598735619),
                                             ("on", 784525456, 784597050, 0.08497705989882713)):
            row = rows[f"SCNN_current_source_B1T20_{arm}"]
            self.assertEqual((row["samples"], row["batch"], row["timesteps_per_sample"]), (1, 1, 20))
            self.assertEqual(row["input"]["sample_index"], 0)
            self.assertEqual(row["input"]["multiplier"], 640)
            self.assertEqual(row["input"]["corruption"], "none; original clean sample0")
            self.assertEqual(sum(r["sample_reset"] for r in row["input"]["frame_schedule"]), 1)
            self.assertEqual(sorted({r["time_base"] for r in row["input"]["frame_schedule"]}), [0, 4, 8, 12, 16])
            self.assertEqual(row["source_family"], "frame_service")
            self.assertTrue(row["full_sample_throughput_is_projection"])
            self.assertEqual(row["cycles"], cycles)
            self.assertEqual(row["cycles_including_initialization"], all_cycles)
            self.assertEqual(row["setup_cycles"], 71594)
            self.assertAlmostEqual(row["full_sample_throughput_samples_s"], rate, places=15)
            self.assertAlmostEqual(row["projected_interval_ms"], cycles * 15 / 1e6)
            for key in ("initial_load_and_refill_cycles", "initial_state_cycles", "host_file_load_seconds",
                        "host_golden_seconds", "isolated_pure_learning_latency_cycles", "platform_publication_latency_cycles"):
                self.assertIsNone(row[key])
            self.assertEqual(row["passive_timing"]["golden_cycles"], 0)
            self.assertEqual(row["passive_timing"]["golden_calls"], 165)
            self.assertEqual((row["counters"]["frames"], row["counters"]["descriptors"], row["counters"]["features"]), (20, 160, 40960))
            self.assertEqual((row["counters"]["currents"], row["counters"]["logits"], row["counters"]["probes"]), (220, 220, 480))
            for stage in ("elab", "build"):
                self.assertEqual(row["execution"]["warnings"][stage], {"MISINDENT": 9, "UNDRIVEN": 2, "UNSIGNED": 7})
            for field in ("state_restore_rows", "trace_restore_rows", "state_spill_rows", "trace_spill_rows"):
                self.assertEqual(row["flow_counts"][field], 1065400)
            self.assertEqual(row["bus_payload"]["state_restore_bytes"], 42616000)
            self.assertEqual(row["bus_payload"]["state_restore_scalar16_words"], 21308000)
            self.assertEqual(row["initial_publication_words"], 1892)
            self.assertEqual(row["flow_counts"]["code_write_words"], 0)
        on, off = rows["SCNN_current_source_B1T20_on"], rows["SCNN_current_source_B1T20_off"]
        self.assertEqual((on["counters"]["traces"], on["counters"]["rmw"], len(on["commits"]), on["counters"]["publications"]),
                         (40960, 85600, 5, 11352))
        self.assertEqual(on["runtime_publication_words"], 9460)
        self.assertEqual(on["flow_counts"]["residual_write_words"], 85600)
        self.assertEqual(on["flow_counts"]["forward_reads"], 184478208)
        self.assertEqual(off["flow_counts"]["forward_reads"], 184365568)
        self.assertEqual((off["counters"]["traces"], off["counters"]["rmw"], off["runtime_publication_words"]), (0, 0, 0))
        self.assertEqual((off["active_counters"]["learning"], off["active_counters"]["publication"]), (0, 0))
        self.assertEqual(rows["SCNN_real_final_B4T12_query4"]["counters"]["frames"], 52)
        self.assertIsNone(rows["SCNN_real_final_B4T12_query4"]["full_sample_throughput_samples_s"])

    def test_current_scnn_stage_partition_overlap_own_activity_and_minimal_raw(self):
        for arm, monitor_cycles in (("off", 781728973), ("on", 784597050)):
            profile = next(r for r in self.report["profiles"] if r["id"] == f"SCNN_current_source_B1T20_{arm}")
            selected = [s for s in self.report["stages"] if s["profile"] == profile["id"]]
            partition = [s for s in selected if s["kind"] == "descriptor" or s["stage"] in
                         ("frame_minus_descriptors", "outside_frames", "initial_setup")]
            self.assertEqual(sum(s["cycles"] for s in partition), monitor_cycles)
            self.assertEqual(len([s for s in selected if s["kind"] == "descriptor"]), 160)
            self.assertEqual(len([s for s in selected if s["kind"] == "control_interval"]), 60)
            self.assertEqual(len([s for s in selected if s["kind"] == "window_interval"]), 5)
            activity = next(r for r in self.report["activity"] if r["id"] == profile["activity_id"])
            self.assertEqual(activity["cycles"], monitor_cycles)
            self.assertEqual(activity["profile_relation"], profile["id"])
            self.assertEqual(activity["physical_tiles"], 12)
            self.assertEqual(activity["tile_cycle_denominator"], 12 * monitor_cycles)
            self.assertNotEqual(activity["physical_lane_cycle_denominator"], 480 * profile["cycles"])
            self.assertEqual(activity["totals"]["lane_fires"], 3118447608)
            self.assertEqual(activity["totals"]["enabled_terms"], 720486568)
            self.assertEqual(activity["source_log"], (CURRENT_SCNN / ("run_" + arm + ".log")).as_posix())
            store_loads = next(s for s in selected if s["kind"] == "flow_counter" and s["stage"] == "store_loads")
            self.assertIn("constant", store_loads["boundary"])
        on = next(r for r in self.report["profiles"] if r["id"] == "SCNN_current_source_B1T20_on")
        self.assertEqual(sum(r["interval_count"] for r in on["learning_busy_by_window"]), 91393)
        self.assertEqual(sum(r["cycles"] for r in on["learning_busy_by_window"]), 611436)
        self.assertEqual(len(on["learning_busy_by_window"]), 5)
        self.assertTrue(all(r["last_end"] - r["first_start"] > r["cycles"] for r in on["learning_busy_by_window"]))
        self.assertEqual(len(on["publication_request_to_done"]), 5)
        self.assertEqual(sum(r["end"]-r["start"] for r in on["phase_intervals"] if r["kind"] == "publication" and r["window"] > 0), 357940)
        pending = [r for r in on["phase_intervals"] if r["kind"] == "commit_pending_to_pulse"]
        self.assertEqual(len(pending), 5)
        for row in pending:
            publication = next(p for p in on["phase_intervals"] if p["kind"] == "publication" and p["window"] == row["window"])
            self.assertLessEqual(row["start"], publication["start"])
            self.assertGreaterEqual(row["end"], publication["end"])
        raw_reads = [p for p in self.report["sources"] if p.startswith(SCNN_RAW.as_posix() + "/")]
        self.assertEqual(raw_reads, sorted((SCNN_RAW / path).as_posix() for path in SCNN_RAW_FILES))

    def test_current_scnn_raw_summary_mismatch_and_missing_completion_rejected(self):
        directory = ROOT / CURRENT_SCNN
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        preparation = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / preparation["fixture"] / "manifest.json").read_text(encoding="utf-8"))
        status = json.loads((directory / "run_off.status.json").read_text(encoding="utf-8"))
        text = (directory / "run_off.log").read_text(encoding="utf-8")
        archived = summary["measurements"]["off"]
        changed = json.loads(json.dumps(archived))
        changed["timing"]["sample_cycles"] += 1
        with self.assertRaises(ValueError):
            parse_current_scnn(text, changed, reference, status, "off")
        for broken in (text + "\n%Error: failed", text.replace("SCNN_PROFILE_FRAME frame=19 ", "REMOVED frame=19 "),
                       text.replace("SCNN_PROFILE_PASS ", "REMOVED "),
                       text.replace("SCNN_PROFILE_DDR bank=8 ", "REMOVED bank=8 ")):
            with self.assertRaises(ValueError):
                parse_current_scnn(broken, archived, reference, status, "off")
        changed = json.loads(json.dumps(archived))
        changed["h40_summary"]["lane_fires"] += 1
        with self.assertRaises(ValueError):
            parse_current_scnn(text, changed, reference, status, "off")

    def test_current_scnn_stage_exits_continuity_and_server_paths(self):
        directory = ROOT / CURRENT_SCNN
        preparation = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / preparation["fixture"] / "manifest.json").read_text(encoding="utf-8"))
        statuses = {key: json.loads((directory / file).read_text(encoding="utf-8")) for key, file in
                    (("elab", "elab.status.json"), ("build", "build.status.json"),
                     ("off", "run_off.status.json"), ("on", "run_on.status.json"))}
        launcher = (directory / "scnn_launcher.status").read_text(encoding="utf-8")
        for arm in ("off", "on"):
            statuses[arm]["command"][0] = "/other-server/workspace/obj/Vscgraph"
        validate_current_scnn_records(preparation, reference, statuses, launcher)
        for stage in statuses:
            changed = json.loads(json.dumps(statuses))
            changed[stage]["exit_code"] = 1
            with self.assertRaises(ValueError):
                validate_current_scnn_records(preparation, reference, changed, launcher)
        with self.assertRaises(ValueError):
            validate_current_scnn_records(preparation, reference, statuses, launcher.replace("off_exit=0", "off_exit=1"))
        changed = json.loads(json.dumps(reference))
        changed["frame_schedule"][4]["sample_reset"] = True
        with self.assertRaises(ValueError):
            validate_current_scnn_records(preparation, changed, statuses, launcher)
        for key, value in (("multiplier", 160), ("timesteps", 12), ("corruption", "shifted")):
            changed = json.loads(json.dumps(reference))
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_current_scnn_records(preparation, changed, statuses, launcher)

    def test_software_accuracy_not_rtl_accuracy(self):
        rows = self.report["software_accuracy"]
        self.assertEqual(len(rows), 8)
        integer = {r["topology"]: r["correct"] for r in rows
                   if r["arm"] in ("integer_raw24", "full_integer")}
        self.assertEqual(integer, {"FC": 9537, "SRNN": 1822, "SCNN": 250})
        self.assertTrue(all(r["execution"] == "software" and r["rtl_dataset_accuracy_percent"] is None for r in rows))

    def test_actual_scnn_stage_accounting_and_traffic(self):
        selected = [r for r in self.report["stages"] if r["profile"] == "SCNN_real_final_B4T12_query4"]
        partition = [r for r in selected if r["kind"] == "layer" or r["stage"] in
                     ("frame_minus_layers", "outside_frames", "initialization")]
        self.assertEqual(sum(r["cycles"] for r in partition), 2038360664)
        layers = [r for r in selected if r["kind"] == "layer"]
        self.assertEqual(len(layers), 416)
        self.assertEqual(sum(r["requests"] for r in layers if r["layer"] == 1), 748800)
        self.assertEqual(sum(r["requests"] for r in layers if r["layer"] == 8), 10303488)
        scnn = next(r for r in self.report["profiles"] if r["id"] == "SCNN_real_final_B4T12_query4")
        self.assertEqual(scnn["crossings"][0]["code_after"], 869)

    def test_outputs_roundtrip_with_nulls_and_readme_links(self):
        # All test artifacts stay in the user-authorized output tree.
        parent = ROOT / OUTPUT
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            output = Path(directory)
            write_report(self.report, output)
            read = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(read, self.report)
            with (output / "profiles.csv").open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), PROFILE_CSV_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row["full_sample_throughput_samples_s"] == "" for row in rows[:5]))
            self.assertAlmostEqual(float(rows[5]["full_sample_throughput_samples_s"]), 1052.6204987, places=7)
            self.assertAlmostEqual(float(rows[6]["full_sample_throughput_samples_s"]), 122.4034405, places=7)
            self.assertAlmostEqual(float(rows[7]["full_sample_throughput_samples_s"]), 0.6035798817723421, places=15)
            self.assertAlmostEqual(float(rows[8]["full_sample_throughput_samples_s"]), 0.5947719798721192, places=15)
            self.assertAlmostEqual(float(rows[9]["full_sample_throughput_samples_s"]), 0.0852888598735619, places=15)
            self.assertAlmostEqual(float(rows[10]["full_sample_throughput_samples_s"]), 0.08497705989882713, places=15)
            self.assertTrue(all(row["period_ns"] == "15.0" for row in rows))
            self.assertEqual(rows[2]["initial_load_and_refill_cycles"], "")
            self.assertEqual(rows[2]["initial_state_cycles"], "")
            self.assertEqual(rows[4]["samples"], "")
            self.assertEqual(rows[4]["training_samples"], "4")
            self.assertEqual(rows[4]["query_samples"], "1")
            self.assertEqual(rows[4]["query_timesteps"], "4")
            self.assertEqual(len(read["profiles"][0]["input"]["support_indices"]), 160)
            self.assertEqual(len(read["profiles"][4]["input"]["frame_schedule"]), 52)
            for path in output.glob("*.csv"):
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        for value in row.values():
                            try:
                                decoded = json.loads(value)
                            except json.JSONDecodeError:
                                continue
                            self.assertNotIsInstance(decoded, (list, dict), path.name)
            with (output / "activity.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 13)
            self.assertNotIn("tiles", rows[0])
            self.assertNotIn("modes", rows[0])
            self.assertEqual(int(rows[0]["total_lane_fires"]), read["activity"][0]["totals"]["lane_fires"])
            text = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("JSON null", text)
            self.assertIn("未测量", text)
            self.assertIn("frame-service", text)
            self.assertIn("精简版本", text)
            self.assertIn("详细版本", text)
            self.assertIn("1052.6204987", text)
            self.assertIn("122.4034405", text)
            self.assertIn("612216重叠忙不可再加", text)
            self.assertIn("0.603579881772", text)
            self.assertIn("0.594771979872", text)
            self.assertIn("不是官方firstbatch B64 padded T89", text)
            self.assertIn("检查周期为0；主机检查耗时未知null", text)
            self.assertIn("73步合计73周期，不能相加为互斥耗时", text)
            self.assertIn("不需要镜像整个raw、obj", text)
            self.assertIn("0.085288859874", text)
            self.assertIn("0.084977059899", text)
            self.assertIn("on有91393段碎片化学习忙区间", text)
            self.assertIn("store_loads=0是monitor打印常量", text)
            for link in re.findall(r"\]\(([^)]+)\)", text):
                self.assertTrue(filesystem_path(output / link).resolve().is_file(), link)
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ["README.md", "activity.csv", "metrics.json", "missing_counters.csv",
                              "profiles.csv", "software_accuracy.csv", "stages.csv"])


if __name__ == "__main__":
    unittest.main()
