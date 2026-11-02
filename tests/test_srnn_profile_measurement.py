"""Focused generator/parser checks; no RTL simulation or GPU execution."""
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_srnn_profile_measurement_reference as reference
import run_srnn_profile_measurement as runner


def metadata(steps=2):
    return dict(batch=1, samples=1, sample_index=0, complete_sample=True,
                steps=steps, native_steps=steps, feedforward_words=17)


def synthetic_log(steps=2, learning=0, missing=None):
    lines = ["SRNN_SAME_DUT_PASS batch=1 steps=%d learning=%d" % (steps, learning)]
    cursor = 0
    for phase in ("reset", "config", "parameter_load", "initial_state"):
        lines.append(f"SRNN_PROFILE_INTERVAL kind=stage phase={phase} timestep=-1 start={cursor} end={cursor+2} cycles=2")
        cursor += 2
    initialized = cursor
    for timestep in range(steps):
        for phase in ("backbone_refill_ff", "backbone_refill_r", "backbone", "m2", "tb_store_check", "boundary"):
            length = 0 if phase == "tb_store_check" else 3
            lines.append(f"SRNN_PROFILE_INTERVAL kind=stage phase={phase} timestep={timestep} start={cursor} end={cursor+length} cycles={length}")
            cursor += length
    for name in runner.COUNTER_NAMES:
        if name == missing:
            continue
        value = steps if name in ("backbone_layer_done", "capture_done", "backbone_replay_done", "m2_forward_done", "timestep_done") else 0
        lines.append(f"SRNN_PROFILE_COUNT name={name} value={value}")
    lines.append(f"SRNN_PROFILE_COMPLETE batch=1 steps={steps} learning={learning} initialized={initialized} end={cursor} store_checks={steps*120320}")
    return "\n".join(lines)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(dataset="SHD official test", time_window_us=10000, include_incomplete=False,
                             samples=[dict(dataset_index=0, clipped_feature_counts=0, time_steps=2, records=3, label=10)])
        self.words = [1, 1 << 18, 2 << 18]

    def test_complete_native_not_t100(self):
        result = reference.sequence_definition(self.manifest, self.words, "native", 89)
        self.assertEqual((result["steps"], result["padded_steps"]), (2, 0))

    def test_explicit_formal_padding(self):
        result = reference.sequence_definition(self.manifest, self.words, "official-b64", 89)
        self.assertEqual((result["steps"], result["padded_steps"]), (89, 87))

    def test_truncated_or_trailing_stream_rejected(self):
        for words in (self.words[:-1], self.words+[1], [1, 2 << 18, 2 << 18]):
            with self.subTest(words=words), self.assertRaises(ValueError):
                reference.sequence_definition(self.manifest, words)

    def test_manifest_mismatch_and_clipping_rejected(self):
        self.manifest["samples"][0]["time_steps"] = 3
        with self.assertRaises(ValueError):
            reference.sequence_definition(self.manifest, self.words)
        self.manifest["samples"][0].update(time_steps=2, clipped_feature_counts=1)
        with self.assertRaises(ValueError):
            reference.sequence_definition(self.manifest, self.words)

    def test_actual_official_sample0_definition(self):
        manifest = json.loads((reference.DATA / "manifest.json").read_text(encoding="utf-8"))
        words = [int(s, 16) for s in (reference.DATA / "sample_0000.hex").read_text().split()]
        padding = json.loads(reference.PAD_REPORT.read_text())["padded_timesteps_per_batch"][0]
        result = reference.sequence_definition(manifest, words, "native", padding)
        self.assertEqual((result["steps"], result["official_b64_first_batch_steps"]), (73, 89))


class GeneratorTests(unittest.TestCase):
    def test_body_preserves_actual_dut_feedback_and_scoreboard(self):
        body = runner.render_body(metadata(73))
        self.assertIn("T = 73,", body)
        self.assertIn("timestep*480+h", body)
        self.assertIn("ff_offsets[timestep+1]", body)
        self.assertNotIn("timestep*960", body)
        self.assertNotIn("ff_offsets[timestep*2", body)
        self.assertIn("idx = timestep*256", body)
        self.assertIn("m_pre[0:T*256-1]", body)
        self.assertIn("m_logit[0:T*20-1]", body)
        self.assertNotIn("T56", body)
        self.assertNotIn("timestep56", body)
        self.assertIn("previous_spikes = next_spikes", body)
        self.assertIn("mask[lane]=previous_spikes", body)
        self.assertIn("BONE_MISMATCH", body)
        self.assertIn("M2_HIDDEN_MISMATCH", body)
        self.assertIn("m_after[timestep*5120", body)
        self.assertNotIn("force ", body)
        self.assertIn('profile_enter("tb_store_check")', body)
        self.assertLess(body.index("profile_finish();"), body.index("`undef RED"))

    def test_changed_template_anchor_is_not_silently_ignored(self):
        body = runner.BODY.read_text(encoding="utf-8").replace("T = 8,", "T = 9,")
        with self.assertRaises(ValueError):
            runner.render_body(metadata(), body)

    def test_105_sources_explicitly_include_store(self):
        tb = runner.OUTPUT / "not_created_test_tb.sv"
        sources = runner.simulation_sources(runner.SOURCE_LIST, tb)
        self.assertEqual(len(sources), 107)
        self.assertEqual(len(sources[:-2]), 105)
        self.assertTrue(all(p.is_relative_to(ROOT / "rtl") for p in sources[:-2]))
        self.assertIn(runner.resolve_verification_store(), sources)

    def test_activity_opt_in_only_adds_passive_source(self):
        tb = runner.OUTPUT / "not_created_test_tb.sv"
        plain = runner.simulation_sources(runner.SOURCE_LIST, tb)
        observed = runner.simulation_sources(runner.SOURCE_LIST, tb, True)
        self.assertEqual(observed, plain+[runner.ACTIVITY_MONITOR])
        self.assertEqual(len(observed), 108)

    def test_parameterized_rom_included(self):
        paths = runner.rom_dependencies([ROOT / "rtl/learning/unified_scnn_tp_surrogate_type1_q8_8.sv"])
        self.assertEqual(paths, [ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"])
        self.assertTrue(paths[0].is_file())

    def test_structural_warnings_block_but_ordinary_warnings_remain(self):
        result = runner.inspect_build_log("%Warning-WIDTH: a\n%Warning-LATCH: b\n%Warning-UNOPTFLAT: c\n%Warning-MULTIDRIVEN: d")
        self.assertEqual(result["blocking"], dict(LATCH=1, UNOPTFLAT=1, MULTIDRIVEN=1))
        self.assertEqual(result["ordinary"], dict(WIDTH=1))
        self.assertFalse(runner.inspect_build_log("%Warning-WIDTH: retained")["blocking"])

    def test_every_observed_name_exists_in_current_source(self):
        top = (ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv").read_text()
        engine = (ROOT / "rtl/learning/unified_srnn_m2_h40_forward_engine_v1.sv").read_text()
        learner = (ROOT / "rtl/learning/unified_srnn_tp_wout_direct_online_learning_engine.sv").read_text()
        import re
        for expression in [s for _, v, r in runner.CHANNELS for s in (v, r)] + [s for _, s in runner.EVENTS]:
            for name in re.findall(r"`ENG\.(\w+)", expression):
                self.assertRegex(engine, rf"\b{name}\b")
            for name in re.findall(r"`RED\.(\w+)", expression):
                self.assertRegex(learner, rf"\b{name}\b")
            for name in re.findall(r"(?:u_final\.)?\b((?:[io]_\w+|link_\w+))\b", expression):
                if "`ENG."+name in expression or "`RED."+name in expression:
                    continue
                self.assertRegex(top, rf"\b{name}\b")

    def test_monitor_has_true_transfers_stalls_and_no_unexpanded_tokens(self):
        monitor = runner.render_monitor()
        self.assertNotIn("@", monitor.replace("@(posedge i_clk)", ""))
        self.assertIn("(o_store_learning_write_valid) && (i_store_learning_write_ready)", monitor)
        self.assertIn("(o_store_forward_read_valid) && !(i_store_forward_read_ready)", monitor)
        self.assertIn(runner.PURE_BUSY, monitor)
        self.assertEqual(len(runner.COUNTER_NAMES), len(set(runner.COUNTER_NAMES)))


class ParserTests(unittest.TestCase):
    def summarize_existing(self, previous, period_ns=None):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = base / "fixture"
            fixture.mkdir()
            (fixture / "reference.json").write_text(json.dumps(metadata()), encoding="utf-8")
            output = base / "pair" / "off"
            output.mkdir(parents=True)
            (output / "run.log").write_text(synthetic_log(), encoding="utf-8")
            (output / "measurements.json").write_text(json.dumps(previous), encoding="utf-8")
            args = ["--fixture", str(fixture), "--summarize", str(output)]
            if period_ns is not None:
                args += ["--period-ns", str(period_ns)]
            with patch.object(runner, "OUTPUT", base), patch.object(runner.subprocess, "run") as simulate:
                with redirect_stdout(StringIO()):
                    runner.main(args)
                simulate.assert_not_called()
            return json.loads((output / "measurements.json").read_text(encoding="utf-8"))

    def test_summarize_preserves_recorded_walltime_and_projection(self):
        result = self.summarize_existing(dict(simulation_wall_seconds=3012.2, period_ns_projection=15))
        self.assertEqual(result["simulation_wall_seconds"], 3012.2)
        self.assertEqual(result["period_ns_projection"], 15)
        self.assertEqual(result["projected_sample_interval_ms"], 30 * 15 * 1e-6)

    def test_summarize_unknown_walltime_stays_null_and_explicit_period_wins(self):
        result = self.summarize_existing(dict(simulation_wall_seconds=None, period_ns_projection=15), 10)
        self.assertIsNone(result["simulation_wall_seconds"])
        self.assertEqual(result["period_ns_projection"], 10)

    def test_complete_sample_allows_only_explicit_projection(self):
        result = runner.parse_log(synthetic_log(), metadata(), 15)
        self.assertTrue(result["status"].startswith("PASS_"))
        self.assertEqual(result["sample_service_cycles_including_refill_excluding_tb_checks"], 30)
        self.assertEqual(result["tb_store_check_cycles"], 0)
        self.assertIsNone(result["tb_store_check_host_seconds"])
        self.assertIsNone(result["measured_hardware_throughput_samples_s"])
        self.assertIsNotNone(result["full_sample_throughput_samples_s_projection"])
        self.assertIsNone(runner.parse_log(synthetic_log(), metadata())["full_sample_throughput_samples_s_projection"])

    def test_missing_counter_is_null_not_zero_or_throughput(self):
        result = runner.parse_log(synthetic_log(missing="store_learning_read_accepted"), metadata(), 15)
        self.assertIsNone(result["counters"]["store_learning_read_accepted"])
        self.assertIn("store_learning_read_accepted", result["missing_counters"])
        self.assertEqual(result["status"], "incomplete_measurement")
        self.assertIsNone(result["full_sample_throughput_samples_s_projection"])

    def test_old_window_log_cannot_supply_new_metrics(self):
        result = runner.parse_log("SRNN_SAME_DUT_PASS batch=1 steps=8 cycles=1234", metadata(73), 15)
        self.assertEqual(result["status"], "incomplete_measurement")
        self.assertTrue(all(value is None for value in result["counters"].values()))
        self.assertIsNone(result["sample_service_cycles_including_refill_excluding_tb_checks"])

    def test_overlap_duplicate_and_wrong_steps_rejected(self):
        for text in (synthetic_log()+"\nSRNN_PROFILE_COUNT name=commit_pulse value=0",
                     synthetic_log().replace("start=0 end=2 cycles=2", "start=0 end=2 cycles=3"),
                     synthetic_log().replace("SRNN_PROFILE_COMPLETE batch=1 steps=2", "SRNN_PROFILE_COMPLETE batch=1 steps=8")):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                runner.parse_log(text, metadata())

    def test_missing_timestep_interval_rejected(self):
        text = synthetic_log().replace("phase=m2 timestep=1", "phase=boundary timestep=1")
        with self.assertRaises(ValueError):
            runner.parse_log(text, metadata())


if __name__ == "__main__":
    unittest.main()
