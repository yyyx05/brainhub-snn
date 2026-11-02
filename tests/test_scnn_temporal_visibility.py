"""Bounded CPU-only preparation tests; no RTL, shell job or GPU invocation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_scnn_temporal_visibility as ref
import numpy as np


class TemporalVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = ref.compute()
        cls.arrays = cls.result["arrays"]
        cls.manifest = cls.result["manifest"]
        cls.schedule = cls.manifest["frame_schedule"][:48]
        cls.features = np.empty((4, 12, 2048), dtype=np.int64)
        for row, values in zip(cls.schedule, cls.arrays["train_features"]):
            cls.features[row["sample_slot"], row["timestep"]] = values

    def test_fixed_shifted_formal_episode(self):
        m = self.manifest
        self.assertEqual((m["seed"], m["support_subject"], m["query_subject"], m["temporal_jitter"]), (38472, 29, 28, 1))
        self.assertEqual(m["support_ids"], [0, 1, 2, 3])
        self.assertEqual(m["labels"], [8, 5, 3, 6])
        self.assertEqual(m["query_id"], 22)
        self.assertEqual(m["support_domain"], "shifted")
        self.assertTrue(m["checks"]["formal_first_batch_counters_match"])
        self.assertEqual(m["checks"]["formal_first_batch_counters"]["code_change_events"], 5190)
        self.assertEqual(m["checks"]["prefix_values"], 3743810)

    def test_actual_jitter_before_prefix_not_clean(self):
        import torch
        from supplement_scnn_weight_quantization import input_batches, PAYLOAD
        import run_scnn_tp_selected_layer_shift_adaptation as api
        with (PAYLOAD / "data/test_264x20_inputs_q8_8.hex").open(encoding="ascii") as stream:
            _, frames = next(input_batches(stream, 264, 4, (20, 2, 32, 32)))
        full = api.transform_event_view(torch.from_numpy(frames), [0, 1, 2, 3], dx=0, dy=0,
                                       corruption=api.EventCorruption(temporal_jitter=1, seed=38472))
        packed = ref.pack_frames((full * 256).to(torch.int64).numpy(), self.schedule)
        np.testing.assert_array_equal(packed, self.arrays["train_raw"])
        self.assertFalse(np.array_equal(packed, ref.pack_frames((frames*256).astype(np.int64), self.schedule)))

    def test_sample_major_schedule_uses_four_contexts(self):
        self.assertEqual(len(self.manifest["frame_schedule"]), 52)
        for w in range(3):
            rows = self.schedule[w*16:(w+1)*16]
            self.assertEqual([r["capture_row"] for r in rows], list(range(16)))
            self.assertEqual([r["sample_id"] for r in rows], [0]*4+[1]*4+[2]*4+[3]*4)
            self.assertEqual([r["timestep"] for r in rows], list(range(w*4, w*4+4))*4)
            self.assertEqual([r["sample_state_reset"] for r in rows], [w == 0, False, False, False]*4)
            self.assertEqual([r["wait_for_commit_after"] for r in rows], [False]*15+[True])
            self.assertEqual([r["completed_windows"] for r in rows], [w]*16)
        self.assertTrue(self.manifest["frame_schedule"][48]["sample_state_reset"])
        self.assertTrue(all(not r["learning"] and r["completed_windows"] == 3 for r in self.manifest["frame_schedule"][48:]))

    def test_hardware_pair_and_version_are_not_algorithm_oracles(self):
        for row in self.manifest["frame_schedule"] + self.manifest["commits"]:
            self.assertFalse(any("pair" in key or "version" in key for key in row))
        self.assertIn("not an algorithm oracle", self.manifest["publication"])
        self.assertIn("it does NOT alternate pairs", ref.INTEGRATION)

    def test_finaltop_runtime_is_pending_actual_measurement(self):
        resource = self.manifest["resource"]
        self.assertIsNone(resource["current_finaltop_estimate_hours"])
        self.assertIn("DDR/refill", resource["current_finaltop_estimate_status"])
        self.assertIn("NOT a current final-top estimate", resource["legacy_graph_online_subtop_only"]["timing_basis"])
        self.assertNotIn("estimated_simulation_hours", resource)

    def test_preparation_dependencies_exist(self):
        dependencies = self.manifest["dependencies"]
        for field in ("python_files", "report_files", "model_files", "raw_files"):
            for path in dependencies[field]:
                self.assertTrue((ROOT / path).is_file(), path)
        self.assertTrue((ROOT / dependencies["prefix_directory"]).is_dir())
        self.assertTrue((ROOT / dependencies["inactive_instance_rom"]).is_file())

    def test_trace_continues_independently_for_each_sample(self):
        old = np.zeros((4, 2048), dtype=np.int64)
        for row, features, expected in zip(self.schedule, self.arrays["train_features"], self.arrays["train_traces"]):
            s = row["sample_slot"]
            old[s] = ((old[s] * 250) >> 8) + features
            np.testing.assert_array_equal(old[s], expected)
            if row["timestep"] in (4, 8):
                self.assertFalse(np.array_equal(expected, features))

    def test_cumulative_logits_continue_independently(self):
        state = np.zeros((4, 11), dtype=np.int64)
        for row, current, cumulative, centered in zip(self.schedule, self.arrays["train_currents"],
                                                      self.arrays["train_cumulative"], self.arrays["train_centered"]):
            s = row["sample_slot"]
            state[s] = (state[s]+current+(1 << 31)) % (1 << 32) - (1 << 31)
            np.testing.assert_array_equal(state[s], cumulative)
            np.testing.assert_array_equal(np.clip(state[s]-state[s].max(), -32768, 32767), centered)

    def test_time_major_kernel_matches_all_commits(self):
        result = ref.load_kernel().replay(self.features, self.arrays["initial_codes"],
                                         labels=self.arrays["labels"], capture=True)
        for w in range(3):
            for ours, theirs in (("codes", "codes"), ("residuals", "residuals"), ("gradient", "gradients")):
                np.testing.assert_array_equal(self.arrays[f"window{w+1:02d}_{ours}"], result["history"][theirs][w])
        np.testing.assert_array_equal(result["codes"], self.arrays["final_codes"])

    def test_cold_membrane_restart_is_not_continuation(self):
        import torch
        from evaluate_scnn_full_integer import IntegerNetwork
        cold = ref.extract(IntegerNetwork("cpu"), torch.tensor(self.arrays["train_raw"][16:20][None]), [0])
        self.assertFalse(np.array_equal(cold["features"][0], self.arrays["train_features"][16:20]))
        self.assertFalse(np.array_equal(cold["membrane_probes"][0], self.arrays["train_membrane_post_probes"][16:20]))

    def test_global_accumulator_and_cold_logit_negative_controls(self):
        for row in self.manifest["negative_controls"].values():
            self.assertGreater(row["changed_centered_values"], 0)
            self.assertEqual(row["first_mismatch"]["frame"], 16)
            self.assertEqual(row["first_mismatch"]["sample_id"], 0)
            self.assertEqual(row["first_mismatch"]["timestep"], 4)

    def test_zero_residual_and_unchanged_multiplier(self):
        self.assertFalse(np.any(self.arrays["initial_residuals"]))
        self.assertEqual(self.manifest["multiplier"], 640)
        self.assertEqual(self.manifest["vote_increment"], 0)
        previous = self.arrays["initial_residuals"]
        for w in range(1, 4):
            total = previous + self.arrays[f"window{w:02d}_delta"]
            code_delta = np.sign(total) * (np.abs(total) >> 15)
            np.testing.assert_array_equal(total - code_delta*32768, self.arrays[f"window{w:02d}_residuals"])
            previous = self.arrays[f"window{w:02d}_residuals"]

    def test_exact_crossing_and_witness(self):
        self.assertEqual([r["code_changes"] for r in self.manifest["commits"]], [0, 0, 662])
        self.assertEqual(self.manifest["first_crossing_window"], 3)
        witness = self.manifest["commits"][2]["witness"]
        self.assertEqual(witness, dict(address=18433, class_id=9, feature=1,
                                     code_before=870, code_after=869, residual_before=-16155,
                                     delta=-21184, residual_after=-4571))
        self.assertEqual(int(np.count_nonzero(self.arrays["final_codes"] != self.arrays["initial_codes"])), 662)

    def test_fresh_fixed_query_observes_updated_codes(self):
        q = self.manifest["query_visibility"]
        self.assertEqual([q[n]["changed"] for n in ("currents", "centered")], [4, 4])
        self.assertEqual(q["currents"]["first_witness"], dict(timestep=0, class_id=9, before=1345, after=1340))
        self.assertEqual(q["centered"]["first_witness"], dict(timestep=0, class_id=9, before=-56, after=-61))
        np.testing.assert_array_equal(self.arrays["query_currents"],
                                     np.clip((self.arrays["query_features"] @ self.arrays["final_codes"].T) >> 15, -32768, 32767))

    def test_independent_scalar_witness_terms_and_query(self):
        scalar = self.manifest["scalar_checks"]
        self.assertEqual(scalar["status"], "pass")
        self.assertEqual((scalar["checked_train_dots"], scalar["checked_query_dots"]), (528, 88))
        self.assertEqual((scalar["checked_gradient_parameters"], scalar["checked_update_parameters"]), (67584, 67584))
        for w, witness in enumerate(scalar["windows"], 1):
            self.assertEqual(len(witness["terms"]), 16)
            self.assertEqual(witness["gradient"], sum(term["contribution"] for term in witness["terms"]))
            self.assertEqual(witness["delta"], int(self.arrays[f"window{w:02d}_delta"][9, 1]))
            self.assertEqual(witness["code_after"], int(self.arrays[f"window{w:02d}_codes"][9, 1]))
            self.assertEqual(witness["residual_after"], int(self.arrays[f"window{w:02d}_residuals"][9, 1]))
        t0 = [row for row in scalar["query_class9"] if row["timestep"] == 0]
        self.assertEqual([row["current"] for row in t0], [1345, 1340])
        self.assertEqual([row["centered"] for row in t0], [-56, -61])

    def test_scalar_checker_rejects_corrupted_update_expectation(self):
        arrays = dict(self.arrays)
        arrays["window03_delta"] = arrays["window03_delta"].copy()
        arrays["window03_delta"][9, 1] += 1
        with self.assertRaises(AssertionError):
            ref.scalar_checks(arrays, self.schedule)

    def assert_bad_schedule(self, frame, key, value):
        schedule = copy.deepcopy(self.schedule)
        schedule[frame][key] = value
        with self.assertRaises(ValueError):
            ref.scheduled_reference(self.features, self.arrays["labels"], self.arrays["initial_codes"], schedule)

    def test_window_reset_is_rejected(self):
        self.assert_bad_schedule(16, "sample_state_reset", True)

    def test_completed_windows_must_not_advance_before_commit(self):
        self.assert_bad_schedule(15, "completed_windows", 1)

    def test_commit_requires_all_four_samples(self):
        self.assert_bad_schedule(3, "wait_for_commit_after", True)

    def test_duplicate_sample_is_rejected(self):
        self.assert_bad_schedule(4, "sample_slot", 0)

    def test_fixture_round_trip_and_stimulus_boundary(self):
        ref.BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ref.BASE, prefix="unit_") as tmp:
            output = Path(tmp) / "fixture"
            manifest = ref.write_fixture(self.result, output)
            self.assertEqual(manifest["status"], ref.STATUS)
            self.assertLess(manifest["fixture_hex_bytes"], 6 * 1024 * 1024)
            for row in manifest["files"]:
                with (output / row["path"]).open(encoding="ascii") as stream:
                    values = np.fromiter((int(v, 16) for v in stream), dtype=np.int64)
                if row["signed"]:
                    sign = 1 << (row["bits"]-1)
                    values = (values ^ sign) - sign
                name = Path(row["path"]).stem
                np.testing.assert_array_equal(values.reshape(row["shape"]), self.arrays[name])
                if row["role"] == "stimulus":
                    self.assertIn(name, ("train_raw", "query_raw", "labels", "initial_codes", "initial_residuals"))
            self.assertEqual(json.loads((output / "manifest.json").read_text())["commits"], self.manifest["commits"])
            with self.assertRaises(FileExistsError):
                ref.write_fixture(self.result, output)

    def test_cannot_overwrite_formal_report_or_existing_runner(self):
        for path in (ref.REPORT / "new", ROOT / "tests/new", ref.BASE):
            with self.assertRaises(ValueError):
                ref.write_fixture(self.result, path)
        self.assertEqual(self.manifest["resource"]["device"], "cpu")
        self.assertEqual(self.manifest["resource"]["cpu_threads"], 1)


if __name__ == "__main__":
    unittest.main()
