"""Small CPU tests: recurrence, signed arithmetic, counts and boundary export."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_srnn_backbone_numeric as ref


class BackboneTests(unittest.TestCase):
    def test_negative_leak_floor_and_inclusive_threshold(self):
        pre, membrane, spike = ref.lif_step(np.array([0, 32, 100]), np.array([-1, 0, 0]))
        np.testing.assert_array_equal(pre, [-1, 32, 100])
        np.testing.assert_array_equal(spike, [0, 1, 1])
        np.testing.assert_array_equal(membrane, [-1, 0, 68])

    def test_mode1_saturates_before_soft_reset(self):
        maximum = (1 << 31) - 1
        pre, membrane, spike = ref.lif_step(np.array([maximum, -maximum]), np.array([maximum, -maximum]))
        np.testing.assert_array_equal(pre, [maximum, -(1 << 31)])
        np.testing.assert_array_equal(membrane, [maximum - 32, -(1 << 31)])
        np.testing.assert_array_equal(spike, [1, 0])

    def test_recurrence_delayed_source_major_and_sample_isolation(self):
        inputs = np.array([[[1], [0]], [[0], [0]], [[0], [0]]])
        w0 = np.array([[32, 0]])
        recurrent = np.array([[0, 32], [0, 0]])
        result = ref.backbone_reference(inputs, w0, recurrent)
        np.testing.assert_array_equal(result["spike"][:, 0], [[1, 0], [0, 1], [0, 0]])
        np.testing.assert_array_equal(result["spike"][:, 1], 0)
        np.testing.assert_array_equal(result["recurrent_current"][0], 0)

    def test_input_multiplicity_is_not_binary(self):
        counts = np.array([[[2]]])
        full = ref.backbone_reference(counts, np.array([[20]]), np.zeros((1, 1), dtype=np.int64))
        binary = ref.backbone_reference(counts > 0, np.array([[20]]), np.zeros((1, 1), dtype=np.int64))
        self.assertEqual(full["spike"].item(), 1)
        self.assertEqual(binary["spike"].item(), 0)

    def test_p4_repeat_preserves_counts_and_bank_parity(self):
        counts = np.array([0, 3, 1, 2, 4, 1, 0, 2])
        weights = np.arange(24).reshape(8, 3) - 12
        groups = ref.p4_groups(counts, row_base=350)
        self.assertEqual(len(groups), 7)
        self.assertEqual(groups[0], (14, 350, 350, 351, 351))
        np.testing.assert_array_equal(ref.group_current(groups, weights, 350), counts @ weights)

    def test_p4_empty_frame_and_tail450(self):
        self.assertEqual(ref.p4_groups(np.zeros(700, dtype=int)), [(0, 0, 0, 0, 0)])
        counts = np.zeros(450, dtype=int)
        counts[449] = 1
        self.assertEqual(ref.p4_groups(counts, 350), [(2, 574, 574, 575, 575)])
        weights = np.arange(450).reshape(450, 1)
        np.testing.assert_array_equal(ref.group_current(ref.p4_groups(counts, 350), weights, 350), [449])

    def test_p4_rejects_invalid_counts(self):
        for counts in ([256], [-1], [0.5]):
            with self.assertRaises(ValueError):
                ref.p4_groups(np.asarray(counts))

    def test_refill_destination_major_address_is_not_model_file_order(self):
        source_major = np.array([[11, 12], [21, 22], [31, 32]])
        scalar = ref.refill_codes(source_major).ravel()
        np.testing.assert_array_equal(scalar, [11, 21, 31, 12, 22, 32])
        for destination in range(2):
            for source in range(3):
                self.assertEqual(scalar[destination * 3 + source], source_major[source, destination])

    def test_first_mismatch_and_no_comparison_across_shapes(self):
        left = np.zeros((2, 2, 3), dtype=np.int64)
        right = left.copy()
        right[1, 0, 2] = 4
        diff = ref.difference(left, right)
        self.assertEqual(diff["first_index"], [1, 0, 2])
        self.assertEqual(diff["per_timestep"], [0, 1])
        self.assertEqual((diff["left"], diff["right"]), (0, 4))
        with self.assertRaises(ValueError):
            ref.difference(left, np.zeros((2, 3)))

    def test_independent_scalar_loop(self):
        rng = np.random.default_rng(45)
        counts = rng.integers(0, 4, (4, 2, 7))
        w0 = rng.integers(-30, 30, (7, 5))
        rc = rng.integers(-30, 30, (5, 5))
        actual = ref.backbone_reference(counts, w0, rc)
        for b in range(2):
            state, previous = [0] * 5, [0] * 5
            for t in range(4):
                spikes = []
                for n in range(5):
                    current = sum(int(counts[t, b, i]) * int(w0[i, n]) for i in range(7))
                    current += sum(previous[i] * int(rc[i, n]) for i in range(5))
                    pre = state[n] * 27853 // 32768 + current
                    spike = int(pre >= 32)
                    state[n] = pre - spike * 32
                    spikes.append(spike)
                    self.assertEqual(actual["pre"][t, b, n], pre)
                    self.assertEqual(actual["membrane"][t, b, n], state[n])
                    self.assertEqual(actual["spike"][t, b, n], spike)
                previous = spikes


class FileTests(unittest.TestCase):
    def setUp(self):
        ref.EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit_", dir=ref.EVIDENCE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_stream_keeps_repeats_empty_frames_and_prefix(self):
        source = self.root / "sample.hex"
        source.write_text("00102\n00103\n40000\n40000\n00201\n80000\n", encoding="ascii")
        counts = ref.read_prefix(source, 3)
        self.assertEqual(counts[0, 1], 5)
        self.assertEqual(counts[1].sum(), 0)
        self.assertEqual(counts[2, 2], 1)
        with self.assertRaises(ValueError):
            ref.read_prefix(source, 4)

    def test_invalid_source_and_boundary_are_rejected(self):
        source = self.root / "sample.hex"
        for text in ("2bc01\n80000\n", "40001\n", "c0000\n", "00000\n"):
            source.write_text(text, encoding="ascii")
            with self.assertRaises(ValueError):
                ref.read_prefix(source, 1)

    def test_output_boundary_and_existing_files_preserved(self):
        with self.assertRaises(ValueError):
            ref.new_output(ROOT / "rtl/test_not_created")
        with self.assertRaises(FileExistsError):
            ref.new_output(self.root)
        with self.assertRaises(ValueError):
            ref.export(self.root / "too_long", 9)
        self.assertFalse((self.root / "too_long").exists())

    def test_signed_hex_roundtrip(self):
        path = self.root / "test.mem"
        ref.write_hex(path, [-32768, -1, 0, 32767], 16)
        np.testing.assert_array_equal(ref.read_signed(path, 16, (2, 2)), [[-32768, -1], [0, 32767]])

    def fixture_and_observed(self):
        shape = (1, 2, 480)
        values = {key: np.zeros(shape, dtype=np.int64) for key in ("current", "pre", "membrane", "spike")}
        np.savez(self.root / "backbone_reference.npz", **values, labels=np.array([0, 1]))
        return values

    def test_observed_mismatch_is_retained_while_exporting_actual_spikes(self):
        values = self.fixture_and_observed()
        values["spike"][0, 1, 3] = 1
        observed = self.root / "observed.npz"
        np.savez(observed, **values)
        with patch.object(ref, "m2_fixture", return_value=({}, {"reference_counters": {}})) as m2:
            result = ref.connect(self.root, observed, self.root / "connected")
        self.assertEqual(result["status"], "observed_backbone_mismatch_not_equivalent")
        self.assertEqual(m2.call_args.args[1][0, 1, 3], 1)
        self.assertEqual(result["differences"]["spike"]["count"], 1)

    def test_observed_padding_shape_and_type_are_not_silently_fixed(self):
        for kind in ("tail", "short", "float", "nonbinary"):
            values = self.fixture_and_observed()
            if kind == "tail":
                values["spike"][0, 0, 450] = 1
            elif kind == "short":
                values["spike"] = values["spike"][:, :, :449]
            elif kind == "float":
                values["pre"] = values["pre"].astype(float)
            else:
                values["spike"][0, 0, 0] = 16384
            observed = self.root / "observed.npz"
            np.savez(observed, **values)
            with self.assertRaises(ValueError):
                ref.connect(self.root, observed, self.root / "not_created")
            self.assertFalse((self.root / "not_created").exists())

    def test_active_only_csv_and_missing_record_detection(self):
        self.fixture_and_observed()
        observed = self.root / "observed.csv"
        with observed.open("w", encoding="ascii", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestep", "sample", "neuron", "current", "pre", "membrane", "spike"])
            for sample in range(2):
                for neuron in range(450):
                    writer.writerow([0, sample, neuron, 0, 0, 0, 0])
        with patch.object(ref, "m2_fixture", return_value=({}, {"reference_counters": {}})):
            result = ref.connect(self.root, observed, self.root / "csv_connected")
        self.assertEqual(result["source"]["observed_neurons"], 450)
        self.assertFalse(result["source"]["inactive_padding_checked"])
        self.assertEqual(result["status"], "observed_trace_matches_python_reference")
        lines = observed.read_text(encoding="ascii").splitlines()
        observed.write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")
        with self.assertRaises(ValueError):
            ref.load_observed(observed, 1)

    def test_public_pre_spike_dump_does_not_invent_internal_state(self):
        self.fixture_and_observed()
        observed = self.root / "public.csv"
        with observed.open("w", encoding="ascii", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestep", "sample", "neuron", "pre", "spike"])
            for sample in range(2):
                for neuron in range(450):
                    writer.writerow([0, sample, neuron, 0, 0])
        with patch.object(ref, "m2_fixture", return_value=({}, {"reference_counters": {}})):
            result = ref.connect(self.root, observed, self.root / "public_connected")
        self.assertEqual(set(result["differences"]), {"pre", "spike"})
        self.assertEqual(result["source"]["unobserved_fields"], ["current", "membrane"])
        self.assertEqual(result["status"], "observed_trace_matches_python_reference")


if __name__ == "__main__":
    unittest.main()
