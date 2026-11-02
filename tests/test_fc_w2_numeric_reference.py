import json

import numpy as np
import unittest

from scripts.build_fc_w2_numeric_reference import (
    accumulate_logits,
    build_fixture,
    lif18,
    load_model,
    sample0_prefix,
    scale_raw_to_q6,
    signed_wrap,
    trace16,
    w2_dot,
)



class FCW2NumericReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_fixture()

    def test_lif_rounding_threshold_soft_reset_and_wrap(self):
        assert lif18(1023) == (1023, 0, 1023)
        assert lif18(1024) == (1024, 1, 0)
        assert lif18(2049) == (2049, 1, 1025)  # One spike, not repeated subtraction.
        assert lif18(0, -1, leak=16384) == (0, 0, 0)  # -0.5 rounds to zero.
        assert lif18(0, 1, leak=16384) == (1, 0, 1)  # +0.5 rounds to one.
        assert lif18(131072) == (-131072, 0, -131072)
        assert signed_wrap(-8388609, 24) == 8388607


    def test_original_trace_is_unsigned_integer_not_q016(self):
        assert trace16(0, 1) == 1
        assert trace16(1, 0) == 1
        assert trace16(65535, 1) == 64225
        assert trace16(0, 1, fractional_bits=10) == 1024
        assert trace16(1024, 0, fractional_bits=10) == 1004
        assert trace16(65535, 1, fractional_bits=10, decay=65535) == 65535


    def test_real_model_and_two_hex_exports_match(self):
        model = load_model()
        assert model["W2"].shape == (160, 10)
        assert int(model["W2"].min()) == -127
        assert int(model["W2"].max()) == 72
        assert int(np.count_nonzero((model["W2"] < -16) | (model["W2"] > 15))) == 990


    def test_real_w2_vectors_against_independent_numpy_dot(self):
        fixture = self.fixture
        weights = np.array(fixture["w2_codes_source_major"], dtype=np.int64)
        assert {0, 39, 40, 79, 80, 119, 120, 159}.issubset(
            {case["active_hidden"][0] for case in fixture["dot_cases"]
             if len(case["active_hidden"]) == 1})
        for case in fixture["dot_cases"]:
            spikes = np.zeros(160, dtype=np.int64)
            spikes[case["active_hidden"]] = 1
            assert case["dot_raw24"] == (spikes @ weights).tolist()
            assert int(case["spikes_hex_lsb_hidden0"], 16) == sum(
                1 << h for h in case["active_hidden"])
        for extreme in ("w2_min", "w2_max"):
            e = fixture["model"][extreme]
            case = next(c for c in fixture["dot_cases"] if c["name"] == f"onehot_{e['hidden']}")
            assert case["dot_raw24"][e["class"]] == e["code"]


    def test_temporal_logits_hold_without_leak_and_reset_per_sample(self):
        fixture = self.fixture
        sequence = fixture["temporal_cases"]
        assert sequence[1]["dot_raw24"] == [0] * 10
        assert sequence[1]["cumulative_raw24"] == sequence[0]["cumulative_raw24"]
        assert sequence[4]["cumulative_raw24"] == sequence[3]["cumulative_raw24"]
        assert sequence[5]["cumulative_raw24"] == [0] * 10
        assert sequence[6]["cumulative_raw24"] == sequence[0]["cumulative_raw24"]
        assert sequence[1]["shared_fc_if_hidden_parameters_reused"]["pre18"] != sequence[1]["cumulative_raw24"]
        assert sequence[7]["dot_raw24"][1] == 928
        assert sequence[8]["cumulative_raw24"][1] == 1856
        shared = sequence[8]["shared_fc_if_hidden_parameters_reused"]
        assert shared["spike"][1] == 1
        assert shared["post18"][1] == shared["pre18"][1] - 1024
        assert sequence[9]["cumulative_raw24"][1] == 1856
        assert sequence[10]["cumulative_raw24"] == [0] * 10
        assert accumulate_logits([8388607, -8388608], [1, -1]) == [-8388608, 8388607]
        assert accumulate_logits([123, 456], [-7, 9], sample_start=True) == [-7, 9]


    def test_prefix_w0_scaling_states_and_linear_w2(self):
        fixture = self.fixture
        w0 = load_model()["W0"].astype(np.int64)
        voltage = np.zeros(160, dtype=np.int64)
        total = np.zeros(10, dtype=np.int64)
        w2 = np.array(fixture["w2_codes_source_major"], dtype=np.int64)
        for step in fixture["sample0_prefix"]:
            current = (w0[step["input_features"]].sum(axis=0) * 64 + 131072) % 262144 - 131072
            pre = (((voltage * 32113 + 16384) // 32768 + current + 131072) % 262144) - 131072
            spikes = (pre >= 1024).astype(np.int64)
            voltage = (pre - spikes * 1024 + 131072) % 262144 - 131072
            total += spikes @ w2
            assert step["w0_current_shift6_wrap18"] == current.tolist()
            assert step["pre18"] == pre.tolist()
            assert step["post18"] == voltage.tolist()
            assert step["active_hidden"] == np.flatnonzero(spikes).tolist()
            assert step["cumulative_raw24"] == total.tolist()
        assert fixture["summary"]["prefix_hidden_spikes"] > 0
        assert fixture["summary"]["prefix_hidden_spikes"] != fixture["summary"]["prefix_unshifted_hidden_spikes"]


    def test_q6_scaler_ties_even_and_saturation(self):
        assert [scale_raw_to_q6(x, 32768) for x in (1, 3, -1, -3)] == [0, 2, 0, -2]
        assert scale_raw_to_q6(8388607, 65536) == 32767
        assert scale_raw_to_q6(-8388608, 65536) == -32768
        assert self.fixture["model"]["w2_scale_times64_q16_rne"] == 47303
        assert scale_raw_to_q6(-127, 47303) == -92
        assert (-127 >> 4) == -8  # Current replay scaling is not W2 dequantization.


    def test_fixture_is_reproducible_and_explicitly_not_rtl(self):
        fixture = self.fixture
        assert json.loads(json.dumps(fixture)) == build_fixture()
        assert not fixture["summary"]["final_top_simulated"]
        assert not fixture["summary"]["accuracy_measured"]
        assert not fixture["numeric"]["output_lif_in_reference"]


    def test_invalid_inputs_do_not_silently_change_semantics(self):
        with self.assertRaises(ValueError):
            w2_dot([[1]], [1024])
        with self.assertRaises(ValueError):
            w2_dot([[128]], [1])
        with self.assertRaises(ValueError):
            sample0_prefix(17)


if __name__ == "__main__":
    unittest.main()
