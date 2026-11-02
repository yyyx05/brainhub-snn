"""Validate counter accounting without launching another RTL simulation."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_scnn_fullgraph_measurements import summarize
from summarize_srnn_m2_online_cycles import main as summarize_legacy_srnn


def fullgraph_log():
    lines = ["SUPPLEMENT_FULLGRAPH_INITIALIZATION cycles=9 weight_fill_words=8 state_write_words=2 trace_write_words=2"]
    for step in range(2):
        for layer in range(1, 9):
            lines.append(
                f"SUPPLEMENT_FULLGRAPH_LAYER timestep={step} layer={layer} cycles=10 "
                f"requests={layer} responses={layer} response_mask_bits=2 "
                "feature_update_words=3 weight_fill_words=4 state_read_words=5 "
                "trace_read_words=6 state_write_words=7 trace_write_words=8"
            )
        cumulative = "+".join(str(layer * (step + 1)) for layer in range(1, 9))
        lines.append(f"SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_TIMESTEP_PASS requests={cumulative}")
        lines.append(f"SUPPLEMENT_FULLGRAPH_TIMESTEP timestep={step} cycles=83 classifier_logits={step+1}")
    lines.append("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS")
    return "\n".join(lines)


class FullgraphAccountingTests(unittest.TestCase):
    def test_all_eight_layers_and_nonoverlapping_window(self):
        result = summarize(fullgraph_log())
        self.assertEqual(result["total_conv_feature_requests"], 72)
        self.assertEqual(result["forward_cycles_excluding_initialization"], 166)
        self.assertEqual(result["steps"][0]["driver_gap_cycles"], 3)
        self.assertEqual(result["layers"][7]["stage"], "Conv8+pool+classifier")
        self.assertEqual(result["layers"][0]["state_spill_restore_port_bytes"], 480)

    def test_missing_conv8_rejected(self):
        text = "\n".join(line for line in fullgraph_log().splitlines()
                         if "timestep=0 layer=8 " not in line)
        with self.assertRaisesRegex(ValueError, "eight-layer"):
            summarize(text)

    def test_wrong_handshake_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "reconcile"):
            summarize(fullgraph_log().replace("requests=3 responses=3", "requests=3 responses=2", 1))

    def test_cumulative_counter_not_treated_as_step_count(self):
        text = fullgraph_log().replace("requests=2+4+6+8+10+12+14+16", "requests=1+2+3+4+5+6+7+8")
        with self.assertRaisesRegex(ValueError, "reconcile"):
            summarize(text)

    def test_functional_failure_overrides_metrics(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            summarize(fullgraph_log() + "\n%Error: reference mismatch")

    def test_incomplete_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            summarize(fullgraph_log().replace("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS", ""))


class HistoricalScopeTests(unittest.TestCase):
    def test_actual_legacy_logs_do_not_certify_primary_model_or_mask(self):
        # Parse old logs without running the DUT or replacing the old report.
        with tempfile.TemporaryDirectory(dir=ROOT / "work/supplement_20261004") as directory:
            with patch.object(sys, "argv", ["summary", "--output-dir", directory]), contextlib.redirect_stdout(io.StringIO()):
                summarize_legacy_srnn()
            result = json.loads((Path(directory) / "summary.json").read_text())
        self.assertEqual(result["m2_shape"], [450, 2, 2])
        self.assertIsNone(result["selected_mask"])
        self.assertFalse(result["final_v2_performance"])
        self.assertEqual(result["train_cycles"], 58368)
        self.assertEqual(result["learning_off_cycles"], 19515)
        self.assertEqual(result["changed_scan_events"], 8)
        self.assertIsNone(result["distinct_changed_addresses"])
        self.assertIn("independent initial-state learning-off control", result["not_evidence_for"])

    def test_historical_output_cannot_be_overwritten_by_erratum(self):
        original = ROOT / "results/paper_evidence/rtl/srnn_m2_current_cycles_20260928"
        with patch.object(sys, "argv", ["summary", "--output-dir", str(original)]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                summarize_legacy_srnn()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
