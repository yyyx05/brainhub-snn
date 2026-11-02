"""Focused B3 parser tests; synthetic fixtures are never paper evidence."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.summarize_supplement_topology_switch import (
    main, parse_simulation, timing_rows, traffic_rows,
)


def event(phase, name, cycle, value=1):
    return dict(phase=phase, event=name, cycle=cycle, value=value)


def metric(events, phase, name, passed=True, status="measured_pass"):
    return next(row for row in timing_rows(events, passed, status)
                if row["phase"] == phase and row["metric"] == name)


class SummaryTests(unittest.TestCase):
    def test_build_failure_does_not_become_zero(self):
        events, counts, passed, status = parse_simulation("", 2)
        self.assertEqual((events, counts, passed, status), ([], [], False, "NA_build_failed"))
        self.assertTrue(all(row["cycles"] == "NA" for row in timing_rows(events, passed, status)))

    def test_pass_without_monitor_is_not_measured(self):
        self.assertEqual(parse_simulation("PASS: descriptor-driven FC", 0)[2:],
                         (False, "NA_measurement_missing"))

    def test_failed_simulation_does_not_publish_pass_cycles(self):
        sim = "B3_EVENT,4,1,dense_layer_accept,1\nB3_EVENT,9,1,dense_done,1\nPASS: descriptor-driven FC"
        events, _, passed, status = parse_simulation(sim, 1)
        self.assertEqual(metric(events, "FC_initial", "execution", passed, status)["cycles"], "NA")

    def test_zero_wait_and_one_beat_span_have_different_conventions(self):
        events = [event(1, "dense_config_request", 10), event(1, "dense_config_accept", 10),
                  event(1, "weight_r0_accept", 5), event(1, "maintenance_release_accept", 8)]
        self.assertEqual(metric(events, "FC_initial", "configuration_wait")["cycles"], 0)
        self.assertEqual(metric(events, "FC_initial", "weight_r0_preload_span")["cycles"], 1)

    def test_missing_and_reversed_endpoints_remain_na(self):
        events = [event(1, "dense_layer_accept", 10), event(1, "dense_done", 9)]
        self.assertEqual(metric(events, "FC_initial", "execution")["status"], "NA_invalid_order")
        self.assertEqual(metric(events, "FC_initial", "configuration_to_owner")["status"], "NA_missing_event")

    def test_maintenance_release_is_not_computation_release(self):
        events = [event(1, "owner", 8, 0), event(1, "array_done", 20)]
        self.assertEqual(metric(events, "FC_initial", "post_array_drain_to_owner_release")["cycles"], "NA")
        events.append(event(1, "owner", 30, 0))
        self.assertEqual(metric(events, "FC_initial", "post_array_drain_to_owner_release")["cycles"], 10)

    def test_return_fc_uses_scnn_done(self):
        events = [event(3, "dense_done", 40), event(3, "scnn_done", 50), event(4, "dense_layer_accept", 70)]
        self.assertEqual(metric(events, "FC_return", "previous_done_to_start")["cycles"], 20)

    def test_scnn_contexts_and_refill_are_separate(self):
        events = [event(3, "weight_r0_accept", 2), event(3, "weight_r0_accept", 4),
                  event(3, "maintenance_release_accept", 6), event(3, "weight_r0_accept", 12),
                  event(3, "context_load_request", 10), event(3, "context_load_accept", 14),
                  event(3, "context_load_request", 30), event(3, "context_load_accept", 31)]
        self.assertEqual(metric(events, "SCNN", "weight_r0_preload_span")["cycles"], 3)
        self.assertEqual(metric(events, "SCNN", "conv_context_service")["cycles"], 4)
        self.assertEqual(metric(events, "SCNN", "classifier_context_service")["cycles"], 1)
        counts = [dict(phase=3, channel="weight_r0", accepts=3)]
        traffic = traffic_rows(events, counts, True, "measured_pass")
        self.assertEqual([row["accepted_beats"] for row in traffic], [2, 1])
        self.assertEqual([row["payload_byte_equivalent"] for row in traffic], [50, 25])

    def test_counter_event_disagreement_is_not_traffic_evidence(self):
        events = [event(1, "weight_r0_accept", 5), event(1, "maintenance_release_accept", 8)]
        traffic = traffic_rows(events, [dict(phase=1, channel="weight_r0", accepts=2)], True, "measured_pass")
        self.assertTrue(all(row["status"] == "NA_counter_mismatch" for row in traffic))
        self.assertTrue(all(row["accepted_beats"] == "NA" for row in traffic))


    def test_cli_separate_output_preserves_inputs(self):
        work = Path(__file__).resolve().parents[1] / "work" / "supplement_20261004" / "topology_switch"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parser_test_", dir=work) as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            output = root / "output" / "nested"
            resources = dict(
                exit_code=0, wall_seconds=0.01, peak_sampled_tree_rss_bytes=1,
                peak_sampled_tree_private_bytes=1, minimum_host_available_bytes=1,
                peak_process_count=1, samples=1,
            )
            originals = {
                "console.log": (
                    "B3_EVENT,1,1,maintenance_release_accept,1\n"
                    "B3_EVENT,2,1,weight_r0_accept,1\n"
                    "B3_COUNT,1,weight_r0,1,0\n"
                    "PASS: descriptor-driven FC\n"
                ),
                "resources.json": json.dumps(resources),
                "timings.csv": "previous result must stay unchanged\n",
            }
            for name, content in originals.items():
                (source / name).write_text(content, encoding="utf-8")
            with redirect_stdout(io.StringIO()) as stdout:
                main([str(source), "--simulation-log", "console.log",
                      "--output-dir", str(output)])
            self.assertEqual(json.loads(stdout.getvalue())["handshake_rows"], 1)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"events.csv", "handshakes.csv", "timings.csv", "traffic.csv", "host_metrics.csv"},
            )
            self.assertEqual({path.name for path in source.iterdir()}, set(originals))
            for name, content in originals.items():
                self.assertEqual((source / name).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
