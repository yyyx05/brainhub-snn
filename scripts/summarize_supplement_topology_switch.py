"""Extract passive B3 logs; missing simulation results stay NA, never zero."""

import argparse
import csv
import json
import re
from pathlib import Path


PHASES = {1: "FC_initial", 2: "SRNN", 3: "SCNN", 4: "FC_return"}
PAYLOAD_BITS = {
    "weight_r0": 200, "weight_r1": 200, "weight_r2": 200, "weight_r3": 200,
    "state_write": 320, "trace_write": 320, "feature_response": 160,
    "feature_update": 64, "feature_wave": 128, "logit_wave": 192,
    "fc_trace": 17, "fc_surrogate": 17, "srnn_backbone": 32,
}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_simulation(sim, exit_code):
    passed = "PASS: descriptor-driven FC" in sim and exit_code == 0
    status = "measured_pass" if passed else "NA_build_failed" if not sim else "incomplete_simulation"
    events, counts = [], []
    for line in sim.splitlines():
        if line.startswith("B3_EVENT,"):
            _, cycle, phase, name, value = next(csv.reader([line]))
            events.append(dict(cycle=int(cycle), phase=int(phase), event=name, value=int(value)))
        elif line.startswith("B3_COUNT,"):
            _, phase, channel, accepts, stalls = next(csv.reader([line]))
            counts.append(dict(phase=int(phase), channel=channel,
                               accepts=int(accepts), stall_cycles=int(stalls), status=status))
    if passed and not events:
        passed, status = False, "NA_measurement_missing"
        for row in counts:
            row["status"] = status
    return events, counts, passed, status


def timing_rows(events, passed, status):
    def at(phase, name, last=False, value=None, after=None):
        selected = [row["cycle"] for row in events if row["phase"] == phase
                    and row["event"] == name and (value is None or row["value"] == value)
                    and (after is None or row["cycle"] >= after)]
        return selected[-1 if last else 0] if selected else None

    timings = []

    def interval(phase, metric, start, end, boundary, inclusive=False):
        row_status = status
        if passed and (start is None or end is None):
            row_status = "NA_missing_event"
        elif passed and end < start:
            row_status = "NA_invalid_order"
        measured = passed and row_status == "measured_pass"
        timings.append(dict(phase=PHASES[phase], metric=metric,
                            start_cycle=start if start is not None else "NA",
                            end_cycle=end if end is not None else "NA",
                            cycles=end - start + int(inclusive) if measured else "NA",
                            status=row_status,
                            boundary=boundary))

    for phase in PHASES:
        scnn = phase == 3
        start_name = "scnn_start" if scnn else "dense_layer"
        done_name = "scnn_done" if scnn else "dense_done"
        config_name = start_name if scnn else "dense_config"
        owner = 2 if scnn else 1
        start = at(phase, start_name + "_accept")
        done = at(phase, done_name)
        array_done = at(phase, "array_done", last=True)
        owner_release = at(phase, "owner", value=0, after=array_done) if array_done is not None else None
        interval(phase, "phase_with_preload", at(phase, "phase_begin"), done,
                 "first maintenance request to externally visible done; includes TB gaps and preload")
        interval(phase, "execution", start, done,
                 "layer/start acceptance to externally visible done; includes SCNN in-owner refill")
        interval(phase, "maintenance_acquire_wait", at(phase, "maintenance_start_request"),
                 at(phase, "maintenance_start_accept"), "request first sampled to handshake; zero means no wait")
        interval(phase, "maintenance_lease", at(phase, "maintenance_start_accept"),
                 at(phase, "maintenance_release_accept"), "includes weights/state/trace writes and TB gaps")
        interval(phase, "configuration_wait", at(phase, config_name + "_request"),
                 at(phase, config_name + "_accept"), "zero means first sampled request was accepted")
        interval(phase, "configuration_to_owner", at(phase, config_name + "_accept"),
                 at(phase, "owner", value=owner), "control only; preloaded data excluded; registered owner sampled next edge")
        interval(phase, "start_to_owner", start, at(phase, "owner", value=owner),
                 "pre-NBA sampling adds one edge for registered owner visibility")
        interval(phase, "last_command_to_array_done", at(phase, "array_command_accept", last=True),
                 at(phase, "array_done", last=True), "last command execution tail; not configuration latency")
        interval(phase, "post_array_drain_to_owner_release", at(phase, "array_done", last=True),
                 owner_release, "observation/client/replay drain plus owner release; not pure switch control")
        interval(phase, "post_array_drain_to_done", at(phase, "array_done", last=True), done,
                 "final command done to externally visible phase done")
        maintenance_end = at(phase, "maintenance_release_accept")
        for name in ("weight_r0", "state_write", "trace_write"):
            accepted = [row["cycle"] for row in events if row["phase"] == phase
                        and row["event"] == name + "_accept" and maintenance_end is not None
                        and row["cycle"] <= maintenance_end]
            interval(phase, name + "_preload_span", accepted[0] if accepted else None,
                     accepted[-1] if accepted else None,
                     "inclusive first-to-last accepted beat; state/trace windows overlap; includes TB gaps", True)
        if phase > 1:
            previous_done = "scnn_done" if phase == 4 else "dense_done"
            interval(phase, "previous_done_to_start", at(phase - 1, previous_done), start,
                     "total observed interphase gap including maintenance/data/configuration and TB pacing")
        if scnn:
            interval(phase, "conv_context_service", at(phase, "context_load_request"),
                     at(phase, "context_load_accept"), "includes TB refill service; not pure control")
            interval(phase, "classifier_context_service", at(phase, "context_load_request", last=True),
                     at(phase, "context_load_accept", last=True), "classifier context acknowledgement; no weight refill")
            interval(phase, "graph_done_to_owner_release", at(phase, "scnn_client_done"),
                     owner_release, "owner release after graph client completes")
    return timings


def traffic_rows(events, counts, passed, status):
    rows = []
    for count in counts:
        phase, channel = count["phase"], count["channel"]
        if phase not in PHASES or channel not in PAYLOAD_BITS:
            continue
        bits = PAYLOAD_BITS[channel]
        accepted = [row["cycle"] for row in events if row["phase"] == phase
                    and row["event"] == channel + "_accept"]
        release = next((row["cycle"] for row in events if row["phase"] == phase
                        and row["event"] == "maintenance_release_accept"), None)
        for segment in ("maintenance", "runtime"):
            selected = [cycle for cycle in accepted if release is not None
                        and ((cycle <= release) == (segment == "maintenance"))]
            row_status = status
            if passed and release is None:
                row_status = "NA_missing_event"
            elif passed and len(accepted) != count["accepts"]:
                row_status = "NA_counter_mismatch"
            measured = passed and row_status == "measured_pass"
            beats = len(selected)
            rows.append(dict(phase=PHASES[phase], segment=segment, channel=channel,
                             accepted_beats=beats if measured else "NA", payload_bits_per_beat=bits,
                             payload_bits=beats * bits if measured else "NA",
                             payload_byte_equivalent=beats * bits / 8 if measured else "NA",
                             status=row_status,
                             scope="accepted fixed-width data payload only; excludes mask/address/control; boundaries must not be summed as unique memory traffic"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--simulation-log", default="simulation.log")
    parser.add_argument("--output-dir", type=Path,
                        help="Write CSVs here instead of the input result directory.")
    args = parser.parse_args(argv)
    result = args.result_dir
    sim_path = result / args.simulation_log
    sim = sim_path.read_text(encoding="utf-8-sig") if sim_path.exists() else ""
    console = (result / "console.log").read_text(encoding="utf-8-sig")
    resources = json.loads((result / "resources.json").read_text(encoding="utf-8"))
    events, counts, passed, status = parse_simulation(sim, resources["exit_code"])
    timings = timing_rows(events, passed, status)
    traffic = traffic_rows(events, counts, passed, status)
    output = args.output_dir if args.output_dir is not None else result
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "events.csv", ["cycle", "phase", "event", "value"], events)
    write_csv(output / "handshakes.csv",
              ["phase", "channel", "accepts", "stall_cycles", "status"], counts)
    write_csv(output / "timings.csv", ["phase", "metric", "start_cycle", "end_cycle", "cycles", "status", "boundary"], timings)
    write_csv(output / "traffic.csv", ["phase", "segment", "channel", "accepted_beats",
                                      "payload_bits_per_beat", "payload_bits", "payload_byte_equivalent",
                                      "status", "scope"], traffic)
    host = []
    for name in ("wall_seconds", "peak_sampled_tree_rss_bytes", "peak_sampled_tree_private_bytes",
                 "minimum_host_available_bytes", "peak_process_count", "samples", "exit_code"):
        host.append(dict(metric=name, value=resources[name], scope="host process-tree profile; not RTL latency"))
    for line in console.splitlines():
        if line.startswith(("B3_BUILD_END", "B3_SIM_END")):
            match = re.search(r"seconds=([0-9.]+)", line)
            if match:
                host.append(dict(metric=line.split()[0].lower() + "_wall_seconds",
                                 value=match.group(1), scope="host stopwatch; not RTL latency"))
    write_csv(output / "host_metrics.csv", ["metric", "value", "scope"], host)
    print(json.dumps(dict(status=status, event_rows=len(events), handshake_rows=len(counts),
                          timing_rows=len(timings), traffic_rows=len(traffic)), indent=2))


if __name__ == "__main__":
    main()
