"""Describe accepted E5/E10 CSV evidence without rerunning or auditing experiments."""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization"
SOURCES = {
    "srnn": RESULT / "srnn_representative_a_r1/sequence_measurements.csv",
    "scnn_frames": RESULT / "scnn_representative_a_r1/frame_measurements.csv",
    "scnn_layers": RESULT / "scnn_representative_a_r1/layer_measurements.csv",
    "scan": RESULT / "spiker_scan_rtl_r1/frame_measurements.csv",
    "scan_index": ROOT / "verification/experiments/nearest_neighbor/scan_fixture_r1/frames.csv",
}


def rows(name):
    with SOURCES[name].open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sum_field(records, field):
    return sum(int(record[field]) for record in records)


def comparison(scan, local):
    return {
        "scan_cycles": scan,
        "local_cycles": local,
        "saved_cycles": scan - local,
        "saved_percent": 100 * (scan - local) / scan,
        "interval_ratio": scan / local,
    }


def summarize():
    srnn = {}
    for arm in ("off", "on"):
        selected = [r for r in rows("srnn") if r["arm"] == arm]
        total = sum(
            sum_field(selected, field)
            for field in ("service_cycles", "state_setup_cycles", "initialization_cycles")
        )
        warm = [r for r in selected if not int(r["cold_weight_context"])]
        warm_total = sum_field(warm, "service_cycles") + sum_field(warm, "state_setup_cycles")
        srnn[arm] = {
            "indices": [int(r["dataset_index"]) for r in selected],
            "native_steps": sum_field(selected, "native_steps"),
            "service_cycles": sum_field(selected, "service_cycles"),
            "initialization_cycles": sum_field(selected, "initialization_cycles"),
            "state_setup_cycles": sum_field(selected, "state_setup_cycles"),
            "total_cycles": total,
            "projected_sequences_per_second_15ns": len(selected) / (total * 15e-9),
            "warm_sequences": len(warm),
            "warm_cycles_including_state_setup": warm_total,
            "warm_projected_sequences_per_second_15ns": len(warm) / (warm_total * 15e-9),
        }

    frames = [{key: int(value) for key, value in r.items()} for r in rows("scnn_frames")]
    layers = rows("scnn_layers")
    layer_fields = [
        "cycles", "idle", "context_load", "feature_request", "feature_response",
        "command_request", "command_wait", "context_release", "classifier",
        "reference_spikes", "reference_scalars",
    ]
    per_sample = []
    for frame in frames:
        selected = [r for r in layers if int(r["sample"]) == frame["sample"]]
        totals = {field: sum_field(selected, field) for field in layer_fields}
        per_sample.append({
            "sample": frame["sample"],
            "layer_totals": totals,
            "outside_layer_cycles": frame["frame_cycles"] - totals["cycles"],
            "conv8_pooled_nonzero": int(next(r for r in selected if r["layer"] == "8")["pooled_nonzero"]),
        })
    service = sum_field(frames, "frame_cycles")
    cold = sum_field(frames, "initialize_start_to_service_end_cycles")
    context = sum_field(layers, "context_load")
    classifier = sum_field(layers, "classifier")
    cycle_range = max(r["frame_cycles"] for r in frames) - min(r["frame_cycles"] for r in frames)
    scnn = {
        "frames": frames,
        "per_sample": per_sample,
        "service_cycles": service,
        "cold_cycles": cold,
        "cold_projected_frames_per_second_15ns": len(frames) / (cold * 15e-9),
        "frame_cycle_range": cycle_range,
        "range_over_min_percent": 100 * cycle_range / min(r["frame_cycles"] for r in frames),
        "service_bucket_percent": {
            field: 100 * sum_field(layers, field) / service
            for field in ("context_load", "command_wait", "classifier")
        },
        "arithmetic_only_not_measured_speedup": {
            "halve_context_load_other_cycles_fixed": service / (service - context / 2),
            "remove_classifier_other_cycles_fixed": service / (service - classifier),
        },
    }

    scan_index = rows("scan_index")
    scan_rows = [{key: int(value) for key, value in r.items()} for r in rows("scan")]
    indices = list(dict.fromkeys(int(r["dataset_index"]) for r in scan_index))
    e10 = {}
    for scenario in (0, 1):
        selected = [r for r in scan_rows if r["scenario"] == scenario and r["frame"] < len(scan_index)]
        lookup = {(r["frame"], r["sparse"]): r for r in selected}
        by_sequence = defaultdict(lambda: [0, 0])
        faster = equal = slower = 0
        for frame, source in enumerate(scan_index):
            scan = lookup[(frame, 0)]["total"]
            local = lookup[(frame, 1)]["total"]
            faster += local < scan
            equal += local == scan
            slower += local > scan
            totals = by_sequence[int(source["dataset_index"])]
            totals[0] += scan
            totals[1] += local
        e10[str(scenario)] = {
            "comparison": comparison(
                sum(r["total"] for r in selected if r["sparse"] == 0),
                sum(r["total"] for r in selected if r["sparse"] == 1),
            ),
            "real_frames": len(scan_index),
            "local_faster_equal_slower": [faster, equal, slower],
            "sequences": [
                {"dataset_index": index, **comparison(*by_sequence[index])} for index in indices
            ],
            "boundaries": [
                {"frame": frame, **comparison(
                    next(r["total"] for r in scan_rows if (r["scenario"], r["sparse"], r["frame"]) == (scenario, 0, frame)),
                    next(r["total"] for r in scan_rows if (r["scenario"], r["sparse"], r["frame"]) == (scenario, 1, frame)),
                )}
                for frame in sorted({r["frame"] for r in scan_rows if r["frame"] >= len(scan_index)})
            ],
        }
    no_stall = [r for r in scan_rows if r["scenario"] == 0 and r["sparse"] == 0 and r["frame"] < len(scan_index)]
    fixed = sum_field(no_stall, "ingress") + sum_field(no_stall, "scans") + sum_field(no_stall, "delimiters") + 2 * len(no_stall)
    return {
        "scope": "Local descriptive aggregation of accepted CSVs; no new RTL, terminal audit, or physical measurement.",
        "sources": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in SOURCES.items()},
        "srnn": srnn,
        "scnn": scnn,
        "e10": e10,
        "e10_no_stall_common_input_scan_tail_drain_cycles": fixed,
        "e10_no_stall_common_percent_of_scan": 100 * fixed / e10["0"]["comparison"]["scan_cycles"],
        "e10_no_stall_common_percent_of_local": 100 * fixed / e10["0"]["comparison"]["local_cycles"],
    }


if __name__ == "__main__":
    output = RESULT / "e5_e10_analysis/derived_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT))
