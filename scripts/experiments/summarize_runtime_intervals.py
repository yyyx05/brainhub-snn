"""Reanalyse six archived profiles only; no simulator or producer imports.

All timestamp algebra uses half-open [start, end) elapsed-cycle intervals.
Counts without endpoints never become fabricated intervals.
"""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CURRENT = Path("results/paper_evidence/supplement_20261004/performance/current_source_measurements")
OUTPUT = Path("results/paper_evidence/supplement_20261012/performance")
STAGES = ("outside", "parameter_load", "initial_refill", "initial_state",
          "runtime", "learning_tail_wait", "runtime_refill", "learning_configuration")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def union(intervals):
    """Merge overlap/adjacency, reject reversed endpoints, discard empty spans."""
    result = []
    for start, end in sorted(intervals):
        require(start >= 0 and end >= start, "invalid interval")
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = union(left), union(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def length(intervals):
    return sum(end - start for start, end in union(intervals))


def partition(bounds, groups):
    """Sweep endpoints into disjoint exact membership sets, including gaps."""
    require(len(union([bounds])) == 1, "empty runtime bounds")
    start, end = bounds
    events = defaultdict(Counter)
    events[start]
    events[end]
    for name, spans in groups.items():
        for a, b in intersection(spans, [bounds]):
            events[a][name] += 1
            events[b][name] -= 1
    active = Counter()
    previous = start
    result = []
    for point in sorted(events):
        if point > previous:
            labels = tuple(sorted(name for name, count in active.items() if count))
            if result and result[-1][2] == labels:
                result[-1] = (result[-1][0], point, labels)
            else:
                result.append((previous, point, labels))
        active.update(events[point])
        previous = point
    return result


def coverage(bounds, groups):
    clipped = {name: intersection(spans, [bounds]) for name, spans in groups.items()}
    segments = partition(bounds, clipped)
    totals = Counter()
    for a, b, labels in segments:
        totals["&".join(labels) if labels else "unclassified"] += b - a
    merged = union([span for spans in clipped.values() for span in spans])
    return {
        "bounds": list(bounds), "runtime_cycles": bounds[1] - bounds[0],
        "union_cycles": length(merged), "union_interval_count": len(merged),
        "unclassified_cycles": totals.get("unclassified", 0),
        "non_overlapping_phase_cycles": dict(sorted(totals.items())),
        "phase_union_cycles": {name: length(spans) for name, spans in clipped.items()},
        "pairwise_intersection_cycles": {
            a + "&" + b: length(intersection(clipped[a], clipped[b]))
            for a, b in combinations(sorted(clipped), 2)},
    }, segments


def records(path, prefixes):
    found = defaultdict(list)
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            marker = line.split(" ", 1)[0].strip()
            if marker not in prefixes:
                continue
            row = {k: int(v) if re.fullmatch(r"-?\d+", v) else v
                   for k, v in re.findall(r"(\w+)=([^\s]+)", line)}
            found[marker].append(row)
    return found


def one(rows, marker):
    require(len(rows[marker]) == 1, "missing/duplicate " + marker)
    return rows[marker][0]


def monitor(rows):
    tiles = rows["H40_ACTIVITY"]
    require(len(tiles) == 12, "expected twelve H40 monitor records")
    counts = {r["cycles"] for r in tiles}
    require(len(counts) == 1, "different monitor denominators")
    return {"cycles_per_tile": counts.pop(), "tiles": 12,
            "scope": "reset-released monitor clocks; includes setup; not runtime-only",
            "absolute_bounds": None}


def fc(path):
    rows = records(path, {"FC_FINAL_COUNTS_PASS", "FC_PROFILE_STAGE", "FC_PROFILE_WINDOW",
                          "FC_PROFILE_MEASUREMENT_PASS", "H40_ACTIVITY"})
    terminal = one(rows, "FC_FINAL_COUNTS_PASS")
    one(rows, "FC_PROFILE_MEASUREMENT_PASS")
    require((terminal["batch"], terminal["batches"], terminal["steps"]) == (16, 1, 99), "FC scope")
    stages = rows["FC_PROFILE_STAGE"]
    require(len(stages) == 8 and {r["id"] for r in stages} == set(range(8)), "FC stages")
    by_id = {r["id"]: r for r in stages}
    phase_counts = {STAGES[i]: by_id[i]["cycles"] for i in (4, 5, 6)}
    require(sum(phase_counts.values()) == terminal["cycles"], "FC runtime count mismatch")
    for row in stages:
        require(row["work_busy"] == row["work_overlap_forward"] + row["work_only"] + row["work_other"],
                "FC overlapping work counts")
    windows = rows["FC_PROFILE_WINDOW"]
    require([w["index"] for w in windows] == list(range(99 if terminal["learning"] else 0)), "FC windows")
    # These are latency envelopes, not a continuous learning-busy waveform.
    envelopes = [(w["observation"], w["commit"]) for w in windows]
    return {
        "runtime_cycles": terminal["cycles"], "runtime_bounds": None,
        "setup_cycles": sum(by_id[i]["cycles"] for i in (1, 2, 3, 7)),
        "setup_scope": "counted load/refill/state/config only; reset/outside not inferred",
        "setup_phase_cycles": {STAGES[i]: by_id[i]["cycles"] for i in (1, 2, 3, 7)},
        "monitor": monitor(rows), "views": {"stage_counts": {
            "basis": "exclusive TB stage counters; timestamp partition unavailable",
            "runtime_cycles": terminal["cycles"], "union_cycles": None,
            "covered_count_cycles": sum(phase_counts.values()), "unclassified_cycles": 0,
            "non_overlapping_phase_cycles": phase_counts}},
        "learning_counts_not_additive": {
            key: sum(by_id[i][key] for i in (4, 5, 6))
            for key in ("work_busy", "work_overlap_forward", "work_only", "work_other", "window_busy")},
        "observation_to_commit": {"bounds_semantics": "[observation,commit) elapsed latency, not busy",
                                  "union_intervals": union(envelopes), "union_cycles": length(envelopes)},
        "missing": ["absolute runtime/stage endpoints", "per-cycle learning work timestamps",
                    "host golden/file-load seconds", "packaged FC monitor include"],
    }, {}


def srnn(path):
    rows = records(path, {"SRNN_PROFILE_INTERVAL", "SRNN_PROFILE_COMPLETE", "SRNN_PROFILE_COUNT", "H40_ACTIVITY"})
    terminal = one(rows, "SRNN_PROFILE_COMPLETE")
    require((terminal["batch"], terminal["steps"]) == (1, 73), "SRNN scope")
    groups = {"stage": defaultdict(list), "hardware": defaultdict(list)}
    for r in rows["SRNN_PROFILE_INTERVAL"]:
        require(r["cycles"] == r["end"] - r["start"] >= 0, "SRNN interval duration")
        groups[r["kind"]][r["phase"]].append((r["start"], r["end"]))
    for name in ("backbone_refill_ff", "backbone_refill_r", "backbone", "m2", "tb_store_check"):
        subset = [r for r in rows["SRNN_PROFILE_INTERVAL"] if r["kind"] == "stage" and r["phase"] == name]
        require([r["timestep"] for r in subset] == list(range(73)), "SRNN incomplete " + name)
    require(length(groups["stage"]["tb_store_check"]) == 0, "nonzero checks need disjoint service bounds")
    bounds = (terminal["initialized"], terminal["end"])
    detail = {k: v for k, v in groups["stage"].items() if k not in ("m2", "tb_store_check")}
    detail.update(groups["hardware"])
    views, parts = {}, {}
    for name, spans in (("stages", groups["stage"]), ("hardware_detail", detail)):
        views[name], parts[name] = coverage(bounds, spans)
    counts = {r["name"]: r["value"] for r in rows["SRNN_PROFILE_COUNT"]}
    return {
        "runtime_cycles": bounds[1] - bounds[0], "runtime_bounds": list(bounds),
        "setup_cycles": terminal["initialized"], "setup_scope": "[0,initialized), includes reset",
        "setup_phase_cycles": {k: length(intersection(v, [(0, bounds[0])]))
                               for k, v in groups["stage"].items() if k in ("reset", "config", "parameter_load", "initial_state")},
        "monitor": monitor(rows), "views": views,
        "learning_counts_not_additive": {k: v for k, v in counts.items() if "busy_cycles" in k},
        "tb_check_simulation_cycles": 0,
        "missing": ["host golden/file-load seconds", "packaged SRNN monitor include",
                    "cause of hardware-detail gaps; edge overlap is not proof of simultaneous arithmetic"],
    }, parts


def scnn(path):
    rows = records(path, {"SCNN_PROFILE_TIMING", "SCNN_PROFILE_ACTIVE", "SCNN_PROFILE_PASS",
                          "SCNN_PROFILE_DESCRIPTOR", "SCNN_PROFILE_CONTROL", "SCNN_PROFILE_INTERVAL", "H40_ACTIVITY"})
    timing, active, terminal = (one(rows, marker) for marker in
                               ("SCNN_PROFILE_TIMING", "SCNN_PROFILE_ACTIVE", "SCNN_PROFILE_PASS"))
    require((terminal["samples"], terminal["frames"], terminal["descriptors"]) == (1, 20, 160), "SCNN scope")
    bounds = (timing["sample_start"], timing["sample_end"])
    require(bounds[1] - bounds[0] == timing["sample_cycles"], "SCNN runtime bounds")
    require(bounds[0] - timing["reset"] == timing["setup_cycles"], "SCNN setup bounds")
    descriptors = rows["SCNN_PROFILE_DESCRIPTOR"]
    require(len(descriptors) == 320, "SCNN descriptor count")
    layers = defaultdict(list)
    for index in range(160):
        a, b = descriptors[2 * index:2 * index + 2]
        frame, layer = divmod(index, 8)
        require((a["edge"], b["edge"]) == ("start", "end") and
                (a["frame"], a["layer"]) == (b["frame"], b["layer"]) == (frame, layer + 1),
                "SCNN unpaired descriptor")
        layers[f"descriptor_layer_{layer + 1}"].append((a["cycle"], b["cycle"]))
    service = {"descriptor": [s for spans in layers.values() for s in spans], "learning": [], "publication": []}
    pending = []
    for r in rows["SCNN_PROFILE_INTERVAL"]:
        span = (r["start"], r["end"])
        union([span])
        if r["kind"] == "commit_pending_to_pulse":
            pending.append(span)
        elif r["kind"] in service:
            service[r["kind"]].append(span)
    views, parts = {}, {}
    views["service"], parts["service"] = coverage(bounds, service)
    v = views["service"]
    for phase, key in (("descriptor", "forward"), ("learning", "learning"), ("publication", "publication")):
        require(v["phase_union_cycles"][phase] == active[key], "SCNN busy/interval mismatch: " + phase)
    for pair, key in (("descriptor&learning", "forward_learning_overlap"),
                      ("learning&publication", "learning_publication_overlap")):
        require(v["pairwise_intersection_cycles"][pair] == active[key], "SCNN overlap mismatch")
    detail = dict(layers)
    detail.update({"learning": service["learning"], "publication": service["publication"],
                   "commit_pending_to_pulse": pending})
    control = rows["SCNN_PROFILE_CONTROL"]
    require([r["frame"] for r in control] == list(range(20)), "SCNN control frames")
    for phase in ("conv8", "pool", "classifier"):
        detail[phase + "_envelope"] = [(r[phase + "_first"], r[phase + "_end"]) for r in control]
    views["control_envelopes"], parts["control_envelopes"] = coverage(bounds, detail)
    return {
        "runtime_cycles": timing["sample_cycles"], "runtime_bounds": list(bounds),
        "setup_cycles": timing["setup_cycles"], "setup_scope": "reset-release to first accepted request",
        "monitor": monitor(rows), "views": views, "active_counts_not_additive": active,
        "tb_check_simulation_cycles": timing["golden_cycles"],
        "missing": ["host golden/file-load seconds", "busy run endpoints inside pool/Conv8 envelopes",
                    "state-service/refill/stall timestamp attribution for unclassified cycles"],
    }, parts


def collect(root=ROOT):
    profiles, segments = [], []
    for mode, parser, template in (
        ("FC", fc, "fc/server_pair_r1/{arm}.log"),
        ("SRNN", srnn, "srnn/server_pair_r1/{arm}/run.log"),
        ("SCNN", scnn, "scnn/server_pair_r2/run_{arm}.log"),
    ):
        for arm in ("off", "on"):
            source = CURRENT / template.format(arm=arm)
            profile, parts = parser(root / source)
            profile.update(mode=mode, arm=arm, source=source.as_posix())
            profiles.append(profile)
            for view, spans in parts.items():
                segments.extend((mode, arm, view, a, b, b - a, "&".join(labels) if labels else "unclassified")
                                for a, b, labels in spans)
    return {"scope": "existing current-profile logs only; not new simulation or board timing",
            "interval_convention": "[start,end); end-start elapsed cycles; no inferred missing endpoints",
            "profiles": profiles}, segments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result, segments = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime_intervals.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "runtime_intervals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "arm", "view", "start_inclusive", "end_exclusive", "cycles", "membership"))
        writer.writerows(segments)
    with (output / "runtime_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "arm", "view", "runtime_cycles", "covered_cycles", "unclassified_cycles", "membership", "cycles"))
        for profile in result["profiles"]:
            for name, view in profile["views"].items():
                covered = view.get("covered_count_cycles", view["union_cycles"])
                for label, count in view["non_overlapping_phase_cycles"].items():
                    writer.writerow((profile["mode"], profile["arm"], name, profile["runtime_cycles"],
                                     covered, view["unclassified_cycles"], label, count))
    for p in result["profiles"]:
        print(p["mode"], p["arm"], p["runtime_cycles"],
              {name: v["unclassified_cycles"] for name, v in p["views"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
