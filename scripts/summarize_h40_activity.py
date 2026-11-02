#!/usr/bin/env python3
"""Summarize actual H40 handshakes, without treating busy time as utilization."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("abort", "compute", "issue_wait", "region_wait", "observation_wait", "fill", "internal", "idle")


def parse_activity(text, pass_marker, expected_tiles=12):
    if not pass_marker or pass_marker not in text or re.search(r"%Error|Assertion failed|\$fatal|Aborting", text):
        raise ValueError("functional completion is missing or failed")
    tiles, modes = {}, {}
    for line in text.splitlines():
        if not line.startswith(("H40_ACTIVITY ", "H40_ACTIVITY_MODE ")):
            continue
        row = dict(re.findall(r"(\w+)=([^\s]+)", line))
        tile = row.pop("tile")
        values = {k: int(v) for k, v in row.items()}
        if any(v < 0 for v in values.values()):
            raise ValueError("negative activity counter")
        if line.startswith("H40_ACTIVITY "):
            if tile in tiles:
                raise ValueError("duplicate tile")
            tiles[tile] = values
        else:
            key = (tile, values.pop("mode"))
            if key in modes or key[1] not in range(4):
                raise ValueError("duplicate or invalid mode")
            modes[key] = values
    if len(tiles) != expected_tiles or len(modes) != 4*expected_tiles:
        raise ValueError("incomplete physical array or per-mode rows")
    clocks = {row["cycles"] for row in tiles.values()}
    if len(clocks) != 1 or not next(iter(clocks)):
        raise ValueError("tile measurement windows differ or are empty")
    for tile, row in tiles.items():
        if sum(row[k] for k in BUCKETS) != row["cycles"]:
            raise ValueError("activity buckets do not partition the measured window")
        if row["contributions"] > row["cycles"] or row["lane_fires"] > row["contributions"]*40:
            raise ValueError("destination lane accounting exceeds capacity")
        if row["enabled_terms"] > row["lane_fires"]*4:
            raise ValueError("source-term accounting exceeds P4 capacity")
        for key in ("contributions", "lane_fires", "enabled_terms"):
            if sum(modes[(tile, m)][key] for m in range(4)) != row[key]:
                raise ValueError("per-mode counters do not match physical totals")
    totals = {key: sum(row[key] for row in tiles.values()) for key in next(iter(tiles.values()))}
    ratio = lambda n, d: n/d if d else None
    return {"status": "PASS", "physical_tiles": expected_tiles, "cycles": next(iter(clocks)),
        "totals": totals, "tiles": tiles,
        "modes": {str(m): {k: sum(modes[(tile, m)][k] for tile in tiles)
                            for k in ("contributions", "lane_fires", "enabled_terms")} for m in range(4)},
        "whole_array_destination_lane_occupancy": ratio(totals["lane_fires"], totals["cycles"]*40),
        "whole_array_p4_term_occupancy": ratio(totals["enabled_terms"], totals["cycles"]*160),
        "accepted_issue_destination_occupancy": ratio(totals["lane_fires"], totals["contributions"]*40),
        "accepted_lane_source_term_occupancy": ratio(totals["enabled_terms"], totals["lane_fires"]*4),
        "scope": "all reset-released clocks in this simulation, including setup/refill/learning/driver gaps",
        "definitions": "lane=destination accumulator lane; enabled term=one enabled P4 source term, including zero weights; not DSP multiplier count, FLOPs or model SOPs",
        "buckets": "exclusive priority: abort, compute, issue-wait, region-wait, observation-wait, fill, internal, idle; observed categories, not exhaustive causal stall attribution",
        "board_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pass-marker", required=True)
    parser.add_argument("--tiles", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = parse_activity(args.log.read_text(encoding="utf-8"), args.pass_marker, args.tiles)
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("output must stay in the project")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
