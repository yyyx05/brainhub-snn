#!/usr/bin/env python3
"""Read existing C1/C2 reports; write small derived CSV/JSON files, never run EDA."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from summarize_fixed_shell_comparison import (
    parse_clock_period,
    parse_timing,
    parse_utilization,
)
from summarize_fixed_shell_hierarchy import parse_hierarchy_report, read_report


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("results/paper_evidence")
OUT = EVIDENCE / "supplement_20261004/comparison"
FIXED = EVIDENCE / "comparison/fixed_shell_comparison_20260928"
PHYSICAL = EVIDENCE / "physical"
TOP = "unified_h40_final_composition_top_v1"
SRNN = TOP + "/u_parameter_plane/g_srnn_client.u_srnn_m2/contribution_frontend"
SHARED = (TOP + "/u_parameter_plane/gen_shared_fc_scnn_output_learner."
          "u_shared_output_learner/g_output_learning_core."
          "u_shared_output_learning_path/u_frontend/contribution_scheduler_i")
REPLAY = (TOP + "/u_system/g_scnn_client.u_train_infer/u_runtime/u_runtime/"
          "u_selected_layers/u_cl/u_observation_replay")
ROUTER = TOP + "/u_system/g_fc_srnn_ingress.u_fc_srnn_learning_ingress/u_router"
SCOPES = {"srnn_frontend": SRNN, "shared_scheduler": SHARED,
          "scnn_replay": REPLAY, "fc_srnn_router": ROUTER}
EXPAND = {SRNN, REPLAY, REPLAY + "/u_trace"}
METRICS = ("slice_luts", "registers", "ramb36", "ramb18", "bram36eq", "dsp")
INPUTS = {
    "learning_on": (PHYSICAL / "run120_current_v2_unified_impl_r2", "post_route"),
    "learning_off": (PHYSICAL / "run122_current_v2_learning_off_impl_r1", "post_route"),
    "Unified": (FIXED / "fixed_shell_comparison_v2/unified_mask7", "post_synth"),
    "FC-only": (FIXED / "fixed_shell_comparison_v2/fc_only_mask1", "post_synth"),
    "SRNN-only": (FIXED / "fixed_shell_comparison_v2/srnn_only_mask2", "post_synth"),
    "SCNN-only": (FIXED / "fixed_shell_comparison_v2/scnn_only_mask4", "post_synth"),
    "private": (PHYSICAL / "run123_current_v2_private_output_learners_synth", "post_synth"),
}
HEADER = ["Instance", "Module", "Total LUTs", "Logic LUTs", "LUTRAMs",
          "SRLs", "FFs", "RAMB36", "RAMB18", "DSP Blocks"]


def metadata(text):
    return dict(re.findall(r"^\|\s*([^:|]+?)\s*:\s*(.*?)\s*$", text, re.M))


def device(meta):
    value = meta.get("Device", "")
    if value.startswith("xc"):
        return value
    speed = meta.get("Speed File", "").split()
    return "xc" + value.replace("-", "") + (speed[0] if speed else "")


def parse_hierarchy(text):
    """Reuse the existing indentation parser, but do not silently lose bad rows."""
    lines = []
    header_seen = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|") or not line.endswith("|"):
            continue
        fields = [field.strip() for field in line[1:-1].split("|")]
        if fields[0] == "Instance":
            if fields != HEADER:
                raise ValueError("unexpected hierarchy resource columns")
            header_seen = True
            continue
        if not header_seen:
            continue
        if len(fields) != 10 or any(
            not re.fullmatch(r"[\d,]+", field) for field in fields[2:]
        ):
            raise ValueError(f"malformed hierarchy row at line {number}")
        if not fields[0].startswith("("):
            lines.append(number)
    parsed = parse_hierarchy_report(text)
    if not header_seen or len(lines) != len(parsed.rows):
        raise ValueError("missing header or unparsed hierarchy rows")
    rows = {row.path: row for row in parsed.rows}
    if len(rows) != len(parsed.rows):
        raise ValueError("duplicate hierarchy path")
    if sum("/" not in path for path in rows) != 1:
        raise ValueError("expected exactly one hierarchy root")
    return rows, dict(zip(rows, lines))


def memory(row):
    return (0, 0) if row is None else row.resources[5:7]


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def add_all(values):
    return tuple(sum(value[i] for value in values) for i in range(2))


def eq(value):
    return value[0] + value[1] / 2


def check_disjoint(paths):
    for index, path in enumerate(paths):
        if any(path == other or path.startswith(other + "/") or
               other.startswith(path + "/") for other in paths[:index]):
            raise ValueError("overlapping hierarchy scopes")


def partition(rows, path, expand):
    """An expanded parent contributes ONLY parent minus immediate children."""
    row = rows.get(path)
    if row is None:
        return {}
    if path not in expand:
        return {(path, "subtree"): memory(row)}
    children = [p for p in rows if p.rpartition("/")[0] == path]
    own = subtract(memory(row), add_all([memory(rows[p]) for p in children]))
    if min(own) < 0:
        raise ValueError(f"negative exclusive memory at {path}")
    result = {(path, "self_minus_children"): own}
    for child in children:
        result.update(partition(rows, child, expand))
    return result


def memory_record(name, path, mode, on, off, line_on=None, line_off=None):
    delta = subtract(on, off)
    return {"name": name, "path": path, "scope": mode,
            "on_line": line_on, "off_line": line_off,
            "on_ramb36": on[0], "on_ramb18": on[1], "on_bram36eq": eq(on),
            "off_ramb36": off[0], "off_ramb18": off[1], "off_bram36eq": eq(off),
            "delta_ramb36": delta[0], "delta_ramb18": delta[1],
            "delta_bram36eq": eq(delta)}


def decompose(on, off, on_lines, off_lines, scopes=SCOPES, expand=EXPAND):
    check_disjoint(list(scopes.values()))
    top_on, top_off = next(iter(on.values())), next(iter(off.values()))
    if top_on.path != top_off.path:
        raise ValueError("different hierarchy roots")
    coarse, details = [], []
    for name, path in scopes.items():
        if path not in on and path not in off:
            raise ValueError(f"scope absent in both reports: {path}")
        coarse.append(memory_record(name, path, "subtree", memory(on.get(path)),
                                    memory(off.get(path)), on_lines.get(path), off_lines.get(path)))
        a, b = partition(on, path, expand), partition(off, path, expand)
        # Refine both runs at the same paths, even if a branch is pruned in one.
        for p, mode in sorted(a.keys() | b.keys()):
            av, bv = a.get((p, mode), (0, 0)), b.get((p, mode), (0, 0))
            if av != (0, 0) or bv != (0, 0):
                details.append(memory_record(name, p, mode, av, bv,
                                             on_lines.get(p), off_lines.get(p)))
    rest_on = subtract(memory(top_on), add_all([memory(on.get(p)) for p in scopes.values()]))
    rest_off = subtract(memory(top_off), add_all([memory(off.get(p)) for p in scopes.values()]))
    remainder = memory_record("other_hierarchy", top_on.path, "root_minus_selected_subtrees",
                              rest_on, rest_off, on_lines[top_on.path], off_lines[top_off.path])
    total = memory_record("total", top_on.path, "root", memory(top_on), memory(top_off),
                          on_lines[top_on.path], off_lines[top_off.path])
    for metric in ("on_ramb36", "on_ramb18", "off_ramb36", "off_ramb18"):
        if sum(r[metric] for r in details) != sum(r[metric] for r in coarse):
            raise ValueError(f"detail reconciliation failed: {metric}")
    return {"coarse": coarse, "details": details, "remainder": remainder, "total": total,
            "unattributed_delta_bram36eq": remainder["delta_bram36eq"]}


def source(root, relative):
    text = read_report(root / relative)
    return text, {"path": relative.as_posix(), "metadata": metadata(text)}


def load_run(root, directory, stage):
    util, up = source(root, directory / f"utilization_{stage}.rpt")
    timing, tp = source(root, directory / f"timing_summary_{stage}.rpt")
    resources = parse_utilization(util)
    if any(value is None for value in resources.values()):
        raise ValueError(f"missing utilization counts: {directory}")
    resources["bram36eq"] = resources["ramb36"] + resources["ramb18"] / 2
    tile_match = re.search(r"^\|\s*Block RAM Tile\s*\|\s*([\d.,]+)\s*\|", util, re.M)
    if tile_match is None:
        raise ValueError(f"missing Block RAM Tile row: {directory}")
    reported_tiles = float(tile_match.group(1).replace(",", ""))
    for key in ("Design", "Tool Version"):
        if up["metadata"].get(key) is None or up["metadata"][key] != tp["metadata"].get(key):
            raise ValueError(f"utilization/timing metadata mismatch: {directory}: {key}")
    if device(up["metadata"]) != device(tp["metadata"]):
        raise ValueError(f"utilization/timing device mismatch: {directory}")
    if not tp["metadata"].get("Command", "").endswith(f"/timing_summary_{stage}.rpt"):
        raise ValueError(f"unexpected timing report command: {directory}")
    expected_state = "Physopt postRoute" if stage == "post_route" else "Synthesized"
    if up["metadata"]["Design State"] != expected_state:
        raise ValueError(f"unexpected measurement state: {directory}")
    timing_values = parse_timing(timing)
    period = parse_clock_period(timing)
    if period is None or any(v is None for v in timing_values.values()):
        raise ValueError(f"missing timing or clock period: {directory}")
    return {"stage": stage, "resources": resources, "timing": timing_values,
            "period_ns": period, "utilization": up, "timing_source": tp,
            "reported_block_ram_tile": reported_tiles,
            "bram36eq_minus_reported_tiles": resources["bram36eq"] - reported_tiles}


def comparison_rows(runs):
    rows = []
    fixed_names = ("FC-only", "SRNN-only", "SCNN-only")
    for kind, target, references in (
        ("unified_vs_sum", "Unified", fixed_names),
        ("unified_vs_maxsingle", "Unified", fixed_names),
        ("learning_on_vs_off", "learning_on", ("learning_off",)),
        ("shared_vs_private", "Unified", ("private",)),
    ):
        for name in references:
            for field in ("Design", "Device", "Tool Version", "Design State"):
                if runs[target]["utilization"]["metadata"][field] != runs[name]["utilization"]["metadata"][field]:
                    raise ValueError(f"incomparable reports: {kind}: {field}")
            if runs[target]["period_ns"] != runs[name]["period_ns"]:
                raise ValueError(f"different clock periods: {kind}")
        for metric in METRICS:
            values = [runs[name]["resources"][metric] for name in references]
            reference = max(values) if kind.endswith("maxsingle") else sum(values)
            denominator = "+".join(references)
            if kind.endswith("maxsingle"):
                denominator = ",".join(name for name, value in zip(references, values) if value == reference)
            value = runs[target]["resources"][metric]
            rows.append({"comparison": kind, "stage": runs[target]["stage"],
                         "metric": metric, "target": target, "reference": denominator,
                         "target_value": value, "reference_value": reference,
                         "delta_target_minus_reference": value - reference,
                         "percent_of_reference": 100 * (value - reference) / reference if reference else None})
    return rows


def site_bounds(resources):
    """Count-only envelope; not a measurement of placed RAMB18 site pairing."""
    b36, b18 = int(resources["ramb36"]), int(resources["ramb18"])
    return [b36 + (b18 + 1) // 2, b36 + b18]


def summarize(root=ROOT):
    runs = {name: load_run(root, directory, stage) for name, (directory, stage) in INPUTS.items()}
    hierarchies, line_maps = {}, {}
    for name in ("learning_on", "learning_off"):
        directory, stage = INPUTS[name]
        text, provenance = source(root, directory / f"utilization_hierarchical_{stage}.rpt")
        for field in ("Design", "Device", "Tool Version", "Design State"):
            if provenance["metadata"][field] != runs[name]["utilization"]["metadata"][field]:
                raise ValueError(f"hierarchy metadata mismatch: {name}: {field}")
        rows, lines = parse_hierarchy(text)
        top = next(iter(rows.values()))
        for metric, index in (("slice_luts", 0), ("registers", 4), ("ramb36", 5), ("ramb18", 6), ("dsp", 7)):
            if top.resources[index] != runs[name]["resources"][metric]:
                raise ValueError(f"hierarchy/flat total mismatch: {name}: {metric}")
        runs[name]["hierarchy_source"] = provenance
        runs[name]["hierarchy_row_count"] = len(rows)
        hierarchies[name], line_maps[name] = rows, lines
    memory_summary = decompose(hierarchies["learning_on"], hierarchies["learning_off"],
                               line_maps["learning_on"], line_maps["learning_off"])
    on_sites = site_bounds(runs["learning_on"]["resources"])
    off_sites = site_bounds(runs["learning_off"]["resources"])
    return {"runs": runs, "memory": memory_summary, "comparisons": comparison_rows(runs),
            "packing": {"learning_on_site_count_bounds": on_sites,
                        "learning_off_site_count_bounds": off_sites,
                        "incremental_site_count_bounds": [on_sites[0] - off_sites[1], on_sites[1] - off_sites[0]],
                        "actual_placed_site_counts": None,
                        "logical_payload_padding_bits": None,
                        "note": "Count-only bounds assuming two RAMB18 per BRAM site; not placed-site measurements. BRAM36eq is capacity, not occupied sites."},
            "boundaries": ["Existing reports only; no EDA, simulation or experiment rerun.",
                           "CSV percent is signed (target-reference)/reference * 100.",
                           "Hierarchy scopes are inclusive subtrees OR exclusive parent-minus-children, never both.",
                           "Missing off row means no separately reported allocation at that path, not a tensor-level proof.",
                           "Module attribution does not establish per-array payload/padding or packing efficiency.",
                           "Whole-design deltas are not standalone learner areas; stages and denominators must not be mixed."]}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    summary = summarize()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / OUT):
        parser.error("outputs must remain inside the supplementary comparison directory")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "learning_memory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    mem = summary["memory"]
    write_csv(destination / "learning_memory_hierarchy.csv", mem["coarse"] + [mem["remainder"]])
    write_csv(destination / "learning_memory_detail.csv", mem["details"])
    write_csv(destination / "resource_comparisons.csv", summary["comparisons"])
    print(f"Written {destination}; delta={mem['total']['delta_bram36eq']} BRAM36eq; "
          f"unattributed={mem['unattributed_delta_bram36eq']} BRAM36eq")


if __name__ == "__main__":
    main()
