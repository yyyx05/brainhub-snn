#!/usr/bin/env python3
"""Compare completed T4 pairs from raw evidence, without rerunning simulation."""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import re

from prepare_scnn_resident_compatibility import parse_log
from run_srnn_resident_remote import ROOT, RESULT, local_path
from summarize_scnn_read_budget import reconcile


def path(value):
    return Path(local_path(value))


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def rows(text, marker):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in text.splitlines()
            if line.startswith(marker + " ")]


def timing(text, parsed):
    if re.search(r"%Error|%Fatal|Assertion failed|out of memory|Killed", text, re.I):
        raise ValueError("runtime diagnostic in complete log")
    setup = rows(text, "SUPPLEMENT_FULLGRAPH_INITIALIZATION")
    if len(setup) != 1:
        raise ValueError("expected one initialization interval")
    setup_cycles = int(setup[0]["cycles"])
    total = parsed["final"]["cycles"]
    frame_cycles = sum(int(r["cycles"]) for r in parsed["frames"])
    if total < setup_cycles + frame_cycles:
        raise ValueError("frame/setup accounting exceeds total")
    return dict(total_cycles=total, initialization_cycles=setup_cycles,
                post_initialization_cycles=total-setup_cycles,
                frame_cycles=frame_cycles,
                inter_frame_boundary_cycles=total-setup_cycles-frame_cycles)


def compare_arm(old, new, old_text, new_text, arm):
    if {k:v for k,v in old["final"].items() if k != "cycles"} != {
            k:v for k,v in new["final"].items() if k != "cycles"}:
        raise ValueError("workload or completion counts changed")
    for key in ("traffic", "banks"):
        if old[key] != new[key]:
            raise ValueError("interface counts changed: " + key)
    for marker in ("SCNN_PROFILE_FRAME_NUMERIC_PASS", "H40_ACTIVITY_MODE"):
        if rows(old_text, marker) != rows(new_text, marker):
            raise ValueError("numeric/activity markers changed: " + marker)
    before, after = timing(old_text, old), timing(new_text, new)
    layers = []
    for i in range(0, len(old["descriptors"]), 2):
        a, b = old["descriptors"][i:i+2], new["descriptors"][i:i+2]
        if any((x["frame"], x["layer"], x["edge"]) != (y["frame"], y["layer"], y["edge"])
               for x,y in zip(a,b)):
            raise ValueError("descriptor identity changed")
        old_cycles = int(a[1]["cycle"])-int(a[0]["cycle"])+1
        new_cycles = int(b[1]["cycle"])-int(b[0]["cycle"])+1
        layers.append(dict(arm=arm, frame=int(a[0]["frame"]), layer=int(a[0]["layer"]),
                           old_cycles=old_cycles, new_cycles=new_cycles,
                           saved_cycles=old_cycles-new_cycles))
    saved = before["total_cycles"]-after["total_cycles"]
    saved_init = before["initialization_cycles"]-after["initialization_cycles"]
    saved_layers = sum(r["saved_cycles"] for r in layers)
    publications = []
    for text in (old_text,new_text):
        publications.append([r for r in rows(text,"SCNN_PROFILE_INTERVAL")
                             if r["kind"] == "publication"])
    if [r["window"] for r in publications[0]] != [r["window"] for r in publications[1]]:
        raise ValueError("publication sequence changed")
    saved_post_publication = sum((int(a["end"])-int(a["start"]))-
                                (int(b["end"])-int(b["start"]))
                                for a,b in zip(*publications) if int(a["window"]) > 0)
    row = dict(arm=arm, old_total_cycles=before["total_cycles"], new_total_cycles=after["total_cycles"],
               total_cycle_speedup=before["total_cycles"]/after["total_cycles"],
               old_initialization_cycles=before["initialization_cycles"],
               new_initialization_cycles=after["initialization_cycles"],
               post_initialization_speedup=before["post_initialization_cycles"]/after["post_initialization_cycles"],
               saved_total_cycles=saved, saved_initialization_cycles=saved_init,
               saved_descriptor_cycles=saved_layers, saved_post_publication_cycles=saved_post_publication,
               saved_outside_those_intervals=saved-saved_init-saved_layers-saved_post_publication,
               forward_reads=new["traffic"]["forward_reads"], physical_timing_validated=False)
    return row, layers, dict(old=before,new=after)


def summarize(name="scnn_issue_sparse_r1"):
    old = path(ROOT/RESULT/"scnn_compat_r3")
    new = path(ROOT/RESULT/name)
    required = ("tb_scnn_profile_measurement.sv", "reference.json", "compatibility_sources.f",
                "scnn_compatibility_build.json")
    for relative in required:
        if (old/relative).read_bytes() != (new/relative).read_bytes():
            raise ValueError("comparison input changed: " + relative)
    dependency_count = 0
    for folder in ("models", "results", "tb", "tests", "scripts", "build"):
        a = {p.relative_to(old).as_posix():p for p in (old/folder).rglob("*") if p.is_file()}
        b = {p.relative_to(new).as_posix():p for p in (new/folder).rglob("*") if p.is_file()}
        if a.keys() != b.keys():
            raise ValueError("dependency set changed: " + folder)
        for relative in a:
            if relative == "scripts/experiments/srnn_resident_server_worker.py":
                # Other workloads were added; compare the actual SCNN build function.
                def build_function(file):
                    tree = ast.parse(file.read_text(encoding="utf-8"))
                    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                                         and n.name == "scnn_build_command"), include_attributes=False)
                if build_function(a[relative]) != build_function(b[relative]):
                    raise ValueError("SCNN build recipe function changed")
                continue
            if a[relative].read_bytes() != b[relative].read_bytes():
                raise ValueError("non-RTL dependency changed: " + relative)
            dependency_count += 1
    a = {p.relative_to(old).as_posix():p for p in (old/"rtl").rglob("*.sv")}
    b = {p.relative_to(new).as_posix():p for p in (new/"rtl").rglob("*.sv")}
    if a.keys() != b.keys():
        raise ValueError("RTL source set changed")
    changed = sorted(k for k in a if a[k].read_bytes() != b[k].read_bytes())
    overlay = read_json(new/"scnn_optimization_overlay.json")
    if changed != sorted(overlay["changed_sources"]) or len(changed) != 2:
        raise ValueError("unexpected RTL difference")
    for experiment, selected, unit_run, module in (
        ("fc_contribution_pipeline", overlay["fc_scanner_candidate"], "fc_contribution_sparse_unit_r2", "unified_fc_tp_w2_contribution_scheduler.sv"),
        ("scnn_publisher_issue", overlay["scnn_publisher_candidate"], "scnn_publisher_issue_unit_r1", "unified_scnn_classifier_cache_refill_publisher_v1.sv")):
        prepared = path(ROOT/"verification/experiments"/experiment/selected/"rtl/learning"/module)
        if prepared.read_bytes() != (new/"rtl/learning"/module).read_bytes():
            raise ValueError("actual module differs from tested candidate")
        if path(ROOT/RESULT/unit_run/module).read_bytes() != prepared.read_bytes():
            raise ValueError("candidate differs from actual unit-test module")
    parsed, texts, states = {}, {}, {}
    for directory in (old,new):
        state = read_json(directory/"status.json")
        if state["status"] != "PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility":
            raise ValueError("pair has not completed")
        if any(state[stage]["exit"] != 0 for stage in ("build","run_off","run_on")):
            raise ValueError("nonzero actual exit")
        if state["structural_warnings"]:
            raise ValueError("structural warnings present")
        states[directory.name] = state
        parsed[directory.name], texts[directory.name] = {}, {}
        reference = read_json(directory/"reference.json")
        for arm in ("off","on"):
            text = (directory/("run_"+arm+".log")).read_text(encoding="utf-8")
            item = parse_log(text,reference["expected"][arm],arm == "on")
            if item != read_json(directory/(arm+"_checks.json")) or item["final"] != state[arm+"_final"]:
                raise ValueError("raw log differs from checks/status")
            parsed[directory.name][arm], texts[directory.name][arm] = item,text
    summary, layers, intervals = [], [], {}
    for arm in ("off","on"):
        row, detail, account = compare_arm(parsed[old.name][arm],parsed[new.name][arm],
                                          texts[old.name][arm],texts[new.name][arm],arm)
        summary.append(row); layers.extend(detail); intervals[arm] = account
    budget, head = reconcile((new/required[0]).read_text(encoding="utf-8"),parsed[new.name])
    for row in layers:
        if row["saved_cycles"] != budget[row["layer"]-1]["unique_code_words"]:
            raise ValueError("layer saving is not one cycle per convolution weight")
    build = (new/"build.log").read_text(encoding="utf-8",errors="replace")
    if re.search(r"%Error|%Fatal|out of memory|Killed",build,re.I):
        raise ValueError("build diagnostic")
    report = dict(status="PASS_reparsed_same_service_T4_comparison_not_physical_QoR",
                  candidate_run=name,reference_run=old.name,source_count=len(b),changed_sources=changed,
                  unchanged_dependency_files=dependency_count,rows=summary,intervals=intervals,
                  layers=layers,states=states,classifier_words_per_publication=head,
                  warnings=dict(Counter(re.findall(r"^%Warning-([^:]+):",build,re.M))),
                  limitations=["Total already includes initialization; do not add it again.",
                    "Same sample0 T4 prefix and service model, not full T20/Test/DDR or peer superiority.",
                    "First window has residual changes but zero code changes; no nonzero-update query.",
                    "SCNN B1 uses the scheduler dense pipeline fallback; bitmap skipping is enabled only for batch > 2.",
                    "No new physical timing or resource result; old 3ps does not apply.",
                    "Remaining on-arm saving is outside descriptor/publication intervals, not a separately measured scanner-only latency.",
                    "Kernel OOM log unavailable; no OOM/Killed diagnostic in fetched logs."])
    output = path(ROOT/RESULT/"scnn_issue_sparse_comparison.json")
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    for suffix, data in (("",summary),("_layers",layers)):
        with output.with_name(output.stem+suffix+".csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:v for k,v in report.items() if k not in ("states","layers")},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="scnn_issue_sparse_r1")
    summarize(parser.parse_args().name)
