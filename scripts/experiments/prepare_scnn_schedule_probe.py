#!/usr/bin/env python3
"""Prepare passive first-frame timing attribution, not a compatibility rerun."""
import argparse
import difflib
import json
import re
from prepare_scnn_resident_compatibility import ROOT

EXPERIMENT = ROOT / "verification/experiments/srnn_resident_weight_context"
ACCEPTED = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/scnn_compat_r3"
BUCKETS = ("idle", "context_load", "feature_request", "feature_response", "command_request", "command_wait", "context_release", "classifier")
PUBLISHER = ("idle", "command", "gather", "read_request", "read_response", "send_row", "wait_complete")


def rows(log, marker):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines() if line.startswith(marker+" ")]


def transform(source, monitor):
    anchor = "        end\n        fg_finish();\n        $finish;"
    if source.count(anchor) != 1 or source.count("endmodule") != 1:
        raise ValueError("unknown accepted SCNN testbench")
    # $finish need not stop the current procedural block immediately in Verilator.
    # Leave the frame loop, then report once without executing the T4 finalizer.
    text = source.replace(anchor, "        break; // First-frame diagnostic only.\n"
                          "        end\n        sp_report();\n        $finish;", 1)
    return text.replace("endmodule", monitor+"\nendmodule", 1)


def first_frame_reference(log):
    descriptors = [r for r in rows(log,"SCNN_PROFILE_DESCRIPTOR") if int(r["frame"]) == 0]
    frames = [r for r in rows(log,"SCNN_PROFILE_FRAME") if int(r["frame"]) == 0]
    numeric = [r for r in rows(log,"SCNN_PROFILE_FRAME_NUMERIC_PASS") if int(r["frame"]) == 0]
    if len(descriptors) != 16 or len(frames) != 1 or len(numeric) != 1:
        raise ValueError("accepted first-frame reference incomplete")
    return dict(descriptors=descriptors, frame=frames[0], numeric=numeric[0])


def parse_log(log, expected, learning=False):
    if learning or re.search(r"%Error|%Fatal|Assertion failed",log):
        raise ValueError("first-frame probe error or unsupported learning mode")
    finals = rows(log,"SCNN_SCHEDULE_PROBE_PASS")
    wanted = dict(learning=0,frames=1,descriptors=8,features=2048,currents=11,logits=11,probes=24,commits=0,rmw=0,publications=1892)
    if len(finals) != 1 or {k:int(v) for k,v in finals[0].items()} != wanted:
        raise ValueError("missing/duplicate first-frame diagnostic completion")
    if (rows(log,"SCNN_PROFILE_PASS") or len(rows(log,"SCNN_PROFILE_FRAME")) != 1 or
            len(rows(log,"SCNN_PROFILE_DESCRIPTOR")) != 16 or len(rows(log,"SCNN_PROFILE_FRAME_NUMERIC_PASS")) != 1):
        raise ValueError("probe must not claim complete T4 compatibility")
    observed = first_frame_reference(log)
    if observed != expected:
        raise ValueError("passive measurement changed descriptor timing or numeric evidence")
    wanted_order = [(0,layer,edge) for layer in range(1,9) for edge in ("start","end")]
    if [(int(r["frame"]),int(r["layer"]),r["edge"]) for r in observed["descriptors"]] != wanted_order:
        raise ValueError("descriptor ordering")
    tables = {}
    for kind in ("BUCKET","CONTEXT","PUBLISHER","HANDSHAKE"):
        table = [{k:int(v) for k,v in r.items()} for r in rows(log,"SCNN_SCHEDULE_"+kind)]
        if [r["layer"] for r in table] != list(range(1,9)) or any(v<0 for r in table for v in r.values()):
            raise ValueError("incomplete probe table: "+kind)
        tables[kind.lower()] = table
    for index in range(8):
        b,c,p,h = (tables[k][index] for k in ("bucket","context","publisher","handshake"))
        desc = observed["descriptors"][2*index:2*index+2]
        if b["cycles"] != int(desc[1]["cycle"])-int(desc[0]["cycle"]) or sum(b[k] for k in BUCKETS) != b["cycles"]:
            raise ValueError("exclusive scheduler buckets do not close")
        if sum(c[k] for k in ("idle","cache","transfer","drain")) != b["context_load"]:
            raise ValueError("context sub-buckets do not close")
        if sum(p[k] for k in PUBLISHER) > b["context_load"] or p["reads"] != p["responses"]:
            raise ValueError("publisher attribution or response mismatch")
        if not (h["features"] == h["responses"] == h["commands"] == h["completions"] > 0):
            raise ValueError("command/feature transaction imbalance")
        if h["loads"] != h["releases"] or h["loads"] != h["weight_loads"]+h["reuses"] or h["forward_reads"] != h["forward_responses"]:
            raise ValueError("context/forward transaction imbalance")
        if p["starts"] != h["weight_loads"] or p["reads"] > h["forward_reads"]:
            raise ValueError("publisher refill/traffic attribution mismatch")
    if log.index("SCNN_PROFILE_FRAME_NUMERIC_PASS") > log.index("SCNN_SCHEDULE_PROBE_PASS"):
        raise ValueError("probe marker precedes numeric checks")
    total = sum(b["cycles"] for b in tables["bucket"])
    aggregate = {k:sum(b[k] for b in tables["bucket"]) for k in BUCKETS}
    return dict(status="PASS_first_frame_passive_attribution_not_full_window",final=wanted,
                original_first_frame=observed,tables=tables,descriptor_cycles=total,
                scheduler_cycles=aggregate,scheduler_fraction={k:v/total for k,v in aggregate.items()},
                scope="One real sample0 frame, learning off. Publisher/context subcounts overlap scheduler buckets; do not sum across tables. No prefetch speedup, DDR bandwidth, full-window or learning claim.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="scnn_schedule_probe_r1")
    args=parser.parse_args()
    if not args.name.replace("_","").isalnum(): parser.error("simple name required")
    source=(ACCEPTED/"tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
    monitor=(EXPERIMENT/"scnn_schedule_probe.svh").read_text(encoding="utf-8")
    text=transform(source,monitor)
    config=json.loads((ACCEPTED/"reference.json").read_text())
    reference=first_frame_reference((ACCEPTED/"run_off.log").read_text())
    output=EXPERIMENT/args.name
    output.mkdir(parents=True,exist_ok=False)
    (output/"tb_scnn_profile_measurement.sv").write_text(text,encoding="utf-8")
    (output/"testbench_changes.diff").write_text("".join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile="accepted_R3/tb",tofile="first_frame_probe/tb")),encoding="utf-8")
    (output/"reference.json").write_text(json.dumps(dict(status="PREPARED_NOT_SIMULATED",fixture=config["fixture"],expected=dict(off=reference),scope="Passive actual sample0 first-frame scheduler/context/refill attribution. Original first-frame numeric checks preserved; ends before remaining T4 steps. No RTL changes, no full learning-window or optimization PASS."),indent=2)+"\n")
    print(str(output).removeprefix("\\\\?\\"))


if __name__ == "__main__": main()
