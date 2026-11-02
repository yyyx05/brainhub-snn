#!/usr/bin/env python3
"""Use the accepted SCNN sample's first complete T4 window for compatibility."""
import argparse
import difflib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
BASE = ROOT / "results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn"


def transform(source):
    text = source
    for old, new in (
        ("timestep_phase < 20;", "timestep_phase < 4;"),
        ("fg_requests != 5 || fg_windows != 5 || fg_descriptors != 160", "fg_requests != 1 || fg_windows != 1 || fg_descriptors != 32"),
        ("fg_classifier_restores != 20 || fg_classifier_spills != 20", "fg_classifier_restores != 4 || fg_classifier_spills != 4"),
        ("tv_features_checked != 40960 || tv_currents_checked != 220 || classifier_logit_wave_count != 20", "tv_features_checked != 8192 || tv_currents_checked != 44 || classifier_logit_wave_count != 4"),
        ("tv_traces_checked != 40960*fg_learn_on || tv_publish_words != 1892*(1+5*fg_learn_on)", "tv_traces_checked != 8192*fg_learn_on || tv_publish_words != 1892*(1+fg_learn_on)"),
        ("fg_commits != 5*fg_learn_on", "fg_commits != fg_learn_on"),
        ("tv_probes_checked != 480", "tv_probes_checked != 96"),
        ("SCNN_PROFILE_PASS learning=%0d samples=1 frames=20", "SCNN_PROFILE_PASS learning=%0d samples=1 frames=4"),
    ):
        if text.count(old) != 1:
            raise ValueError("SCNN prefix anchor not unique: " + old)
        text = text.replace(old, new, 1)
    # The candidate wraps the original SCNN cache in Context0; these two
    # read-only failure diagnostics must follow that hierarchy, not change data.
    old = ".u_weight_store.g_bank"
    if text.count(old) != 2:
        raise ValueError("SCNN packing probe anchor not unique")
    return text.replace(old,".u_weight_store.g_context[0].g_enabled.u_store.g_bank")


def parse_log(log, expected, learning):
    if re.search(r"%Error|%Fatal|Assertion failed", log):
        raise ValueError("simulator error in SCNN compatibility log")
    def rows(marker):
        return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines() if line.startswith(marker+" ")]
    final_rows = rows("SCNN_PROFILE_PASS")
    if len(final_rows) != 1:
        raise ValueError("missing/duplicate complete SCNN compatibility result")
    final = {k: int(v) for k, v in final_rows[0].items()}
    enabled = int(learning)
    exact = dict(learning=enabled, samples=1, frames=4, descriptors=32, requests=1,
                 features=8192, currents=44, logits=44, traces=8192*enabled,
                 probes=96, commits=enabled, publications=1892*(1+enabled), rmw=expected["rmw"])
    if any(final.get(k) != v for k, v in exact.items()):
        raise ValueError("incomplete SCNN T4 numeric/update counts")
    frames = rows("SCNN_PROFILE_FRAME")
    if [(int(r["frame"]),int(r["sample"]),int(r["timestep"])) for r in frames] != [(i,0,i) for i in range(4)]:
        raise ValueError("missing/duplicate/out-of-order SCNN frames")
    descriptors = rows("SCNN_PROFILE_DESCRIPTOR")
    wanted = [(frame,layer,edge) for frame in range(4) for layer in range(1,9) for edge in ("start","end")]
    if [(int(r["frame"]),int(r["layer"]),r["edge"]) for r in descriptors] != wanted:
        raise ValueError("incomplete SCNN descriptors")
    for i in range(0,len(descriptors),2):
        if int(descriptors[i+1]["cycle"]) <= int(descriptors[i]["cycle"]):
            raise ValueError("invalid descriptor interval")
    traffic_rows = rows("SCNN_PROFILE_TRAFFIC")
    if len(traffic_rows) != 1:
        raise ValueError("missing SCNN interface counters")
    traffic = {k:int(v) for k,v in traffic_rows[0].items()}
    for req,rsp in (("forward_reads","forward_responses"),("learning_reads","learning_responses"),
                    ("learning_reads","learning_writes"),("state_spill_requests","state_spill_rows"),
                    ("trace_spill_requests","trace_spill_rows")):
        if traffic[req] != traffic[rsp]:
            raise ValueError("undrained SCNN interface: "+req)
    if traffic["learning_writes"] != exact["rmw"] or traffic["store_loads"] != 0:
        raise ValueError("unexpected SCNN store traffic")
    if (traffic["code_write_words"],traffic["residual_write_words"]) != (expected["code_changes"],expected["residual_changes"]):
        raise ValueError("SCNN code/residual write counts differ from fixed reference")
    banks = rows("SCNN_PROFILE_DDR")
    if [int(r["bank"]) for r in banks] != list(range(9)) or any(int(r["forward_reads"])<=0 or int(r["forward_reads"])!=int(r["forward_responses"]) for r in banks):
        raise ValueError("incomplete actual bank traffic")
    if sum(int(r["forward_reads"]) for r in banks)!=traffic["forward_reads"]:
        raise ValueError("bank/aggregate traffic mismatch")
    tiles = rows("H40_ACTIVITY")
    if len(tiles)!=12 or len({r["tile"] for r in tiles})!=12:
        raise ValueError("missing twelve-tile activity")
    return dict(status="PASS_fixed_sample0_T4_candidate_SCNN_compatibility",final=final,
                frames=frames,descriptors=descriptors,traffic=traffic,banks=banks,h40=tiles)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="scnn_compat_r2")
    args=parser.parse_args()
    if not args.name.replace("_","").isalnum(): parser.error("simple name required")
    source=(BASE/"server_pair_r2/tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
    text=transform(source)
    manifest=json.loads((BASE/"fixed_sample0_b1t20_r1/manifest.json").read_text())
    output=ROOT/"verification/experiments/srnn_resident_weight_context"/args.name
    output.mkdir(parents=True,exist_ok=False)
    (output/"tb_scnn_profile_measurement.sv").write_text(text,encoding="utf-8")
    (output/"testbench_changes.diff").write_text("".join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile="accepted_T20/tb",tofile="candidate_T4/tb")),encoding="utf-8")
    report=dict(status="PREPARED_NOT_SIMULATED",expected={arm:manifest["windows"][arm][0] for arm in ("off","on")},
        fixture="results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn/fixed_sample0_b1t20_r1",
        scope="Candidate RTL, original sample0 first complete B1T4 window, original weights/640 rule, cold off/on. Full layer numeric/state/trace checks and head update checks retained. Not full T20, official Test, M2/SCNN cross-profile switching, board or a throughput optimization claim. First window reference has zero code changes; nonzero residual writes are checked.")
    (output/"reference.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
