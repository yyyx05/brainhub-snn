"""Derive first-frame and startup-inclusive comparisons without changing raw data."""
import argparse
import csv
import json
from pathlib import Path

from run_srnn_resident_remote import ROOT, RESULT, local_path
from prepare_scnn_schedule_probe import BUCKETS, rows
from run_scnn_turnover_frame import parse_log


def compare(before, after, logs, budget):
    if before["banks"] != after["banks"] or before["tables"]["handshake"] != after["tables"]["handshake"]:
        raise ValueError("different paired work")
    contract=("reads","responses","slots","code_bits","address_bits","tag_bits","minimum_latency","replaceable","m2_cache_depth")
    if any(before["service"][k]!=after["service"][k] for k in contract):
        raise ValueError("different host service")
    metrics=[]
    def metric(name,b,a):
        if a<=0 or b<=0:raise ValueError("nonpositive measurement")
        metrics.append(dict(metric=name,baseline_cycles=b,candidate_cycles=a,saved_cycles=b-a,ratio=b/a))
    metric("raw_frame_only",before["frame_cycles"],after["frame_cycles"])
    metric("initialization_interval_only",before["initialization_cycles"],after["initialization_cycles"])
    starts=[]
    for log in logs:
        events=[r for r in rows(log,"SCNN_PROFILE_EVENT") if r["kind"]=="initialize_start"]
        if len(events)!=1:raise ValueError("ambiguous initialization anchor")
        starts.append(int(events[0]["cycle"]))
    elapsed=[c["service"]["end_cycle"]-s for c,s in zip((before,after),starts)]
    metric("initialize_start_to_final_service_report",*elapsed)
    layers=[]
    for i,geometry in enumerate(budget["layers"]):
        b,a=before["tables"]["bucket"][i],after["tables"]["bucket"][i]
        if b["layer"]!=i+1 or a["layer"]!=i+1:raise ValueError("layer ordering")
        saved=b["cycles"]-a["cycles"]
        delta={k:b[k]-a[k] for k in BUCKETS}
        layers.append(dict(layer=i+1,baseline_cycles=b["cycles"],candidate_cycles=a["cycles"],saved_cycles=saved,
            ratio=b["cycles"]/a["cycles"],context_load_saved=delta["context_load"],
            other_scheduler_saved=sum(v for k,v in delta.items() if k!="context_load"),
            conditional_prior_budget=geometry["max_turnovers"],budget_matches=saved==geometry["max_turnovers"],
            reads=before["tables"]["publisher"][i]["reads"],responses=after["tables"]["publisher"][i]["responses"]))
    frame_saved=metrics[0]["saved_cycles"]
    replacement_delta=after["service"]["replacements"]-before["service"]["replacements"]
    return dict(status="DERIVED_from_audited_actual_first_frame_pair",metrics=metrics,layers=layers,
        frame_saved_cycles=frame_saved,descriptor_saved_cycles=sum(r["saved_cycles"] for r in layers),
        context_load_saved_cycles=sum(r["context_load_saved"] for r in layers),
        initialization_interval_saved_cycles=metrics[1]["saved_cycles"],
        startup_inclusive_saved_cycles=metrics[2]["saved_cycles"],replacement_delta=replacement_delta,
        budget_reconciliation=dict(prior_conv_budget=budget["max_turnover_savings_per_frame"],
            actual_frame_savings_match_prior_budget=frame_saved==budget["max_turnover_savings_per_frame"],
            prior_classifier_budget=budget["classifier_savings_per_publish"],
            actual_initialization_savings_match_prior_budget=metrics[1]["saved_cycles"]==budget["classifier_savings_per_publish"],
            startup_savings_match_replacement_count=metrics[2]["saved_cycles"]==replacement_delta),
        boundary_cycles_outside_frame_and_initialization_interval=[e-c["frame_cycles"]-c["initialization_cycles"] for e,c in zip(elapsed,(before,after))],
        service=dict(identical_contract={k:before["service"][k] for k in contract},
            baseline_replacements=before["service"]["replacements"],candidate_replacements=after["service"]["replacements"],
            code_payload_bytes_per_arm=2*before["service"]["reads"],conv_code_payload_bytes_per_frame=2*sum(r["reads"] for r in layers)),
        boundary="One actual sample0 raw Conv1..8 frame, pool/classifier, learning off, identical replaceable latency1 16-bit one-slot host. Frame-only excludes initialization. Inclusive interval is directly measured from initialize_start to final service report; initialization is not added twice. Legacy II2 module result remains zero benefit. Not full T20/Test, learning updates, abort, physical DDR bandwidth, whole-core mapped area or new STA. No old physical margin inherited.")


def summarize(name):
    archive=Path(local_path(ROOT/RESULT/name))
    audit=json.loads((archive/"terminal_audit.json").read_text())
    if audit["status"]!="PASS_raw_first_frame_pair_archive_audit":raise ValueError("complete archive audit required")
    checks=[];logs=[]
    for arm in ("baseline","candidate"):
        log=(archive/arm/"run.log").read_text()
        result=parse_log(log,json.loads((archive/arm/"reference.json").read_text()))
        if result!=json.loads((archive/arm/"checks.json").read_text()):raise ValueError("raw reparse mismatch")
        checks.append(result);logs.append(log)
    budget=json.loads(Path(local_path(ROOT/RESULT/"scnn_forward_turnover_unit_r1/cycle_and_service_budget.json")).read_text())
    comparison=compare(*checks,logs,budget)
    raw=json.loads((archive/"comparison.json").read_text())
    if raw["baseline_frame_cycles"]!=comparison["metrics"][0]["baseline_cycles"] or raw["candidate_frame_cycles"]!=comparison["metrics"][0]["candidate_cycles"]:
        raise ValueError("worker comparison mismatch")
    (archive/"frame_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    for name,table in (("frame_comparison.csv",comparison["metrics"]),("layer_comparison.csv",comparison["layers"])):
        with (archive/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({k:comparison[k] for k in ("metrics","budget_reconciliation","service","boundary_cycles_outside_frame_and_initialization_interval")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="scnn_turnover_frame_r2")
    summarize(parser.parse_args().name)
