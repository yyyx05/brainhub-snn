#!/usr/bin/env python3
"""Reconcile accepted SCNN traffic with actual matrix sizes; no simulation."""
import csv
import json
from pathlib import Path
import re

from prepare_scnn_resident_compatibility import parse_log
from run_srnn_resident_remote import ROOT, RESULT, local_path


def numeric_constant(source, name):
    matches = re.findall(r"localparam int " + re.escape(name) + r"\s*=\s*(\d+)\s*;", source)
    if len(matches) != 1:
        raise ValueError("expected one literal integer declaration: " + name)
    return int(matches[0])


def reconcile(source, measurements):
    result = []
    for layer in range(1, 9):
        channels = numeric_constant(source, f"CONV{layer}_CHANNELS")
        contributions = numeric_constant(source, f"CONV{layer}_CONTRIBUTIONS")
        weights = channels * contributions
        row = dict(layer=layer, output_channels=channels, contributions=contributions,
                   unique_code_words=weights, code_payload_bytes=weights*2)
        for arm in ("off", "on"):
            parsed = measurements[arm]
            bank = parsed["banks"][layer-1]
            frames = parsed["final"]["frames"]
            actual = int(bank["forward_reads"])
            if int(bank["bank"]) != layer-1 or actual != int(bank["forward_responses"]):
                raise ValueError("wrong or undrained bank")
            if actual != weights * frames:
                raise ValueError("convolution reads are not exactly one matrix per frame")
            row[arm+"_frames"] = frames
            row[arm+"_actual_reads"] = actual
            row[arm+"_reads_per_unique_weight_per_frame"] = actual/(weights*frames)
        result.append(row)
    head = int(measurements["off"]["final"]["features"]) // measurements["off"]["final"]["frames"]
    head *= numeric_constant(source, "CLASSES")
    for arm in ("off", "on"):
        parsed = measurements[arm]
        publications = 1 + parsed["final"]["commits"]
        if int(parsed["banks"][8]["forward_reads"]) != head*publications:
            raise ValueError("classifier read budget differs from initial/commit publications")
        if sum(int(b["forward_reads"]) for b in parsed["banks"]) != parsed["traffic"]["forward_reads"]:
            raise ValueError("aggregate traffic mismatch")
    return result, head


def main():
    directory = Path(local_path(ROOT/RESULT/"scnn_compat_r3"))
    read_json = lambda name: json.loads((directory/name).read_text(encoding="utf-8"))
    state = read_json("status.json")
    if state["status"] != "PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility":
        raise ValueError("accepted R3 pair required")
    if any(state[stage]["exit"] != 0 for stage in ("build", "run_off", "run_on")):
        raise ValueError("nonzero process exit")
    expected = read_json("reference.json")["expected"]
    measurements = {}
    for arm in ("off", "on"):
        parsed = parse_log((directory/("run_"+arm+".log")).read_text(encoding="utf-8"), expected[arm], arm == "on")
        if parsed != read_json(arm+"_checks.json"):
            raise ValueError("saved parser output differs from actual log")
        measurements[arm] = parsed
    source = (directory/"tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
    rows, head = reconcile(source, measurements)
    report = dict(status="ANALYSIS_from_accepted_R3_no_new_RTL_run", source_run="scnn_compat_r3",
        rows=rows, convolution_code_words_per_frame=sum(r["unique_code_words"] for r in rows),
        convolution_code_payload_bytes=sum(r["code_payload_bytes"] for r in rows),
        classifier_words_per_publication=head,
        host_service=dict(minimum_response_latency_cycles=1, minimum_request_interval_cycles=2,
            ready_expression="ft_i_store_forward_read_ready = !fg_forward_valid",
            basis="Accepted TB sets response valid on request edge and clears it on response acceptance; no same-cycle replacement."),
        conclusions=[
            "Each convolution matrix is read exactly once per frame in both accepted arms; no spatial-window reload multiplication.",
            "Scheduler iterates spatial windows inside each output-channel wave; load_weights is asserted only at spatial_base=0.",
            "Current scalar service, backend outstanding bit, and shared response-owner router all prevent same-cycle response/request replacement.",
            "The verification-support BRAM store already permits response replacement, but it is not the T4 external service and is not evidence of a deployed DDR engine.",
            "Further request pipelining must compare old/new RTL under identical service conditions; wider reads need explicit payload bandwidth, buffering, mapping and resource costs."],
        limitations=["Traffic reconciliation is not a new throughput, cycle-attribution, board, DDR or physical-resource result.",
            "16-bit code payload excludes address/tag/residual/protocol overhead; this is not measured external-memory byte traffic.",
            "No use of the failed schedule-probe R1 or unfinished R2 as accepted measurement."])
    output = Path(local_path(ROOT/RESULT/"scnn_read_budget.json"))
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
