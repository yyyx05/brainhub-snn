#!/usr/bin/env python3
"""Retrieve actual XPM sources/logs and distinguish characterization from PASS."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import stat

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, local_path, run
from run_m2_xpm_context import parse_run, transform_wrapper


def main():
    name = "m2_xpm_context_r1"
    destination = Path(local_path(ROOT/RESULT/name))
    remote = REMOTE+"/"+name
    c = connect()
    files = []
    try:
        with c.open_sftp() as s:
            with s.open(remote+"/status.json") as f:
                status = json.load(f)
            if status["status"] not in ("diagnostic_complete_baseline_hold_failure_candidates_pass", "PASS_exercised_XPM_scope"):
                raise ValueError("not completed characterization")

            def retrieve(path, relative=""):
                for item in s.listdir_attr(path):
                    if item.filename in ("xsim.dir", "obj", "elab_obj", ".Xil", "__pycache__"):
                        continue
                    rel = relative+item.filename
                    if stat.S_ISDIR(item.st_mode):
                        retrieve(path+"/"+item.filename, rel+"/")
                    else:
                        local = destination/rel
                        local.parent.mkdir(parents=True,exist_ok=True)
                        s.get(path+"/"+item.filename,str(local))
                        if local.stat().st_size != item.st_size:
                            raise ValueError("size mismatch "+rel)
                        files.append({"path":rel,"size":item.st_size})
            retrieve(remote)
        environment = {"disk":run(c,"df -B1 /root/autodl-tmp"),
                       "processes":run(c,"ps -eo pid,comm,args"),
                       "dmesg":run(c,"dmesg 2>&1 || true")}
    finally:
        c.close()
    if list(destination.rglob("*.part")):
        raise ValueError("partial download")
    source = ROOT/RESULT/"scnn_issue_banked_synth_r1/rtl/memory"
    candidate = ROOT/"verification/experiments/m2_xpm_context/candidate_r1"
    inputs = {
        "baseline.sv":source/"unified_dual_role_weight_store_p4all_candidate_v1.sv",
        "original_wrapper.sv":source/"experimental_dual_context_weight_store_p4all_v1.sv",
        "m2_compact_store.sv":ROOT/"verification/experiments/m2_storage_shape/candidate_r1/m2_compact_store.sv",
        "candidate_wrapper.sv":candidate/"candidate_wrapper.sv",
        "tb_m2_xpm_context.sv":candidate/"tb_m2_xpm_context.sv",
        "worker.py":Path(__file__).with_name("run_m2_xpm_context.py"),
        "run_fc_bitmap_mapping.py":Path(__file__).with_name("run_fc_bitmap_mapping.py"),
    }
    for file,path in inputs.items():
        if (destination/file).read_bytes() != Path(local_path(path)).read_bytes():
            raise ValueError("executed input differs "+file)
    if (destination/"candidate_wrapper.sv").read_text() != transform_wrapper((destination/"original_wrapper.sv").read_text()):
        raise ValueError("wrapper transformation differs")
    rows, diagnostics = [], []
    logs = ""
    for row in status["variants"]:
        variant = row["variant"]
        folder = destination/f"variant{variant}"
        if set(row["stages"]) != {"compile","elaborate","run"}:
            raise ValueError("missing actual execution")
        for stage, execution in row["stages"].items():
            if execution["exit"] != 0:
                raise ValueError("tool failed")
            text = (folder/(stage+"_stdout.log")).read_text()
            logs += text
            diag = {"variant":variant,"stage":stage}
            for key,pattern in (("errors",r"(?im)^\s*ERROR:"),("fatals",r"(?im)^\s*Fatal:"),
                                ("critical_warnings",r"(?m)^CRITICAL WARNING:"),("warnings",r"(?m)^WARNING:")):
                diag[key] = len(re.findall(pattern,text))
            if diag["errors"] or diag["fatals"] or diag["critical_warnings"]:
                raise ValueError("unexpected simulator diagnostic")
            diagnostics.append(diag)
        elaboration = (folder/"elaborate_stdout.log").read_text()
        if "xpm.xpm_memory_tdpram" not in elaboration:
            raise ValueError("missing real Context0 TDP XPM")
        if variant and "xpm.xpm_memory_sdpram" not in elaboration:
            raise ValueError("missing real candidate SDP XPM")
        checks = parse_run((folder/"run_stdout.log").read_text(),variant)
        if checks != row["checks"] or checks != json.loads((folder/"checks.json").read_text()):
            raise ValueError("strict reparse disagrees")
        rows.append(checks)
    if [row["variant"] for row in rows] != [0,1,2]:
        raise ValueError("variant coverage")
    with (destination/"comparison.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"checked_at":datetime.now().astimezone().isoformat(),
        "status":"PASS_evidence_audit_baseline_hold_failure_preserved",
        "raw_wrapper_status":status["status"],"raw_files":files,
        "file_count":len(files),"byte_count":sum(f["size"] for f in files),
        "executed_inputs_byte_equal":list(inputs),"checks_reparsed":rows,
        "diagnostics":diagnostics,"environment":environment,
        "no_oom_killed_in_logs":not re.search(r"out of memory|\bKilled\b|oom-kill",logs,re.I),
        "boundaries":["Real Vivado 2022.2 XPM library simulation; not a behavioral RAM stand-in or gate simulation.",
            "Common M2/FC-INT8 dual-context tests pass; old TDP fails the named same-context fill under held-response probe.",
            "Context0 storage and outer context arbitration are unchanged; FC-INT5/SRNN/SCNN Context0 numerical modes not revalidated here.",
            "The final top has refill safety/maintenance ownership. This module counterexample alone does not prove final-top reachability.",
            "512 is an explicit restricted-capacity alternative, not a general 578-row replacement.",
            "No full core RTL adoption, new utilization/STA or throughput claim."]}
    (destination/"terminal_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({k:audit[k] for k in ("status","file_count","byte_count","checks_reparsed","diagnostics","no_oom_killed_in_logs")},indent=2))


if __name__ == "__main__": main()
