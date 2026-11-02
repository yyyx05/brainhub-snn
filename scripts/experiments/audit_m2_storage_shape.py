#!/usr/bin/env python3
"""Fetch and reconcile the isolated M2 memory experiment with its raw reports."""
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import stat
import zipfile

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, fetch_tree, local_path, run
from run_m2_storage_shape import parse_unit


def utilization(text):
    def value(label):
        matches = re.findall(r"^\|\s*"+re.escape(label)+r"\s*\|\s*([0-9.]+)\s*\|", text, re.M)
        if len(matches) != 1:
            raise ValueError("ambiguous utilization row "+label)
        return float(matches[0])
    return {"lut": int(value("Slice LUTs*")), "ff": int(value("Slice Registers")),
            "bram36eq": value("Block RAM Tile"), "bram36": int(value("RAMB36/FIFO*")),
            "bram18": int(value("RAMB18")), "dsp": int(value("DSPs"))}


def timing(text):
    header = text.index("WNS(ns)")
    for line in text[header:].splitlines()[1:]:
        fields = line.split()
        if len(fields) == 12 and re.fullmatch(r"-?\d+\.\d+", fields[0]):
            return dict(zip(("wns", "tns", "setup_fail", "setup_endpoints", "whs", "ths",
                             "hold_fail", "hold_endpoints", "wpws", "tpws", "pulse_fail", "pulse_endpoints"),
                            [float(v) if "." in v else int(v) for v in fields]))
    raise ValueError("timing summary not found")


def main():
    name = "m2_storage_shape_r1"
    destination = Path(local_path(ROOT/RESULT/name))
    remote = REMOTE+"/"+name
    c = connect()
    files = []
    try:
        with c.open_sftp() as s:
            with s.open(remote+"/status.json") as f:
                status = json.load(f)
            if status["status"] != "PASS_module_equivalence_and_synthesis_not_full_core":
                raise ValueError("not successful terminal state")
            fetch_tree(s, remote, destination)

            def visit(path, relative=""):
                for item in s.listdir_attr(path):
                    if item.filename in ("obj", "elab_obj", ".Xil", "__pycache__"):
                        continue
                    rel = relative+item.filename
                    if stat.S_ISDIR(item.st_mode):
                        visit(path+"/"+item.filename, rel+"/")
                    else:
                        local = destination/rel
                        if local.stat().st_size != item.st_size:
                            raise ValueError("size mismatch "+rel)
                        files.append({"path":rel,"size":item.st_size})
            visit(remote)
        environment = {"disk":run(c,"df -B1 /root/autodl-tmp"),
                       "processes":run(c,"ps -eo pid,comm,args"),
                       "dmesg":run(c,"dmesg 2>&1 || true")}
    finally:
        c.close()
    if list(destination.rglob("*.part")):
        raise ValueError("partial download")
    if any(status[stage]["exit"] != 0 for stage in ("build","run","mapping")):
        raise ValueError("nonzero real exit")
    checks = parse_unit((destination/"run.log").read_text())
    if checks != status["checks"]:
        raise ValueError("strict reparse differs")
    inputs = {"baseline.sv": ROOT/RESULT/"scnn_issue_banked_synth_r1/rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv",
              "worker.py":Path(__file__).with_name("run_m2_storage_shape.py"),
              "run_fc_bitmap_mapping.py":Path(__file__).with_name("run_fc_bitmap_mapping.py"),
              "mapping.tcl":Path(__file__).with_name("m2_storage_shape.tcl")}
    for item in ("m2_compact_store.sv","m2_storage_shape.sv","tb_m2_storage_shape.sv"):
        inputs[item] = ROOT/"verification/experiments/m2_storage_shape/candidate_r1"/item
    for name, path in inputs.items():
        if (destination/name).read_bytes() != Path(local_path(path)).read_bytes():
            raise ValueError("executed source differs "+name)
    if (destination/"clock.xdc").read_text() != "create_clock -name core_clk -period 15.000 [get_ports i_clk]\n":
        raise ValueError("unexpected clock")
    mapping = (destination/"mapping.log").read_text()
    diagnostics = {name:len(re.findall(pattern,mapping,re.M)) for name,pattern in (
        ("errors",r"^ERROR:"),("critical_warnings",r"^CRITICAL WARNING:"),("warnings",r"^WARNING:"))}
    if diagnostics != status["mapping_diagnostics"] or diagnostics["errors"] or diagnostics["critical_warnings"]:
        raise ValueError("mapping diagnostics differ")
    build = (destination/"build.log").read_text()
    ordinary = Counter(re.findall(r"%Warning-([A-Z0-9_]+)", build))
    if any(ordinary[k] for k in ("MULTIDRIVEN","LATCH","UNOPTFLAT")):
        raise ValueError("structural warning")
    rows, primitive_shapes, dcps = [], {}, []
    for variant in ("baseline","full578","selected512"):
        row = {"variant":variant, "depth":512 if variant == "selected512" else 578}
        row.update(utilization((destination/(variant+"_utilization.rpt")).read_text()))
        row.update(timing((destination/(variant+"_timing.rpt")).read_text()))
        with (destination/(variant+"_memory_primitives.tsv")).open() as f:
            primitives = list(csv.DictReader(f,delimiter="\t"))
        counts = Counter(p["ref"] for p in primitives)
        if counts["RAMB36E1"] != row["bram36"] or counts["RAMB18E1"] != row["bram18"]:
            raise ValueError("primitive count mismatch")
        if row["bram36eq"] != row["bram36"]+row["bram18"]/2:
            raise ValueError("BRAM equivalence mismatch")
        if (destination/(variant+"_structure.txt")).read_text() != "black_boxes=0\n":
            raise ValueError("black boxes")
        shape = Counter(tuple(p[key] for key in ("ref","read_width_a","read_width_b","write_width_a","write_width_b","ram_mode")) for p in primitives)
        primitive_shapes[variant] = [{"shape":list(k),"count":v} for k,v in shape.items()]
        dcp = destination/(variant+".dcp")
        with zipfile.ZipFile(dcp) as archive:
            bad = archive.testzip()
        if bad:
            raise ValueError("DCP CRC "+bad)
        dcps.append({"path":dcp.name,"size":dcp.stat().st_size,"zip_crc":"PASS"})
        rows.append(row)
    with (destination/"comparison.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    audit = {"checked_at":datetime.now().astimezone().isoformat(),
             "status":"PASS_raw_module_evidence_audit_not_full_core", "raw_files":files,
             "file_count":len(files),"byte_count":sum(f["size"] for f in files),
             "executed_inputs_byte_equal":list(inputs),"checks_reparsed":checks,
             "build_warnings":dict(ordinary),"mapping_diagnostics":diagnostics,
             "no_oom_killed_in_logs":not re.search(r"out of memory|\bKilled\b|oom-kill",build+mapping+(destination/"run.log").read_text(),re.I),
             "dcps":dcps,"rows":rows,"primitive_shapes":primitive_shapes,"environment":environment,
             "boundaries":["Behavioral-array simulation, not XPM simulation or gate simulation.",
                           "578-row variant preserves all addresses and all 40 observable Q14 lanes, not unused padding bits.",
                           "512-row variant excludes addresses 512..577; only current zero-base <=450-source use fits.",
                           "Single M2-only module synthesis, not full core resources, route timing, DDR, or throughput.",
                           "Context0, dual-context arbitration and production RTL unchanged; no candidate adoption."]}
    (destination/"terminal_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({k:audit[k] for k in ("status","file_count","byte_count","checks_reparsed","build_warnings","mapping_diagnostics","rows","dcps","primitive_shapes")},indent=2))


if __name__ == "__main__":
    main()
