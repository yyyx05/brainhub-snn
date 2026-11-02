"""Audit R1's completed reports without accepting its failed checkpoint delivery."""
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import zipfile

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from audit_m2_compact_synthesis import hierarchy, timing

NAME = "m2_compact_impl_r1"
OUT = "server_results/final_composition_impl"


def checkpoint(path):
    if not path.is_file():
        return dict(present=False, bytes=None, crc_pass=None)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        return dict(present=True, bytes=path.stat().st_size, crc_pass=bad is None,
                    bad_member=bad, members=[dict(name=i.filename, bytes=i.file_size,
                    compressed_bytes=i.compress_size) for i in archive.infolist()])


def resources(text):
    result = {}
    for key, label in (("lut", "Slice LUTs"), ("lut_logic", "LUT as Logic"),
                       ("lut_memory", "LUT as Memory"), ("ff", "Slice Registers"),
                       ("bram36eq", "Block RAM Tile"), ("dsp", "DSPs")):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|", text, re.M)
        if not match:
            raise ValueError("missing resource: " + label)
        result[key] = float(match[1]) if "." in match[1] else int(match[1])
    return result


def violations(text):
    rules = [dict(rule=m[1], severity=m[2], count=int(m[3])) for m in re.finditer(
        r"^\|\s*([A-Z]+-\d+)\s*\|\s*(Warning|Critical Warning|Error)\s*\|[^|]*\|\s*(\d+)\s*\|", text, re.M)]
    total = int(re.search(r"Violations found:\s*(\d+)", text)[1])
    if sum(x["count"] for x in rules) != total:
        raise ValueError("incomplete diagnostic rule table")
    return dict(total=total, rules=rules)


def report_checks_pass(metrics, route):
    return (all(metrics[k] >= 0 for k in ("wns_ns", "whs_ns", "wpws_ns"))
            and all(metrics[k] == 0 for k in ("setup_failing", "hold_failing", "pulse_failing"))
            and route["routable"] == route["fully_routed"] and route["errors"] == 0)


def main():
    archive = Path(local_path(ROOT / RESULT))
    local, previous = archive / NAME, archive / "scnn_issue_banked_impl_r1"
    remote_code = r'''
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path(ROOT)
files={p.relative_to(r).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
 for p in r.rglob('*') if p.is_file() and not set(p.relative_to(r).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
disk={str(p):dict(total=shutil.disk_usage(p).total,used=shutil.disk_usage(p).used,free=shutil.disk_usage(p).free)
 for p in (r,pathlib.Path('/tmp'),pathlib.Path('/dev/shm'))}
scratch=[dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size)
 for p in r.rglob('*') if p.is_file() and '.Xil' in p.relative_to(r).parts]
print(json.dumps(dict(checked_at=datetime.datetime.now().astimezone().isoformat(),files=files,
 disk=disk,active_eda=active,dmesg_exit=k.returncode,dmesg_error=k.stderr,scratch_files=scratch)))
'''
    client = connect()
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\n" + remote_code)))
    finally:
        client.close()
    assert not observation["active_eda"]
    for relative, info in observation["files"].items():
        assert (local / relative).is_file() and (local / relative).stat().st_size == info["size"], relative
    assert not list(local.rglob("*.part"))
    recipe = ["vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl",
              "server/run_unified_h40_final_composition_v1_impl_from_dcp.sh"]
    for relative in recipe:
        assert (local / relative).read_bytes() == (previous / relative).read_bytes()
        assert (local / relative).read_bytes() == (local / "launch_inputs" / relative).read_bytes()
    report = local / OUT
    def read(name):
        return (report / name).read_text(errors="replace")
    states = {name: read(name).strip() for name in ("wrapper_status.txt", "vivado_exit_code.txt",
              "error_count.txt", "critical_warning_count.txt", "start_time.txt", "end_time.txt")}
    assert states["wrapper_status.txt"] == "status=fail" and states["vivado_exit_code.txt"] == "1"
    assert not (report / "vivado_status.txt").exists()
    log = read("vivado.log")
    errors, critical = [re.findall(r"^" + label + r":.*$", log, re.M) for label in ("ERROR", "CRITICAL WARNING")]
    assert len(errors) == int(states["error_count.txt"]) == 1
    assert len(critical) == int(states["critical_warning_count.txt"]) == 0
    assert "Common 17-141" in errors[0] and "zip archive" in errors[0]
    assert '"write_checkpoint -force [file join $out_dir post_route_physopt.dcp]"' in log
    launch = json.loads((local / "launch_observation.json").read_text())
    source = archive / "m2_compact_synth_r2/server_results/final_composition_synth/post_synth.dcp"
    assert launch["source_dcp"] == REMOTE + "/m2_compact_synth_r2/server_results/final_composition_synth/post_synth.dcp"
    source_checkpoint = checkpoint(source)
    assert source_checkpoint["crc_pass"] and source_checkpoint["bytes"] == launch["source_dcp_bytes"] == 142604223
    dcps = {name: checkpoint(report / name) for name in ("post_place_physopt.dcp", "post_route_physopt.dcp")}
    assert dcps["post_place_physopt.dcp"]["crc_pass"] and not dcps["post_route_physopt.dcp"]["present"]
    assert [p for p in observation["files"] if p.endswith(".dcp")] == [OUT + "/post_place_physopt.dcp"]
    route_text = read("route_status_post_route.rpt")
    route = {key: int(re.search(r"# of " + label + r"[^:]*:\s*(\d+)", route_text)[1])
             for key, label in (("routable", "routable nets"), ("fully_routed", "fully routed nets"),
                                ("errors", "nets with routing errors"))}
    metrics = timing(read("timing_summary_post_route.rpt"))
    rows = []
    for label, folder in (("accepted_banked578", previous), ("compact512_failed_checkpoint_diagnostic", local)):
        rpt = folder / OUT
        rows.append(dict(variant=label, stage="post_route_physopt", accepted=folder == previous,
                         **resources((rpt / "utilization_post_route.rpt").read_text()),
                         **timing((rpt / "timing_summary_post_route.rpt").read_text())))
    items = hierarchy(read("utilization_hierarchical_post_route.rpt"))
    arrays = [x["path"] for x in items if x["module"] == "unified_dual_role_h40_stationary_array12_candidate_v1"]
    tiles = [x["path"] for x in items if re.fullmatch(r"g_tile\[\d+\]\.u_tile", x["name"])]
    assert len(arrays) == 1 and len(tiles) == 12
    banks = [x for x in items if "g_context[1].g_enabled.g_compact.u_store/" in x["path"]
             and re.fullmatch(r"g_bank\[\d+\]\.u_memory", x["name"])]
    context = {i: [x for x in items if re.fullmatch(r"g_context\[" + str(i) + r"\]\.g_enabled\.g_(?:original|compact)\.u_store", x["name"])] for i in (0, 1)}
    assert len(context[0]) == 12 and len(context[1]) == 8 and len(banks) == 32
    totals = {i: sum(x["ramb36"] + x["ramb18"] / 2 for x in context[i]) for i in (0, 1)}
    assert totals == {0: 288, 1: 80}
    assert all(x["ramb36"] == (1 if "g_bank[3]" in x["name"] else 3) and x["ramb18"] == 0 for x in banks)
    protected = []
    trace = []
    for stage in ("pre_impl", "post_impl"):
        lines = read("protected_primitives_" + stage + ".rpt").splitlines()
        names = [line for line in lines if "=" not in line]
        assert len(names) == 15 and len(set(names)) == 15
        assert "protected_primitive_count=15" in lines
        protected.append(names)
        text = read("trace_payload_control_" + stage + ".rpt")
        payload = int(re.search(r"trace_payload_cell_count=(\d+)", text)[1])
        nonconstant = int(re.search(r"trace_payload_nonconstant_control_count=(\d+)", text)[1])
        controls = [line for line in text.splitlines() if line.startswith("cell=")]
        assert payload == len(controls) == 6270 and nonconstant == 0
        assert all(line.endswith("drivers=GND") for line in controls)
        trace.append(dict(stage=stage, cells=payload, nonconstant_controls=nonconstant))
    assert protected[0] == protected[1]
    assert "protection_preserved=1" in read("protected_primitives_post_impl.rpt")
    clocks = re.findall(r"^(\w+)\s+(\d+\.\d+)\s+\{([^}]+)\}\s+P\s+\{([^}]+)\}", read("clocks_post_route.rpt"), re.M)
    assert clocks == [("core_clk", "15.000", "0.000 7.500", "i_clk")]
    assert "No valid timing exceptions found." in read("exceptions_post_route.rpt")
    checks = {name: int(re.search(r"checking " + name + r" \((\d+)\)", read("timing_summary_post_route.rpt"))[1])
              for name in ("no_clock", "unconstrained_internal_endpoints", "no_input_delay", "no_output_delay", "loops")}
    path = {key: re.search(r"^\s*" + key + r":\s*(.+)$", read("timing_setup_paths_post_route.rpt"), re.M)[1]
            for key in ("Source", "Destination", "Data Path Delay", "Logic Levels", "Requirement")}
    diagnostics = {label: violations(read(label + "_post_route.rpt")) for label in ("drc", "methodology")}
    assert all(rule["severity"] == "Warning" for d in diagnostics.values() for rule in d["rules"])
    blackbox_section = read("utilization_post_route.rpt").rsplit("9. Black Boxes", 1)[1].split("10. Instantiated", 1)[0]
    assert re.search(r"\|\s*Black Boxes\s*\|\s*0\s*\|", blackbox_section) or not re.search(r"\|.*\d+.*\|", blackbox_section)
    audit = dict(status="FAIL_final_checkpoint_write_reports_archived", accepted_implementation=False,
        observed_report_checks_met=report_checks_pass(metrics, route), raw_status=states,
        final_tcl_status=None, raw_file_count=len(observation["files"]),
        raw_bytes=sum(x["size"] for x in observation["files"].values()), all_remote_sizes_equal=True, partial_files=[],
        wall_seconds=(dt.datetime.fromisoformat(states["end_time.txt"]) - dt.datetime.fromisoformat(states["start_time.txt"])).total_seconds(),
        actual_scripts_byte_equal_banked_and_launch=recipe, source_dcp=launch["source_dcp"],
        source_checkpoint=source_checkpoint, checkpoints=dcps, errors=errors, critical_warnings=critical,
        ordinary_warning_count=len(re.findall(r"^WARNING:", log, re.M)), diagnostics=diagnostics,
        diagnostic_resources=rows[1], diagnostic_timing=metrics, route=route,
        diagnostic_comparison=rows, context_bram36eq=totals, other_bram36eq=rows[1]["bram36eq"]-sum(totals.values()),
        protected_primitives=protected[0], trace_control=trace, clocks=clocks,
        physical_array_paths=arrays, physical_tile_paths=tiles,
        valid_timing_exceptions=0, timing_checks=checks, worst_setup_path=path,
        post_route_black_boxes_reported=0, final_unplaced_query_count=None,
        final_unplaced_boundary="Tcl computed but did not persist the count before checkpoint error; do not infer zero from echoed Tcl.",
        oom_or_killed_log_lines=re.findall(r"^.*(?:Out of memory|\bOOM\b|\bKilled\b).*$", log, re.M | re.I),
        direct_disk_full_log_lines=re.findall(r"^.*(?:ENOSPC|No space left|Disk quota exceeded).*$", log, re.M | re.I),
        abnormal_termination_log_lines=re.findall(r"^.*Abnormal program termination.*$", log, re.M),
        remote_observation=observation,
        boundary="Reports are post-route diagnostics from a complete-input design, not an accepted implementation delivery. Final routed DCP is absent, so no reopen verification is possible. WNS0.000 is the displayed report precision, not positive margin. Capacity512 differs from578. Original15ns/protection retained. OOC I/O delays absent. No direct ENOSPC attribution; dmesg unavailable. No production edit or EDA restart.")
    (local / "terminal_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for name, data in (("diagnostic_comparison.csv", rows), ("context1_banks.csv", banks)):
        with (local / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("remote_observation", "source_checkpoint", "checkpoints", "protected_primitives")}, indent=2))
    print(json.dumps({k: {n: v for n, v in record.items() if n != "members"} for k, record in dcps.items()}, indent=2))


if __name__ == "__main__":
    main()
