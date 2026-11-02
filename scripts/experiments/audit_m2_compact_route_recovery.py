"""Audit fetched route-only recovery, including the delivered DCP's reopen."""
import csv
import json
from pathlib import Path
import re
import shlex
import zipfile
import zlib

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_m2_compact_route_recovery import NAME, OUT, SOURCE, SCRATCH, TCL, WORKER, ORIGINAL_TCL, route_tcl
from audit_m2_compact_implementation import checkpoint, resources, violations, report_checks_pass
from audit_m2_compact_synthesis import hierarchy, timing


def fields(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def route_metrics(text):
    return {key: int(re.search(r"# of " + label + r"[^:]*:\s*(\d+)", text)[1])
            for key, label in (("routable", "routable nets"), ("fully_routed", "fully routed nets"),
                               ("errors", "nets with routing errors"))}


def reopened_equal(before, after):
    keys = ("primitive_count", "unplaced_primitives", "black_boxes", "wns_ns", "whs_ns")
    if not all(key in before and key in after for key in keys):
        return False
    return (all(before[key] == after[key] for key in keys)
            and after.get("status") == "pass" and after.get("protected_set_preserved") == "1"
            and after.get("trace_payload_cell_count") == "6270"
            and after.get("trace_payload_nonconstant_control_count") == "0")


def delivery_pass(state, checks):
    return (state.get("status") == "PASS_route_recovery_checkpoint_reopened"
            and state.get("exit") == 0 and bool(checks) and all(value is True for value in checks.values()))


def checkpoint_record(path):
    try:
        return checkpoint(path)
    except (zipfile.BadZipFile, RuntimeError, zlib.error, EOFError) as error:
        return dict(present=True, bytes=path.stat().st_size, crc_pass=False, archive_error=str(error))


def nonnegative_slacks(state):
    try:
        return all(float(state[key]) >= 0 for key in ("wns_ns", "whs_ns"))
    except (KeyError, TypeError, ValueError):
        return False


def main():
    local = Path(local_path(ROOT / RESULT / NAME))
    original = Path(local_path(ROOT / RESULT / "m2_compact_impl_r1" / ORIGINAL_TCL))
    client = connect()
    try:
        code = r'''
import datetime,json,pathlib,shutil,subprocess,zipfile
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);source=pathlib.Path(SOURCE)
state=json.loads((r/'status.json').read_text())
if state['status']=='running':raise ValueError('still running: do not audit a partial run')
def inventory(root):
 return {p.relative_to(root).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
  for p in root.rglob('*') if p.is_file() and not set(p.relative_to(root).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
with zipfile.ZipFile(source) as z:source_crc=z.testzip() is None
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(r),
 scratch_files=inventory(s),active_eda=active,source_bytes=source.stat().st_size,source_crc_pass=source_crc,
 data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free,dmesg_exit=k.returncode,dmesg_error=k.stderr)))
'''
        remote = "ROOT=" + repr(REMOTE + "/" + NAME) + "\nSCRATCH=" + repr(SCRATCH) + "\nSOURCE=" + repr(REMOTE + "/" + SOURCE) + "\n"
        observation = json.loads(run(client, "python3 -c " + shlex.quote(remote + code)))
    finally:
        client.close()
    if observation["active_eda"]:
        raise ValueError("EDA still active; defer terminal audit")
    for path, info in observation["files"].items():
        if not (local / path).is_file() or (local / path).stat().st_size != info["size"]:
            raise ValueError("fetch incomplete: " + path)
    if list(local.rglob("*.part")):
        raise ValueError("partial fetch files remain")
    state = json.loads((local / "status.json").read_text())
    if state["status"] == "running":
        raise ValueError("local status is stale")
    checks, parse_errors, missing = {}, [], []
    checks["original_script_byte_equal"] = (local / "original_impl.tcl").read_bytes() == original.read_bytes()
    checks["actual_worker_byte_equal"] = (local / WORKER).read_bytes() == Path(local_path(ROOT / WORKER)).read_bytes()
    checks["actual_tcl_byte_equal_prepared"] = (local / TCL).read_bytes() == Path(local_path(ROOT / TCL)).read_bytes()
    checks["actual_tcl_is_route_only_transform"] = (local / TCL).read_text() == route_tcl(original.read_text())
    checks["source_original_size_and_crc"] = observation["source_bytes"] == 176061306 and observation["source_crc_pass"] is True
    source = checkpoint_record(Path(local_path(ROOT / RESULT / SOURCE)))
    checks["local_source_crc"] = source["bytes"] == 176061306 and source["crc_pass"] is True
    scratch_missing = [path for path, info in observation["scratch_files"].items()
                       if path not in observation["files"] or observation["files"][path]["size"] != info["size"]]
    checks["scratch_archive_complete"] = not scratch_missing and state.get("scratch_preserved") is True
    report = local / OUT
    def read(name):
        path = report / name
        return path.read_text(errors="replace") if path.is_file() else ""
    def parsed(name, parser):
        text = read(name)
        if not text:
            missing.append(name)
            return None
        try:
            return parser(text)
        except (ValueError, IndexError, TypeError, KeyError, AttributeError) as error:
            parse_errors.append(dict(report=name, error=str(error)))
            return None
    log = read("vivado.log")
    errors = re.findall(r"^ERROR:.*$", log, re.M)
    critical = re.findall(r"^CRITICAL WARNING:.*$", log, re.M)
    checks["log_errors_and_critical_warnings_zero"] = not errors and not critical and bool(log)
    tcl, before, reopened = [fields(read(name)) for name in
                            ("vivado_status.txt", "before_checkpoint_status.txt", "checkpoint_reopen_status.txt")]
    checks["actual_status_matches_wrapper"] = tcl == state.get("tcl_status") and reopened == state.get("checkpoint_reopen")
    checks["tcl_pass_and_exact_timing"] = (tcl.get("status") == "pass"
        and all(tcl.get(key) == "0" for key in ("black_boxes", "unplaced_primitives", "routing_error_nets", "drc_errors", "drc_critical_warnings"))
        and tcl.get("route_complete") == "1" and nonnegative_slacks(tcl))
    checks["checkpoint_reopen_matches"] = reopened_equal(before, reopened)
    dcp = checkpoint_record(report / "post_route_physopt.dcp")
    checks["final_dcp_crc"] = dcp.get("crc_pass") is True and state.get("output_crc_pass") is True and state.get("output_bytes") == dcp.get("bytes")
    reports = {}
    for stage, prefix in (("exported", ""), ("reopened", "reopened/")):
        record = dict(resources=parsed(prefix + "utilization_post_route.rpt", resources),
                      timing=parsed(prefix + "timing_summary_post_route.rpt", timing),
                      route=parsed(prefix + "route_status_post_route.rpt", route_metrics))
        record["clocks"] = re.findall(r"^(\w+)\s+(\d+\.\d+)\s+\{([^}]+)\}\s+P\s+\{([^}]+)\}",
                                      read(prefix + "clocks_post_route.rpt"), re.M)
        record["no_valid_exceptions"] = "No valid timing exceptions found." in read(prefix + "exceptions_post_route.rpt")
        checks[stage + "_route_setup_hold_pulse"] = bool(record["timing"] and record["route"] and report_checks_pass(record["timing"], record["route"]))
        checks[stage + "_clock_and_exceptions"] = record["clocks"] == [("core_clk", "15.000", "0.000 7.500", "i_clk")] and record["no_valid_exceptions"]
        reports[stage] = record
    checks["reopened_reports_equal"] = reports["exported"] == reports["reopened"]
    protected = []
    traces = {}
    for stage in ("pre_impl", "post_impl"):
        names = [line for line in read("protected_primitives_" + stage + ".rpt").splitlines() if line and "=" not in line]
        protected.append(names)
        text = read("trace_payload_control_" + stage + ".rpt")
        controls = [line for line in text.splitlines() if line.startswith("cell=")]
        traces[stage] = dict(cells=len(controls), all_gnd=bool(controls) and all(line.endswith("drivers=GND") for line in controls))
        checks[stage + "_trace"] = (len(controls) == 6270 and traces[stage]["all_gnd"]
            and "trace_payload_cell_count=6270" in text and "trace_payload_nonconstant_control_count=0" in text)
    checks["protected_set_preserved"] = len(protected[0]) == len(set(protected[0])) == 15 and protected[0] == protected[1]
    items = parsed("utilization_hierarchical_post_route.rpt", hierarchy) or []
    arrays = [x["path"] for x in items if x["module"] == "unified_dual_role_h40_stationary_array12_candidate_v1"]
    tiles = [x["path"] for x in items if re.fullmatch(r"g_tile\[\d+\]\.u_tile", x["name"])]
    checks["one_array_twelve_tiles"] = len(arrays) == 1 and len(tiles) == 12
    context = {i: [x for x in items if re.fullmatch(r"g_context\[" + str(i) + r"\]\.g_enabled\.g_(?:original|compact)\.u_store", x["name"])] for i in (0, 1)}
    totals = {i: sum(x["ramb36"] + x["ramb18"] / 2 for x in context[i]) for i in (0, 1)}
    banks = [x for x in items if "g_context[1].g_enabled.g_compact.u_store/" in x["path"] and re.fullmatch(r"g_bank\[\d+\]\.u_memory", x["name"])]
    checks["context_mapping_preserved"] = (len(context[0]) == 12 and len(context[1]) == 8 and len(banks) == 32
        and totals == {0: 288, 1: 80} and all(x["ramb36"] == (1 if x["name"].startswith("g_bank[3]") else 3) and x["ramb18"] == 0 for x in banks))
    diagnostics = {name: parsed(name + "_post_route.rpt", violations) for name in ("drc", "methodology")}
    checks["no_drc_methodology_severe"] = all(d is not None and all(x["severity"] == "Warning" for x in d["rules"]) for d in diagnostics.values())
    timing_text = read("timing_summary_post_route.rpt")
    constraints = {name: int(match[1]) if (match := re.search(r"checking " + name + r" \((\d+)\)", timing_text)) else None
                   for name in ("no_clock", "unconstrained_internal_endpoints", "no_input_delay", "no_output_delay", "loops")}
    checks["internal_constraints_complete"] = all(constraints[x] == 0 for x in ("no_clock", "unconstrained_internal_endpoints", "loops"))
    worst = {key: match[1] if (match := re.search(r"^\s*" + key + r":\s*(.+)$", read("timing_setup_paths_post_route.rpt"), re.M)) else None
             for key in ("Source", "Destination", "Data Path Delay", "Logic Levels", "Requirement")}
    samples_path = local / "disk_samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.is_file() else []
    disk = {key: min(x[key] for x in samples) if samples else None for key in ("data_free", "tmp_free")}
    checks["disk_samples_match"] = bool(samples) and state.get("minimum_data_free") == disk["data_free"] and state.get("minimum_tmp_free") == disk["tmp_free"]
    checks["all_reports_parsed"] = not missing and not parse_errors
    audit = dict(status="PASS_route_recovery_delivery" if delivery_pass(state, checks) else "NOT_ACCEPTED_route_recovery_archived",
        accepted_implementation=delivery_pass(state, checks), raw_status=state, checks=checks,
        raw_file_count=len(observation["files"]), raw_bytes=sum(x["size"] for x in observation["files"].values()),
        all_remote_sizes_equal=True, partial_files=[], scratch_archive_missing=scratch_missing,
        source_checkpoint=source, final_checkpoint=dcp, tcl_status=tcl, before_checkpoint=before, checkpoint_reopen=reopened,
        reports=reports, context_bram36eq=totals, context_store_counts={i:len(rows) for i,rows in context.items()},
        protected_primitives=protected[0], trace_controls=traces, physical_array_paths=arrays, physical_tile_paths=tiles,
        diagnostics=diagnostics, timing_checks=constraints, worst_setup_path=worst, errors=errors, critical_warnings=critical,
        ordinary_warning_count=len(re.findall(r"^WARNING:", log, re.M)), missing_reports=missing, parse_errors=parse_errors,
        disk_sample_count=len(samples), sampled_minimum_free=disk,
        observed_peak_additional_bytes={key:samples[0][key]-disk[key] for key in disk} if samples else {},
        oom_or_disk_error_lines=re.findall(r"^.*(?:Out of memory|\bOOM\b|\bKilled\b|ENOSPC|No space left|Disk quota exceeded).*$", log, re.M | re.I),
        remote_observation=observation,
        boundary="This run resumes original saved placement through unchanged routing/route-physopt, not synthesis or placement. Capacity512 differs from578. Single internal15ns OOC implementation; input/output delays absent. DCP reopen and equal reports are not formal equivalence, board timing or a unique historical root-cause proof. One-second sampling may miss short disk peaks. Original failures, production and historical tables unchanged.")
    if reports["exported"]["resources"]:
        audit["other_bram36eq"] = reports["exported"]["resources"]["bram36eq"] - sum(totals.values())
    (local / "terminal_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if banks:
        with (local / "context1_banks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(banks[0]))
            writer.writeheader()
            writer.writerows(banks)
    print(json.dumps({key:audit[key] for key in ("status", "accepted_implementation", "checks", "raw_file_count", "raw_bytes", "errors", "critical_warnings", "missing_reports", "parse_errors")}, indent=2))


if __name__ == "__main__":
    main()
