"""Audit experiment A's fetched implementation using its own synthesis source."""
import csv
import json
from pathlib import Path
import re
import shlex
import types

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_scnn_cfg_payload_implementation import NAME, SCRATCH, SOURCE, SYNTHESIS, EXPERIMENT, RECIPE, INPUTS
from audit_m2_compact_implementation import resources, violations, report_checks_pass
from audit_m2_compact_synthesis import hierarchy, timing
from audit_m2_compact_route_recovery import fields, route_metrics, reopened_equal, checkpoint_record

OUT = "server_results/final_composition_impl"


def load_worker(path):
    module = types.ModuleType("archived_cfg_payload_implementation_worker")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def inspect_reports(report, state, worker):
    """Read immutable reports; this function never runs EDA or writes evidence."""
    checks, missing, parse_errors = {}, [], []

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
    errors = re.findall(r"^ERROR:.*", log, re.M)
    critical = re.findall(r"^CRITICAL WARNING:.*", log, re.M)
    ordinary = len(re.findall(r"^WARNING:", log, re.M))
    tcl, before, reopened = [fields(read(name)) for name in
                            ("vivado_status.txt", "before_checkpoint_status.txt", "checkpoint_reopen_status.txt")]
    checks["tool_exit_and_diagnostics"] = state.get("exit") == 0 and bool(log) and not errors and not critical
    checks["actual_status_records"] = tcl == state.get("tcl_status") and reopened == state.get("checkpoint_reopen")
    checks["tcl_acceptance"] = (tcl.get("status") == "pass" and tcl.get("route_complete") == "1"
        and all(tcl.get(key) == "0" for key in
                ("black_boxes", "unplaced_primitives", "routing_error_nets", "drc_errors", "drc_critical_warnings")))
    checks["reopen_counts_timing_protection_trace"] = reopened_equal(before, reopened)
    checkpoints = {name: checkpoint_record(report / name)
                   for name in ("post_place_physopt.dcp", "post_route_physopt.dcp")}
    for row in checkpoints.values():
        row["zip_members"] = len(row["members"]) if "members" in row else None
    checks["delivered_checkpoint_crcs"] = all(row.get("crc_pass") is True
        and all(state.get("checkpoints", {}).get(name, {}).get(key) == row.get(key)
                for key in ("bytes", "zip_members", "crc_pass")) for name, row in checkpoints.items())
    original_verdict = worker.implementation_status(state.get("exit"), tcl, reopened,
        checkpoints["post_route_physopt.dcp"].get("crc_pass"), errors, critical)
    checks["archived_worker_strict_reparse"] = (
        original_verdict == state.get("tool_result_status", state.get("status"))
        and errors == state.get("errors") and critical == state.get("critical_warnings")
        and ordinary == state.get("ordinary_warning_count"))
    reports = {}
    for stage, prefix in (("exported", ""), ("reopened", "reopened/")):
        row = dict(resources=parsed(prefix + "utilization_post_route.rpt", resources),
                   timing=parsed(prefix + "timing_summary_post_route.rpt", timing),
                   route=parsed(prefix + "route_status_post_route.rpt", route_metrics))
        row["clocks"] = re.findall(r"^(\w+)\s+(\d+\.\d+)\s+\{([^}]+)\}\s+P\s+\{([^}]+)\}",
                                   read(prefix + "clocks_post_route.rpt"), re.M)
        row["no_valid_exceptions"] = "No valid timing exceptions found." in read(prefix + "exceptions_post_route.rpt")
        checks[stage + "_route_setup_hold_pulse"] = bool(row["timing"] and row["route"]
                                                       and report_checks_pass(row["timing"], row["route"]))
        checks[stage + "_15ns_clock_no_exceptions"] = (
            row["clocks"] == [("core_clk", "15.000", "0.000 7.500", "i_clk")] and row["no_valid_exceptions"])
        reports[stage] = row
    checks["reopened_reports_equal"] = reports["exported"] == reports["reopened"]
    protected, traces = [], {}
    for stage in ("pre_impl", "post_impl"):
        protected.append([line for line in read("protected_primitives_" + stage + ".rpt").splitlines()
                          if line and "=" not in line])
        text = read("trace_payload_control_" + stage + ".rpt")
        controls = [line for line in text.splitlines() if line.startswith("cell=")]
        traces[stage] = dict(cells=len(controls), all_gnd=bool(controls)
                            and all(line.endswith("drivers=GND") for line in controls))
        checks[stage + "_trace"] = (len(controls) == 6270 and traces[stage]["all_gnd"]
                                     and "trace_payload_nonconstant_control_count=0" in text)
    checks["protected_primitive_set"] = len(protected[0]) == len(set(protected[0])) == 15 and protected[0] == protected[1]
    items = parsed("utilization_hierarchical_post_route.rpt", hierarchy) or []
    arrays = [row["path"] for row in items if row["module"] == "unified_dual_role_h40_stationary_array12_candidate_v1"]
    tiles = [row["path"] for row in items if re.fullmatch(r"g_tile\[\d+\]\.u_tile", row["name"])]
    checks["one_array_twelve_tiles"] = len(arrays) == 1 and len(tiles) == 12
    contexts = {i: [row for row in items if re.fullmatch(r"g_context\[" + str(i)
                   + r"\]\.g_enabled\.g_(?:original|compact)\.u_store", row["name"])] for i in (0, 1)}
    totals = {i: sum(row["ramb36"] + row["ramb18"] / 2 for row in contexts[i]) for i in (0, 1)}
    banks = [row for row in items if "g_context[1].g_enabled.g_compact.u_store/" in row["path"]
             and re.fullmatch(r"g_bank\[\d+\]\.u_memory", row["name"])]
    checks["context_mapping"] = (len(contexts[0]) == 12 and len(contexts[1]) == 8 and len(banks) == 32
        and totals == {0: 288, 1: 80} and all(row["ramb36"] == (1 if row["name"].startswith("g_bank[3]") else 3)
                                            and row["ramb18"] == 0 for row in banks))
    diagnostics = {name: parsed(name + "_post_route.rpt", violations) for name in ("drc", "methodology")}
    checks["no_severe_drc_methodology"] = all(record is not None
        and all(row["severity"] == "Warning" for row in record["rules"]) for record in diagnostics.values())
    timing_text = read("timing_summary_post_route.rpt")
    constraints = {key: int(match[1]) if (match := re.search(r"checking " + key + r" \((\d+)\)", timing_text))
                   else None for key in ("no_clock", "unconstrained_internal_endpoints", "no_input_delay", "no_output_delay", "loops")}
    checks["internal_constraints"] = all(constraints[key] == 0 for key in ("no_clock", "unconstrained_internal_endpoints", "loops"))
    worst = {key: match[1] if (match := re.search(r"^\s*" + key + r":\s*(.+)$",
             read("timing_setup_paths_post_route.rpt"), re.M)) else None
             for key in ("Source", "Destination", "Data Path Delay", "Logic Levels", "Requirement")}
    checks["all_reports_parsed"] = not missing and not parse_errors
    evidence = dict(checks=checks, checkpoints=checkpoints, tcl_status=tcl, before_checkpoint=before,
        checkpoint_reopen=reopened, archived_worker_verdict=original_verdict, reports=reports,
        protected_primitives=protected[0], trace_controls=traces, physical_array_paths=arrays, physical_tile_paths=tiles,
        context_bram36eq=totals, diagnostics=diagnostics, timing_checks=constraints, worst_setup_path=worst,
        errors=errors, critical_warnings=critical, ordinary_warnings=ordinary, missing_reports=missing, parse_errors=parse_errors,
        oom_or_disk_error_lines=re.findall(r"^.*(?:out of memory|\bOOM\b|\bKilled\b|ENOSPC|No space left|Disk quota exceeded).*$", log, re.I | re.M))
    return evidence, banks


def observe_archive():
    code = '''import datetime,json,pathlib,shutil,subprocess,zipfile
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);source=pathlib.Path(SOURCE)
state=json.loads((r/'status.json').read_text())
if not state.get('end') or state['status']=='running':raise ValueError('not terminal')
def inventory(p):
 return {q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
with zipfile.ZipFile(source) as z:crc=z.testzip() is None
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),state=state,files=inventory(r),scratch_files=inventory(s),active_eda=active,
 source_bytes=source.stat().st_size,source_crc_pass=crc,data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free,
 dmesg_exit=k.returncode,dmesg_error=k.stderr)))
'''
    constants = "ROOT=" + repr(REMOTE + "/" + NAME) + "\nSCRATCH=" + repr(SCRATCH) + "\nSOURCE=" + repr(REMOTE + "/" + SOURCE) + "\n"
    client = connect()
    try:
        return json.loads(run(client, "python3 -c " + shlex.quote(constants + code)))
    finally:
        client.close()


def audit():
    base = Path(local_path(ROOT / RESULT))
    local, prepared = base / NAME, Path(local_path(ROOT / EXPERIMENT))
    state = json.loads((local / "status.json").read_text())
    if not state.get("end") or state["status"] == "running":
        raise ValueError("natural terminal fetch required")
    observation = observe_archive()
    if observation["active_eda"] or observation["state"] != state:
        raise ValueError("EDA still active or local status stale; defer audit")
    files = observation["files"]
    if list(local.rglob("*.part")) or any(name.endswith(".part") for name in files):
        raise ValueError("partial archive")
    for rel, size in files.items():
        if not (local / rel).is_file() or (local / rel).stat().st_size != size:
            raise ValueError("incomplete fetch: " + rel)
    if any(files.get(rel) != size for rel, size in observation["scratch_files"].items()):
        raise ValueError("scratch archive incomplete; preserve failure and fetch missing originals before audit")
    checks = {}
    checks["six_prepared_input_bytes"] = all((local / rel).read_bytes() == (prepared / rel).read_bytes() for rel in INPUTS)
    checks["original_recipe_bytes"] = all((local / rel).read_bytes() == (base / RECIPE / rel).read_bytes() for rel in INPUTS[:3])
    checks["actual_original_worker_bytes"] = ((local / "run_scnn_turnover_implementation.py").read_bytes()
                                             == (base / RECIPE / "launch_worker.py").read_bytes())
    worker = load_worker(local / "run_scnn_turnover_implementation.py")
    checks["only_original_export_reopen_transform"] = ((local / "implementation.tcl").read_text()
        == worker.implementation_tcl((local / "original_impl.tcl").read_text()))
    synthesis = json.loads((base / SYNTHESIS / "terminal_audit.json").read_text())
    source = checkpoint_record(base / SOURCE)
    checks["accepted_own_A_synthesis_source"] = (
        synthesis["status"] == "PASS_cfg_payload_synthesis_evidence_not_routed_STA"
        and source.get("crc_pass") is True and observation["source_crc_pass"] is True
        and source.get("bytes") == synthesis["dcp"]["bytes"] == observation["source_bytes"])
    prep = json.loads((local / "preparation.json").read_text())
    checks["actual_source_and_command"] = (state.get("source") == REMOTE + "/" + SOURCE and state.get("scratch") == SCRATCH
        and prep.get("source") == SOURCE and prep.get("source_checkpoint", {}).get("bytes") == source.get("bytes")
        and state.get("command") == [worker.VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source",
                                    REMOTE + "/" + NAME + "/implementation.tcl", "-tclargs", SCRATCH, REMOTE + "/" + SOURCE])
    evidence, banks = inspect_reports(local / OUT, state, worker)
    checks.update(evidence.pop("checks"))
    samples = json.loads((local / "disk_samples.json").read_text())
    minima = {key: min(row[key] for row in samples) if samples else None for key in ("data_free", "tmp_free")}
    checks["disk_samples_match"] = (bool(samples) and state.get("minimum_data_free") == minima["data_free"]
                                    and state.get("minimum_tmp_free") == minima["tmp_free"])
    checks["scratch_preserved"] = state.get("scratch_preserved") is True
    accepted = state["status"] == "PASS_candidate_internal15ns_checkpoint_reopened" and all(value is True for value in checks.values())
    result = dict(status="PASS_cfg_payload_implementation_delivery" if accepted else "NOT_ACCEPTED_cfg_payload_implementation_archived",
        accepted_implementation=accepted, original_status=state, checks=checks, raw_files=len(files), raw_bytes=sum(files.values()),
        all_raw_sizes_match=True, no_part=True, source_checkpoint=source, disk_sample_count=len(samples), sampled_minimum_free=minima,
        remote_observation=observation, **evidence,
        boundary="A13payload-mux change only, own108SV+ROM/512 synthesis DCP. Original8threads/15ns/protection/optimization/reopen/all-log CW rejection unchanged. Single internal OOC, no IO delays, board Fmax, multi-seed stability, new throughput or production adoption. Old R2 NOT_ACCEPTED and all scratch/failures preserved. One-second disk samples may miss short peaks; log absence of OOM is not kernel evidence.")
    exported = evidence["reports"]["exported"]
    if exported["resources"] and exported["timing"]:
        result["other_bram36eq"] = exported["resources"]["bram36eq"] - sum(evidence["context_bram36eq"].values())
        previous = base / RECIPE
        previous_audit = json.loads((previous / "terminal_audit.json").read_text())
        table = [dict(variant="turnover512_before_A", stage="post_route",
                      accepted_implementation=previous_audit["accepted_implementation"],
                      **resources((previous / OUT / "utilization_post_route.rpt").read_text()),
                      **timing((previous / OUT / "timing_summary_post_route.rpt").read_text())),
                 dict(variant="turnover512_cfg_payload_A", stage="post_route", accepted_implementation=accepted,
                      **exported["resources"], **exported["timing"])]
        result["same_stage_comparison"] = table
        result["delta_A_minus_turnover"] = {key: table[1][key] - table[0][key] for key in ("lut", "ff", "bram36eq", "dsp")}
        with (local / "implementation_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (local / "terminal_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if banks:
        with (local / "context1_banks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(banks[0]))
            writer.writeheader()
            writer.writerows(banks)
    print(json.dumps({key: result[key] for key in ("status", "accepted_implementation", "checks", "raw_files", "raw_bytes",
                                                  "errors", "critical_warnings", "missing_reports", "parse_errors")}, indent=2))


if __name__ == "__main__":
    audit()
