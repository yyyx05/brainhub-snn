"""Audit experiment A's own synthesis archive, without inheriting routed STA."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import shlex
import zipfile

from audit_m2_compact_synthesis import resources, timing, hierarchy
from run_scnn_cfg_payload_synthesis import NAME, FUNCTIONAL, RECIPE
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path


def load_worker(path):
    spec = importlib.util.spec_from_file_location("archived_payload_synth_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def report_metrics(report):
    use = resources((report / "utilization_post_synth.rpt").read_text())
    summary = (report / "timing_summary_post_synth.rpt").read_text()
    sta = timing(summary)
    items = hierarchy((report / "utilization_hierarchical_post_synth.rpt").read_text())
    contexts = {
        i: [row for row in items if re.fullmatch(
            rf"g_context\[{i}\]\.g_enabled(?:\.g_original|\.g_compact)?\.u_store", row["name"])]
        for i in (0, 1)
    }
    if len(contexts[0]) != 12 or len(contexts[1]) != 8:
        raise ValueError("context hierarchy coverage")
    totals = {i: sum(row["ramb36"] + row["ramb18"] / 2 for row in contexts[i]) for i in (0, 1)}
    banks = [row for row in items if "g_context[1].g_enabled.g_compact.u_store/" in row["path"]
             and re.fullmatch(r"g_bank\[\d+\]\.u_memory", row["name"])]
    if len(banks) != 32:
        raise ValueError("Context1 bank coverage")
    worst = (report / "timing_setup_paths_post_synth.rpt").read_text()
    path = {key: re.search(r"^\s*" + key + r":\s*(.+)$", worst, re.M)[1]
            for key in ("Source", "Destination", "Data Path Delay", "Logic Levels", "Requirement")}
    checks = {key: int(re.search(r"checking " + key + r" \((\d+)\)", summary)[1])
              for key in ("no_clock", "unconstrained_internal_endpoints", "no_input_delay", "no_output_delay", "loops")}
    methodology = (report / "methodology_post_synth.rpt").read_text()
    rules = [dict(rule=m[1], severity=m[2], count=int(m[3])) for m in re.finditer(
        r"^\|\s*([A-Z]+-\d+)\s*\|\s*(Warning|Critical Warning|Error)\s*\|[^|]*\|\s*(\d+)\s*\|",
        methodology, re.M)]
    total = int(re.search(r"Violations found:\s*(\d+)", methodology)[1])
    if sum(row["count"] for row in rules) != total:
        raise ValueError("methodology warning accounting")
    return dict(stage="post_synth", **use, **sta, context0_bram36eq=totals[0],
                context1_bram36eq=totals[1], other_bram36eq=use["bram36eq"] - totals[0] - totals[1]), dict(
                    worst_path=path, timing_checks=checks, methodology_rules=rules, methodology_total=total), banks


def audit():
    base = Path(local_path(ROOT / RESULT))
    local = base / NAME
    state = json.loads((local / "status.json").read_text())
    if not state.get("end"):
        raise ValueError("active experiment; terminal fetch required")
    if state["status"] == "FAIL_archive_space":
        raise ValueError("raw files remain in scratch; finish archival before auditing")
    prepared = Path(local_path(ROOT / "verification/experiments/scnn_control_payload/core_synthesis_r1"))
    for archived, original in (("worker.py", "launch_worker.py"),
                               ("run_scnn_turnover_synthesis.py", "run_scnn_turnover_synthesis.py")):
        if (local / archived).read_bytes() != (prepared / original).read_bytes():
            raise ValueError("worker bytes changed: " + archived)
    worker = load_worker(local / "run_scnn_turnover_synthesis.py")
    source = base / FUNCTIONAL
    paths = sorted(p.relative_to(source).as_posix() for p in (source / "rtl").rglob("*") if p.is_file())
    actual = sorted(p.relative_to(local).as_posix() for p in (local / "rtl").rglob("*") if p.is_file())
    if actual != paths or len(paths) != 109 or sum(p.endswith(".sv") for p in paths) != 108:
        raise ValueError("source tree differs from actual R4 tree")
    for rel in paths:
        if (local / rel).read_bytes() != (source / rel).read_bytes():
            raise ValueError("actual R4 source bytes changed: " + rel)
    recipes = (worker.TCL, worker.XDC, worker.SHELL)
    for rel in recipes:
        if (local / rel).read_bytes() != (base / RECIPE / rel).read_bytes():
            raise ValueError("original turnover recipe changed: " + rel)

    client = connect()
    code = '''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
excluded={"obj","elab_obj",".Xil","__pycache__"}
files={q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not excluded.intersection(q.relative_to(p).parts)}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files,active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\n" + code)))
    finally:
        client.close()
    files = observation.pop("files")
    if list(local.rglob("*.part")):
        raise ValueError("partial local archive")
    for rel, size in files.items():
        if rel.endswith(".part") or not (local / rel).is_file() or (local / rel).stat().st_size != size:
            raise ValueError("raw archive mismatch: " + rel)
    result = dict(status="NOT_ACCEPTED_payload_synthesis_archived", accepted_synthesis=False,
                  original_status=state, raw_files=len(files), raw_bytes=sum(files.values()),
                  raw_sizes_match=True, no_part=True, source_files_byte_equal=len(paths),
                  input_files_byte_equal=len(paths) + len(recipes), recipe_byte_equal=True,
                  workers_byte_equal=2, remote_observation=observation,
                  boundary="Experiment A only, R4 actual108SV+ROM; unchanged512/15ns/8-thread recipe and source protections. Synthesis is not routed STA, new throughput, full-profile acceptance or production adoption. Old R2 NOT_ACCEPTED retained. No power/bitstream. Netlist protections/trace controls require this candidate's own implementation.")
    report = local / worker.OUT
    log_path = report / "vivado.log"
    if log_path.is_file():
        log = log_path.read_text(errors="replace")
        errors = re.findall(r"^ERROR:.*", log, re.M)
        critical = re.findall(r"^CRITICAL WARNING:.*", log, re.M)
        ordinary = len(re.findall(r"^WARNING:", log, re.M))
        dcp = report / "post_synth.dcp"
        crc = None
        members = None
        if dcp.is_file():
            with zipfile.ZipFile(dcp) as checkpoint:
                crc = checkpoint.testzip() is None
                members = len(checkpoint.infolist())
        tcl = worker.parse_state(report / "vivado_status.txt")
        parsed = worker.result_status(state.get("exit"), tcl, crc, errors, critical)
        result.update(reparsed_status=parsed, errors=errors, critical_warnings=critical,
                      ordinary_warnings=ordinary, tcl_status=tcl,
                      dcp=dict(bytes=dcp.stat().st_size if dcp.is_file() else None, zip_crc_pass=crc, members=members),
                      oom_or_killed_in_log=bool(re.search(r"out of memory|\bOOM\b|\bKilled\b", log, re.I)))
        if state["status"].startswith("PASS"):
            expected = dict(status=parsed, tcl=tcl, checkpoint_crc_pass=crc, errors=errors,
                            critical_warnings=critical, ordinary_warning_count=ordinary,
                            checkpoint_bytes=dcp.stat().st_size, checkpoint_zip_members=members)
            if any(state.get(key) != value for key, value in expected.items()) or not parsed.startswith("PASS"):
                raise ValueError("archived worker reparse differs from original status")
            for filename, value in (("wrapper_status.txt", "status=pass"), ("vivado_exit_code.txt", "0"),
                                    ("error_count.txt", "0"), ("critical_warning_count.txt", "0")):
                if (report / filename).read_text().strip() != value:
                    raise ValueError("actual terminal diagnostic: " + filename)
            if state.get("vivado_exit") != "0" or re.search(r"Abnormal program termination", log):
                raise ValueError("actual Vivado termination")
            for index, rel in enumerate(p for p in paths if p.endswith(".sv")):
                compat = report / "vivado_2022_2_sources" / f"{index:03d}_{Path(rel).name}"
                if compat.read_text() != (local / rel).read_text().replace("`default_nettype none", "`default_nettype wire"):
                    raise ValueError("compatibility source changed: " + rel)
            comparison = []
            for label, folder in (("turnover512_before_A", base / RECIPE), ("turnover512_payload_A", local)):
                metrics, details, banks = report_metrics(folder / worker.OUT)
                comparison.append(dict(variant=label, **metrics))
                if folder == local:
                    result.update(details)
                    with (local / "context1_banks.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(banks[0]))
                        writer.writeheader()
                        writer.writerows(banks)
            result.update(status="PASS_cfg_payload_synthesis_evidence_not_routed_STA", accepted_synthesis=True,
                          compatibility_sources_byte_checked=108, archived_worker_reparse_equal=True,
                          comparison=comparison, delta_candidate_minus_baseline={
                              key: comparison[1][key] - comparison[0][key] for key in ("lut", "ff", "bram36eq", "dsp")})
            with (local / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
                writer.writeheader()
                writer.writerows(comparison)
    elif state["status"].startswith("PASS"):
        raise ValueError("PASS missing actual Vivado log")
    (local / "terminal_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("original_status", "remote_observation")}, indent=2))


if __name__ == "__main__":
    audit()
