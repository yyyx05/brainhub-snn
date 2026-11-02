"""Archive and audit the incomplete-input R1 synthesis without accepting its QoR."""
import datetime
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_m2_compact_synthesis import selected_tcl, TCL

NAME = "m2_compact_synth_r1"
OUT = "server_results/final_composition_synth"
ROM = "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"


def main():
    local = Path(local_path(ROOT / RESULT / NAME))
    remote_code = r'''
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path(ROOT)
files={p.relative_to(r).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
       for p in r.rglob('*') if p.is_file() and not set(p.relative_to(r).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
processes=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):
   processes.append(dict(pid=p.name,comm=comm))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
kernel=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 files=files,free_bytes=shutil.disk_usage(r).free,active_eda=processes,
 dmesg_exit=kernel.returncode,dmesg_error=kernel.stderr)))
'''
    client = connect()
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\n" + remote_code)))
    finally:
        client.close()
    assert not observation["active_eda"], observation["active_eda"]
    for relative, meta in observation["files"].items():
        path = local / relative
        assert path.is_file() and path.stat().st_size == meta["size"], relative
    assert not list(local.rglob("*.part"))
    assert ROM not in observation["files"]
    dcps = [p for p in observation["files"] if p.endswith(".dcp")]
    assert dcps == [], dcps
    native = Path(local_path(ROOT / RESULT / "m2_compact_core_r1"))
    banked = Path(local_path(ROOT / RESULT / "scnn_issue_banked_synth_r1"))
    sources = sorted(p.relative_to(local).as_posix() for p in (local / "rtl").rglob("*.sv"))
    assert len(sources) == 108
    assert all((local / p).read_bytes() == (native / p).read_bytes() for p in sources)
    assert (local / TCL).read_text() == selected_tcl((banked / TCL).read_text())
    same_recipe = ["server/run_unified_h40_final_composition_v1_synth.sh"]
    same_recipe += [p.relative_to(banked).as_posix() for p in (banked / "vivado").rglob("*.xdc")]
    assert all((local / p).read_bytes() == (banked / p).read_bytes() for p in same_recipe)
    report = local / OUT
    states = {n: (report / n).read_text().strip() for n in (
        "wrapper_status.txt", "vivado_exit_code.txt", "error_count.txt",
        "critical_warning_count.txt", "start_time.txt", "end_time.txt")}
    assert states["wrapper_status.txt"] == "status=fail"
    assert states["vivado_exit_code.txt"] == "139"
    assert not (report / "vivado_status.txt").exists()
    log = (report / "vivado.log").read_text(errors="replace")
    errors = re.findall(r"^ERROR:.*$", log, re.M)
    critical = re.findall(r"^CRITICAL WARNING:.*$", log, re.M)
    warnings = re.findall(r"^WARNING:.*$", log, re.M)
    assert len(errors) == int(states["error_count.txt"]) == 1
    assert len(critical) == int(states["critical_warning_count.txt"]) == 1
    assert "Common 17-141" in errors[0] and "zip archive" in errors[0]
    assert "Synth 8-4445" in critical[0] and ROM in critical[0]
    utilization = (report / "utilization_post_synth.rpt").read_text()
    def resource(label):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|", utilization, re.M)
        assert match, label
        return float(match[1]) if "." in match[1] else int(match[1])
    timing = (report / "timing_summary_post_synth.rpt").read_text()
    summary = timing.split("| Design Timing Summary", 1)[1]
    timing_row = next(line.split() for line in summary.splitlines()
                      if re.match(r"^\s+-?\d+\.\d+\s+-?\d+\.\d+", line))
    keys = ("wns_ns", "tns_ns", "setup_failing", "setup_total", "whs_ns", "ths_ns",
            "hold_failing", "hold_total", "wpws_ns", "tpws_ns", "pulse_failing", "pulse_total")
    assert len(timing_row) == len(keys)
    audit = dict(
        status="REJECTED_incomplete_input_and_checkpoint_failure",
        acceptable_for_qor_comparison=False, acceptable_for_implementation=False,
        remote_observation=observation, raw_file_count=len(observation["files"]),
        raw_bytes=sum(p["size"] for p in observation["files"].values()),
        all_remote_file_sizes_equal=True, partial_files=[], raw_status=states,
        dcp_files=dcps, dcp_crc="not_applicable_no_checkpoint_generated",
        native_rtl_byte_equal=len(sources), recipe_byte_equal=same_recipe,
        tcl_matches_only_selected_generic_and_status_change=True,
        omitted_original_source_data=dict(path=ROM, bytes=(banked / ROM).stat().st_size),
        errors=errors, critical_warnings=critical, ordinary_warning_count=len(warnings),
        oom_or_killed_log_lines=re.findall(r"^.*(?:Out of memory|OOM|\bKilled\b).*$", log, re.M | re.I),
        diagnostic_only_resources=dict(lut=resource("Slice LUTs*"), ff=resource("Slice Registers"),
            bram36eq=resource("Block RAM Tile"), dsp=resource("DSPs")),
        diagnostic_only_timing=dict(zip(keys, map(float, timing_row))),
        boundary="Reports belong to a missing-ROM design and failed checkpoint stage, not accepted QoR/STA. No DCP exists. Disk was low; a direct ENOSPC cause is not established. No OOM claim beyond log scan; kernel evidence unavailable. No production edits or run restart.",
    )
    (local / "terminal_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "remote_observation"}, indent=2))


if __name__ == "__main__":
    main()
