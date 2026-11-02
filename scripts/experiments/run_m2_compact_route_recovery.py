"""Continue the saved compact placement through unchanged route commands."""
import argparse
import datetime as dt
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile

NAME = "m2_compact_route_recovery_r1"
SCRATCH = "/tmp/snn_m2_compact_route_recovery_r1"
SOURCE = "m2_compact_impl_r1/server_results/final_composition_impl/post_place_physopt.dcp"
ORIGINAL_TCL = "vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl"
EXPERIMENT = "verification/experiments/m2_compact_route_recovery"
TCL = EXPERIMENT + "/route_from_placement.tcl"
WORKER = "scripts/experiments/run_m2_compact_route_recovery.py"
OUT = "server_results/final_composition_impl"
VIVADO = "/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado"

REOPEN = r'''
set saved_primitive_count [llength [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1}]]
set saved_clock_period [get_property PERIOD [get_clocks core_clk]]
set saved_handle [open [file join $out_dir before_checkpoint_status.txt] w]
puts $saved_handle "black_boxes=[llength $final_black_boxes]"
puts $saved_handle "unplaced_primitives=[llength $final_unplaced]"
puts $saved_handle "wns_ns=$setup_slack"
puts $saved_handle "whs_ns=$hold_slack"
puts $saved_handle "primitive_count=$saved_primitive_count"
close $saved_handle
write_checkpoint -force [file join $out_dir post_route_physopt.dcp]
close_design
open_checkpoint [file join $out_dir post_route_physopt.dcp]
set reopened_protected [lsort [get_property NAME [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && DONT_TOUCH == 1}]]]
set reopened_primitives [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1}]
set reopened_unplaced [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && IS_PLACED == 0}]
set reopened_black_boxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
set reopened_setup [timing_slack max]
set reopened_hold [timing_slack min]
set reopened_trace_cells [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && REF_NAME =~ FD* && (NAME =~ *trace_product_q_reg* || NAME =~ *trace_old_flat_q_reg* || NAME =~ *o_trace_next_flat_reg*)}]
set reopened_trace_nonconstant 0
foreach payload_cell $reopened_trace_cells {
    foreach pin_name {R S CLR PRE} {
        set pin [get_pins -quiet "[get_property NAME $payload_cell]/$pin_name"]
        if {[llength $pin] == 0} { continue }
        set net [get_nets -quiet -of_objects $pin]
        set drivers [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]]
        if {[lsearch -exact [get_property REF_NAME $drivers] GND] < 0} { incr reopened_trace_nonconstant }
    }
}
if {$reopened_protected ne $original_protected_names ||
    [llength $reopened_primitives] != $saved_primitive_count ||
    [llength $reopened_unplaced] != [llength $final_unplaced] ||
    [llength $reopened_black_boxes] != [llength $final_black_boxes] ||
    $reopened_setup != $setup_slack || $reopened_hold != $hold_slack ||
    [llength $reopened_trace_cells] != [llength $final_trace_cells] || $reopened_trace_nonconstant != 0 ||
    [llength [get_clocks -quiet]] != $clock_count ||
    [get_property PERIOD [get_clocks core_clk]] != $saved_clock_period} {
    error "routed checkpoint reopen mismatch"
}
set reopen_dir [file join $out_dir reopened]
file mkdir $reopen_dir
write_timing_reports post_route $reopen_dir
report_route_status -file [file join $reopen_dir route_status_post_route.rpt]
report_utilization -file [file join $reopen_dir utilization_post_route.rpt]
report_clocks -file [file join $reopen_dir clocks_post_route.rpt]
report_exceptions -file [file join $reopen_dir exceptions_post_route.rpt]
set reopen_handle [open [file join $out_dir checkpoint_reopen_status.txt] w]
puts $reopen_handle "status=pass"
puts $reopen_handle "primitive_count=[llength $reopened_primitives]"
puts $reopen_handle "unplaced_primitives=[llength $reopened_unplaced]"
puts $reopen_handle "black_boxes=[llength $reopened_black_boxes]"
puts $reopen_handle "wns_ns=$reopened_setup"
puts $reopen_handle "whs_ns=$reopened_hold"
puts $reopen_handle "protected_set_preserved=1"
puts $reopen_handle "trace_payload_cell_count=[llength $reopened_trace_cells]"
puts $reopen_handle "trace_payload_nonconstant_control_count=$reopened_trace_nonconstant"
close $reopen_handle
'''


def route_tcl(original):
    start = original.index("\nopt_design\n")
    tail = original.index("# RUN115 and its route-only RUN116")
    resume = '''
# Input is the original post-place/physopt DCP. Only route and its following stages resume.
set post_place_congestion_report [file join $out_dir congestion_post_place.rpt]
report_design_analysis -congestion -file $post_place_congestion_report
if {$clock_count != 1 || [get_property PERIOD [get_clocks core_clk]] != 15.000} {
    error "unexpected inherited clock constraint"
}
'''
    text = original[:start] + "\n" + resume + "\n" + original[tail:]
    anchor = "write_checkpoint -force [file join $out_dir post_route_physopt.dcp]"
    if text.count(anchor) != 1:
        raise ValueError("ambiguous routed checkpoint command")
    text = text.replace(anchor, REOPEN.strip(), 1)
    return text


def implementation_status(exit_code, log, tcl_status, reopened, crc):
    if exit_code != 0 or re.search(r"^ERROR:", log, re.M) or crc is not True or reopened.get("status") != "pass":
        return "FAIL"
    if tcl_status.get("status") == "pass" and not re.search(r"^CRITICAL WARNING:", log, re.M):
        return "PASS_route_recovery_checkpoint_reopened"
    return "DIAGNOSTIC_route_recovery_checkpoint_reopened"


def worker(source):
    root, scratch = Path.cwd(), Path(SCRATCH)
    scratch.mkdir(exist_ok=False)
    out = root / OUT
    out.mkdir(parents=True)
    state = dict(status="running", start=dt.datetime.now().astimezone().isoformat(), source=source,
                 scratch=str(scratch), scope="Route AggressiveExplore + physopt Explore from saved original placement; no synthesis or placement.")
    def save():
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    def read_state(path):
        return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line) if path.is_file() else {}
    save()
    samples, done = [], threading.Event()
    def sample():
        while not done.is_set():
            observation = dict(time=dt.datetime.now().astimezone().isoformat(), data_free=shutil.disk_usage(root).free,
                               tmp_free=shutil.disk_usage(scratch).free)
            samples.append(observation)
            if len(samples) % 15 == 1:
                (root / "disk_observation.json").write_text(json.dumps(observation) + "\n")
            done.wait(1)
    thread = threading.Thread(target=sample)
    thread.start()
    env = os.environ.copy()
    install = str(Path(VIVADO).parents[1])
    env.update(XILINX_VIVADO=install, PATH=install+"/bin:"+env.get("PATH", ""),
               LD_LIBRARY_PATH=install+"/lib/lnx64.o/SuSE:"+env.get("LD_LIBRARY_PATH", ""),
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    if Path("/lib/x86_64-linux-gnu/libudev.so.1").is_file():
        env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libudev.so.1"
    command = [VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source", str(root / TCL),
               "-tclargs", str(scratch), source]
    started = time.monotonic()
    try:
        with (out / "vivado.log").open("w") as log:
            result = subprocess.run(command, cwd=scratch, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        state.update(exit=result.returncode, wall_seconds=time.monotonic()-started,
                     tool_end=dt.datetime.now().astimezone().isoformat())
    finally:
        done.set()
        thread.join()
        (root / "disk_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    state.update(command=command, minimum_data_free=min(x["data_free"] for x in samples),
                 minimum_tmp_free=min(x["tmp_free"] for x in samples))
    try:
        retained = [p for p in scratch.rglob("*") if p.is_file() and ".Xil" not in p.relative_to(scratch).parts]
        size = sum(p.stat().st_size for p in retained)
        if shutil.disk_usage(root).free < size + 268435456:
            state.update(status="FAIL_archive_space", scratch_bytes=size)
            save()
            return
        inventory = []
        for path in retained:
            relative, target = path.relative_to(scratch), root / path.relative_to(scratch)
            if path.is_symlink() or target.exists():
                raise ValueError("unexpected archive target: " + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assert target.stat().st_size == path.stat().st_size
            inventory.append(dict(source=str(path), relative=relative.as_posix(), bytes=path.stat().st_size))
        (root / "scratch_archive.json").write_text(json.dumps(inventory, indent=2) + "\n")
        checkpoint = out / "post_route_physopt.dcp"
        state.update(output_bytes=checkpoint.stat().st_size if checkpoint.exists() else None, output_crc_pass=None)
        if checkpoint.is_file():
            with zipfile.ZipFile(checkpoint) as archive:
                state["output_crc_pass"] = archive.testzip() is None
                state["output_zip_members"] = len(archive.infolist())
        log = (out / "vivado.log").read_text(errors="replace")
        state.update(tcl_status=read_state(out / "vivado_status.txt"),
                     checkpoint_reopen=read_state(out / "checkpoint_reopen_status.txt"),
                     errors=re.findall(r"^ERROR:.*$", log, re.M),
                     critical_warnings=re.findall(r"^CRITICAL WARNING:.*$", log, re.M),
                     ordinary_warnings=len(re.findall(r"^WARNING:", log, re.M)))
        state["status"] = implementation_status(state["exit"], log, state["tcl_status"],
                                                state["checkpoint_reopen"], state["output_crc_pass"])
    except Exception as error:
        state.update(status="FAIL_archive_or_validation", archive_error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(), data_free_after=shutil.disk_usage(root).free,
                 tmp_free_after=shutil.disk_usage(scratch).free, scratch_preserved=True)
    save()


def launch():
    import run_srnn_resident_remote as m
    archive = Path(m.local_path(m.ROOT / m.RESULT))
    accepted = json.loads((archive / "m2_checkpoint_roundtrip_r2/terminal_audit.json").read_text())
    if accepted["status"] != "PASS_checkpoint_export_reopen_only" or accepted["output_crc_pass"] is not True:
        raise ValueError("scratch export/reopen diagnosis not accepted")
    original = (archive / "m2_compact_impl_r1" / ORIGINAL_TCL).read_text()
    generated = route_tcl(original)
    prepared = Path(m.local_path(m.ROOT / EXPERIMENT))
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "route_from_placement.tcl").write_text(generated, encoding="utf-8", newline="\n")
    changes = "".join(difflib.unified_diff(original.splitlines(True), generated.splitlines(True),
                    fromfile="original_impl.tcl", tofile="route_from_placement.tcl"))
    (prepared / "route_changes.diff").write_text(changes, encoding="utf-8", newline="\n")
    target, source = m.REMOTE + "/" + NAME, m.REMOTE + "/" + SOURCE
    code = r'''
import datetime,json,pathlib,shutil,zipfile
target=pathlib.Path(TARGET);source=pathlib.Path(SOURCE);scratch=pathlib.Path(SCRATCH)
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
if target.exists() or scratch.exists():raise FileExistsError('recovery destination exists')
data_free=shutil.disk_usage(target.parent).free;tmp_free=shutil.disk_usage(scratch.parent).free
if tmp_free<8*1024**3 or data_free<768*1024**2:raise RuntimeError('insufficient scratch/archive space')
with zipfile.ZipFile(source) as archive:
 if archive.testzip() is not None:raise ValueError('source checkpoint CRC failed')
if source.stat().st_size!=176061306:raise ValueError('unexpected placed source')
target.mkdir()
record=dict(time=datetime.datetime.now().astimezone().isoformat(),source=str(source),source_bytes=source.stat().st_size,
 source_crc_pass=True,data_free=data_free,tmp_free=tmp_free,scratch=str(scratch),active_eda=active,
 boundary='Recover only route/route-physopt from original placed checkpoint. Cwd/temp/output use demonstrated system-disk scratch. Original failed run immutable. All noncache outputs archived; scratch preserved.')
(target/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
    client = m.connect()
    try:
        print(m.run(client, "python3 -c " + shlex.quote("TARGET="+repr(target)+"\nSOURCE="+repr(source)+"\nSCRATCH="+repr(SCRATCH)+"\n"+code)))
        with client.open_sftp() as sftp:
            for relative in (TCL, WORKER, EXPERIMENT + "/route_changes.diff"):
                m.upload(sftp, m.local_path(m.ROOT / relative), target + "/" + relative)
            m.upload(sftp, str(archive / "m2_compact_impl_r1" / ORIGINAL_TCL), target + "/original_impl.tcl")
            m.upload(sftp, str(archive / "m2_checkpoint_roundtrip_r2/terminal_audit.json"), target + "/accepted_export_audit.json")
        command = "cd " + shlex.quote(target) + " && exec python3 " + WORKER + " worker --source " + shlex.quote(source)
        m.run(client, "screen -dmS snn_m2_route_recovery bash -lc " + shlex.quote(command))
        print(json.dumps(dict(name=NAME, remote=target, scratch=SCRATCH, launched=True)))
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker"))
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.action == "launch":
        launch()
    elif args.source:
        worker(args.source)
    else:
        parser.error("worker requires --source")
