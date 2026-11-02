"""Measure the exact raw-frame candidate with the unchanged compact synth recipe."""
import argparse
import datetime as dt
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

NAME="scnn_turnover_core_synth_r1"
SCRATCH="/tmp/snn_"+NAME
TCL="vivado/run_unified_h40_final_composition_v1_synth.tcl"
XDC="vivado/unified_h40_final_composition_v1_ooc.xdc"
SHELL="server/run_unified_h40_final_composition_v1_synth.sh"
OUT="server_results/final_composition_synth"


def parse_state(path):
    return dict(line.split("=",1) for line in path.read_text().splitlines() if "=" in line) if path.is_file() else None


def result_status(exit_code,tcl,crc,errors,critical):
    if exit_code!=0:return "FAIL_tool_exit"
    if errors or critical:return "FAIL_tool_diagnostic"
    if not tcl or tcl.get("status")!="pass" or tcl.get("black_boxes")!="0":return "FAIL_tcl_or_blackbox"
    expected=dict(period_ns="15.000",source_count="108",enable_online_learning="1",m2_cache_depth="512")
    if any(tcl.get(k)!=v for k,v in expected.items()):return "FAIL_configuration"
    if crc is not True:return "FAIL_checkpoint_crc"
    return "PASS_synthesis_candidate_only_not_STA_or_final_functional_release"


def worker():
    root=Path.cwd(); scratch=Path(SCRATCH)
    state=dict(status="running",start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch))
    def save():
        (root/"status.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        scratch.mkdir(exist_ok=False)
        for folder in ("rtl","vivado","server"):
            shutil.copytree(root/folder,scratch/folder)
        out=scratch/OUT
        observations=[];done=threading.Event()
        def sample():
            while not done.is_set():
                observation=dict(time=dt.datetime.now().astimezone().isoformat(),data_free=shutil.disk_usage(root).free,tmp_free=shutil.disk_usage(scratch).free)
                observations.append(observation)
                if len(observations)%15==1:(root/"disk_observation.json").write_text(json.dumps(observation)+"\n")
                done.wait(1)
        thread=threading.Thread(target=sample);thread.start()
        env=os.environ.copy();env.update(TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch))
        command=["bash",str(scratch/SHELL),str(scratch),"1"]
        started=time.monotonic()
        try:
            with (root/"worker_wrapper.log").open("w") as stream:
                completed=subprocess.run(command,cwd=scratch,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            state.update(exit=completed.returncode,command=command,wall_seconds=time.monotonic()-started,tool_end=dt.datetime.now().astimezone().isoformat())
        finally:
            done.set();thread.join()
            (root/"disk_samples.json").write_text(json.dumps(observations,indent=2)+"\n")
        state.update(minimum_data_free=min(r["data_free"] for r in observations),minimum_tmp_free=min(r["tmp_free"] for r in observations))
        files=[p for p in scratch.rglob("*") if p.is_file() and ".Xil" not in p.relative_to(scratch).parts]
        needed=sum(p.stat().st_size for p in files if not (root/p.relative_to(scratch)).exists())
        if shutil.disk_usage(root).free<needed+128*1024**2:
            state.update(status="FAIL_archive_space",archive_bytes_needed=needed)
        else:
            inventory=[]
            for path in files:
                rel=path.relative_to(scratch);target=root/rel
                if path.is_symlink():raise ValueError("unexpected scratch symlink: "+str(path))
                if target.exists():
                    if target.read_bytes()!=path.read_bytes():raise ValueError("input changed in scratch: "+str(rel))
                    copied=False
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);copied=True
                if target.stat().st_size!=path.stat().st_size:raise ValueError("archive size mismatch")
                inventory.append(dict(relative=rel.as_posix(),bytes=path.stat().st_size,copied=copied))
            (root/"scratch_archive.json").write_text(json.dumps(inventory,indent=2)+"\n")
            log=(out/"vivado.log").read_text(errors="replace")
            errors=re.findall(r"^ERROR:.*",log,re.M);critical=re.findall(r"^CRITICAL WARNING:.*",log,re.M)
            dcp=out/"post_synth.dcp";crc=None
            if dcp.is_file():
                with zipfile.ZipFile(dcp) as archive:
                    crc=archive.testzip() is None;state["checkpoint_zip_members"]=len(archive.infolist())
            tcl=parse_state(out/"vivado_status.txt")
            state.update(vivado_exit=(out/"vivado_exit_code.txt").read_text().strip() if (out/"vivado_exit_code.txt").exists() else None,
                tcl=tcl,checkpoint_bytes=dcp.stat().st_size if dcp.exists() else None,checkpoint_crc_pass=crc,
                errors=errors,critical_warnings=critical,ordinary_warning_count=len(re.findall(r"^WARNING:",log,re.M)),
                status=result_status(state["exit"],tcl,crc,errors,critical))
    except Exception as error:
        state.update(status="FAIL_worker_or_archive",error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True,data_free_after=shutil.disk_usage(root).free,tmp_free_after=shutil.disk_usage('/tmp').free)
    save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    local=Path(local_path(ROOT/RESULT/"scnn_turnover_frame_r2"))
    if json.loads((local/"terminal_audit.json").read_text())["status"]!="PASS_raw_first_frame_pair_archive_audit":
        raise ValueError("raw-frame pair evidence incomplete")
    reference=Path(local_path(ROOT/RESULT/"m2_compact_synth_r2"))
    prepared=Path(local_path(ROOT/"verification/experiments/scnn_forward_turnover/core_synthesis_r1"))
    prepared.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),prepared/"launch_worker.py")
    dest=REMOTE+"/"+NAME
    code='''import datetime,json,pathlib,shutil
base=pathlib.Path(BASE);target=base/NAME;scratch=pathlib.Path(SCRATCH)
if target.exists() or scratch.exists():raise FileExistsError('unique synthesis destination exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('another EDA active: '+repr(active))
data_free=shutil.disk_usage(base).free;tmp_free=shutil.disk_usage('/tmp').free
if data_free<512*1024**2 or tmp_free<4*1024**3:raise RuntimeError('insufficient scratch/archive headroom')
target.mkdir()
shutil.copytree(base/'scnn_turnover_frame_r2/candidate/rtl',target/'rtl')
for rel in RECIPES:
 source=base/'m2_compact_synth_r2'/rel;destination=target/rel
 destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data_free,tmp_free=tmp_free,
 source='scnn_turnover_frame_r2/candidate/rtl',recipe='m2_compact_synth_r2',scratch=str(scratch),
 boundary='Exact raw-frame candidate and unchanged original compact synth Tcl/shell/XDC. 8 threads,15ns,512,online learning enabled. Synthesis only; no route, power, bitstream or final functional release. Cwd/output/temp use system-disk scratch; all non-Xil raw files archived; scratch preserved.')
(target/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
    client=connect()
    try:
        record=json.loads(run(client,"python3 -c "+shlex.quote("BASE="+repr(REMOTE)+"\nNAME="+repr(NAME)+"\nSCRATCH="+repr(SCRATCH)+"\nRECIPES="+repr([TCL,XDC,SHELL])+"\n"+code)))
        compared=[]
        with client.open_sftp() as s:
            paths=[p.relative_to(local/"candidate").as_posix() for p in (local/"candidate/rtl").rglob("*") if p.is_file()]
            if len(paths)!=109:raise ValueError("source tree incomplete")
            for rel in paths+[TCL,XDC,SHELL]:
                expected=(local/"candidate"/rel) if rel in paths else reference/rel
                if s.open(dest+"/"+rel,"rb").read()!=expected.read_bytes():raise ValueError("actual input mismatch "+rel)
                compared.append(rel)
            upload(s,local_path(Path(__file__).resolve()),dest+"/worker.py")
            if s.open(dest+"/worker.py","rb").read()!=(prepared/"launch_worker.py").read_bytes():raise ValueError("worker mismatch")
        command="cd "+shlex.quote(dest)+" && exec python3 worker.py worker"
        run(client,"screen -dmS snn_turnover_synth bash -lc "+shlex.quote(command))
        record.update(remote=dest,launched=True,screen="snn_turnover_synth",actual_input_byte_equal=compared,actual_worker_byte_equal=True)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/"launch_observation.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps({k:v for k,v in record.items() if k!="actual_input_byte_equal"}));print("byte_equal_inputs",len(compared))
    finally:client.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("launch","worker"))
    if parser.parse_args().action=="launch":launch()
    else:worker()
