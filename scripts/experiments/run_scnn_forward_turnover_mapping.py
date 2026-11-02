"""Paired module mapping, after accepting the same-service chain simulation."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import zipfile

NAME="scnn_forward_turnover_mapping_r1"
SCRATCH="/tmp/snn_scnn_forward_turnover_mapping_r1"
VIVADO="/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado"


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    root=Path.cwd();scratch=Path(SCRATCH)
    scratch.mkdir(exist_ok=False)
    compatible=root/"sources_vivado";compatible.mkdir()
    originals=list((root/"rtl").rglob("*.sv"))+list(root.glob("reference_*.sv"))
    originals += [root/"unified_array16_weight_refill_dispatcher_v1.sv",root/"turnover_chain.sv"]
    for index,path in enumerate(originals):
        (compatible/(str(index)+"_"+path.name)).write_text(path.read_text().replace("`default_nettype none","`default_nettype wire"))
    env=vivado_environment(VIVADO)
    env.update(TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch))
    command=[VIVADO,"-mode","batch","-nojournal","-nolog","-source",str(root/"mapping.tcl"),"-tclargs",str(root),str(scratch)]
    state=dict(status="running",start=dt.datetime.now().astimezone().isoformat(),command=command,scratch=SCRATCH)
    (root/"status.json").write_text(json.dumps(state,indent=2)+"\n")
    start=time.monotonic()
    try:
        with (root/"vivado.log").open("w") as log:
            result=subprocess.run(command,cwd=scratch,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        state.update(exit=result.returncode,wall_seconds=time.monotonic()-start)
        text=(root/"vivado.log").read_text(errors="replace")
        state.update(errors=re.findall(r"^ERROR:.*$",text,re.M),critical_warnings=re.findall(r"^CRITICAL WARNING:.*$",text,re.M),
                     ordinary_warning_count=len(re.findall(r"^WARNING:",text,re.M)))
        retained=[p for p in scratch.rglob("*") if p.is_file() and ".Xil" not in p.relative_to(scratch).parts]
        if shutil.disk_usage(root).free < sum(p.stat().st_size for p in retained)+128*1024**2:
            raise RuntimeError("insufficient archive space; scratch retained")
        for path in retained:
            target=root/path.relative_to(scratch)
            if target.exists():raise FileExistsError(str(target))
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        state["checkpoints"]={}
        for variant in ("baseline","candidate"):
            path=root/(variant+".dcp")
            if path.is_file():
                with zipfile.ZipFile(path) as archive:
                    state["checkpoints"][variant]=dict(bytes=path.stat().st_size,crc_pass=archive.testzip() is None,members=len(archive.infolist()))
        if (state["exit"] or state["errors"] or state["critical_warnings"] or
            "SCNN_TURNOVER_MAPPING_COMPLETE\n" not in text or
            len(state["checkpoints"])!=2 or not all(v["crc_pass"] for v in state["checkpoints"].values())):
            raise RuntimeError("paired mapping/export not accepted")
        state["status"]="PASS_module_mapping_not_whole_core_or_implementation"
    except subprocess.TimeoutExpired:
        state.update(status="FAIL",exit=None,timeout_seconds=600,wall_seconds=time.monotonic()-start)
    except Exception as error:
        state.update(status="FAIL",error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),data_free_after=shutil.disk_usage(root).free,tmp_free_after=shutil.disk_usage(scratch).free,scratch_preserved=True)
    (root/"status.json").write_text(json.dumps(state,indent=2)+"\n")
    print(json.dumps(state,indent=2))
    return 0 if state["status"].startswith("PASS") else 1


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,fetch_tree,local_path
    from prepare_scnn_forward_turnover import OUTPUT
    unit=Path(local_path(ROOT/RESULT/"scnn_forward_turnover_unit_r1/terminal_audit.json"))
    if not json.loads(unit.read_text())["status"].startswith("PASS"):
        raise ValueError("paired chain simulation not accepted")
    source=Path(local_path(ROOT/OUTPUT));target=REMOTE+"/"+NAME
    c=connect()
    try:
        probe=r'''
import datetime,json,pathlib,shutil
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH)
if r.exists() or s.exists():raise FileExistsError('mapping destination exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','xelab','xsim','verilator_bin') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
data=shutil.disk_usage(r.parent).free;tmp=shutil.disk_usage(s.parent).free
if data<512*1024**2 or tmp<2*1024**3:raise RuntimeError('insufficient mapping/archive space')
r.mkdir();record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=tmp,scratch=str(s))
(r/'preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
        print(run(c,"python3 -c "+shlex.quote("ROOT="+repr(target)+"\nSCRATCH="+repr(SCRATCH)+"\n"+probe)))
        with c.open_sftp() as s:
            for path in source.rglob("*"):
                if path.is_file():upload(s,str(path),target+"/"+path.relative_to(source).as_posix())
            upload(s,str(Path(__file__).resolve()),target+"/worker.py")
            upload(s,str(Path(__file__).with_name("run_fc_bitmap_mapping.py")),target+"/run_fc_bitmap_mapping.py")
            upload(s,str(Path(__file__).with_name("scnn_forward_turnover_mapping.tcl")),target+"/mapping.tcl")
            upload(s,str(unit),target+"/accepted_unit_audit.json")
            with s.open(target+"/clock.xdc","w") as stream:
                stream.write("create_clock -name core_clk -period 15.000 [get_ports i_clk]\n")
        try:print(run(c,"cd "+shlex.quote(target)+" && python3 worker.py --worker"))
        finally:
            with c.open_sftp() as s:fetch_tree(s,target,ROOT/RESULT/NAME)
    finally:c.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker())
    launch()
