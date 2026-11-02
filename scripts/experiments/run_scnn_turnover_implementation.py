"""Implement the accepted turnover synthesis with the original compact recipe."""
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
import zlib

NAME='scnn_turnover_core_impl_r1'
SCRATCH='/tmp/snn_'+NAME
SOURCE='scnn_turnover_core_synth_r1/server_results/final_composition_synth/post_synth.dcp'
EXPERIMENT=Path('verification/experiments/scnn_forward_turnover/core_implementation_r1')
ORIGINAL='vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl'
OUT='server_results/final_composition_impl'
VIVADO='/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado'


def run_paths(name):
    match=re.fullmatch(r'scnn_turnover_core_impl_r([1-9][0-9]*)',name)
    if not match:raise ValueError('expected an isolated turnover implementation revision')
    return name,'/tmp/snn_'+name,Path('verification/experiments/scnn_forward_turnover/core_implementation_r'+match[1])


def implementation_tcl(original):
    from run_m2_compact_route_recovery import REOPEN
    anchor='write_checkpoint -force [file join $out_dir post_route_physopt.dcp]'
    if original.count(anchor)!=1:raise ValueError('ambiguous final checkpoint command')
    return original.replace(anchor,REOPEN.strip(),1)


def implementation_status(exit_code,tcl,reopened,crc,errors,critical):
    if exit_code!=0 or errors or critical or crc is not True or reopened.get('status')!='pass':return 'FAIL_tool_or_checkpoint'
    if tcl.get('status')=='pass':return 'PASS_candidate_internal15ns_checkpoint_reopened'
    return 'DIAGNOSTIC_candidate_impl_checkpoint_reopened_not_timing_closed'


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    base=Path(local_path(ROOT/RESULT))
    audit=json.loads((base/'scnn_turnover_core_synth_r1/terminal_audit.json').read_text())
    if audit['status']!='PASS_turnover_synthesis_evidence_not_routed_STA':raise ValueError('synthesis evidence incomplete')
    original_path=base/'m2_compact_impl_r1'/ORIGINAL
    original=original_path.read_text()
    generated=implementation_tcl(original)
    out=Path(local_path(ROOT/EXPERIMENT));out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(original_path,out/'original_impl.tcl')
    (out/'implementation.tcl').write_text(generated,newline='\n')
    (out/'implementation_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),generated.splitlines(True),fromfile='accepted_compact_impl',tofile='turnover_impl_with_reopen')))
    record=dict(status='PREPARED_NOT_IMPLEMENTED',source=SOURCE,source_checkpoint=audit['dcp'],
        original_recipe='m2_compact_impl_r1/'+ORIGINAL,
        recipe='opt_design; place_design ExtraNetDelay_high; phys_opt_design ExploreWithHoldFix; route_design AggressiveExplore; phys_opt_design Explore',
        delta='Only persist pre-export counts and close/open the final checkpoint with matching reports. No optimization/constraint/protection/congestion-acceptance changes.',
        boundary='Own candidate implementation, not inherited timing or final functional release. 15ns, original protection, capacity512, no model-key, power or bitstream. Scratch preserved.')
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    print(str(out))


def fields(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line) if path.is_file() else {}


def worker(source):
    root=Path.cwd();scratch=Path(SCRATCH)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=SCRATCH,source=source)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        scratch.mkdir(exist_ok=False);out=scratch/OUT;out.mkdir(parents=True)
        observations=[];done=threading.Event()
        def sample():
            while not done.is_set():
                row=dict(time=dt.datetime.now().astimezone().isoformat(),data_free=shutil.disk_usage(root).free,tmp_free=shutil.disk_usage(scratch).free)
                observations.append(row)
                if len(observations)%15==1:(root/'disk_observation.json').write_text(json.dumps(row)+'\n')
                done.wait(1)
        thread=threading.Thread(target=sample);thread.start()
        install=str(Path(VIVADO).parents[1]);env=os.environ.copy()
        env.update(XILINX_VIVADO=install,PATH=install+'/bin:'+env.get('PATH',''),
            LD_LIBRARY_PATH=install+'/lib/lnx64.o/SuSE:'+env.get('LD_LIBRARY_PATH',''),
            TMPDIR=SCRATCH,TMP=SCRATCH,TEMP=SCRATCH)
        if Path('/lib/x86_64-linux-gnu/libudev.so.1').is_file():env['LD_PRELOAD']='/lib/x86_64-linux-gnu/libudev.so.1'
        command=[VIVADO,'-mode','batch','-nojournal','-nolog','-source',str(root/'implementation.tcl'),'-tclargs',SCRATCH,source]
        started=time.monotonic()
        try:
            with (out/'vivado.log').open('w') as log:
                result=subprocess.run(command,cwd=scratch,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(exit=result.returncode,command=command,wall_seconds=time.monotonic()-started,tool_end=dt.datetime.now().astimezone().isoformat())
        finally:
            done.set();thread.join()
            (root/'disk_samples.json').write_text(json.dumps(observations,indent=2)+'\n')
        state.update(minimum_data_free=min(x['data_free'] for x in observations),minimum_tmp_free=min(x['tmp_free'] for x in observations))
        log=(out/'vivado.log').read_text(errors='replace')
        errors=re.findall(r'^ERROR:.*',log,re.M);critical=re.findall(r'^CRITICAL WARNING:.*',log,re.M)
        checkpoints={}
        for name in ('post_place_physopt.dcp','post_route_physopt.dcp'):
            path=out/name
            if path.is_file():
                try:
                    with zipfile.ZipFile(path) as z:checkpoints[name]=dict(bytes=path.stat().st_size,zip_members=len(z.infolist()),crc_pass=z.testzip() is None)
                except (zipfile.BadZipFile,RuntimeError,zlib.error,EOFError) as error:
                    checkpoints[name]=dict(bytes=path.stat().st_size,crc_pass=False,archive_error=str(error))
        tcl=fields(out/'vivado_status.txt');reopened=fields(out/'checkpoint_reopen_status.txt')
        crc=checkpoints.get('post_route_physopt.dcp',{}).get('crc_pass')
        state.update(tcl_status=tcl,checkpoint_reopen=reopened,checkpoints=checkpoints,errors=errors,critical_warnings=critical,
            ordinary_warning_count=len(re.findall(r'^WARNING:',log,re.M)),
            status=implementation_status(state['exit'],tcl,reopened,crc,errors,critical))
        files=[p for p in scratch.rglob('*') if p.is_file() and '.Xil' not in p.relative_to(scratch).parts]
        needed=sum(p.stat().st_size for p in files if not (root/p.relative_to(scratch)).exists())
        if shutil.disk_usage(root).free<needed+128*1024**2:
            state.update(tool_result_status=state['status'],status='FAIL_archive_space',archive_bytes_needed=needed)
        else:
            inventory=[]
            for path in files:
                rel=path.relative_to(scratch);target=root/rel
                if path.is_symlink():raise ValueError('unexpected scratch symlink: '+str(rel))
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if target.read_bytes()!=path.read_bytes():raise ValueError('archive collision: '+str(rel))
                else:shutil.copy2(path,target)
                if target.stat().st_size!=path.stat().st_size:raise ValueError('archive size mismatch: '+str(rel))
                inventory.append(dict(relative=rel.as_posix(),bytes=path.stat().st_size))
            (root/'scratch_archive.json').write_text(json.dumps(inventory,indent=2)+'\n')
    except Exception as error:state.update(status='FAIL_worker_or_archive',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True,
        data_free_after=shutil.disk_usage(root).free,tmp_free_after=shutil.disk_usage('/tmp').free)
    save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));record=json.loads((prepared/'preparation.json').read_text())
    destination=REMOTE+'/'+NAME;source=REMOTE+'/'+SOURCE
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil,zipfile
p=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);source=pathlib.Path(SOURCE)
if p.exists() or s.exists():raise FileExistsError('unique implementation destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('another EDA active: '+repr(active))
data_free=shutil.disk_usage(p.parent).free;tmp_free=shutil.disk_usage('/tmp').free
if data_free<768*1024**2 or tmp_free<6*1024**3:raise RuntimeError('insufficient archive/scratch margin')
with zipfile.ZipFile(source) as z:crc=z.testzip() is None
if source.stat().st_size!=EXPECTED_BYTES or not crc:raise ValueError('source DCP size/CRC mismatch')
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data_free,tmp_free=tmp_free,
 source=str(source),source_bytes=source.stat().st_size,source_crc_pass=crc,scratch=SCRATCH,
 boundary='Original compact implementation optimization commands and constraints. Only final-checkpoint export/reopen evidence added. 768MiB public and6GiB scratch launch checks are margins, not measured peak guarantees; actual archive bytes checked before copy. No source DCP mutation, power, bitstream or production merge.')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
        before=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(destination)+'\nSCRATCH='+repr(SCRATCH)+'\nSOURCE='+repr(source)+'\nEXPECTED_BYTES='+repr(record['source_checkpoint']['bytes'])+'\n'+code)))
        with client.open_sftp() as s:
            for rel in ('original_impl.tcl','implementation.tcl','implementation_changes.diff','preparation.json','launch_worker.py'):
                upload(s,str(prepared/rel),destination+'/'+rel)
                if s.open(destination+'/'+rel,'rb').read()!=(prepared/rel).read_bytes():raise ValueError('actual input mismatch: '+rel)
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker --name '+shlex.quote(NAME)+' --source '+shlex.quote(source)
        run(client,'screen -dmS snn_turnover_impl bash -lc '+shlex.quote(command))
        before.update(launched=True,remote=destination,screen='snn_turnover_impl',five_prepared_inputs_byte_equal=True)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(before,indent=2)+'\n');print(json.dumps(before))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','worker'))
    parser.add_argument('--source');parser.add_argument('--name',default=NAME);args=parser.parse_args()
    NAME,SCRATCH,EXPERIMENT=run_paths(args.name)
    if args.action=='prepare':prepare()
    elif args.action=='launch':launch()
    else:worker(args.source)
