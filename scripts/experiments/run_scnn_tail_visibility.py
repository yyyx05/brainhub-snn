#!/usr/bin/env python3
"""Launch one bounded final-top Conv8 tail visibility experiment, not throughput."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def worker():
    from prepare_scnn_tail_testbench import parse_log
    from srnn_resident_server_worker import scnn_build_command
    config=json.loads(Path('reference.json').read_text())
    status=dict(status='starting',scope=config['scope'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    def save():
        status['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        Path('status.json').write_text(json.dumps(status,indent=2)+'\n')
    def execute(name,command,timeout):
        status['status']=name+'_running';save();started=time.monotonic()
        with Path(name+'.log').open('x') as stream:
            try:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                status[name]=dict(command=command,exit=None,timed_out=True,timeout_seconds=timeout,wall_seconds=time.monotonic()-started)
                raise
        status[name]=dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-started);save()
        if result.returncode:raise RuntimeError(name+' failed')
    try:
        accepted=json.loads(Path('scnn_compatibility_build.json').read_text())
        command=scnn_build_command(accepted['command'],str(Path('scripts/supplement_scnn_graph_host.mk').resolve()))
        command[command.index('--top-module')+1]='tb_scnn_tail_visibility'
        execute('build',command,config['build_timeout'])
        log=Path('build.log').read_text(errors='replace')
        status['structural_warnings']=re.findall(r'%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log)
        if status['structural_warnings']:raise RuntimeError('structural warnings')
        execute('run',['obj/Vscgraph','+SCNN_TEMPORAL_FIXTURE=fixture',
                       '+SCNN_TEMPORAL_CONTROL=control','+TAIL_FIXTURE=tail_reference'],config['run_timeout'])
        checks=parse_log(Path('run.log').read_text(errors='replace'))
        Path('checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        status.update(status=checks['status'],final=checks['final'])
    except Exception as error:
        status.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    finally:save()
    return 0 if status['status'].startswith('PASS_') else 1


def launch(name,candidate):
    from run_srnn_resident_remote import connect,run,upload,local_path,ROOT,REMOTE
    from prepare_scnn_tail_testbench import EXPERIMENT,ORIGINAL
    target=REMOTE+'/'+name
    client=connect()
    try:
        print(run(client,'df -B1 /root/autodl-tmp'))
        bootstrap=f'''import pathlib,shutil,json
t=pathlib.Path({target!r});s=pathlib.Path({(REMOTE+'/scnn_issue_sparse_r1')!r})
b=pathlib.Path({(REMOTE+'/scnn_issue_banked_synth_r1')!r})
n=pathlib.Path({(REMOTE+'/scnn_natural_component_r4')!r})
a=pathlib.Path('/root/autodl-tmp/SNN_SCNN_REAL_FINAL_B4T12_R2')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator'):active.append((p.name,comm))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError('existing EDA/simulation process: '+str(active))
if shutil.disk_usage(t.parent).free < 1000000000:raise RuntimeError('insufficient build headroom')
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(b/'rtl',t/'rtl')
rtls=sorted((t/'rtl').rglob('*.sv'))
if len(rtls)!=107 or any(p.read_bytes()!=(n/p.relative_to(t)).read_bytes() for p in rtls):raise ValueError('actual synthesis/component RTL mismatch')
shutil.copytree(s/'models',t/'models')
shutil.copytree(n/'fixture',t/'fixture')
shutil.copytree(a/'results/paper_evidence/supplement_20261004/rtl/scnn_real_final_graph/server_b4t12_r2/control',t/'control')
lines=[]
for line in (s/'compatibility_sources.f').read_text().splitlines():
 rel=pathlib.PurePosixPath(line.strip().strip('"'))
 if rel.is_absolute() or '..' in rel.parts:raise ValueError(rel)
 if rel.name=='tb_scnn_profile_measurement.sv':rel=pathlib.PurePosixPath('tb_scnn_tail_visibility.sv')
 elif rel.parts[0]!='rtl':
  q=t/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,q)
 lines.append(rel.as_posix())
(t/'compatibility_sources.f').write_text(''.join('"'+p+'"\\n' for p in lines))
for rel in ('scnn_compatibility_build.json','scripts/supplement_scnn_graph_host.mk'):
 q=t/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,q)
(t/'candidate_source.json').write_text(json.dumps(dict(actual_rtl_source=str(b/'rtl'),compared_component=str(n/'rtl'),rtl_files=len(rtls),same_bytes=True,scope='Physical candidate source identity, not a new functional PASS.'),indent=2)+'\\n')
'''
        run(client,'python3 -c '+shlex.quote(bootstrap))
        with client.open_sftp() as sftp:
            prepared=ROOT/EXPERIMENT/candidate
            for filename in ('tb_scnn_tail_visibility.sv','testbench_changes.diff','reference.json'):
                upload(sftp,local_path(prepared/filename),target+'/'+filename)
            for filename in ('prepare_scnn_tail_testbench.py','prepare_scnn_tail_visibility.py',
                             'run_scnn_tail_visibility.py','srnn_resident_server_worker.py'):
                upload(sftp,local_path(ROOT/'scripts/experiments'/filename),target+'/scripts/experiments/'+filename)
            for path in Path(local_path(ROOT/EXPERIMENT/'tail_reference_r1')).iterdir():
                if path.is_file():upload(sftp,str(path),target+'/tail_reference/'+path.name)
            upload(sftp,local_path(ROOT/ORIGINAL),target+'/original_tb_scnn_real_final_b4t12.sv')
        shell='cd '+shlex.quote(target)+' && exec python3 scripts/experiments/run_scnn_tail_visibility.py --worker'
        run(client,'screen -dmS '+shlex.quote('snn_resident_'+name)+' bash -lc '+shlex.quote(shell))
        print(json.dumps(dict(launched=True,remote=target,screen='snn_resident_'+name)))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--name',default='scnn_tail_visibility_r1')
    parser.add_argument('--candidate',default='candidate_r1')
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker())
    if not args.name.replace('_','').isalnum():parser.error('simple experiment name required')
    launch(args.name,args.candidate)
