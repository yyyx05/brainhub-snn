#!/usr/bin/env python3
"""Run the scoped natural-update component integration in an isolated server root."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def worker(verilator):
    from prepare_scnn_natural_component import parse_log
    status={'status':'running'}
    def save(): Path('status.json').write_text(json.dumps(status,indent=2)+'\n')
    def execute(name,command,timeout):
        status['status']=name+'_running';save(); started=time.monotonic()
        with Path(name+'.log').open('x') as log:
            try:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                status[name]=dict(command=command,exit_code=None,timed_out=True,timeout_seconds=timeout,elapsed_seconds=time.monotonic()-started)
                raise
        status[name]=dict(command=command,exit_code=result.returncode,elapsed_seconds=time.monotonic()-started);save()
        if result.returncode: raise RuntimeError(name+' failed')
    try:
        execute('build',[verilator,'--binary','--timing','--assert','-j','2','-Wno-fatal',
                        '--top-module','tb_scnn_natural_component','--Mdir','obj','-CFLAGS','-O1','-f','sources.f'],900)
        log=Path('build.log').read_text(errors='replace')
        status['structural_warnings']=re.findall(r'%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log)
        if status['structural_warnings']: raise RuntimeError('structural warnings')
        execute('run',['obj/Vtb_scnn_natural_component','+FIXTURE=fixture'],300)
        checks=parse_log(Path('run.log').read_text(errors='replace'))
        Path('checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        status.update(status=checks['status'],final=checks['final'])
    except Exception as error:
        status.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    finally:save()
    print(json.dumps(status,indent=2))
    return 0 if status['status'].startswith('PASS_') else 1


def launch(name,candidate):
    from run_srnn_resident_remote import connect,run,upload,fetch_tree,local_path,ROOT,REMOTE,RESULT,VERILATOR
    from prepare_scnn_natural_component import EXPERIMENT,FIXTURE
    target=REMOTE+'/'+name
    client=connect()
    try:
        print(run(client,'df -B1 /root/autodl-tmp'))
        code=f'''import pathlib,shutil,json
t=pathlib.Path({target!r});s=pathlib.Path({(REMOTE+'/scnn_issue_sparse_r1')!r})
a=pathlib.Path('/root/autodl-tmp/SNN_SCNN_REAL_FINAL_B4T12_R2')
if not json.loads((s/'status.json').read_text())['status'].startswith('PASS_'): raise ValueError('unaccepted source')
t.mkdir(parents=True,exist_ok=False)
shutil.copytree(s/'rtl',t/'rtl')
shutil.copytree(a/{FIXTURE!r},t/'fixture')
lines=[str(p.relative_to(t)) for p in sorted((t/'rtl').rglob('*.sv'))]
lines+=['tb_scnn_natural_component.sv']
(t/'sources.f').write_text(''.join('"'+p+'"\\n' for p in lines))
'''
        run(client,'python3 -c '+shlex.quote(code))
        with client.open_sftp() as sftp:
            for local,destination in (
                (ROOT/EXPERIMENT/candidate/'tb_scnn_natural_component.sv','tb_scnn_natural_component.sv'),
                (ROOT/EXPERIMENT/candidate/'reference.json','reference.json'),
                (ROOT/EXPERIMENT/'body.svh','body.svh'),
                (Path(__file__),'worker.py'),
                (ROOT/'scripts/experiments/prepare_scnn_natural_component.py','prepare_scnn_natural_component.py'),
                (ROOT/'verification/experiments/fc_contribution_pipeline/candidate_banked_r3/rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv','rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv'),
            ): upload(sftp,local_path(local),target+'/'+destination)
        try:
            print(run(client,'cd '+shlex.quote(target)+' && python3 worker.py --worker --verilator '+shlex.quote(VERILATOR)))
        finally:
            with client.open_sftp() as sftp: fetch_tree(sftp,target,ROOT/RESULT/name)
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--name',default='scnn_natural_component_r1')
    parser.add_argument('--candidate',default='candidate_r1')
    parser.add_argument('--verilator',default='verilator')
    args=parser.parse_args()
    if args.worker: raise SystemExit(worker(args.verilator))
    if not args.name.replace('_','').isalnum(): parser.error('simple name required')
    launch(args.name,args.candidate)
