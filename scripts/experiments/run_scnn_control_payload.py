"""Run and archive the isolated real-core cfg-payload A/B experiment."""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time

NAME='scnn_cfg_payload_unit_r1'


def parse_log(log):
    if re.search(r'%Error|%Fatal|Assertion failed',log): raise ValueError('runtime diagnostic')
    def records(marker):
        return [dict((k,int(v)) for k,v in re.findall(r'(\w+)=(-?\d+)',line))
                for line in log.splitlines() if line.startswith(marker+' ')]
    if records('CFG_PAYLOAD_PASS') != [dict(modes=4,real_core=1,cycle_miter=1)]:
        raise ValueError('missing or duplicate completion')
    rows=sorted(records('CFG_PAYLOAD_CHECKER_PASS'),key=lambda r:r['mode'])
    if [r['mode'] for r in rows]!=list(range(4)): raise ValueError('enable matrix incomplete')
    for row in rows:
        clients=(row['mode']&1)+((row['mode']>>1)&1)
        expected=dict(configs=10*clients+int(clients>0),rejected=8*clients,
                      releases=2*clients+int(clients>0),windows=2*clients,
                      reads=16*clients,results=16*clients,code_writes=16*clients)
        if any(row[k]!=v for k,v in expected.items()): raise ValueError('incomplete real-core cases')
        if clients and any(row[k]<=0 for k in ('request_stalls','result_stalls','payload_changes')):
            raise ValueError('missing busy/backpressure coverage')
    windows=records('CFG_PAYLOAD_WINDOW')
    expected={(m,c,p) for m in (1,2,3) for c in (0,1) for p in (0,1) if m&(1<<c)}
    if len(windows)!=8 or {(r['mode'],r['client'],r['padded']) for r in windows}!=expected:
        raise ValueError('layout/client coverage')
    if any(r['results']!=8 or r['code_writes']!=8 for r in windows): raise ValueError('nonzero learning missing')
    return dict(checkers=rows,windows=windows,
        boundary='Cycle-by-cycle public-output equivalence for actual small shared learners; 32 invalid configurations, three early releases, eight nonzero learning windows with default/padded FC/SCNN layouts. Not final-top/profile regression, mapped timing or throughput.')


def worker(verilator,name):
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat())
    def save(): Path('status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();start=time.monotonic()
        with Path(stage+'.log').open('x') as log:
            try:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,timeout_seconds=timeout,wall_seconds=time.monotonic()-start)
                raise
        state[stage]=dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-start);save()
        if result.returncode: raise RuntimeError(stage+' exit '+str(result.returncode))
    try:
        obj=Path('/tmp')/('snn_'+name)/'obj'
        obj.parent.mkdir(exist_ok=False)
        execute('build',[verilator,'--binary','--timing','--assert','-Wno-fatal','-j','2',
                         '--top-module','tb_scnn_control_payload','--Mdir',str(obj),'-CFLAGS','-O1','-f','sources.f'],900)
        build=Path('build.log').read_text(errors='replace')
        state['warnings']=dict((kind,len(re.findall(r'^%Warning-'+re.escape(kind)+':',build,re.M)))
            for kind in sorted(set(re.findall(r'^%Warning-([A-Z0-9_]+):',build,re.M))))
        state['structural_warnings']=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',build,re.M)
        if state['structural_warnings'] or '%Error' in build: raise RuntimeError('structural build diagnostic')
        execute('run',[str(obj/'Vtb_scnn_control_payload')],120)
        state['checks']=parse_log(Path('run.log').read_text(errors='replace'))
        Path('checks.json').write_text(json.dumps(state['checks'],indent=2)+'\n')
        state['status']='PASS_real_shared_core_cfg_payload_cycle_miter_not_QoR'
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
    state['end']=dt.datetime.now().astimezone().isoformat();save()
    print(json.dumps(state,indent=2))
    return 0 if state['status'].startswith('PASS') else 1


def launch(name,candidate):
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,VERILATOR,connect,run,upload,local_path
    from prepare_scnn_control_payload import EXPERIMENT
    local=Path(local_path(ROOT/EXPERIMENT/candidate)); dest=REMOTE+'/'+name
    if not (local/'preparation.json').is_file(): raise ValueError('not prepared')
    client=connect()
    try:
        probe=f'''
import pathlib,shutil,json,datetime
r=pathlib.Path({dest!r});tmp=pathlib.Path('/tmp')/{('snn_'+name)!r}
if r.exists() or tmp.exists():raise FileExistsError('unique run root exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','xelab','xsim','verilator_bin') or comm.startswith(('Vtb_','Vscgraph')):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA '+repr(active))
data=shutil.disk_usage(r.parent).free;temp=shutil.disk_usage('/tmp').free
if data<128*1024**2 or temp<512*1024**2:raise RuntimeError('insufficient small unit build space')
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=temp)
for n in ('cpu.max','memory.max'):
 p=pathlib.Path('/sys/fs/cgroup')/n
 record[n]=p.read_text().strip() if p.exists() else None
r.mkdir();(r/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
        print(run(client,'python3 -c '+shlex.quote(probe)),flush=True)
        with client.open_sftp() as sftp:
            for path in local.rglob('*'):
                if path.is_file(): upload(sftp,str(path),dest+'/'+path.relative_to(local).as_posix())
            for script,remote in [(Path(__file__),'worker.py'),(ROOT/'scripts/experiments/prepare_scnn_control_payload.py','prepare_scnn_control_payload.py')]:
                upload(sftp,str(script),dest+'/'+remote)
            # Verify the actual transfer bytes before this root becomes active.
            for path in local.rglob('*'):
                if path.is_file():
                    with sftp.open(dest+'/'+path.relative_to(local).as_posix(),'rb') as f:
                        if f.read()!=path.read_bytes(): raise ValueError('upload mismatch '+str(path))
        print(run(client,'cd '+shlex.quote(dest)+' && python3 worker.py --worker --name '+shlex.quote(name)+
                  ' --verilator '+shlex.quote(VERILATOR)),flush=True)
    finally: client.close()


def audit(name,candidate):
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,local_path
    from prepare_scnn_control_payload import EXPERIMENT,SOURCE,SERVICE,candidate as transform
    archive=Path(local_path(ROOT/RESULT/name)); prepared=Path(local_path(ROOT/EXPERIMENT/candidate))
    state=json.loads((archive/'status.json').read_text())
    spec=importlib.util.spec_from_file_location('archived_cfg_worker',archive/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checks=module.parse_log((archive/'run.log').read_text())
    if state['checks']!=checks or json.loads((archive/'checks.json').read_text())!=checks:
        raise ValueError('archive parser/status mismatch')
    if any(state[s]['exit']!=0 for s in ('build','run')) or not state.get('end'): raise ValueError('nonterminal or nonzero exit')
    if list(archive.rglob('*.part')):raise ValueError('partial download')
    sizes=[];client=connect()
    try:
        with client.open_sftp() as sftp:
            def walk(remote,rel=Path()):
                for e in sftp.listdir_attr(remote):
                    if e.filename in ('obj','elab_obj','.Xil','__pycache__'):continue
                    child=rel/e.filename
                    if stat.S_ISDIR(e.st_mode):walk(remote+'/'+e.filename,child)
                    else:
                        if (archive/child).stat().st_size!=e.st_size:raise ValueError('archive size '+str(child))
                        sizes.append(dict(path=child.as_posix(),bytes=e.st_size))
            walk(REMOTE+'/'+name)
    finally:client.close()
    inputs=[]
    for path in prepared.rglob('*'):
        if path.is_file():
            rel=path.relative_to(prepared)
            if (archive/rel).read_bytes()!=path.read_bytes():raise ValueError('input differs '+str(rel))
            inputs.append(rel.as_posix())
    for local,remote in [(Path(__file__),'worker.py'),(ROOT/'scripts/experiments/prepare_scnn_control_payload.py','prepare_scnn_control_payload.py')]:
        if local.read_bytes()!=(archive/remote).read_bytes():raise ValueError('worker/preparer differs')
    source=Path(local_path(ROOT/SOURCE));changed=[]
    for path in (source/'rtl').rglob('*'):
        if path.is_file():
            rel=path.relative_to(source)
            if path.read_bytes()!=(archive/rel).read_bytes():changed.append(rel.as_posix())
    if changed!=[SERVICE.as_posix()]:raise ValueError('unexpected RTL differences '+repr(changed))
    if (archive/SERVICE).read_text()!=transform((source/SERVICE).read_text()):raise ValueError('candidate transformation')
    log='\n'.join((archive/(s+'.log')).read_text() for s in ('build','run'))
    if re.search(r'%Error|%Fatal|%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT)|Assertion failed',log):raise ValueError('diagnostics')
    result=dict(status='PASS_cfg_payload_real_core_archive_audit',time=dt.datetime.now().astimezone().isoformat(),
        raw_files=len(sizes),raw_bytes=sum(r['bytes'] for r in sizes),files=sizes,prepared_inputs=len(inputs),
        changed_rtl=changed,checks=checks,warnings=state['warnings'],
        oom_killed_in_logs=bool(re.search(r'Out of memory|oom-kill|Killed',log)),dcp_crc='not applicable: no DCP',
        production_adopted=False,timing_or_throughput_measured=False)
    (archive/'terminal_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','checks')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--audit',action='store_true')
    parser.add_argument('--name',default=NAME);parser.add_argument('--candidate',default='candidate_r1')
    parser.add_argument('--verilator',default='verilator');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+',args.name):parser.error('simple name required')
    if args.worker:raise SystemExit(worker(args.verilator,args.name))
    if args.audit:audit(args.name,args.candidate)
    else:launch(args.name,args.candidate)
