"""Reuse the limited public M2 early-cancel/retry check for the final candidate."""
import argparse
import datetime as dt
import difflib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME='scnn_turnover_recovery_r1'
EXPERIMENT=Path('verification/experiments/scnn_forward_turnover/recovery_r1/candidate_r1')
TB='tb_srnn_profile_measurement.sv'
PARSER='recovery_parser_snapshot.py'
SCOPE='Exact final compact512/turnover candidate; original public native M2 first-step START_LEARNING early cancellation and retry, learning off/on, no global reset or parameter reload. Not SCNN cancellation, arbitrary-cycle abort, mid-update rollback, full T73, throughput or STA.'


def transform(text):
    old='.TOPOLOGY_MASK(TOPOLOGY_MASK),'
    if text.count(old)!=1:raise ValueError('ambiguous final-top parameter anchor')
    return text.replace(old,old+'\n        .M2_CACHE_DEPTH(512),',1)


def load_parser(path):
    spec=importlib.util.spec_from_file_location('actual_recovery_parser',path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module.parse_recovery_log


def parse_log(log,arm,expected,original_parser=None):
    if original_parser is None:
        from srnn_resident_server_worker import parse_recovery_log as original_parser
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b',log):raise ValueError('runtime diagnostic')
    checks=original_parser(log,arm,expected)
    markers=[line.split()[0] for line in log.splitlines() if line.startswith(('RESIDENT_ABORT_ACCEPTED ','RESIDENT_ABORT_RECOVERY_PASS '))]
    if markers!=['RESIDENT_ABORT_ACCEPTED','RESIDENT_ABORT_RECOVERY_PASS']:raise ValueError('early abort/retry order')
    tiles=re.findall(r'^H40_ACTIVITY tile=(\S+)',log,re.M)
    coordinates=[(int(m[1]),int(m[2])) for name in tiles if (m:=re.search(r'g_region\[(\d+)\].*g_tile\[(\d+)\]',name))]
    if len(tiles)!=12 or len(set(tiles))!=12 or sorted(coordinates)!=[(r,t) for r in range(3) for t in range(4)]:
        raise ValueError('incomplete12-tile activity')
    checks.update(tiles=tiles,scope=SCOPE)
    return checks


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    base=Path(local_path(ROOT/RESULT));old=base/'final_combination_recovery_r1';source=base/'scnn_turnover_frame_r2/candidate'
    out=Path(local_path(ROOT/EXPERIMENT));out.mkdir(parents=True,exist_ok=False)
    original=(old/TB).read_text();changed=transform(original)
    (out/TB).write_text(changed,newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='accepted_recovery/tb',tofile='final_compact_turnover_recovery/tb')))
    shutil.copytree(source/'rtl',out/'rtl')
    paths=sorted(p.relative_to(source).as_posix() for p in (source/'rtl').rglob('*') if p.is_file())
    if len(paths)!=109 or sum(x.endswith('.sv') for x in paths)!=108:raise ValueError('incomplete final candidate source')
    accepted=json.loads((old/'accepted_source_status.json').read_text())
    commands={stage:accepted[stage]['command'] for stage in ('build','run_off','run_on')}
    (out/'accepted_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    fixtures=set()
    for arm in ('off','on'):
        for arg in commands['run_'+arm]:
            if arg.startswith(('+RAW=','+FIXTURE=')):
                rel=Path(arg.split('=',1)[1])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('unexpected external fixture')
                fixtures.add(rel.as_posix())
    if len(fixtures)!=1:raise ValueError('paired original fixture changed')
    dependencies=[]
    for folder in ['tb']+sorted(fixtures):
        shutil.copytree(old/folder,out/folder)
        dependencies.extend(p.relative_to(old).as_posix() for p in (old/folder).rglob('*') if p.is_file())
    (out/'dependencies.json').write_text(json.dumps(sorted(dependencies),indent=2)+'\n')
    shutil.copy2(old/'recovery/reference.json',out/'expected.json')
    shutil.copy2(old/'scripts/experiments/srnn_resident_server_worker.py',out/PARSER)
    (out/'sources.f').write_text(''.join('"'+rel+'"\n' for rel in paths if rel.endswith('.sv'))+'"'+TB+'"\n"tb/monitors/supplement_h40_activity_monitor.sv"\n')
    record=dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,source='scnn_turnover_frame_r2/candidate/rtl',source_files=109,
        original_test='final_combination_recovery_r1',dependencies=len(dependencies),fixture=next(iter(fixtures)),
        tb_delta='Only explicit final-top M2_CACHE_DEPTH512; all original cancellation, numeric, parameter, reset/load and sequence assertions remain unchanged.',
        boundary='Local preparation only while implementation is active. No upload/compile/simulation, no terminal audit. Do not run concurrently with any EDA or rerun original historical task.')
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py');print(str(out))


def worker():
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    expected=json.loads((root/'expected.json').read_text());commands=json.loads((root/'accepted_commands.json').read_text())
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();started=time.monotonic()
        with (root/(stage+'.log')).open('w') as log:
            try:result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,wall_seconds=time.monotonic()-started,timeout_seconds=timeout);save();raise
        state[stage]=dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-started);save()
        if result.returncode:raise RuntimeError(stage+' exit '+str(result.returncode))
    try:
        build=commands['build'].copy();build[build.index('-j')+1]='2';build[build.index('--Mdir')+1]=str(scratch/'obj')
        execute('build',build,1200)
        log=(root/'build.log').read_text(errors='replace')
        state['ordinary_warnings']=len(re.findall(r'^%Warning-',log,re.M));state['structural_warnings']=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M)
        if state['structural_warnings']:raise ValueError('structural warnings')
        parser=load_parser(root/PARSER);checks={}
        for arm in ('off','on'):
            command=commands['run_'+arm].copy();command[0]=str(scratch/'obj/Vtb_srnn_profile_measurement')
            execute('run_'+arm,command,600)
            checks[arm]=parse_log((root/('run_'+arm+'.log')).read_text(errors='replace'),arm,expected['expected'][arm],parser)
            state[arm]=checks[arm];save()
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state['status']='PASS_final_turnover_public_M2_early_abort_first_step_retry_pair'
    except Exception as error:state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    if not (prepared/'preparation.json').is_file():raise ValueError('prepare first')
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
if p.exists() or pathlib.Path(SCRATCH).exists():raise FileExistsError('unique recovery destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA; do not overlap: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p.parent).free,tmp_free=shutil.disk_usage('/tmp').free)
if record['data_free']<128*1024**2 or record['tmp_free']<1024**3:raise RuntimeError('insufficient recovery input/build margin')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
        record=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\n'+code)))
        inputs=[]
        with client.open_sftp() as s:
            for path in prepared.rglob('*'):
                if not path.is_file() or '__pycache__' in path.relative_to(prepared).parts:continue
                rel=path.relative_to(prepared).as_posix();upload(s,str(path),destination+'/'+rel)
                if s.open(destination+'/'+rel,'rb').read()!=path.read_bytes():raise ValueError('actual input mismatch: '+rel)
                inputs.append(rel)
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_turnover_recovery bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_turnover_recovery',prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','worker'));args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':launch()
    else:worker()
