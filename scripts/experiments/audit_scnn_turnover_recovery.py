"""Audit the final candidate's bounded public M2 early-abort/retry evidence."""
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_scnn_turnover_recovery import NAME,EXPERIMENT,TB,PARSER
from audit_scnn_turnover_tail import load_module


def audit():
    archive=Path(local_path(ROOT/RESULT/NAME));prepared=Path(local_path(ROOT/EXPERIMENT))
    state=json.loads((archive/'status.json').read_text())
    if not state.get('end') or state['status']=='running' or state['status'].endswith('_running'):
        raise ValueError('natural terminal fetch required')
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
files={q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files,active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
        observation=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\n'+code)))
    finally:client.close()
    if observation['active_eda']:raise ValueError('EDA/possible timeout child remains active')
    files=observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):raise ValueError('partial fetch')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:raise ValueError('raw file incomplete: '+rel)
    inputs=[]
    for path in prepared.rglob('*'):
        if not path.is_file() or '__pycache__' in path.relative_to(prepared).parts:continue
        rel=path.relative_to(prepared).as_posix()
        if path.read_bytes()!=(archive/rel).read_bytes():raise ValueError('prepared/actual bytes mismatch: '+rel)
        inputs.append(rel)
    source=Path(local_path(ROOT/RESULT/'scnn_turnover_frame_r2/candidate'))
    source_paths=[p.relative_to(source).as_posix() for p in (source/'rtl').rglob('*') if p.is_file()]
    if len(source_paths)!=109 or sum(p.endswith('.sv') for p in source_paths)!=108:raise ValueError('final source incomplete')
    for rel in source_paths:
        if (archive/rel).read_bytes()!=(source/rel).read_bytes():raise ValueError('final candidate changed: '+rel)
    original=Path(local_path(ROOT/RESULT/'final_combination_recovery_r1'))
    deps=json.loads((archive/'dependencies.json').read_text())
    for rel in deps:
        if (archive/rel).read_bytes()!=(original/rel).read_bytes():raise ValueError('numeric dependency changed: '+rel)
    for actual,old in (('expected.json','recovery/reference.json'),(PARSER,'scripts/experiments/srnn_resident_server_worker.py')):
        if (archive/actual).read_bytes()!=(original/old).read_bytes():raise ValueError('original reference/parser changed')
    text=(archive/TB).read_text();parameter='\n        .M2_CACHE_DEPTH(512),'
    if text.count(parameter)!=1 or text.replace(parameter,'')!=(original/TB).read_text():raise ValueError('unexpected testbench change')
    launched=load_module('audited_recovery_worker',archive/'launch_worker.py');parser=launched.load_parser(archive/PARSER)
    reference=json.loads((archive/'expected.json').read_text());commands=json.loads((archive/'accepted_commands.json').read_text())
    diagnostics={}
    for stage in ('build','run_off','run_on'):
        path=archive/(stage+'.log')
        if path.exists():
            log=path.read_text(errors='replace')
            diagnostics[stage]=dict(errors=len(re.findall(r'^%Error',log,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed',log)),
                ordinary_warnings=len(re.findall(r'^%Warning-',log,re.M)),
                structural_warnings=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M),
                oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',log,re.I)))
    checks=None
    if state['status'].startswith('PASS'):
        if any(state.get(stage,{}).get('exit')!=0 for stage in ('build','run_off','run_on')):raise ValueError('nonzero actual tool exit')
        expected_build=commands['build'].copy();expected_build[expected_build.index('-j')+1]='2';expected_build[expected_build.index('--Mdir')+1]=state['scratch']+'/obj'
        if state['build']['command']!=expected_build:raise ValueError('build recipe changed')
        checks={}
        for arm in ('off','on'):
            expected_run=commands['run_'+arm].copy();expected_run[0]=state['scratch']+'/obj/Vtb_srnn_profile_measurement'
            if state['run_'+arm]['command']!=expected_run:raise ValueError('numeric run arguments changed')
            checks[arm]=launched.parse_log((archive/('run_'+arm+'.log')).read_text(errors='replace'),arm,reference['expected'][arm],parser)
            if checks[arm]!=state[arm]:raise ValueError('arm strict reparse/state mismatch')
        if checks!=json.loads((archive/'checks.json').read_text()):raise ValueError('paired checks mismatch')
        if any(x['errors'] or x['fatal'] or x['structural_warnings'] for x in diagnostics.values()):raise ValueError('tool diagnostic')
    result=dict(status='PASS_final_turnover_M2_early_recovery_archive_audit' if checks else 'TERMINAL_FAIL_original_preserved',
        original_status=state,raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
        prepared_input_files_byte_equal=len(inputs),final_candidate_source_files_byte_equal=109,original_numeric_dependency_files_byte_equal=len(deps),
        testbench_only_capacity_changed=True,archived_worker_and_parser_used=True,checks=checks,diagnostics=diagnostics,remote_observation=observation,
        boundary='Exact final candidate only, public native M2 legal START_LEARNING early cancel/retry off/on. No new baseline or historical task rerun, original numeric/sequence/reset/load assertions retained. Not SCNN cancel, arbitrary-cycle abort, mid-update rollback, fullT73, new throughput, DDR or STA. No DCP; CRC not applicable. Scratch retained; log OOM absence is not kernel proof.')
    (archive/'terminal_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','raw_files','raw_bytes','prepared_input_files_byte_equal','diagnostics')},indent=2))


if __name__=='__main__':audit()
