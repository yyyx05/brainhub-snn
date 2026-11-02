"""Audit only the fully fetched, naturally terminal final-candidate tail check."""
import importlib.util
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_scnn_turnover_tail import NAME,EXPERIMENT,TB,PARSER


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def audit():
    archive=Path(local_path(ROOT/RESULT/NAME));prepared=Path(local_path(ROOT/EXPERIMENT))
    state=json.loads((archive/'status.json').read_text())
    if not state.get('end') or state['status'].endswith('_running') or state['status']=='running':
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
    if observation['active_eda']:raise ValueError('EDA/possible timeout child remains active; defer audit')
    files=observation.pop('files')
    if any(p.endswith('.part') for p in files) or list(archive.rglob('*.part')):raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:raise ValueError('incomplete raw file: '+rel)
    compared=[]
    for path in prepared.rglob('*'):
        if not path.is_file() or '__pycache__' in path.relative_to(prepared).parts:continue
        rel=path.relative_to(prepared).as_posix()
        if (archive/rel).read_bytes()!=path.read_bytes():raise ValueError('prepared/actual bytes differ: '+rel)
        compared.append(rel)
    source=Path(local_path(ROOT/RESULT/'scnn_turnover_frame_r2/candidate'))
    rtls=[p.relative_to(source).as_posix() for p in (source/'rtl').rglob('*') if p.is_file()]
    if len(rtls)!=109 or sum(p.endswith('.sv') for p in rtls)!=108:raise ValueError('final source tree incomplete')
    for rel in rtls:
        if (source/rel).read_bytes()!=(archive/rel).read_bytes():raise ValueError('final RTL/ROM changed: '+rel)
    original=Path(local_path(ROOT/RESULT/'scnn_tail_visibility_r1'))
    deps=json.loads((archive/'dependencies.json').read_text())
    for rel in deps:
        if (archive/rel).read_bytes()!=(original/rel).read_bytes():raise ValueError('original numeric input changed: '+rel)
    if (archive/PARSER).read_bytes()!=(original/'corrected_local_parser.py').read_bytes():raise ValueError('accepted strict parser snapshot changed')
    launched=load_module('audited_tail_worker',archive/'launch_worker.py')
    parser=load_module('audited_tail_parser',archive/PARSER)
    diagnostics={}
    for stage in ('build','run'):
        path=archive/(stage+'.log')
        if path.exists():
            log=path.read_text(errors='replace')
            diagnostics[stage]=dict(errors=len(re.findall(r'^%Error',log,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed',log)),
                ordinary_warnings=len(re.findall(r'^%Warning-',log,re.M)),
                structural_warnings=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M),
                oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',log,re.I)))
    checks=None
    if state['status'].startswith('PASS'):
        if any(state.get(stage,{}).get('exit')!=0 for stage in ('build','run')):raise ValueError('actual tool exit is not zero')
        checks=launched.parse_log((archive/'run.log').read_text(errors='replace'),original_parser=parser.parse_log)
        if checks!=json.loads((archive/'checks.json').read_text()) or checks['status']!=state['status'] or checks['final']!=state['final']:
            raise ValueError('original strict reparse/checks/state mismatch')
        if any(x['errors'] or x['fatal'] or x['structural_warnings'] for x in diagnostics.values()):raise ValueError('tool diagnostic')
    result=dict(status='PASS_final_turnover_tail_archive_audit' if checks else 'TERMINAL_FAIL_original_preserved',
        original_status=state,raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
        prepared_input_files_byte_equal=len(compared),final_candidate_source_files_byte_equal=len(rtls),
        original_numeric_dependency_files_byte_equal=len(deps),archived_worker_and_parser_used=True,
        diagnostics=diagnostics,checks=checks,remote_observation=observation,
        boundary='Final candidate only; no baseline rerun, raw state/log changes, production merge, model/mask/rule edits or assertion relaxation. Necessary cold three-window Conv8 tail fixture, CPU Conv7 service input. Not full raw52-frame eight-layer graph, new throughput, DDR, arbitrary abort/rollback or STA. No DCP; CRC not applicable. Scratch preserved; log OOM absence is not kernel proof.')
    (archive/'terminal_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','raw_files','raw_bytes','prepared_input_files_byte_equal','diagnostics')},indent=2))


if __name__=='__main__':audit()
