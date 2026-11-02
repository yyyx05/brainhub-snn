"""Audit a naturally terminal, fully fetched final-router owner test."""
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,local_path,run
from run_scnn_router_owners import NAME,EXPERIMENT,ROUTER,TB,parse_log


def audit():
    archive=Path(local_path(ROOT/RESULT/NAME))
    prepared=Path(local_path(ROOT/EXPERIMENT))
    state=json.loads((archive/'status.json').read_text())
    if not state.get('end') or state['status'].endswith('_running') or state['status']=='running':
        raise ValueError('not terminal; do not audit an active run')
    client=connect()
    try:
        probe='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
files={q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts).intersection({'obj','elab_obj','__pycache__','.Xil'})}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
        remote=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\n'+probe)))
        files=remote.pop('files')
        if any(p.endswith('.part') for p in files) or list(archive.rglob('*.part')):
            raise ValueError('partial archive')
        for rel,size in files.items():
            if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:
                raise ValueError('raw file size mismatch: '+rel)
        with client.open_sftp() as s:
            for rel in (ROUTER,TB,'preparation.json','launch_worker.py','status.json'):
                if s.open(REMOTE+'/'+NAME+'/'+rel,'rb').read()!=(archive/rel).read_bytes():
                    raise ValueError('actual input/status mismatch: '+rel)
    finally:client.close()
    for rel in (ROUTER,TB,'preparation.json','launch_worker.py'):
        if (prepared/rel).read_bytes()!=(archive/rel).read_bytes():
            raise ValueError('prepared/actual mismatch: '+rel)
    source=Path(local_path(ROOT/RESULT/'scnn_turnover_frame_r2/candidate'/ROUTER))
    if source.read_bytes()!=(archive/ROUTER).read_bytes():raise ValueError('final candidate router changed')
    diagnostics={}
    for stage in ('build','run'):
        path=archive/(stage+'.log')
        if path.exists():
            log=path.read_text(errors='replace')
            diagnostics[stage]=dict(errors=len(re.findall(r'^%Error',log,re.M)),
                fatal=bool(re.search(r'%Fatal|Assertion failed',log)),
                ordinary_warnings=len(re.findall(r'^%Warning-',log,re.M)),
                structural_warnings=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M),
                oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',log,re.I)))
    checks=None
    if state['status'].startswith('PASS'):
        if any(state.get(stage,{}).get('exit')!=0 for stage in ('build','run')):raise ValueError('nonzero actual tool exit')
        checks=parse_log((archive/'run.log').read_text())
        if checks!=json.loads((archive/'checks.json').read_text()) or checks['status']!=state['status']:
            raise ValueError('strict reparse/state mismatch')
        if any(r['errors'] or r['fatal'] or r['structural_warnings'] for r in diagnostics.values()):
            raise ValueError('tool diagnostics')
    result=dict(status='PASS_final_router_three_owner_archive_audit' if checks else 'TERMINAL_FAIL_original_preserved',
        original_status=state,raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
        prepared_four_inputs_and_remote_status_byte_equal=True,final_candidate_router_byte_equal=True,
        diagnostics=diagnostics,checks=checks,remote_observation=remote,
        boundary='No rerun or original status/log changes. Exact final forward router only; learning inputs idle. Module drain/clear/reissue is not final-top abort or application compatibility. No DCP, CRC not applicable; no resource/STA conclusion. Log OOM absence is not kernel proof. Scratch retained.')
    (archive/'terminal_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','raw_files','raw_bytes','diagnostics')},indent=2))


if __name__=='__main__':audit()
