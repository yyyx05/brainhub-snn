#!/usr/bin/env python3
"""Read back the exact final candidate's limited early-abort recovery evidence."""
import collections
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import stat

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, local_path, run
from srnn_resident_server_worker import parse_recovery_log


def main():
    name='final_combination_recovery_r1'
    archive=Path(local_path(ROOT/RESULT))
    result=archive/name
    old=archive/'abort_before_learning_r1'
    candidate=archive/'scnn_issue_banked_synth_r1'
    status=json.loads((result/'status.json').read_text())
    reference=json.loads((result/'recovery/reference.json').read_text())
    assert status['status']=='PASS_final_top_abort_before_learning_retry_first_step_pair'
    assert all(status[stage]['exit']==0 for stage in ('build','run_off','run_on'))
    build=(result/'build.log').read_text()
    warnings=dict(collections.Counter(re.findall(r'%Warning-(\w+)',build)))
    assert not re.search(r'%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT)',build)
    checks={}
    tile_counts={}
    for arm in ('off','on'):
        log=(result/('run_'+arm+'.log')).read_text()
        assert not re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b',build+log)
        assert not re.search(r'Out of memory|\bOOM\b|\bKilled\b',build+log,re.I)
        assert log.index('RESIDENT_ABORT_ACCEPTED ')<log.index('RESIDENT_ABORT_RECOVERY_PASS ')
        checks[arm]=parse_recovery_log(log,arm,reference['expected'][arm])
        assert checks[arm]==status[arm]==json.loads((old/'status.json').read_text())[arm]
        tile_counts[arm]=len(re.findall(r'^H40_ACTIVITY tile=',log,re.M))
        assert tile_counts[arm]==12
    files=sorted((result/'rtl').rglob('*.sv'))
    assert len(files)==107
    for path in files:
        assert path.read_bytes()==(candidate/path.relative_to(result)).read_bytes(),str(path)
    sources=[p.strip().strip('"') for p in (result/'sources.f').read_text().splitlines()]
    assert sorted(sources)==sorted([p.relative_to(result).as_posix() for p in files]+[
        'tb_srnn_profile_measurement.sv','tb/monitors/supplement_h40_activity_monitor.sv'])
    unchanged=[]
    excluded={'status.json','build.log','run_off.log','run_on.log','sources.f',
              'candidate_source.json','accepted_original_recovery_status.json'}
    for path in result.rglob('*'):
        rel=path.relative_to(result)
        if not path.is_file() or rel.parts[0]=='rtl' or '__pycache__' in rel.parts or rel.as_posix() in excluded:
            continue
        # Only inherited raw inputs are compared; later local audit artifacts are not inputs.
        if (old/rel).is_file():
            assert path.read_bytes()==(old/rel).read_bytes(),str(path)
            unchanged.append(rel.as_posix())
    for needed in ('tb_srnn_profile_measurement.sv','recovery/reference.json','scripts/experiments/srnn_resident_server_worker.py'):
        assert needed in unchanged
    inventory=[]
    client=connect()
    try:
        with client.open_sftp() as sftp:
            def visit(relative):
                remote=REMOTE+'/'+name+('/'+relative if relative else '')
                for entry in sftp.listdir_attr(remote):
                    if entry.filename in ('obj','elab_obj','__pycache__','.Xil'):continue
                    child=(relative+'/' if relative else '')+entry.filename
                    if stat.S_ISDIR(entry.st_mode):visit(child)
                    else:
                        assert stat.S_ISREG(entry.st_mode)
                        assert (result/child).stat().st_size==entry.st_size,child
                        inventory.append(dict(path=child,bytes=entry.st_size))
            visit('')
        code='''import json,pathlib,shutil,subprocess
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
d=subprocess.run(['dmesg'],capture_output=True,text=True)
print(json.dumps(dict(active=active,free_bytes=shutil.disk_usage('/root/autodl-tmp').free,dmesg_readable=d.returncode==0,dmesg_error=d.stderr.strip())))'''
        remote=json.loads(run(client,'python3 -c '+shlex.quote(code)))
        assert not remote['active'],remote
    finally:client.close()
    assert not list(result.rglob('*.part'))
    report=dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status=status['status'],
        raw_files=len(inventory),raw_bytes=sum(p['bytes'] for p in inventory),all_raw_sizes_match=True,partial_files=0,
        actual_rtl_files=107,rtl_byte_identical_to_banked_synthesis=True,unchanged_input_files=sorted(unchanged),
        stages={s:{k:v for k,v in status[s].items() if k!='command'} for s in ('build','run_off','run_on')},
        build_warnings=warnings,structural_warnings=0,runtime_errors=0,oom_or_killed_in_logs=False,
        checks=checks,tile_activity_records=tile_counts,remote_at_audit=remote,scope=status['scope'],
        raw_inventory=sorted(inventory,key=lambda row:row['path']))
    (result/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_inventory','unchanged_input_files')},indent=2))
    print('unchanged_input_files='+str(len(unchanged)))


if __name__=='__main__':main()
