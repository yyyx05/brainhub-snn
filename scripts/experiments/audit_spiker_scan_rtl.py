"""Check a fetched source-scan measurement without rerunning tools or fixtures."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode=True
from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_spiker_scan_rtl import NAME,EXPERIMENT,SOURCES


def main():
    archive=Path(local_path(ROOT/RESULT/NAME)); prepared=Path(local_path(ROOT/EXPERIMENT))
    if (archive/'terminal_audit.json').exists():raise FileExistsError('retain completed audit; do not repeat')
    state=json.loads((archive/'status.json').read_text())
    if not state.get('end') or not state['status'].startswith('PASS_') or any(state[s]['exit']!=0 for s in ('build','run')):
        raise ValueError('natural successful build/run required; preserve original status')
    code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);scratch=pathlib.Path(SCRATCH);skip={'obj','elab_obj','__pycache__','.Xil'}
def inventory(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),scratch_noncache_files=inventory(scratch),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    client=connect()
    try:
        observation=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\nSCRATCH='+repr(state['scratch'])+'\n'+code)))
    finally:client.close()
    if observation['active_eda'] or observation['scratch_noncache_files']:raise ValueError('active EDA or scratch evidence remains')
    files=observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:raise ValueError('raw size: '+rel)
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes()!=(archive/p.relative_to(prepared)).read_bytes():raise ValueError('prepared input: '+str(p))
    for rel in SOURCES:
        original=Path(local_path(ROOT/'verification/experiments/scnn_control_payload/candidate_r4/rtl'/rel))
        if original.read_bytes()!=(archive/'rtl'/rel).read_bytes():raise ValueError('original consumer changed')
    fixture=Path(local_path(ROOT/'verification/experiments/nearest_neighbor/scan_fixture_r1'))
    fixture_files=[p for p in fixture.iterdir() if p.is_file()]
    for p in fixture_files:
        if p.read_bytes()!=(archive/'fixture'/p.name).read_bytes():raise ValueError('original fixture changed')
    if (archive/'launch_worker.py').read_bytes()!=(ROOT/'scripts/experiments/run_spiker_scan_rtl.py').read_bytes():raise ValueError('actual worker mismatch')
    spec=importlib.util.spec_from_file_location('archived_scan_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    parsed=worker.parse_log((archive/'run.log').read_text(),archive/'fixture')
    if parsed!=json.loads((archive/'checks.json').read_text()) or parsed['real_frame_totals']!=state['real_frame_totals']:
        raise ValueError('actual worker strict reparse differs')
    diagnostics={}
    for stage in ('build','run'):
        text=(archive/(stage+'.log')).read_text(errors='replace')
        warnings=collections.Counter(re.findall(r'^%Warning-([^:]+):',text,re.M))
        diagnostics[stage]=dict(ordinary_warnings=sum(warnings.values()),warning_categories=dict(sorted(warnings.items())),
            errors=len(re.findall(r'^%Error',text,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b',text)),
            structural=[k for k in warnings if k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
            oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',text,re.I)),
            disk_exhaustion=bool(re.search(r'no space left on device',text,re.I)))
    if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):raise ValueError('diagnostics')
    if diagnostics['build']['warning_categories']!=state['warnings']:raise ValueError('warning summary mismatch')
    rows=parsed['frames']
    # Check actual cycle conservation independently of the remote parser. Every
    # frame includes ingress, every inspected block, emitted groups and drain.
    for r in rows:
        base=r['ingress']+r['scans']+r['groups']+r['delimiters']+r['group_stalls']+r['result_stalls']
        if r['total']<base or (r['scenario']==0 and r['total']-base!=2):raise ValueError('cycle conservation')
    aggregate=parsed['real_frame_totals']; comparison=[]
    for s in range(2):
        dense=next(r for r in aggregate if r['scenario']==s and r['sparse']==0)
        packed=next(r for r in aggregate if r['scenario']==s and r['sparse']==1)
        comparison.append(dict(scenario=s,dense_cycles=dense['total'],packed_cycles=packed['total'],
            saved_cycles=dense['total']-packed['total'],frame_interval_ratio=dense['total']/packed['total'],
            reduction_fraction=1-packed['total']/dense['total']))
    with (archive/'frame_measurements.csv').open('x',newline='',encoding='ascii') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='PASS_pre_LIF_scan_RTL_archive_audit_not_author_implementation',
        raw_files=len(files),raw_bytes=sum(files.values()),all_sizes_match=True,no_part=True,
        prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        original_consumer_files_byte_equal=len(SOURCES),original_fixture_files_byte_equal=len(fixture_files),
        archived_worker_strict_reparse_matches=True,start=state['start'],end=state['end'],
        stages={s:state[s] for s in ('build','run')},diagnostics=diagnostics,remote_observation=observation,
        real_frame_totals=aggregate,comparison=comparison,current_checks=parsed['current_checks'],
        common_preload_bank_writes=16800,scope=worker.SCOPE,dcp_crc='not applicable: no DCP',
        boundary='Ratios of summed per-frame ingress-through-result intervals, excluding common preload and harness idle gaps. No author implementation, whole-network speedup, continuous service, area, new STA, DDR or board claims. No production merge or cleanup. Log absence is not kernel OOM evidence.')
    with (archive/'terminal_audit.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
