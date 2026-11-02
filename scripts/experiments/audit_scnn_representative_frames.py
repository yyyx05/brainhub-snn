"""Terminal archive audit for new SCNN frames; does not rerun RTL or references."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_scnn_representative_frames import NAME, EXPERIMENT


def main():
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():
        raise FileExistsError('completed audit retained; do not repeat')
    state=json.loads((archive/'status.json').read_text())
    config=json.loads((archive/'preparation.json').read_text())
    samples=[r['sample'] for r in config['selected']]
    stages=['build']+[f'run_{i:03d}' for i in samples]
    if not state.get('end') or not state['status'].startswith('PASS_') or any(state[s]['exit']!=0 for s in stages):
        raise ValueError('natural successful exits missing; retain FAIL and inspect residual processes before any new launch')
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
    if observation['active_eda'] or observation['scratch_noncache_files']:
        raise ValueError('active EDA or scratch evidence not yet archived')
    files=observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):
        raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:
            raise ValueError('raw file size differs: '+rel)
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes()!=(archive/p.relative_to(prepared)).read_bytes():
            raise ValueError('prepared input differs: '+str(p))
    candidate=root/'verification/experiments/scnn_control_payload/candidate_r4/rtl'
    sources=[p for p in candidate.rglob('*') if p.is_file()]
    if len(sources)!=109 or any(p.read_bytes()!=(archive/'rtl'/p.relative_to(candidate)).read_bytes() for p in sources):
        raise ValueError('exact A source set differs')
    for bank in list(range(1,9))+[12]:
        rel=f'models/scnn_dvsgesture_vgg9_q0_15/bank_{bank:02d}.hex'
        if (root/rel).read_bytes()!=(archive/rel).read_bytes():
            raise ValueError('original numerical model differs')
    original=root/'data/experiments/scnn_subjects_264'
    raw=original/'test_264x20_inputs_q8_8.hex'
    if raw.stat().st_size!=264*20*2048*5: raise ValueError('original fixed hex layout changed')
    labels=(original/'test_264_labels.hex').read_text().splitlines()
    raw_slice_checks=[]
    with raw.open('rb') as stream:
        for sample in samples:
            stream.seek(sample*20*2048*5)
            original_slice=stream.read(2048*5)
            fixture=(archive/'fixtures'/f'sample_{sample:03d}'/'raw.hex').read_bytes()
            # The original Windows writer expands LF to CRLF; allow only that encoding change.
            if original_slice!=fixture.replace(b'\r\n',b'\n'):
                raise ValueError('raw frame slice differs from official package export')
            raw_slice_checks.append(dict(sample=sample,original_bytes=len(original_slice),fixture_bytes=len(fixture),
                exact_bytes_equal=original_slice==fixture,crlf_to_lf_bytes_equal=True,
                fixture_crlf_count=fixture.count(b'\r\n'),words=2048))
            label=int((archive/'fixtures'/f'sample_{sample:03d}'/'label.hex').read_text().strip(),16)
            if label!=int(labels[sample],16): raise ValueError('original label differs')
    if (archive/'launch_worker.py').read_bytes()!=(root/'scripts/experiments/run_scnn_representative_frames.py').read_bytes():
        raise ValueError('actual worker differs')
    # Resolve parser helpers from the fetched archive, not a later local revision.
    sys.path.insert(0,str(archive))
    for name in ('run_scnn_turnover_frame','prepare_scnn_schedule_probe','prepare_scnn_resident_compatibility'):
        sys.modules.pop(name,None)
    spec=importlib.util.spec_from_file_location('archived_representative_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    geometry=json.loads((archive/'geometry_reference.json').read_text())
    measurements=[];layer_rows=[]
    for row,record in zip(config['selected'],state['runs'],strict=True):
        sample=row['sample'];stage=f'run_{sample:03d}'
        checks=worker.parse_log((archive/(stage+'.log')).read_text(),sample,geometry)
        if checks!=json.loads((archive/(stage+'_checks.json')).read_text()) or record['sample']!=sample:
            raise ValueError('archived parser/checks mismatch')
        if record['frame_cycles']!=checks['frame_cycles'] or record['initialization_cycles']!=checks['initialization_cycles']:
            raise ValueError('status cycle mismatch')
        log=(archive/(stage+'.log')).read_text()
        starts=[int(re.search(r'cycle=(\d+)',line).group(1)) for line in log.splitlines()
                if line.startswith('SCNN_PROFILE_EVENT ') and 'kind=initialize_start' in line]
        if len(starts)!=1: raise ValueError('initialization boundary missing')
        total=checks['service']['end_cycle']-starts[0]
        gaps=total-checks['frame_cycles']-checks['initialization_cycles']
        if gaps<0: raise ValueError('overlapping cold service intervals')
        measurements.append(dict(sample=sample,label=row['label'],stratum=row['stratum'],
            raw_nonzero=row['first_frame_nonzero'],raw_q8_mass=row['first_frame_q8_mass'],
            initialization_cycles=checks['initialization_cycles'],frame_cycles=checks['frame_cycles'],
            boundary_gap_cycles=gaps,initialize_start_to_service_end_cycles=total,scalar_checks=409600,
            parameter_reads=checks['service']['reads'],responses=checks['service']['responses'],
            replacements=checks['service']['replacements']))
        density=json.loads((archive/'fixtures'/f'sample_{sample:03d}'/'reference.json').read_text())['layers']
        for bucket,reference in zip(checks['tables']['bucket'],density,strict=True):
            layer_rows.append(dict(sample=sample,**bucket,reference_spikes=reference['spikes'],
                                  reference_scalars=reference['scalars'],pooled_nonzero=reference['pooled_nonzero']))
    diagnostics={}
    for stage in stages:
        text=(archive/(stage+'.log')).read_text(errors='replace')
        warnings=collections.Counter(re.findall(r'^%Warning-([^:]+):',text,re.M))
        diagnostics[stage]=dict(ordinary_warnings=sum(warnings.values()),warning_categories=dict(sorted(warnings.items())),
            errors=len(re.findall(r'^%Error',text,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b',text)),
            structural=[k for k in warnings if k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
            oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',text,re.I)),
            disk_exhaustion=bool(re.search(r'no space left on device',text,re.I)))
    if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):
        raise ValueError('tool diagnostics')
    if diagnostics['build']['warning_categories']!=state['build_warnings']:
        raise ValueError('warning summary differs')
    for name,rows in (('frame_measurements.csv',measurements),('layer_measurements.csv',layer_rows)):
        with (archive/name).open('x',newline='',encoding='ascii') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    total=sum(r['initialize_start_to_service_end_cycles'] for r in measurements)
    report=dict(status='PASS_three_representative_A_cold_raw_frames_archive_audit',
        raw_files=len(files),raw_bytes=sum(files.values()),all_sizes_match=True,no_part=True,
        prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        exact_A_sources_byte_equal=109,original_model_banks_byte_equal=9,
        original_raw_frame_slices_equal_after_crlf_to_lf=3,raw_slice_checks=raw_slice_checks,
        archived_worker_strict_reparse_matches=True,start=state['start'],end=state['end'],
        stages={s:state[s] for s in stages},diagnostics=diagnostics,remote_observation=observation,
        samples=samples,scalar_checks=1228800,measurements=measurements,
        cold_initialize_start_to_service_end_cycles=total,cold_frame_rate_projection_15ns=len(samples)/(total*15e-9),
        dcp_crc='not applicable: no DCP',scope=worker.SCOPE,
        boundary='Three independent cold first frames, not continuous sequences. Input loading, reference generation, file IO and process gaps outside RTL intervals. Existing warning suppression flags retained. No new speedup, P95, DDR, board or physical claim. No cleanup or production change. Log absence is not kernel evidence.')
    with (archive/'terminal_audit.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
