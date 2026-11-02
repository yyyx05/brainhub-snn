"""Audit a fully fetched, naturally completed T20 experiment without rerunning it."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
from run_srnn_resident_remote import ROOT,RESULT,connect,local_path
from run_scnn_representative_sequence import NAME,EXPERIMENT,SAMPLE
from audit_source_stream_comparison import inventory


def main():
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():raise FileExistsError('terminal audit exists; do not repeat')
    state=json.loads((archive/'status.json').read_text())
    if not state.get('end') or state['status']!='PASS_one_representative_complete_T20_window' or any(state[s]['exit']!=0 for s in ('build','run')):
        raise ValueError('natural successful terminal exits missing; retain failures and check residual processes')
    client=connect()
    try: observation=inventory(client,NAME,state['scratch'])
    finally:client.close()
    if observation['active_eda'] or observation['scratch_noncache']:
        raise ValueError('active EDA or noncache scratch originals not archived')
    files=observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:raise ValueError('raw file size mismatch: '+rel)
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes()!=(archive/p.relative_to(prepared)).read_bytes():raise ValueError('prepared input differs: '+str(p))
    candidate=root/'verification/experiments/scnn_control_payload/candidate_r4/rtl'
    sources=[p for p in candidate.rglob('*') if p.is_file()]
    if len(sources)!=109 or any(p.read_bytes()!=(archive/'rtl'/p.relative_to(candidate)).read_bytes() for p in sources):
        raise ValueError('exact A sources differ')
    for i in list(range(1,9))+[12]:
        rel=f'models/scnn_dvsgesture_vgg9_q0_15/bank_{i:02d}.hex'
        if (root/rel).read_bytes()!=(archive/rel).read_bytes():raise ValueError('original model bank differs')
    source=root/'data/experiments/scnn_subjects_264'
    with (source/'test_264x20_inputs_q8_8.hex').open('rb') as f:
        f.seek(SAMPLE*20*2048*5);raw=f.read(20*2048*5)
    fixture=(archive/'fixture/raw.hex').read_bytes()
    if len(raw)!=204800 or raw!=fixture.replace(b'\r\n',b'\n'):raise ValueError('official complete raw window differs')
    if int((archive/'fixture/label.hex').read_text().strip(),16)!=int((source/'test_264_labels.hex').read_text().splitlines()[SAMPLE],16):
        raise ValueError('official label differs')
    if (archive/'launch_worker.py').read_bytes()!=(root/'scripts/experiments/run_scnn_representative_sequence.py').read_bytes():
        raise ValueError('actual worker differs')
    sys.path.insert(0,str(archive))
    for name in ('prepare_scnn_schedule_probe','prepare_scnn_resident_compatibility','run_source_stream_comparison'):
        sys.modules.pop(name,None)
    spec=importlib.util.spec_from_file_location('actual_sequence_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    checks=worker.parse_log((archive/'run.log').read_text(),json.loads((archive/'geometry_reference.json').read_text()))
    if checks!=json.loads((archive/'checks.json').read_text()) or checks['cold_interval_cycles']!=state['cold_interval_cycles']:
        raise ValueError('archived parser/checks/status mismatch')
    from run_source_stream_comparison import diagnostics
    diag={s:diagnostics((archive/(s+'.log')).read_text()) for s in ('build','run')}
    if any(diag[s]!=state[s]['diagnostics'] for s in diag):raise ValueError('diagnostic summary mismatch')
    if any(d['errors'] or d['fatal'] or d['structural'] for d in diag.values()):raise ValueError('tool diagnostics')
    reference=json.loads((archive/'integer_reference_checks.json').read_text())
    measurements=[];layers=[]
    for measured,ref in zip(checks['frames'],reference['frames'],strict=True):
        if measured['frame']!=ref['frame']:raise ValueError('reference frame order')
        measurements.append(dict(sample=SAMPLE,**measured,input_nonzero=ref['input_nonzero'],input_q8_mass=ref['input_q8_mass']))
        for layer in ref['layers']:
            index=ref['frame']*8+layer['layer']-1
            layers.append(dict(sample=SAMPLE,**checks['tables']['bucket'][index],reference_spikes=layer['spikes'],
                reference_values=layer['values'],reference_nonzero_state=layer['nonzero_state']))
    cold=checks['cold_interval_cycles'];warm=checks['post_first_frame_interval_cycles']
    if cold!=checks['initialization_cycles']+checks['initialization_end_to_frame0']+checks['frame_cycles']+checks['between_frame_cycles']:
        raise ValueError('cold cycle accounting')
    if any(r['input_nonzero']==0 for r in measurements):
        # Zero input is valid and must be retained, never removed as a fast frame.
        zero_input_frames=[r['frame'] for r in measurements if r['input_nonzero']==0]
    else:zero_input_frames=[]
    for name,table in (('frame_measurements.csv',measurements),('layer_measurements.csv',layers)):
        with (archive/name).open('x',newline='',encoding='ascii') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    report=dict(status='PASS_one_representative_T20_raw_sequence_archive_audit',raw_files=len(files),raw_bytes=sum(files.values()),
        all_sizes_match=True,no_part=True,prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        exact_A_sources_byte_equal=109,original_model_banks_byte_equal=9,
        official_raw_window=dict(sample=SAMPLE,words=40960,original_bytes=len(raw),fixture_bytes=len(fixture),
            exact_bytes_equal=raw==fixture,crlf_to_lf_bytes_equal=True),archived_worker_strict_reparse=True,
        start=state['start'],end=state['end'],stages={s:state[s] for s in ('build','run')},diagnostics=diag,
        remote_observation=observation,frames=20,scalar_checks=8192000,zero_input_frames=zero_input_frames,
        cold_interval_cycles=cold,cold_sequence_rate_projection_15ns=1/(cold*15e-9),cold_frame_rate_projection_15ns=20/(cold*15e-9),
        post_first_frame_interval_cycles=warm,post_first_19_frame_rate_projection_15ns=19/(warm*15e-9),
        initialization_cycles=checks['initialization_cycles'],initialization_end_to_frame0=checks['initialization_end_to_frame0'],
        between_frame_cycles=checks['between_frame_cycles'],service=checks['service'],dcp_crc='not applicable: no DCP',scope=worker.SCOPE,
        boundary='Original exported most-active20-frame window only. Host state spill/restore retained, not on-chip full-state capacity proof. File IO/reference generation excluded from RTL intervals. Not an A/B speedup, IID/P95, full Test, learning, DDR or board result. No production merge, physical rerun or cleanup.')
    with (archive/'terminal_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
