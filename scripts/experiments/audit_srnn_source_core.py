"""One terminal audit of the new raw-count source/whole-core comparison."""
import argparse
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
from run_srnn_source_core import NAME,EXPERIMENT


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    global NAME,EXPERIMENT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=NAME,choices=('srnn_source_core_r1','srnn_source_core_r2'))
    parser.add_argument('--failed-build',action='store_true')
    args=parser.parse_args();NAME=args.name;EXPERIMENT=Path('verification/experiments')/NAME
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    report_name='failure_archive_audit.json' if args.failed_build else 'terminal_audit.json'
    if (archive/report_name).exists():raise FileExistsError('already audited; do not repeat')
    state=json.loads((archive/'status.json').read_text())
    arms=('buffered_off','buffered_on','stream_off','stream_on')
    stages=('build',)+tuple('run_'+a for a in arms)
    if args.failed_build:
        if state.get('status')!='FAIL' or not state.get('end') or any(s in state for s in stages[1:]):
            raise ValueError('natural pre-simulation failure required')
        stages=('build',)
    elif not state.get('end') or not state['status'].startswith('PASS_'):raise ValueError('natural complete PASS required')
    if any(state.get(s,{}).get('exit')!=0 for s in stages):raise ValueError('real exit not zero')
    code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);scratch=pathlib.Path(SCRATCH);active=[]
skip={'obj','elab_obj','__pycache__','.Xil'}
def inventory(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),
 scratch_noncache_files=inventory(scratch),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    client=connect()
    try:observation=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\nSCRATCH='+repr(state['scratch'])+'\n'+code)))
    finally:client.close()
    if observation['active_eda'] or observation['scratch_noncache_files']:raise ValueError('active EDA or missing scratch evidence')
    files=observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size!=size:raise ValueError('raw size '+rel)
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes()!=(archive/p.relative_to(prepared)).read_bytes():raise ValueError('input bytes '+str(p))
    candidate=root/'verification/experiments/scnn_control_payload/candidate_r4'
    originals=[p for p in (candidate/'rtl').rglob('*') if p.is_file()]
    changed=[p.relative_to(candidate).as_posix() for p in originals if p.read_bytes()!=(archive/p.relative_to(candidate)).read_bytes()]
    wanted={'rtl/control/unified_fc_srnn_h40_p4_layer_client_v1.sv',
            'rtl/top/unified_h40_final_composition_top_v1.sv',
            'rtl/top/unified_h40_array16_three_topology_top_v1.sv',
            'rtl/top/unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv'}
    if len(originals)!=109 or set(changed)!=wanted:raise ValueError('candidate source boundary')
    prior=root/'verification/experiments/srnn_representative_inputs/a_r1'
    numeric=[p for p in (prior/'multi_fixture').rglob('*') if p.is_file()]
    for p in numeric:
        if p.read_bytes()!=(archive/p.relative_to(prior)).read_bytes():raise ValueError('original numerical input changed')
    for name in ('source_scan.sv','source_stream.sv'):
        if (archive/name).read_bytes()!=(root/'verification/experiments/nearest_neighbor/buffered_ram_r1'/name).read_bytes():
            raise ValueError('accepted producer differs')
    if (archive/'rtl/control/unified_fc_srnn_h40_p4_layer_client_v1.sv').read_text()!=(root/'verification/experiments/srnn_endmarked_layer_r1/endmarked.sv').read_text():
        raise ValueError('locally tested layer client differs')
    load('run_srnn_representative_inputs',archive/'run_srnn_representative_inputs.py')
    worker=load('archived_source_core_worker',archive/'launch_worker.py')
    info=json.loads((archive/'multi_fixture/reference.json').read_text())
    checks={} if args.failed_build else json.loads((archive/'checks.json').read_text())
    diagnostics={};seq_rows=[];frame_rows=[]
    for stage in stages:
        text=(archive/(stage+'.log')).read_text(errors='replace')
        warnings=collections.Counter(re.findall(r'^%Warning-([^:]+):',text,re.M))
        diagnostics[stage]=dict(warning_categories=dict(sorted(warnings.items())),ordinary_warnings=sum(warnings.values()),
            errors=len(re.findall(r'^%Error',text,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b',text)),
            structural=[k for k in warnings if k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
            oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',text,re.I)),
            disk_exhaustion=bool(re.search(r'no space left on device',text,re.I)))
        if stage=='build':continue
        name=stage.removeprefix('run_');policy,arm=name.split('_');streaming=int(policy=='stream')
        parsed=worker.parse_log(text,arm,info,streaming)
        if parsed!=checks[name] or parsed!=state[name]:raise ValueError('strict reparse '+name)
        for bad in (text.replace('SOURCE_CORE_PASS','MISSING'),text.replace('SOURCE_CORE_FRAME','MISSING',1),text+'\n%Error: synthetic\n'):
            try:worker.parse_log(bad,arm,info,streaming)
            except ValueError:pass
            else:raise AssertionError('damaged transcript accepted')
        seq_rows.extend(dict(arm=name,**r) for r in parsed['sequences'])
        frame_rows.extend(dict(arm=name,**r) for r in parsed['source_frames'])
    if args.failed_build:
        build_text=(archive/'build.log').read_text(errors='replace')
        observed=dict(ordinary=len(re.findall(r'^%Warning-',build_text,re.M)),
                      structural=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',build_text,re.M))
        if observed!=state['build_warnings'] or state['error']!="ValueError('structural warning')":
            raise ValueError('original structural failure does not reproduce')
        if not observed['structural'] or list(archive.glob('run_*.log')):
            raise ValueError('unexpected failure scope')
        report=dict(status='ARCHIVED_NOT_PASSED_pre_simulation_structural_failure',
            raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
            prepared_input_files_byte_equal=len(inputs),prepared_input_bytes=sum(p.stat().st_size for p in inputs),
            changed_core_files=changed,unchanged_A_source_files=105,numerical_files_byte_equal=len(numeric),
            original_status=state['status'],start=state['start'],end=state['end'],build=state['build'],
            original_structural_failure_reproduced=True,simulation_started=False,
            diagnostics=diagnostics,remote_observation=observation,dcp_crc='not applicable: no DCP',
            scope=worker.SCOPE,boundary='Original worker/status/logs retained. No RTL numerical or throughput result. No cleanup.')
        with (archive/report_name).open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
        print(json.dumps(report,indent=2));return
    if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):raise ValueError('diagnostic failure')
    comparison=[]
    for arm in ('off','on'):
        a,b=checks['buffered_'+arm],checks['stream_'+arm]
        for field in ('initialization_including_reset_cycles','state_setup_cycles'):
            if a[field]!=b[field]:raise ValueError('initialization or reset schedule changed')
        for x,y in zip(a['sequences'],b['sequences']):
            if any(x[k]!=y[k] for k in ('bone','hidden','logits','store','code','residual','commits','backbone_loads','wh1_loads','wout_loads')):
                raise ValueError('same numerical outcomes required')
        ac,bc=a['total']['cycles'],b['total']['cycles']
        comparison.append(dict(learning=arm,buffered_total_cycles=ac,stream_total_cycles=bc,
            saved_cycles=ac-bc,cycle_ratio=ac/bc,cycle_reduction_percent=100*(ac-bc)/ac,
            buffered_service_cycles=a['service_cycles'],stream_service_cycles=b['service_cycles']))
    for name,rows in (('sequence_measurements.csv',seq_rows),('source_frame_measurements.csv',frame_rows),('comparison.csv',comparison)):
        with (archive/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='PASS_raw_source_A_core_comparison_terminal_audit',
        raw_files=len(files),raw_bytes=sum(files.values()),all_raw_sizes_match=True,no_part=True,
        prepared_input_files_byte_equal=len(inputs),prepared_input_bytes=sum(p.stat().st_size for p in inputs),
        changed_core_files=changed,unchanged_A_source_files=105,numerical_files_byte_equal=len(numeric),
        archived_worker_strict_reparse_matches=True,damaged_transcripts_rejected=12,
        original_status=state['status'],start=state['start'],end=state['end'],
        stages={s:state[s] for s in stages},diagnostics=diagnostics,remote_observation=observation,
        comparison=comparison,dcp_crc='not applicable: no DCP',scope=worker.SCOPE,
        boundary='New protocol and raw producers have no whole-core physical result. Fixed15ns values are nominal projections only, not inherited A timing. Counts include accepted FF tail; original recurrent/host/state protocol remains. Existing representative input set, not new distribution. No cleanup/production merge/old table replacement.')
    with (archive/'terminal_audit.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
