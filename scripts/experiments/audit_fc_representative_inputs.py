"""One terminal archive acceptance of the new FC batches; no RTL rerun."""
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
from run_fc_representative_inputs import NAME, EXPERIMENT


def main():
    root = Path(local_path(ROOT))
    archive, prepared = root/RESULT/NAME, root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():
        raise FileExistsError('terminal audit already completed; do not repeat')
    state = json.loads((archive/'status.json').read_text())
    prep = json.loads((archive/'preparation.json').read_text())
    stages = ['build']+[f'run_{c["batch"]}_{arm}' for c in prep['configs'] for arm in ('off','on')]
    if not state.get('end') or not state['status'].startswith('PASS_') or any(state[s]['exit'] != 0 for s in stages):
        raise ValueError('successful terminal build/six run exits missing')
    code = '''import datetime,json,pathlib,shutil
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
    client = connect()
    try:
        observation = json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+NAME)+'\nSCRATCH='+repr(state['scratch'])+'\n'+code)))
    finally: client.close()
    if observation['active_eda'] or observation['scratch_noncache_files']:
        raise ValueError('active EDA or missing non-cache scratch originals')
    files = observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):
        raise ValueError('partial archive')
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size != size:
            raise ValueError('archive size mismatch: '+rel)
    inputs = [p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for path in inputs:
        if path.read_bytes() != (archive/path.relative_to(prepared)).read_bytes():
            raise ValueError('prepared input mismatch: '+str(path))
    candidate = root/'verification/experiments/scnn_control_payload/candidate_r4/rtl'
    sources = [p for p in candidate.rglob('*') if p.is_file()]
    if len(sources) != 109 or any(p.read_bytes() != (archive/'rtl'/p.relative_to(candidate)).read_bytes() for p in sources):
        raise ValueError('exact A source tree mismatch')
    provenance = json.loads((archive/'source_provenance.json').read_text())
    for row in provenance['raw_files']:
        if Path(row['source']).read_bytes() != (archive/row['archive']).read_bytes():
            raise ValueError('original official raw bytes mismatch')
    model = Path(provenance['original_model'])
    if model.read_bytes() != (archive/'model'/model.name).read_bytes():
        raise ValueError('original model NPZ bytes mismatch')
    if (archive/'launch_worker.py').read_bytes() != (root/'scripts/experiments/run_fc_representative_inputs.py').read_bytes():
        raise ValueError('actual worker revision differs')
    sys.path.insert(0,str(archive/'scripts'))
    for module in ('run_fc_profile_measurement','run_fc_final_counts','summarize_h40_activity'):
        sys.modules.pop(module,None)
    spec = importlib.util.spec_from_file_location('archived_fc_worker',archive/'launch_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    rows = []
    for config in prep['configs']:
        for arm in ('off','on'):
            bi = config['batch']
            stage = f'run_{bi}_{arm}'
            checks = worker.parse_log((archive/(stage+'.log')).read_text(),arm,config)
            if checks != json.loads((archive/(stage+'_checks.json')).read_text()):
                raise ValueError('actual archived parser/checks differ')
            record = next(r for r in state['runs'] if r['batch']==bi and r['arm']==arm)
            if any(record[k] != state[stage][k] for k in ('exit','command','wall_seconds')) or record['runtime_cycles'] != checks['runtime_cycles']:
                raise ValueError('status run record differs')
            buckets = checks['stages']
            legacy = checks['legacy']
            cold = sum(r['cycles'] for r in buckets.values())
            setup = sum(buckets[k]['cycles'] for k in ('parameter_load','initial_refill','initial_state','learning_configuration'))
            if cold != setup+buckets['outside']['cycles']+checks['runtime_cycles']:
                raise ValueError('post-reset accounting does not conserve cycles')
            rows.append(dict(batch=bi,arm=arm,steps=config['scope']['steps'],samples=16,
                input_events=legacy['input_events'],hidden_events=legacy['hidden_events'],
                parameter_load_cycles=buckets['parameter_load']['cycles'],initial_refill_cycles=buckets['initial_refill']['cycles'],
                initial_state_cycles=buckets['initial_state']['cycles'],learning_configuration_cycles=buckets['learning_configuration']['cycles'],
                outside_cycles=buckets['outside']['cycles'],runtime_cycles=checks['runtime_cycles'],
                forward_runtime_cycles=buckets['runtime']['cycles'],learning_tail_cycles=buckets['learning_tail_wait']['cycles'],
                runtime_refill_cycles=buckets['runtime_refill']['cycles'],counted_postreset_cycles=cold,
                commits=legacy['commits'],code_writes=legacy['code_writes'],residual_writes=legacy['residual_writes'],
                hidden_checks=legacy['hidden_checks'],dot_checks=legacy['dot_checks'],q6_checks=legacy['q6_checks'],
                w2_code_residual_values_checked=config['scope']['steps']*1600*2,
                frozen_w0_values_checked=369920,forward_reads=legacy['forward_reads'],learning_reads=legacy['learning_reads']))
    diagnostics = {}
    for stage in stages:
        text = (archive/(stage+'.log')).read_text(errors='replace')
        warnings = collections.Counter(re.findall(r'^%Warning-([^:]+):',text,re.M))
        diagnostics[stage] = dict(ordinary_warnings=sum(warnings.values()),warning_categories=dict(sorted(warnings.items())),
            errors=len(re.findall(r'^%Error',text,re.M)),fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b',text)),
            structural=[k for k in warnings if k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
            oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',text,re.I)),
            disk_exhaustion=bool(re.search(r'no space left on device',text,re.I)))
    if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):
        raise ValueError('fatal/structural tool diagnostics')
    if diagnostics['build']['warning_categories'] != state['build_warnings']:
        raise ValueError('build warnings differ')
    with (archive/'batch_measurements.csv').open('x',newline='',encoding='ascii') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {}
    for arm in ('off','on'):
        selected = [r for r in rows if r['arm']==arm]
        totals[arm] = {k:sum(r[k] for r in selected) for k in ('samples','steps','runtime_cycles','counted_postreset_cycles','commits','code_writes','residual_writes')}
        totals[arm]['cold_samples_per_second_15ns'] = totals[arm]['samples']/(totals[arm]['counted_postreset_cycles']*15e-9)
        totals[arm]['runtime_samples_per_second_15ns'] = totals[arm]['samples']/(totals[arm]['runtime_cycles']*15e-9)
    report = dict(status='PASS_three_new_FC_A_batch_pairs_archive_audit',raw_files=len(files),raw_bytes=sum(files.values()),
        all_sizes_match=True,no_part=True,prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        exact_A_sources_byte_equal=109,official_raw_files_byte_equal=len(provenance['raw_files']),original_model_npz_byte_equal=True,
        archived_worker_strict_reparse_matches=True,start=state['start'],end=state['end'],stages={s:state[s] for s in stages},
        diagnostics=diagnostics,remote_observation=observation,measurements=rows,totals=totals,dcp_crc='not applicable: no DCP',
        scope=worker.SCOPE,boundary='Monitor-counted post-reset cycles include parameter/state setup, configuration and runtime; initial six reset clocks excluded. File IO, reference generation and inter-process gaps are not RTL. All ordinary warnings retained; no new suppression. No cleanup or production change. Log absence is not kernel evidence.')
    with (archive/'terminal_audit.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
