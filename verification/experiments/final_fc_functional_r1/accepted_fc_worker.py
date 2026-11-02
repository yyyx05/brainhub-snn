"""New class/density-stratified N-MNIST Test batches on the isolated A core."""
import argparse
import csv
import datetime as dt
import difflib
from fractions import Fraction
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NAME = 'fc_representative_a_r1'
EXPERIMENT = Path('verification/experiments/fc_representative_inputs/a_r1')
TB = 'tb_fc_profile_measurement.sv'
SCOPE = ('Three independent cold B16 batches, 48 official clean N-MNIST Test records; '
         'class/density-stratified central ranks selected before RTL timing. '
         'Original H160 INT5/INT8 model, first saccade, 1ms counts, complete native '
         'batch time extent including padding. Exact A/512/no-model-key; learning '
         'off/on start independently from the same model. Not full-Test RTL, IID, '
         'stable P95, ongoing adaptation, DDR, board throughput or a new speedup comparison.')


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def select_rows(rows):
    """Read only identity and input statistics, never predictions or RTL timing."""
    fields = ('dataset_index', 'label', 'valid_timesteps', 'events', 'repeated_extra_events')
    clean = [dict({k: int(row[k]) for k in fields}, rawfilename=row['rawfilename']) for row in rows]
    if len({r['dataset_index'] for r in clean}) != len(clean):
        raise ValueError('duplicate official index')
    batches = [[], [], []]
    strata = []
    for label in range(10):
        ordered = sorted((r for r in clean if r['label'] == label),
                         key=lambda r: (Fraction(r['events']-r['repeated_extra_events'], r['valid_timesteps']),
                                        r['dataset_index']))
        count = 2 if label < 6 else 1
        q, rem = divmod(len(ordered), 3)
        start = 0
        for density in range(3):
            size = q + int(density < rem)
            group = ordered[start:start+size]
            if len(group) < count:
                raise ValueError('insufficient class/density stratum')
            middle = (len(group)-count)//2
            chosen = group[middle:middle+count]
            batches[density].extend(chosen)
            strata.append(dict(label=label, density_tertile=density, population=size,
                               selected_indices=[r['dataset_index'] for r in chosen],
                               start_rank=start, selected_ranks=list(range(start+middle, start+middle+count))))
            start += size
    if any(len(b) != 16 for b in batches) or len({r['dataset_index'] for b in batches for r in b}) != 48:
        raise ValueError('selection does not contain three disjoint B16 batches')
    return batches, strata


def prepare(data_root, model_path, source_root):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import tonic
    from unittest.mock import patch
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    sys.path.insert(0, str(root))
    from scripts import build_fc_final_counts_fixture as reference
    from scripts import build_fc_w2_numeric_reference as numeric
    from scripts.run_fc_profile_measurement import inspect_pair
    out = root/EXPERIMENT
    out.mkdir(parents=True, exist_ok=False)
    old = root/RESULT/'fc_banked_scanner_r3'
    snapshot = out/'reference_scripts'
    snapshot.mkdir()
    for name in ('evaluate_fc_formal_test_numeric.py', 'run_fc_tp_supervised_shift_adaptation.py'):
        shutil.copy2(source_root/'scripts'/name, snapshot/name)
    for name in ('build_fc_final_counts_fixture.py', 'build_fc_w2_numeric_reference.py',
                 'build_fc_tp_b16_softmax_rtl_fixture.py'):
        shutil.copy2(root/'scripts'/name, snapshot/name)
    spec = importlib.util.spec_from_file_location('archived_fc_scalar_reference', snapshot/'evaluate_fc_formal_test_numeric.py')
    scalar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scalar)
    model_dir = out/'model'
    model_dir.mkdir()
    for path in [model_path]+[model_path.parent/f'w2_cluster_{i:02d}.hex' for i in range(4)]:
        shutil.copy2(path, model_dir/path.name)
        if path.read_bytes() != (model_dir/path.name).read_bytes():
            raise ValueError('original model import changed bytes')
    archived_params = old/'results/paper_evidence/supplement_20261004/algorithm/fc_final_counts_reference/t99_off/parameters.mem'
    codes = np.array([int(x, 16) for x in archived_params.read_text().split()], dtype=np.int64)
    codes = (codes+32768) % 65536-32768
    if codes.shape != (371520,):
        raise ValueError('archived original parameter extent')
    exported = out/'model_archived_tb_w2'
    exported.mkdir()
    w2 = codes[369920:].reshape(160, 10)
    for tile in range(4):
        lines = [f'{sum((int(v)&255) << (8*c) for c,v in enumerate(row)):020x}'
                 for row in w2[tile*40:(tile+1)*40]]
        (exported/f'w2_cluster_{tile:02d}.hex').write_text('\n'.join(lines)+'\n')
    numeric.MODEL_DIR, numeric.MODEL, numeric.EXPORTED = model_dir, model_dir/model_path.name, exported
    reference.MODEL = numeric.MODEL
    model = numeric.load_model()
    np.testing.assert_array_equal(np.concatenate((model['W0'].ravel(), model['W2'].ravel())), codes)
    if round(float(model['W2_scale'])*64*65536) != 47303:
        raise ValueError('original model scale changed')
    with (root/'results/paper_evidence/supplement_20261004/algorithm/fc_formal_test_a1/predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        batches, strata = select_rows(list(csv.DictReader(stream)))
    dump(out/'selection.json', dict(scope=SCOPE, batches=batches, strata=strata,
        policy='Within each class sort official Test by occupied input bins / native T, tie by index; split into three equal-rank strata; take central adjacent two ranks for labels0..5 and central one rank for6..9. Same class quota and class/density-rank execution order in all batches. No prediction/cycle used.',
        statistics_source='results/paper_evidence/supplement_20261004/algorithm/fc_formal_test_a1/predictions.csv',
        limitations='Purposive 48-record subset; fixed class quotas are not population weighting; no whole-sample IID/tail claims.',
        human_direction='2026-10-19: user does not know source paths and delegates choice; only previously documented exact external sources read.'))
    (out/'raw').mkdir()
    (out/'counts').mkdir()
    transform = tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000)
    with patch.object(tonic.datasets.NMNIST, 'download', side_effect=RuntimeError('read-only source; no download/extraction')):
        dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=False, first_saccade_only=True)
    if len(dataset) != 10000:
        raise ValueError('expected complete locally available official Test inventory')
    provenance = dict(data_root=str(data_root), original_model=str(model_path), model_bytes=model_path.stat().st_size,
        parameter_values_compared=371520, parameter_mismatches=0, tonic_version=tonic.__version__, raw_files=[],
        preprocessing='Original first_saccade_only=True (t<100000 us), ToFrame(time_window=1000), p/y/x flatten, no shift, no event clipping, batch max native T padding.',
        scalar_checks=[], production_changed=False)
    for i, path in enumerate(dict.fromkeys((inspect.getsourcefile(tonic.datasets.NMNIST),
                                            inspect.getsourcefile(tonic.transforms.ToFrame),
                                            inspect.getsourcefile(tonic.functional.to_frame_numpy)))):
        shutil.copy2(path, snapshot/f'tonic_{i}_{Path(path).name}')
    reference.OUTPUT = out/'fixtures'
    configs = []
    for bi, rows in enumerate(batches):
        frames = []
        for row in rows:
            index = row['dataset_index']
            rawpath = Path(dataset.data[index])
            if rawpath.resolve().relative_to(data_root.resolve()).as_posix() != row['rawfilename']:
                raise ValueError('dataset ordering/path changed')
            dest = out/'raw'/f'{index:05d}.bin'
            shutil.copy2(rawpath, dest)
            if dest.read_bytes() != rawpath.read_bytes():
                raise ValueError('raw source copy differs')
            # Read the archived input, not an uncaptured external copy.
            dataset.data[index] = str(dest)
            events, label = dataset[index]
            counts = transform(events).reshape(-1, 2312).astype(np.int64)
            stats = dict(valid_timesteps=len(counts), events=int(counts.sum()),
                         repeated_extra_events=int(counts.sum())-int(np.count_nonzero(counts)))
            if label != row['label'] or any(stats[k] != row[k] for k in stats):
                raise ValueError('raw input no longer matches historical input statistics')
            if counts.min() < 0 or not 1 <= len(counts) <= 99:
                raise ValueError('unsupported count/native time extent')
            frames.append(counts)
            provenance['raw_files'].append(dict(dataset_index=index, source=str(rawpath), archive=dest.relative_to(out).as_posix(),
                bytes=dest.stat().st_size, label=int(label), **stats))
        steps = max(map(len, frames))
        dense = np.zeros((16, steps, 2312), dtype=np.uint32)
        for si, counts in enumerate(frames):
            dense[si, :len(counts)] = counts
        npz = out/'counts'/f'batch_{bi}.npz'
        np.savez_compressed(npz, dense_counts=dense, labels=np.array([r['label'] for r in rows]),
                            lengths=np.array(list(map(len, frames))), dataset_indices=np.array([r['dataset_index'] for r in rows]))
        generated = {}
        for arm in ('off', 'on'):
            directory = reference.OUTPUT/f'batch_{bi}'/arm
            generated[arm] = reference.build_fixture(directory, steps, arm == 'on', npz)
        # Existing independently written scalar equations, full time axis of one
        # new record per stratum; all records still get full RTL scalar checks.
        observed = scalar.run_batch(dense[:1], model, capture=(0,))
        check = scalar.check_scalar(dense[0], model, observed['history'][0])
        provenance['scalar_checks'].append(dict(batch=bi, dataset_index=rows[0]['dataset_index'], **check))
        for key, file, shape, bits in (('pre','pre',(steps,16,160),32), ('trace','trace',(steps,16,160),None),
                                     ('spike','spike',(steps,16,160),None), ('dot','dot',(steps,16,10),32),
                                     ('logits','cumulative',(steps,16,10),32), ('q6','q6',(steps,16,10),16)):
            values = np.array([int(x,16) for x in (reference.OUTPUT/f'batch_{bi}/off/{file}.mem').read_text().split()],dtype=np.int64).reshape(shape)
            if bits:
                values = (values+(1 << (bits-1))) % (1 << bits)-(1 << (bits-1))
            np.testing.assert_array_equal(values[:,0], np.array([r[key] for r in observed['history'][0]]))
        scope = inspect_pair(reference.OUTPUT/f'batch_{bi}/on', reference.OUTPUT/f'batch_{bi}/off', steps)
        configs.append(dict(batch=bi, scope=scope, expected_totals={a: generated[a]['totals'] for a in generated}))
        print(json.dumps(dict(prepared_batch=bi,steps=steps,indices=[r['dataset_index'] for r in rows],scalar=check)),flush=True)
    dump(out/'source_provenance.json', provenance)
    original = (old/TB).read_text()
    anchor = '.TOPOLOGY_MASK(TOPOLOGY_MASK),'
    if original.count(anchor) != 1:
        raise ValueError('ambiguous cache capacity anchor')
    text = original.replace(anchor, anchor+'\n        .M2_CACHE_DEPTH(512),',1)
    (out/TB).write_text(text, newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
        fromfile='accepted_fc_banked_scanner_r3/tb', tofile='representative_A/tb')))
    shutil.copytree(root/'verification/experiments/scnn_control_payload/candidate_r4/rtl', out/'rtl')
    sources = sorted(p.relative_to(out).as_posix() for p in (out/'rtl').rglob('*.sv'))
    if len(sources) != 108:
        raise ValueError('incomplete A source tree')
    shutil.copytree(old/'tb',out/'tb')
    (out/'scripts').mkdir()
    for name in ('run_fc_profile_measurement.py','run_fc_final_counts.py','summarize_h40_activity.py'):
        shutil.copy2(old/'scripts'/name,out/'scripts'/name)
    (out/'sources.f').write_text(''.join('"'+p+'"\n' for p in sources+[TB,'tb/monitors/supplement_h40_activity_monitor.sv']))
    dump(out/'build_command.json', json.loads((old/'status.json').read_text())['build']['command'])
    dump(out/'preparation.json',dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,configs=configs,
        candidate='verification/experiments/scnn_control_payload/candidate_r4/rtl',m2_cache_depth=512,model_key=False,
        reset_between_batches=True,reset_between_arms=True,source_files=109,production_change=False))
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    files = [p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    print(json.dumps(dict(prepared=str(out),files=len(files),bytes=sum(p.stat().st_size for p in files))),flush=True)


def parse_log(log, arm, config):
    from run_fc_profile_measurement import parse_log as parse_profile
    result = parse_profile(log, arm, config['scope'], True)
    expected = config['expected_totals'][arm]
    legacy = result['legacy']
    checks = dict(input_events=expected['input_events'], hidden_events=expected['hidden_spikes'],
                  code_writes=expected['code_changes'] if arm == 'on' else 0,
                  residual_writes=expected['residual_changes'] if arm == 'on' else 0)
    if any(legacy[k] != v for k,v in checks.items()):
        raise ValueError('independent input/spike/update totals differ: '+repr((checks,legacy)))
    if re.search(r'^%Warning|%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b', log,re.M):
        raise ValueError('runtime diagnostic')
    result.update(scope_limit=SCOPE,batch_index=config['batch'],independent_totals=checks)
    return result


def worker():
    root = Path.cwd()
    sys.path.insert(0,str(root/'scripts'))
    scratch = Path('/tmp/snn_'+NAME)
    scratch.mkdir(exist_ok=False)
    state = dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE,runs=[])
    def save(): dump(root/'status.json',state)
    def execute(stage, command, timeout):
        state['status'] = stage+'_running'
        save()
        begin = time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try: code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage] = dict(command=command,exit=None,pid=process.pid,timed_out=True,wall_seconds=time.monotonic()-begin)
                save()
                raise
        state[stage] = dict(command=command,exit=code,wall_seconds=time.monotonic()-begin)
        save()
        if code: raise RuntimeError(stage+' exit '+str(code))
    try:
        prep = json.loads((root/'preparation.json').read_text())
        command = json.loads((root/'build_command.json').read_text())
        command[command.index('--Mdir')+1] = str(scratch/'obj')
        command[command.index('-f')+1] = 'sources.f'
        execute('build',command,1800)
        build = (root/'build.log').read_text(errors='replace')
        state['build_warnings'] = {k:len(re.findall(r'^%Warning-'+k+r':',build,re.M)) for k in sorted(set(re.findall(r'^%Warning-(\w+):',build,re.M)))}
        if '%Error' in build or any(state['build_warnings'].get(k) for k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')):
            raise ValueError('structural build diagnostics')
        for config in prep['configs']:
            bi = config['batch']
            for arm in ('off','on'):
                stage = f'run_{bi}_{arm}'
                execute(stage,[str(scratch/'obj/Vtb_fc_profile_measurement'),f'+FIXTURE=fixtures/batch_{bi}/{arm}',
                    f'+STEPS={config["scope"]["steps"]}',f'+LEARNING={int(arm=="on")}', '+BATCHES=1'],1800)
                checks = parse_log((root/(stage+'.log')).read_text(errors='replace'),arm,config)
                dump(root/(stage+'_checks.json'),checks)
                state['runs'].append(dict(batch=bi,arm=arm,**state[stage],runtime_cycles=checks['runtime_cycles']))
                save()
        state['status'] = 'PASS_three_representative_FC_A_cold_batch_pairs'
    except Exception as error: state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True)
    save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared = Path(local_path(ROOT/EXPERIMENT))
    destination = REMOTE+'/'+NAME
    inputs = sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    total = sum(p.stat().st_size for p in inputs)
    probe = '''import datetime,json,pathlib,shutil
p=pathlib.Path(DEST)
if p.exists() or pathlib.Path(SCRATCH).exists():raise FileExistsError('destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p.parent).free,tmp_free=shutil.disk_usage('/tmp').free,input_bytes=INPUT_BYTES)
if record['data_free'] < INPUT_BYTES+128*1024**2 or record['tmp_free'] < 1024**3:raise RuntimeError('insufficient headroom for prior 2-job FC build')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
    client = connect()
    try:
        record = json.loads(run(client,'python3 -c '+shlex.quote('DEST='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nINPUT_BYTES='+str(total)+'\n'+probe)))
        with client.open_sftp() as sftp:
            for path in inputs:
                remote = destination+'/'+path.relative_to(prepared).as_posix()
                upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual:
                    if actual.read() != path.read_bytes(): raise ValueError('uploaded input differs: '+str(path))
        command = 'cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_fc_representative bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_fc_representative',prepared_input_files_byte_equal=len(inputs))
        out = Path(local_path(ROOT/RESULT/NAME))
        out.mkdir(parents=True,exist_ok=True)
        dump(out/'launch_observation.json',record)
        print(json.dumps(record))
    finally: client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch'))
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--source-root',type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not all((args.data_root,args.model,args.source_root)):
            parser.error('explicit previously documented source paths required')
        prepare(args.data_root,args.model,args.source_root)
    else: {'worker':worker,'launch':launch}[args.action]()
