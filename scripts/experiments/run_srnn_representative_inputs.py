"""Measure eight input-stratified native SHD sequences on the isolated A core."""
import argparse
import datetime as dt
import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME = 'srnn_representative_a_r1'
EXPERIMENT = Path('verification/experiments/srnn_representative_inputs/a_r1')
TB = 'tb_srnn_profile_measurement.sv'
SCOPE = ('Eight new official SHD samples from the existing first-64 pool, selected '
         'by native length and occupied input-bin density before RTL timing. '
         'Exact payload-A RTL, compact512, no model-key. Independent dynamics; '
         'learning state persists in ascending official-index order. '
         'Not full Test, IID accuracy, stable P95, DDR, or a new speedup comparison.')


def select_rows(rows):
    import numpy as np
    eligible = sorted((r.copy() for r in rows if r['dataset_index'] not in (0, 1)),
                      key=lambda r: (r['time_steps'], r['dataset_index']))
    chosen, strata = [], []
    for quartile, group in enumerate(np.array_split(eligible, 4)):
        ordered = sorted(group, key=lambda r: (r['occupied_fraction'], r['dataset_index']))
        for half, subset in enumerate(np.array_split(ordered, 2)):
            # A central rank in each stratum, never a timing-selected sample.
            row = subset[(len(subset) - 1) // 2].copy()
            row.update(length_quartile=quartile, density_half=half)
            chosen.append(row)
            strata.append(dict(length_quartile=quartile, density_half=half,
                               population=[r['dataset_index'] for r in subset],
                               selected=row['dataset_index']))
    return sorted(chosen, key=lambda r: r['dataset_index']), strata


def transform_tb(source, info):
    text = source
    def once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('ambiguous native TB anchor: ' + old)
        text = text.replace(old, new, 1)
    n = len(info['native_steps'])
    once('localparam int H = 450, T = 78, BONE_BASE = 256;',
         f"localparam int H = 450, T = {info['capacity_steps']}, BONE_BASE = 256;")
    once('logic [63:0] ff_groups[0:5176];',
         f"logic [63:0] ff_groups[0:{info['maximum_feedforward_groups']-1}];")
    once("integer native_steps[0:1] = '{73,78};",
         "integer native_steps[0:" + str(n-1) + "] = '{" +
         ','.join(map(str, info['native_steps'])) + '};')
    once('.TOPOLOGY_MASK(TOPOLOGY_MASK),',
         '.TOPOLOGY_MASK(TOPOLOGY_MASK),\n        .M2_CACHE_DEPTH(512),')
    once('sequence_index<2;', f'sequence_index<{n};')
    once('raw={fixture_root,"/sequence_1"};',
         'raw=$sformatf("%s/sequence_%0d",fixture_root,sequence_index);')
    once('RESIDENT_NATIVE_PAIR_PASS sequences=2', f'RESIDENT_NATIVE_PAIR_PASS sequences={n}')
    once('            T,learning,initialized_cycle,cycles,store_checks);',
         '            native_steps[sequence_index],learning,initialized_cycle,cycles,store_checks);')
    once('        $finish;', '''        $display("REPRESENTATIVE_TOTAL learning=%0d cycles=%0d reset_cycles=%0d parameter_loads=%0d store_loads=%0d backbone_reads=%0d backbone_responses=%0d",
            learning,cycles,profile_count[86],profile_count[0],profile_count[2],profile_count[6],profile_count[8]);
        $finish;''')
    once('        initialized_cycle = cycles; profile_enter("boundary");', '''        initialized_cycle = cycles; profile_enter("boundary");
        $display("REPRESENTATIVE_BEGIN sequence=%0d cycle=%0d setup_begin=%0d",sequence_index,cycles,setup_begin);''')
    return text


def prepare():
    import sys
    import h5py
    import numpy as np
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    from build_srnn_resident_multisequence import reference, pad_rows, FIELDS, bone, m2, FIXTURE_REL
    sys.path.insert(0, str(ROOT/'scripts'))
    from export_shd_h5_events import encode_sample
    root = Path(local_path(ROOT))
    old = root/RESULT/'multi_native_r1'
    accepted = root/RESULT/'srnn_specialized_r1'
    base = accepted/FIXTURE_REL
    out = root/EXPERIMENT
    out.mkdir(parents=True, exist_ok=False)
    fixture = out/'multi_fixture'
    fixture.mkdir()
    h5 = root/'imports/srnn/dataset_cache/SHD/shd_test.h5'
    archived_manifest = json.loads((old/'multi_fixture/source_manifest.json').read_text())
    rows, encoded = [], {}
    with h5py.File(h5, 'r') as dataset:
        total = len(dataset['labels'])
        for i in range(64):
            data, stats = encode_sample(dataset['spikes/times'][i], dataset['spikes/units'][i], 10000)
            label = int(dataset['labels'][i])
            original = archived_manifest['samples'][i]
            if original['dataset_index'] != i or original['label'] != label or any(original[k] != v for k,v in stats.items()):
                raise ValueError('official input metadata changed: '+str(i))
            if stats['clipped_feature_counts'] or not 0 < stats['time_steps'] <= 128:
                raise ValueError('input clipping or unsupported native length')
            words = [int(x, 16) for x in data.split()]
            counts = np.zeros((stats['time_steps'], 700), dtype=np.int64)
            frame = 0
            for word in words:
                if word >> 18:
                    frame += 1
                else:
                    counts[frame, (word >> 8) & 1023] = word & 255
            if frame != len(counts) or words[-1] != 2 << 18:
                raise ValueError('serialized sample boundary')
            occupied = int(np.count_nonzero(counts))
            if occupied != stats['records']-stats['time_steps']:
                raise ValueError('duplicate or missing occupied feature record')
            rows.append(dict(dataset_index=i, label=label, **stats,
                             represented_events=int(counts.sum()), occupied_bins=occupied,
                             occupied_fraction=occupied/counts.size))
            encoded[i] = (data, counts[:, None, :])
    selected, strata = select_rows(rows)
    selection = dict(dataset='SHD official test', source='imports/srnn/dataset_cache/SHD/shd_test.h5',
                     source_bytes=h5.stat().st_size, official_sample_count=total,
                     time_window_us=10000, include_incomplete=False, pool=rows,
                     excluded_previously_measured_indices=[0,1], strata=strata, selected=selected,
                     policy='Sort 62 eligible samples by (native steps,index), split rank quartiles; within each split by occupied-bin density into two rank halves; choose lower central rank. Run selected indices ascending. No RTL timing used.',
                     boundary='Purposive eight-point distribution within first64, not full-test random sampling or a stable tail percentile.')
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    steps = [r['time_steps'] for r in selected]
    counts = [encoded[r['dataset_index']][1] for r in selected]
    w0 = np.zeros((700,480), dtype=np.int64)
    w0[:,:450] = bone.read_signed(base/'w0_refill_dest_major.mem',16,(450,700)).T
    recurrent = np.zeros((480,480), dtype=np.int64)
    recurrent[:450,:450] = bone.read_signed(base/'r_refill_dest_major.mem',16,(450,450)).T
    wh1 = bone.read_signed(accepted/'tb/data/srnn_m2_v28_wh1_initial_q14.mem',16,(256,450))
    initial = bone.read_signed(accepted/'tb/data/srnn_m2_v28_wout_initial_q14.mem',16,(20,256))
    backs = [bone.backbone_reference(c,w0,recurrent) for c in counts]
    maximum_groups = max(sum(len(bone.p4_groups(f)) for f in c[:,0]) for c in counts)
    for i, (sample, c, back) in enumerate(zip(selected, counts, backs)):
        directory = fixture/f'sequence_{i}'
        directory.mkdir()
        (directory/'source_sample.hex').write_bytes(encoded[sample['dataset_index']][0])
        # Only sequence0 loads model parameters; later sequences retain them.
        if i == 0:
            for name in ('w0_refill_dest_major.mem','r_refill_dest_major.mem'):
                shutil.copy2(base/name,directory/name)
        bone.write_hex(directory/'raw700_counts.mem',c,8)
        for key in ('pre','spike'):
            bone.write_hex(directory/f'backbone_{key}.mem',pad_rows(back[key],max(steps)),32)
        offsets, group_counts, packed = [0], [], []
        for frame in c[:,0]:
            groups = bone.p4_groups(frame)
            group_counts.append(len(groups))
            packed.extend(mask | sum(a << (8+11*lane) for lane,a in enumerate(addresses))
                          for mask,*addresses in groups)
            offsets.append(len(packed))
        bone.write_hex(directory/'feedforward_p4.mem',pad_rows(np.array(packed,dtype=np.int64),maximum_groups),64)
        bone.write_hex(directory/'feedforward_offsets.mem',offsets+[offsets[-1]]*(max(steps)-steps[i]),32)
        bone.write_hex(directory/'feedforward_group_counts.mem',group_counts+[0]*(max(steps)-steps[i]),16)
        bone.write_hex(directory/'labels.mem',[sample['label']],8)
    expected = {}
    for learning in (False,True):
        arm = 'on' if learning else 'off'
        code, residual, records, changed_steps = initial.copy(), np.zeros_like(initial), [], 0
        for i, (sample,back) in enumerate(zip(selected,backs)):
            before_nonzero = int(np.count_nonzero(residual))
            arrays, traffic = reference(back['spike'][:,:,:450]*16384,sample['label'],wh1,code,residual,learning)
            for key in FIELDS:
                bits = 64 if key in ('logit','gradient') else 16 if key in ('wout_before','wout_after','residual') else 32
                bone.write_hex(fixture/f'sequence_{i}/{arm}_{key}.mem',pad_rows(arrays[key],max(steps)),bits)
            changed = [int(t['code_writes']>0 or t['residual_writes']>0) for t in traffic]
            loads_at_end = 1+changed_steps+sum(changed[:-1])
            changed_steps += sum(changed)
            code, residual = arrays['wout_after'][-1], arrays['residual'][-1]
            traces = arrays['trace']
            records.append(dict(sequence=i,dataset_index=sample['dataset_index'],label=sample['label'],
                steps=steps[i],learning=int(learning),bone=steps[i]*450,hidden=steps[i]*256,
                logits=steps[i]*20,store=steps[i]*120320,commits=steps[i] if learning else 0,
                code=sum(t['code_writes'] for t in traffic),residual=sum(t['residual_writes'] for t in traffic),
                backbone_loads=2,wh1_loads=1,wout_loads=loads_at_end,
                initial_residual_nonzero=before_nonzero,final_residual_nonzero=int(np.count_nonzero(residual)),
                backbone_spikes=int(np.count_nonzero(back['spike'][:,:,:450])),
                hidden_spikes=int(np.count_nonzero(arrays['spike'])),
                nonzero_trace_elements=int(np.count_nonzero(traces)),trace_elements=int(traces.size)))
        expected[arm] = records
    info = dict(status='REFERENCE_ONLY_NOT_RTL',scope=SCOPE,native_steps=steps,capacity_steps=max(steps),
                maximum_feedforward_groups=maximum_groups,expected=expected,
                model_source='srnn_specialized_r1 original backbone/WH1/Wout',
                candidate_source='verification/experiments/scnn_control_payload/candidate_r4/rtl',
                capacity=512,parameter_reload_between_sequences=False,global_reset_between_sequences=False)
    (fixture/'reference.json').write_text(json.dumps(info,indent=2)+'\n')
    source = (old/TB).read_text()
    text = transform_tb(source,info)
    (out/TB).write_text(text,newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),
        fromfile='accepted_native_pair/tb',tofile='representative_A/tb')))
    shutil.copytree(root/info['candidate_source'],out/'rtl')
    shutil.copytree(old/'tb',out/'tb')
    paths = sorted(p.relative_to(out).as_posix() for p in (out/'rtl').rglob('*') if p.is_file())
    if len(paths)!=109 or sum(p.endswith('.sv') for p in paths)!=108:
        raise ValueError('incomplete A source tree')
    (out/'sources.f').write_text(''.join('"'+p+'"\n' for p in paths if p.endswith('.sv'))+
                              '"'+TB+'"\n"tb/monitors/supplement_h40_activity_monitor.sv"\n')
    command = json.loads((old/'status.json').read_text())['build']['command']
    (out/'build_command.json').write_text(json.dumps(command,indent=2)+'\n')
    snapshot = out/'reference_scripts'
    snapshot.mkdir()
    for path in ('scripts/export_shd_h5_events.py','scripts/supplement_srnn_backbone_numeric.py',
                 'scripts/build_srnn_formal_m2_fixture.py','scripts/screen_srnn_m2_direct_optimizer.py',
                 'scripts/experiments/build_srnn_resident_multisequence.py'):
        shutil.copy2(root/path,snapshot/Path(path).name)
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,
        selected_indices=[r['dataset_index'] for r in selected],native_steps=steps,source_files=109,
        old_testbench='multi_native_r1',numerics='Original integer reference; no old sample0/1 numerical rerun',
        service='Original accepted one-response-slot backbone host and shared RMW store; learning on/off use identical inputs and initial weights.'),indent=2)+'\n')
    print(json.dumps(dict(prepared=str(out),indices=[r['dataset_index'] for r in selected],steps=steps,
                         files=sum(p.is_file() for p in out.rglob('*')),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))))


def parse_log(log,arm,info):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b',log):
        raise ValueError('runtime diagnostic')
    def records(marker):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',s)}
                for s in re.findall(r'^'+marker+r' (.+)$',log,re.M)]
    rows = records('RESIDENT_NATIVE_SEQUENCE')
    starts = records('REPRESENTATIVE_BEGIN')
    totals = records('REPRESENTATIVE_TOTAL')
    wanted = info['expected'][arm]
    keys = ('sequence','steps','learning','bone','hidden','logits','store','code','residual','commits',
            'backbone_loads','wh1_loads','wout_loads')
    if len(rows)!=len(wanted) or len(starts)!=len(rows) or len(totals)!=1:
        raise ValueError('incomplete representative sequence set')
    for i,(row,expect,start) in enumerate(zip(rows,wanted,starts)):
        if any(row.get(k)!=expect[k] for k in keys):
            raise ValueError(('native numeric/update/residency counters',i,row,expect))
        if start['sequence']!=i or row['cycles']<=0 or row['state_setup_cycles']!=start['cycle']-start['setup_begin']:
            raise ValueError('native interval accounting')
        if i and start['setup_begin'] != starts[i-1]['cycle']+rows[i-1]['cycles']:
            raise ValueError('sequence boundary gap/overlap')
        row['dataset_index'] = expect['dataset_index']
    marker = f"RESIDENT_NATIVE_PAIR_PASS sequences={len(rows)} learning={int(arm=='on')} steps={sum(info['native_steps'])}"
    if log.count(marker)!=1 or len(records('SRNN_SAME_DUT_PASS'))!=len(rows):
        raise ValueError('missing final numeric completion')
    total = totals[0]
    if total['learning']!=int(arm=='on') or total['parameter_loads']!=120320 or total['store_loads']!=120320:
        raise ValueError('parameters reloaded or incomplete')
    if total['backbone_reads']!=517500 or total['backbone_responses']!=517500:
        raise ValueError('backbone service accounting')
    service = sum(r['cycles'] for r in rows)
    setup = sum(r['state_setup_cycles'] for r in rows)
    initial = starts[0]['setup_begin']
    if initial+service+setup!=total['cycles']:
        raise ValueError('total latency does not reconcile')
    tiles = re.findall(r'^H40_ACTIVITY tile=(\S+)',log,re.M)
    coords = [tuple(map(int,m.groups())) for t in tiles if (m:=re.search(r'g_region\[(\d+)\].*g_tile\[(\d+)\]',t))]
    if sorted(coords)!=[(r,t) for r in range(3) for t in range(4)]:
        raise ValueError('incomplete tile accounting')
    return dict(sequences=rows,total=total,initialization_including_reset_cycles=initial,
                state_setup_cycles=setup,service_cycles=service,
                all_sequences_per_second_15ns_projection=len(rows)*1e9/(15*total['cycles']),
                warm_sequences_per_second_with_setup_15ns_projection=(len(rows)-1)*1e9/
                    (15*sum(r['cycles']+r['state_setup_cycles'] for r in rows[1:])),
                tiles=tiles,scope=SCOPE)


def worker():
    root = Path.cwd()
    scratch = Path('/tmp/snn_'+NAME)
    scratch.mkdir(exist_ok=False)
    state = dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE)
    def save():
        (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(name,command,timeout):
        state['status']=name+'_running'
        save()
        begin=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[name]=dict(command=command,exit=None,pid=process.pid,timed_out=True,
                                 wall_seconds=time.monotonic()-begin)
                save()
                raise
        state[name]=dict(command=command,exit=code,wall_seconds=time.monotonic()-begin)
        save()
        if code:
            raise RuntimeError(name+' exit '+str(code))
    try:
        info=json.loads((root/'multi_fixture/reference.json').read_text())
        build=json.loads((root/'build_command.json').read_text())
        build[build.index('--Mdir')+1]=str(scratch/'obj')
        execute('build',build,1800)
        text=(root/'build.log').read_text(errors='replace')
        state['build_warnings']=dict(ordinary=len(re.findall(r'^%Warning-',text,re.M)),
            structural=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',text,re.M))
        if state['build_warnings']['structural']:
            raise ValueError('structural warning')
        checks={}
        for arm in ('off','on'):
            execute('run_'+arm,[str(scratch/'obj/Vtb_srnn_profile_measurement'),'+RAW=multi_fixture',
                    '+FIXTURE=multi_fixture','+LEARNING='+str(int(arm=='on'))],7200)
            checks[arm]=parse_log((root/('run_'+arm+'.log')).read_text(errors='replace'),arm,info)
            state[arm]=checks[arm]
            save()
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state['status']='PASS_eight_native_SHD_sequences_A_pair_not_full_Test'
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True)
    save()


def launch():
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    prepared=Path(local_path(ROOT/EXPERIMENT))
    if not (prepared/'preparation.json').is_file():
        raise ValueError('prepare first')
    inputs=sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    input_bytes=sum(p.stat().st_size for p in inputs)
    destination=REMOTE+'/'+NAME
    code='''import datetime,json,pathlib,shutil
p=pathlib.Path(DEST)
if p.exists() or pathlib.Path(SCRATCH).exists():raise FileExistsError('new destination already exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA; no overlap: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p.parent).free,tmp_free=shutil.disk_usage('/tmp').free,input_bytes=INPUT_BYTES)
if record['data_free'] < INPUT_BYTES+512*1024**2 or record['tmp_free'] < 2*1024**3:raise RuntimeError('insufficient input/log/build headroom')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
    client=connect()
    try:
        record=json.loads(run(client,'python3 -c '+shlex.quote('DEST='+repr(destination)+
            '\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nINPUT_BYTES='+str(input_bytes)+'\n'+code)))
        with client.open_sftp() as sftp:
            for path in inputs:
                relative=path.relative_to(prepared).as_posix()
                remote=destination+'/'+relative
                upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual, path.open('rb') as expected:
                    while chunk:=expected.read(1024*1024):
                        if actual.read(len(chunk))!=chunk:
                            raise ValueError('actual prepared input mismatch: '+relative)
                    if actual.read(1):raise ValueError('remote input trailing data: '+relative)
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_srnn_representative bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_srnn_representative',
                      prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME))
        out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
    finally:
        client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args()
    {'prepare':prepare,'launch':launch,'worker':worker}[args.action]()
