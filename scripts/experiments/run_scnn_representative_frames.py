#!/usr/bin/env python3
"""Three input-stratified cold raw SCNN frames on exact candidate A, not a speedup pair."""
import argparse
import csv
import datetime as dt
import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NAME = 'scnn_representative_a_r1'
EXPERIMENT = Path('verification/experiments/scnn_representative_inputs/a_r1')
SCOPE = ('Three density-stratified first frames from the existing balanced 264x20 export; '
         'cold raw Conv1..8 plus classifier, learning off, exact A/512/no-model-key. '
         'Not full sequences, IID accuracy, P95, a new speedup pair, DDR or board throughput.')
SCALARS = [65536, 131072, 65536, 65536, 32768, 32768, 8192, 8192]


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('ambiguous TB anchor: '+before[:100])
    return text.replace(before, after, 1)


def transform(text):
    begin = text.index('    task automatic fg_load_expected;')
    end = text.index('    endtask', begin) + len('    endtask')
    loader = '''    task automatic fg_load_expected;
        string fixture;
        if (!$value$plusargs("SCNN_PROFILE_FIXTURE=%s", fixture) ||
            !$value$plusargs("REP_SAMPLE=%d", rep_sample))
            $fatal(1,"representative fixture and sample required");
        fg_learn_on = 0;
        fg_label[1] = 0; fg_label[2] = 0; fg_label[3] = 0;
        $readmemh({fixture,"/label.hex"}, fg_label, 0, 0);
        $readmemh({fixture,"/initial_codes.hex"}, fg_initial_codes);
        $readmemh({fixture,"/raw.hex"}, tv_raw, 0, 2047);
        $readmemh({fixture,"/features.hex"}, fg_features, 0, 2047);
        $readmemh({fixture,"/probes.hex"}, tv_membrane_probes, 0, 23);
        $readmemh({fixture,"/currents.hex"}, fg_currents, 0, 10);
        $readmemh({fixture,"/centered.hex"}, fg_logits, 0, 10);
        $readmemh({fixture,"/pre.hex"}, rep_pre);
        $readmemh({fixture,"/spikes.hex"}, rep_spikes);
        for (int index=0; index<PARAMS; index++) begin
            if (classifier_weights[index] !== fg_initial_codes[index])
                $fatal(1,"initial bank mismatch");
            fg_store_residual[index] = 0;
        end
        $display("SCNN_REP_INPUT sample=%0d frame=0 learning=0",rep_sample);
    endtask'''
    text = text[:begin]+loader+text[end:]
    text = once(text, '"SCNN_PROFILE_FRAME frame=%0d sample=0 timestep=%0d cycles=%0d commits=%0d",\n            timestep_phase, timestep_phase, cycle_count-perf_frame_start, fg_commits);',
                '"SCNN_PROFILE_FRAME frame=%0d sample=%0d timestep=%0d cycles=%0d commits=%0d",\n            timestep_phase, rep_sample, timestep_phase, cycle_count-perf_frame_start, fg_commits);')
    text = once(text, '        int area, size, channels, post, probe_index;',
                '        int area, size, channels, post, probe_index, expected_offset;')
    anchor = '        if (spike != (pre >= 16\'sd256)) $fatal(1, "scalar threshold mismatch");'
    offsets = [sum(SCALARS[:i]) for i in range(8)]
    cases = '\n'.join(f'            {i+1}: expected_offset = {v};' for i,v in enumerate(offsets))
    check = '''        case (layer)
CASES
            default: $fatal(1,"invalid independent scalar layer");
        endcase
        if (rep_seen[expected_offset+index] || pre !== rep_pre[expected_offset+index] ||
            spike !== rep_spikes[expected_offset+index])
            $fatal(1,"representative scalar sample=%0d layer=%0d index=%0d pre=%0d expected=%0d spike=%0b expected=%0b",
                rep_sample,layer,index,pre,rep_pre[expected_offset+index],spike,rep_spikes[expected_offset+index]);
        rep_seen[expected_offset+index] = 1;
        rep_checked++;
'''.replace('CASES', cases)
    text = once(text, anchor, check+anchor)
    text = once(text, '        st_report();\n        $finish;', '''        st_report();
        if (rep_checked != 409600 || rep_tiles !== 16'h0fff)
            $fatal(1,"representative scalar/tile coverage");
        for (int i=0; i<409600; i++)
            if (!rep_seen[i]) $fatal(1,"missing independent scalar %0d",i);
        $display("SCNN_REP_PASS sample=%0d frames=1 scalar_checks=%0d tile_mask=%0d learning=0",
            rep_sample,rep_checked,rep_tiles);
        $finish;''')
    return once(text, 'endmodule', '''    int rep_sample = -1;
    int rep_checked = 0;
    logic signed [15:0] rep_pre [0:409599];
    logic rep_spikes [0:409599];
    bit rep_seen [0:409599];
    logic [15:0] rep_tiles = 0;
    always @(posedge clk) begin
        if (rst_n && state_write_valid && state_write_ready)
            rep_tiles[maintenance_tile] <= 1'b1;
    end
endmodule''')


def prepare():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT)); out = root/EXPERIMENT
    if out.exists():
        raise FileExistsError('representative preparation already exists')
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from scripts.evaluate_scnn_full_integer import IntegerNetwork, read_hex
    from scripts.build_scnn_vgg9_full_frame_t2_reference import write_i16_hex, write_bits
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    source = root/'data/experiments/scnn_subjects_264'
    metadata = json.loads((source/'test_264x20_metadata.json').read_text(encoding='utf-8'))
    # Selection uses only input occupancy, never measured RTL cycles or predictions.
    frames = read_hex(source/'test_264x20_inputs_q8_8.hex').reshape(264,20,2,32,32)
    labels = read_hex(source/'test_264_labels.hex',bits=8)
    if bool((frames < 0).any()):
        raise ValueError('unexpected negative activity')
    pool = []
    for i in range(264):
        pool.append(dict(sample=i,label=int(labels[i]),first_frame_nonzero=int(np.count_nonzero(frames[i,0])),
            first_frame_q8_mass=int(frames[i,0].sum()),sequence_nonzero=int(np.count_nonzero(frames[i])),
            sequence_q8_mass=int(frames[i].sum()),eligible=i>=4))
    ordered = sorted((r for r in pool if r['eligible']),key=lambda r:(r['first_frame_nonzero'],r['sample']))
    selected = []
    for stratum in range(3):
        group = ordered[len(ordered)*stratum//3:len(ordered)*(stratum+1)//3]
        selected.append(dict(group[(len(group)-1)//2],stratum=stratum,
                             stratum_n=len(group),stratum_min=group[0]['first_frame_nonzero'],
                             stratum_max=group[-1]['first_frame_nonzero']))
    selected.sort(key=lambda r:r['sample'])
    out.mkdir(parents=True)
    with (out/'selection_pool.csv').open('w',newline='',encoding='ascii') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pool[0]));writer.writeheader();writer.writerows(pool)
    (out/'selection.json').write_text(json.dumps(dict(selected=selected,pool_size=264,eligible=260,
        excluded=[0,1,2,3],selection='Three equal-rank strata of frame0 nonzero-bin count; lower median within each, ties by sample ID; run in sample-ID order.',
        exclusion_reason='Exclude the first four sample positions conservatively rather than reuse old sample0/B4 scope.',
        dataset_scope='Existing balanced 264x20 export, not complete Test. Export already chose each recording\'s most active contiguous20-frame interval.',
        time_scope='Only frame0, cold states and original model for each separate process. No temporal prefix skipped.',
        source_metadata='source_metadata.json',scope=SCOPE),indent=2)+'\n')
    shutil.copy2(source/'test_264x20_metadata.json',out/'source_metadata.json')
    shutil.copy2(source/'test_264_labels.hex',out/'source_labels.hex')
    model=root/'models/scnn_dvsgesture_vgg9_q0_15'
    (out/'models/scnn_dvsgesture_vgg9_q0_15').mkdir(parents=True)
    for bank in list(range(1,9))+[12]:
        shutil.copy2(model/f'bank_{bank:02d}.hex',out/'models/scnn_dvsgesture_vgg9_q0_15'/f'bank_{bank:02d}.hex')
    network=IntegerNetwork('cpu')
    ids=[r['sample'] for r in selected]
    captured={}
    network.forward(torch.tensor(frames[ids,:1],dtype=torch.int64),sample_ids=ids,verify_ids=ids,
                    capture=lambda t,l,v:captured.update({l:{k:x.cpu().numpy() for k,x in v.items()}}))
    for bi,row in enumerate(selected):
        dest=out/'fixtures'/f'sample_{row["sample"]:03d}';dest.mkdir(parents=True)
        probes=[]; layer_stats=[]; pre=[];spikes=[]
        for layer in range(8):
            value=captured[layer]; state=value['next'][bi]; co,h,w=state.shape
            probes.extend(int(state[c,y,x]) for c,y,x in ((0,0,0),(co//2,h//2,w//2),(co-1,h-1,w-1)))
            pre.append(value['pre'][bi].reshape(-1));spikes.append(value['spike'][bi].reshape(-1))
            layer_stats.append(dict(layer=layer+1,scalars=state.size,spikes=int(value['spike'][bi].sum()),
                pooled_nonzero=int(np.count_nonzero(value['pooled'][bi])),pooled_values=value['pooled'][bi].size,
                range_events=int(value['range'][bi].sum())))
        for filename,values in dict(raw=frames[row['sample'],0],features=captured[7]['pooled'][bi],
            probes=np.array(probes),currents=captured[8]['current'][bi],centered=captured[8]['centered'][bi],
            pre=np.concatenate(pre),label=np.array([row['label']]),initial_codes=network.codes[-1]).items():
            write_i16_hex(dest/(filename+'.hex'),values)
        write_bits(dest/'spikes.hex',np.concatenate(spikes))
        (dest/'reference.json').write_text(json.dumps(dict(input=row,layers=layer_stats,
            all_scalar_checks=409600,scope=SCOPE),indent=2)+'\n')
    (out/'integer_reference_checks.json').write_text(json.dumps(dict(checks=network.checks,
        layers=network.layers,classifier=network.classifier,scope='Original integer reference, three new cold first frames; no RTL result.'),indent=2)+'\n')
    accepted=root/'verification/experiments/scnn_forward_turnover/frame_r2/candidate'
    original=(accepted/'tb_scnn_profile_measurement.sv').read_text(encoding='utf-8')
    changed=transform(original)
    (out/'tb_scnn_profile_measurement.sv').write_text(changed,encoding='utf-8',newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='accepted_frame_r2/tb',tofile='representative_A/tb')),encoding='utf-8')
    shutil.copytree(root/'verification/experiments/scnn_control_payload/candidate_r4/rtl',out/'rtl')
    sources=sorted(p.relative_to(out).as_posix() for p in (out/'rtl').rglob('*.sv'))
    if len(sources)!=108: raise ValueError('A source set incomplete')
    (out/'sources.f').write_text(''.join('"'+r+'"\n' for r in sources+['tb_scnn_profile_measurement.sv']))
    dependencies=json.loads((accepted.parent/'dependencies.json').read_text())
    for rel in dependencies:
        # The expanded TB uses no old data fixtures; retain only its compile includes/make helper.
        if rel.startswith(('tb/','tests/','build/','scripts/')):
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(accepted/rel,dest)
    shutil.copy2(accepted/'accepted_build.json',out/'accepted_build.json')
    shutil.copy2(accepted/'reference.json',out/'geometry_reference.json')
    for script in ('run_scnn_turnover_frame.py','prepare_scnn_schedule_probe.py','prepare_scnn_resident_compatibility.py'):
        shutil.copy2(root/'scripts/experiments'/script,out/script)
    (out/'reference_source/scripts').mkdir(parents=True)
    for script in ('evaluate_scnn_full_integer.py','build_scnn_vgg9_full_frame_t2_reference.py','supplement_scnn_weight_quantization.py'):
        shutil.copy2(root/'scripts'/script,out/'reference_source/scripts'/script)
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,
        exact_candidate='verification/experiments/scnn_control_payload/candidate_r4',rtl_files=109,
        m2_cache_depth=512,model_key=False,learning=False,selected=selected,
        service='Unchanged replaceable latency1 one-response-slot code16 address24/tag16 host.',
        testbench='Accepted frameR2 schedule/body/host plus independent all-scalar pre/spike checks; no producer optimization.',
        fc_gap='No fresh FC raw counts/model NPZ present at package defaults. Old B16T99 is not reused as a new distribution.',
        production_change=False),indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(prepared=str(out),samples=ids,files=len(files),bytes=sum(p.stat().st_size for p in files),reference_checks=network.checks)))


def parse_log(text, sample, geometry):
    from run_scnn_turnover_frame import parse_log as parse_frame
    geometry = json.loads(json.dumps(geometry))
    geometry['frame']['sample'] = str(sample)
    checks=parse_frame(text,geometry)
    marker=re.findall(r'^SCNN_REP_PASS sample=(\d+) frames=1 scalar_checks=409600 tile_mask=4095 learning=0$',text,re.M)
    if marker!=[str(sample)] or text.count(f'SCNN_REP_INPUT sample={sample} frame=0 learning=0')!=1:
        raise ValueError('representative independent scalar/input/tile completion missing')
    checks.update(status='PASS_representative_cold_raw_frame',scope=SCOPE,sample=sample,scalar_checks=409600,tile_mask=4095)
    return checks


def worker():
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE,runs=[])
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();begin=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,pid=process.pid,timed_out=True,wall_seconds=time.monotonic()-begin);save();raise
        record=dict(command=command,exit=code,wall_seconds=time.monotonic()-begin)
        state[stage]=record;save()
        if code: raise RuntimeError(stage+' exit '+str(code))
        return record
    try:
        config=json.loads((root/'preparation.json').read_text())
        command=json.loads((root/'accepted_build.json').read_text())
        command[command.index('--Mdir')+1]=str(scratch/'obj')
        command[command.index('-f')+1]='sources.f'
        command[command.index('-MAKEFLAGS')+1]='OPT_FAST=-O1 OPT_GLOBAL=-O1 -f '+str(root/'scripts/supplement_scnn_graph_host.mk')
        execute('build',command,5400)
        build=(root/'build.log').read_text(errors='replace')
        state['build_warnings']={k:len(re.findall(r'^%Warning-'+k+r':',build,re.M)) for k in sorted(set(re.findall(r'^%Warning-(\w+):',build,re.M)))}
        if '%Error' in build or any(state['build_warnings'].get(k) for k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')):
            raise ValueError('structural build diagnostics')
        geometry=json.loads((root/'geometry_reference.json').read_text())
        for row in config['selected']:
            sample=row['sample'];stage=f'run_{sample:03d}'
            run=execute(stage,[str(scratch/'obj/Vscgraph'),f'+SCNN_PROFILE_FIXTURE=fixtures/sample_{sample:03d}',f'+REP_SAMPLE={sample}'],5400)
            checks=parse_log((root/(stage+'.log')).read_text(errors='replace'),sample,geometry)
            (root/(stage+'_checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
            state['runs'].append(dict(sample=sample,**run,frame_cycles=checks['frame_cycles'],initialization_cycles=checks['initialization_cycles']));save()
        state['status']='PASS_three_representative_A_cold_frames_not_full_sequences'
    except Exception as error: state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    inputs=sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    total=sum(p.stat().st_size for p in inputs)
    probe='''import datetime,json,pathlib,shutil
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
if record['data_free'] < INPUT_BYTES+128*1024**2 or record['tmp_free'] < 3*1024**3:raise RuntimeError('insufficient headroom for one full-core build')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
    client=connect()
    try:
        record=json.loads(run(client,'python3 -c '+shlex.quote('DEST='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nINPUT_BYTES='+str(total)+'\n'+probe)))
        with client.open_sftp() as sftp:
            for path in inputs:
                remote=destination+'/'+path.relative_to(prepared).as_posix();upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual:
                    if actual.read()!=path.read_bytes():raise ValueError('uploaded input differs: '+str(path))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_scnn_representative bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_scnn_representative',prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch'))
    args=parser.parse_args();{'prepare':prepare,'worker':worker,'launch':launch}[args.action]()
