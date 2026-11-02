#!/usr/bin/env python3
"""One complete existing representative T20 window, exact A, learning off."""
import argparse
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

NAME = 'scnn_representative_t20_a_r1'
EXPERIMENT = Path('verification/experiments/scnn_representative_sequence/a_r1')
SAMPLE = 199
STEPS = 20
SCOPE = ('Sample199, the previously selected middle input-occupancy stratum representative, '
         'all20 exported frames in order, raw Conv1..8/pool/classifier, five original four-frame '
         'requests, continuous membrane and cumulative logits, learning off, exact A/512/no-model-key. '
         'One complete exported most-active20-frame window, not a complete raw recording, full Test, '
         'IID/P95, an A/B speedup, DDR, learning or board throughput.')
BANK_READS = [1152, 73728, 294912, 589824, 1179648, 2359296, 2359296, 2359296, 22528]


def transform(text):
    from run_scnn_representative_frames import once
    for name, array, end in [('raw','tv_raw',2047),('features','fg_features',2047),
                             ('probes','tv_membrane_probes',23),('currents','fg_currents',10),
                             ('centered','fg_logits',10)]:
        text = once(text, f'$readmemh({{fixture,"/{name}.hex"}}, {array}, 0, {end});',
                    f'$readmemh({{fixture,"/{name}.hex"}}, {array});')
    text = once(text, '        $readmemh({fixture,"/pre.hex"}, rep_pre);\n        $readmemh({fixture,"/spikes.hex"}, rep_spikes);',
                '        seq_fixture = fixture;')
    text = once(text, 'SCNN_REP_INPUT sample=%0d frame=0 learning=0',
                'SCNN_SEQUENCE_INPUT sample=%0d frames=20 learning=0')
    text = once(text, 'timestep_phase < 4;', 'timestep_phase < 20;')
    text = once(text, '        perf_frame_start = cycle_count;', '''        perf_frame_start = cycle_count;
        $display("SCNN_SEQUENCE_FRAME_BEGIN frame=%0d cycle=%0d", timestep_phase, cycle_count);
        seq_load_scalar_reference();
        sp_descriptors = 0;
        for (int l=1; l<=8; l++) begin
            sp_cycles[l]=0; sp_pub_reads[l]=0; sp_pub_responses[l]=0;
            sp_pub_stalls[l]=0; sp_pub_starts[l]=0;
            for (int k=0; k<8; k++) sp_bucket[l][k]=0;
            for (int k=0; k<4; k++) sp_context[l][k]=0;
            for (int k=0; k<7; k++) sp_publisher[l][k]=0;
            for (int k=0; k<10; k++) sp_events[l][k]=0;
        end''')
    begin = text.index('        break; // First-frame diagnostic only.')
    end = text.index('        $finish;', begin) + len('        $finish;')
    text = text[:begin]+'''        sp_report();
        if (rep_checked != (timestep_phase+1)*409600 || rep_tiles !== 16'h0fff)
            $fatal(1,"sequence scalar/tile coverage");
        for (int i=0; i<409600; i++)
            if (!rep_seen[i]) $fatal(1,"missing independent scalar %0d",i);
        $display("SCNN_SEQUENCE_FRAME_END frame=%0d cycle=%0d scalar_checks=%0d", timestep_phase, cycle_count, rep_checked);
        $fflush();
        end
        fg_finish();
        st_report();
        $display("SCNN_SEQUENCE_PASS sample=%0d frames=20 scalar_checks=%0d tile_mask=%0d learning=0 reset_releases=%0d",rep_sample,rep_checked,rep_tiles,seq_reset_releases);
        $fflush();
        $finish;'''+text[end:]
    text = once(text, 'sp_active || timestep_phase != 0 || layer_phase != sp_descriptors+1',
                'sp_active || layer_phase != sp_descriptors+1')
    text = once(text, 'fg_learn_on || timestep_phase != 0 || sp_active || sp_descriptors != 8 ||\n            tv_features_checked != 2048 || tv_currents_checked != 11 ||\n            classifier_logit_wave_count != 1 || tv_probes_checked != 24 ||',
                'fg_learn_on || sp_active || sp_descriptors != 8 ||\n            tv_features_checked != (timestep_phase+1)*2048 || tv_currents_checked != (timestep_phase+1)*11 ||\n            classifier_logit_wave_count != timestep_phase+1 || tv_probes_checked != (timestep_phase+1)*24 ||')
    for kind in ('BUCKET','CONTEXT','PUBLISHER','HANDSHAKE'):
        text = once(text, f'SCNN_SCHEDULE_{kind} layer=%0d', f'SCNN_SCHEDULE_{kind} frame=%0d layer=%0d')
    for anchor in ('layer,sp_cycles[layer]', 'layer,sp_context[layer][0]',
                   'layer,sp_publisher[layer][0]', 'layer,sp_events[layer][0]'):
        text = once(text, anchor, 'timestep_phase,'+anchor)
    text = once(text, 'SCNN_SCHEDULE_PROBE_PASS learning=0 frames=1',
                'SCNN_SEQUENCE_PROBE_PASS frame=%0d learning=0 frames=1')
    text = once(text, 'currents=%0d logits=11 probes=%0d', 'currents=%0d logits=%0d probes=%0d')
    text = once(text, 'sp_descriptors,tv_features_checked,tv_currents_checked,tv_probes_checked',
                'sp_descriptors,tv_features_checked,tv_currents_checked,classifier_logit_wave_count*11,tv_probes_checked')
    text = once(text, '            sp_descriptors,tv_features_checked',
                '            timestep_phase,sp_descriptors,tv_features_checked')
    begin=text.index('    task automatic fg_check_store;')
    end=text.index('    endtask',begin)+len('    endtask')
    text=text[:begin]+'''    task automatic fg_check_store;
        for (int index=0; index<PARAMS; index++)
            if (classifier_weights[index] !== fg_initial_codes[index] || fg_store_residual[index] !== 0 ||
                fg_rmw_seen[index] || tv_gradient_seen[index] || tv_delta_seen[index] ||
                tv_observed_gradient[index] !== 0 || tv_contributions[index] != 0)
                $fatal(1,"learning-off sequence changed model or gradient state");
        if (fg_rmw || fg_learning_reads || fg_rmw_outstanding || fg_commits || tv_total_rmw)
            $fatal(1,"learning-off sequence issued updates");
    endtask'''+text[end:]
    begin=text.index('    task automatic fg_finish;')
    end=text.index('    endtask',begin)+len('    endtask')
    text=text[:begin]+'''    task automatic fg_finish;
        if (fg_learn_on || fg_requests != 5 || fg_windows != 5 || fg_descriptors != 160 ||
            fg_classifier_restores != 20 || fg_classifier_spills != 20 ||
            tv_features_checked != 40960 || tv_currents_checked != 220 || classifier_logit_wave_count != 20 ||
            tv_traces_checked != 0 || tv_publish_words != 1892 || fg_commits != 0 || fg_detaches != 0 ||
            cache_prime_count != 1 || tv_descriptor_pending || tv_probes_checked != 480 || tv_trace_resets != 0 ||
            rep_checked != 8192000 || seq_reset_releases != 1)
            $fatal(1,"sequence final completion counts");
        pm_report();
        $display("SCNN_PROFILE_PASS learning=0 samples=1 frames=20 descriptors=%0d requests=%0d features=%0d currents=%0d logits=%0d traces=%0d probes=%0d rmw=%0d commits=%0d publications=%0d cycles=%0d",
            fg_descriptors,fg_requests,tv_features_checked,tv_currents_checked,classifier_logit_wave_count*CLASSES,
            tv_traces_checked,tv_probes_checked,tv_total_rmw,fg_commits,tv_publish_words,cycle_count);
    endtask'''+text[end:]
    text=once(text, 'st_reads != 9239680', 'st_reads != 184365568')
    text=once(text, 'endmodule', '''    string seq_fixture;
    int seq_reset_releases = 0;
    always @(posedge rst_n) begin
        seq_reset_releases++;
        if (seq_reset_releases != 1) $fatal(1,"unexpected sequence global reset");
    end
    task automatic seq_load_scalar_reference;
        string prefix;
        prefix = $sformatf("%s/frame_%02d",seq_fixture,timestep_phase);
        $readmemh({prefix,"_pre.hex"},rep_pre);
        $readmemh({prefix,"_spikes.hex"},rep_spikes);
        for (int i=0; i<409600; i++) rep_seen[i]=0;
    endtask
endmodule''')
    return text


def prepare():
    from run_srnn_resident_remote import ROOT, local_path
    root=Path(local_path(ROOT)); out=root/EXPERIMENT
    if out.exists(): raise FileExistsError('sequence preparation exists')
    sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from scripts.evaluate_scnn_full_integer import IntegerNetwork, read_hex
    from scripts.build_scnn_vgg9_full_frame_t2_reference import write_i16_hex, write_bits
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    previous=root/'verification/experiments/scnn_representative_inputs/a_r1'
    selection=json.loads((previous/'selection.json').read_text())
    row=[r for r in selection['selected'] if r['stratum']==1]
    if len(row)!=1 or row[0]['sample']!=SAMPLE: raise ValueError('original input selection changed')
    source=root/'data/experiments/scnn_subjects_264'
    frames=read_hex(source/'test_264x20_inputs_q8_8.hex').reshape(264,20,2,32,32)[SAMPLE:SAMPLE+1]
    out.mkdir(parents=True)
    for folder in ('rtl','models','tb','tests','build','scripts','reference_source'):
        if (previous/folder).exists(): shutil.copytree(previous/folder,out/folder)
    for filename in ('sources.f','accepted_build.json','geometry_reference.json','source_metadata.json',
                     'source_labels.hex','selection.json','selection_pool.csv','prepare_scnn_resident_compatibility.py',
                     'prepare_scnn_schedule_probe.py','run_scnn_turnover_frame.py'):
        shutil.copy2(previous/filename,out/filename)
    original=(previous/'tb_scnn_profile_measurement.sv').read_text(encoding='utf-8')
    changed=transform(original)
    (out/'tb_scnn_profile_measurement.sv').write_text(changed,encoding='utf-8',newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='accepted_cold_frame/tb',tofile='complete_T20/tb')),encoding='utf-8')
    dest=out/'fixture'; dest.mkdir()
    write_i16_hex(dest/'raw.hex',frames[0])
    write_i16_hex(dest/'label.hex',np.array([row[0]['label']]))
    network=IntegerNetwork('cpu')
    write_i16_hex(dest/'initial_codes.hex',network.codes[-1])
    features=[]; currents=[]; logits=[]; probes=[]; frame_stats=[]; pres=[]; spikes=[]; per_frame=[]
    def capture(t,layer,value):
        nonlocal pres,spikes,per_frame
        value={k:x.cpu().numpy()[0] for k,x in value.items()}
        if layer<8:
            state=value['next']; co,h,w=state.shape
            probes.extend(int(state[c,y,x]) for c,y,x in ((0,0,0),(co//2,h//2,w//2),(co-1,h-1,w-1)))
            pres.append(value['pre'].reshape(-1));spikes.append(value['spike'].reshape(-1))
            per_frame.append(dict(layer=layer+1,values=state.size,spikes=int(value['spike'].sum()),
                                  nonzero_state=int(np.count_nonzero(state))))
            if layer==7: features.append(value['pooled'])
        else:
            currents.append(value['current']);logits.append(value['centered'])
            write_i16_hex(dest/f'frame_{t:02d}_pre.hex',np.concatenate(pres))
            write_bits(dest/f'frame_{t:02d}_spikes.hex',np.concatenate(spikes))
            frame_stats.append(dict(frame=t,input_nonzero=int(np.count_nonzero(frames[0,t])),
                input_q8_mass=int(frames[0,t].sum()),layers=per_frame))
            pres=[];spikes=[];per_frame=[]
            print(f'REFERENCE_FRAME {t+1}/20',flush=True)
    cumulative=network.forward(torch.tensor(frames,dtype=torch.int64),sample_ids=[SAMPLE],verify_ids=[SAMPLE],capture=capture)
    for filename,values in dict(features=np.array(features),currents=np.array(currents),centered=np.array(logits),probes=np.array(probes)).items():
        write_i16_hex(dest/(filename+'.hex'),values)
    (out/'integer_reference_checks.json').write_text(json.dumps(dict(checks=network.checks,layers=network.layers,
        classifier=network.classifier,frames=frame_stats,final_cumulative=cumulative.tolist(),scope=SCOPE),indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    shutil.copy2(root/'scripts/experiments/run_source_stream_comparison.py',out/'run_source_stream_comparison.py')
    (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,sample=SAMPLE,
        selection=row[0],selection_reason='Original middle occupancy stratum representative; reduce sequence count, not temporal length, to bound simulation cost.',
        frames=20,learning=False,rtl_files=109,m2_cache_depth=512,model_key=False,
        time_protocol='Five original requests of4 frames, absolute timesteps0..19; no global reset or parameter reload between frames/windows.',
        state='Original host spill/restore for all eight layers and classifier; independent reference keeps states across20 frames.',
        cost_estimate='One build about20min and20 frames about4.2h based on earlier cold-frame walltimes; estimate only, timeout12h leaves child alive.',
        production_change=False),indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(prepared=str(out),files=len(files),bytes=sum(p.stat().st_size for p in files),checks=network.checks)))


def parse_log(text, geometry):
    from prepare_scnn_schedule_probe import rows, BUCKETS, PUBLISHER
    from run_source_stream_comparison import diagnostics
    diag=diagnostics(text)
    if diag['errors'] or diag['fatal'] or diag['structural'] or re.search(r'Assertion failed|%Error',text):
        raise ValueError('runtime diagnostics')
    def ints(marker): return [{k:int(v) for k,v in r.items()} for r in rows(text,marker)]
    wanted=dict(sample=SAMPLE,frames=20,scalar_checks=8192000,tile_mask=4095,learning=0,reset_releases=1)
    if ints('SCNN_SEQUENCE_PASS')!=[wanted]: raise ValueError('sequence final marker')
    if ints('SCNN_SEQUENCE_INPUT')!=[dict(sample=SAMPLE,frames=20,learning=0)]: raise ValueError('sequence input marker')
    final=ints('SCNN_PROFILE_PASS')
    expected=dict(learning=0,samples=1,frames=20,descriptors=160,requests=5,features=40960,currents=220,
                  logits=220,traces=0,probes=480,rmw=0,commits=0,publications=1892)
    if len(final)!=1 or any(final[0].get(k)!=v for k,v in expected.items()): raise ValueError('final counts')
    begins=ints('SCNN_SEQUENCE_FRAME_BEGIN'); ends=ints('SCNN_SEQUENCE_FRAME_END')
    frames=ints('SCNN_PROFILE_FRAME'); numeric=rows(text,'SCNN_PROFILE_FRAME_NUMERIC_PASS')
    probes=ints('SCNN_SEQUENCE_PROBE_PASS')
    if any([r['frame'] if isinstance(r['frame'],int) else int(r['frame']) for r in table]!=list(range(20))
           for table in (begins,ends,frames,numeric,probes)): raise ValueError('missing/duplicate frames')
    desc=rows(text,'SCNN_PROFILE_DESCRIPTOR')
    if [(int(r['frame']),int(r['layer']),r['edge']) for r in desc]!=[(t,l,e) for t in range(20) for l in range(1,9) for e in ('start','end')]:
        raise ValueError('descriptor order')
    tables={k.lower():ints('SCNN_SCHEDULE_'+k) for k in ('BUCKET','CONTEXT','PUBLISHER','HANDSHAKE')}
    for table in tables.values():
        if [(r['frame'],r['layer']) for r in table]!=[(t,l) for t in range(20) for l in range(1,9)]:
            raise ValueError('missing/duplicate scheduler table')
    measured=[]
    for t in range(20):
        if probes[t]!=dict(frame=t,learning=0,frames=1,descriptors=8,features=(t+1)*2048,currents=(t+1)*11,
                           logits=(t+1)*11,probes=(t+1)*24,commits=0,rmw=0,publications=1892):
            raise ValueError('per-frame cumulative checks')
        if ends[t]['scalar_checks']!=(t+1)*409600 or frames[t]['sample']!=SAMPLE or frames[t]['timestep']!=t or frames[t]['commits']:
            raise ValueError('frame scope/counts')
        if frames[t]['cycles']!=ends[t]['cycle']-begins[t]['cycle'] or frames[t]['cycles']<=0:
            raise ValueError('frame cycle boundary')
        for k,v in geometry['numeric'].items():
            actual=numeric[t][k]
            if k=='frame': wanted_value=str(t)
            elif k in ('requests','contexts','scalars'): wanted_value='+'.join(str(int(n)*(t+1)) for n in v.split('+'))
            elif k in ('pooled_q8_8_values','classifier_logits','classifier_contexts'): wanted_value=str(int(v)*(t+1))
            else: wanted_value=v
            if actual!=wanted_value: raise ValueError('numeric geometry '+k)
        for l in range(8):
            i=t*8+l; b,c,p,h=(tables[k][i] for k in ('bucket','context','publisher','handshake'))
            if any(v<0 for r in (b,c,p,h) for v in r.values()): raise ValueError('negative count')
            if b['cycles']!=int(desc[2*i+1]['cycle'])-int(desc[2*i]['cycle']) or sum(b[k] for k in BUCKETS)!=b['cycles']:
                raise ValueError('scheduler cycle closure')
            if sum(c[k] for k in ('idle','cache','transfer','drain'))!=b['context_load'] or sum(p[k] for k in PUBLISHER)>b['context_load']:
                raise ValueError('context cycle closure')
            if not h['features']==h['responses']==h['commands']==h['completions']>0 or h['loads']!=h['releases'] or h['loads']!=h['weight_loads']+h['reuses']:
                raise ValueError('handshake closure')
            if p['reads']!=p['responses'] or p['reads']!=BANK_READS[l] or h['forward_reads']!=p['reads'] or h['forward_responses']!=p['reads'] or p['starts']!=h['weight_loads']:
                raise ValueError('forward/publisher work')
        measured.append(dict(frame=t,start=begins[t]['cycle'],end=ends[t]['cycle'],cycles=frames[t]['cycles'],
            **{k:sum(r[k] for r in tables['bucket'][t*8:t*8+8]) for k in BUCKETS}))
    service=ints('SCNN_TURNOVER_SERVICE_PASS')
    expected_banks=[n*20 if i<8 else n for i,n in enumerate(BANK_READS)]
    if ints('SCNN_TURNOVER_BANK')!=[dict(bank=i,reads=n,responses=n) for i,n in enumerate(expected_banks)]: raise ValueError('bank work')
    wanted_service=dict(reads=sum(expected_banks),responses=sum(expected_banks),slots=1,code_bits=16,address_bits=24,tag_bits=16,minimum_latency=1,replaceable=1,m2_cache_depth=512)
    if len(service)!=1 or any(service[0].get(k)!=v for k,v in wanted_service.items()) or not 0<service[0]['replacements']<service[0]['reads']:
        raise ValueError('native service contract')
    events=rows(text,'SCNN_PROFILE_EVENT')
    starts=[int(r['cycle']) for r in events if r['kind']=='initialize_start']
    stops=[int(r['cycle']) for r in events if r['kind']=='initialize_done']
    if len(starts)!=1 or len(stops)!=1 or starts[0]>=stops[0]: raise ValueError('initialization count')
    for kind in ('window_request','window_done'):
        if [int(r['window']) for r in events if r['kind']==kind]!=list(range(1,6)): raise ValueError('five windows missing')
    if any(r['kind'] in ('learning_start','commit') for r in events): raise ValueError('unexpected learning')
    if any(begins[t]['cycle']<ends[t-1]['cycle'] for t in range(1,20)): raise ValueError('overlapping frames')
    return dict(status='PASS_one_representative_complete_T20_window',scope=SCOPE,frames=measured,tables=tables,
        service=service[0],initialization_cycles=stops[0]-starts[0],initialization_end_to_frame0=begins[0]['cycle']-stops[0],
        frame_cycles=sum(r['cycles'] for r in frames),between_frame_cycles=sum(begins[t]['cycle']-ends[t-1]['cycle'] for t in range(1,20)),
        cold_interval_cycles=ends[-1]['cycle']-starts[0],post_first_frame_interval_cycles=ends[-1]['cycle']-ends[0]['cycle'],
        diagnostics=diag,final=final[0])


def worker():
    from run_source_stream_comparison import diagnostics
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE)
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();start=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,pid=process.pid,timed_out=True,wall_seconds=time.monotonic()-start);save();raise
        diag=diagnostics((root/(stage+'.log')).read_text(errors='replace'))
        state[stage]=dict(command=command,exit=code,wall_seconds=time.monotonic()-start,diagnostics=diag);save()
        if code or diag['errors'] or diag['fatal'] or diag['structural']: raise RuntimeError(stage+' failed')
    try:
        command=json.loads((root/'accepted_build.json').read_text())
        command[command.index('--Mdir')+1]=str(scratch/'obj')
        command[command.index('-f')+1]='sources.f'
        command[command.index('-MAKEFLAGS')+1]='OPT_FAST=-O1 OPT_GLOBAL=-O1 -f '+str(root/'scripts/supplement_scnn_graph_host.mk')
        execute('build',command,5400)
        execute('run',[str(scratch/'obj/Vscgraph'),'+SCNN_PROFILE_FIXTURE=fixture',f'+REP_SAMPLE={SAMPLE}'],43200)
        checks=parse_log((root/'run.log').read_text(),json.loads((root/'geometry_reference.json').read_text()))
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state.update(status=checks['status'],cold_interval_cycles=checks['cold_interval_cycles'])
    except Exception as error: state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    files=sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    size=sum(p.stat().st_size for p in files)
    probe='''import datetime,json,pathlib,shutil
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage('/root/autodl-tmp').free,tmp_free=shutil.disk_usage('/tmp').free)
if active or record['data_free']<REQUIRED_DATA or record['tmp_free']<3*1024**3:raise RuntimeError(record)
print(json.dumps(record))
'''
    client=connect()
    try:
        before=json.loads(run(client,'python3 -c '+shlex.quote('REQUIRED_DATA='+str(size+256*1024**2)+'\n'+probe)))
        run(client,'python3 -c '+shlex.quote('from pathlib import Path\np=Path('+repr(destination)+')\ns=Path('+repr('/tmp/snn_'+NAME)+')\nassert not p.exists() and not s.exists()\np.mkdir()'))
        with client.open_sftp() as sftp:
            for path in files:
                remote=destination+'/'+path.relative_to(prepared).as_posix();upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual:
                    actual.prefetch(file_size=path.stat().st_size,max_concurrent_requests=16)
                    if actual.read()!=path.read_bytes():raise ValueError('uploaded input differs: '+str(path))
        at_start=json.loads(run(client,'python3 -c '+shlex.quote('REQUIRED_DATA='+str(256*1024**2)+'\n'+probe)))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_scnn_t20 bash -lc '+shlex.quote(command))
        record=dict(launched=True,remote=destination,screen='snn_scnn_t20',before_upload=before,before_start=at_start,
                    prepared_input_files_byte_equal=len(files),prepared_input_bytes=size)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch'))
    args=parser.parse_args();{'prepare':prepare,'worker':worker,'launch':launch}[args.action]()
