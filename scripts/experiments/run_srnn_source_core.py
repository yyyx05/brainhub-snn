"""New raw-count source comparison around the isolated A whole core.

Both producers use the same end-marked client. Neither receives an oracle
sparse group count. Original reference arrays are reused, not regenerated.
"""
import argparse
import datetime as dt
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NAME = 'srnn_source_core_r1'
EXPERIMENT = Path('verification/experiments/srnn_source_core_r1')
TB = 'tb_srnn_profile_measurement.sv'
SCOPE = ('Same eight accepted SHD sequences, new raw-count producer-to-whole-core comparison, '
         'not a new distribution. Buffered RAM sparse and streaming sparse share an end-marked '
         'FF layer client, original recurrent suffix, A/512/no-model-key, original learning off/on. '
         'Explicit zero-mask FF marker is charged. No oracle sparse group count. Not DDR, '
         'board timing, author RTL, or production adoption; no new physical results.')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('ambiguous anchor: '+old)
    return text.replace(old, new, 1)


def prepare():
    from run_srnn_resident_remote import ROOT, local_path
    root = Path(local_path(ROOT))
    prior = root/'verification/experiments/srnn_representative_inputs/a_r1'
    out = root/EXPERIMENT
    shutil.copytree(prior, out, ignore=shutil.ignore_patterns('__pycache__'))
    changes = []
    control = Path('rtl/control/unified_fc_srnn_h40_p4_layer_client_v1.sv')
    old = (out/control).read_text()
    new = (root/'verification/experiments/srnn_endmarked_layer_r1/endmarked.sv').read_text()
    (out/control).write_text(new, newline='\n')
    changes.append(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                             fromfile='A/'+str(control), tofile='stream/'+str(control))))
    tops = ('unified_h40_final_composition_top_v1.sv',
            'unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv',
            'unified_h40_array16_three_topology_top_v1.sv')
    for filename in tops:
        path = out/'rtl/top'/filename
        source = text = path.read_text()
        # Existing declarations have different alignment, but one port per line.
        matches = re.findall(r'^    input[^\n]*\bi_fc_srnn_layer_valid,\s*$', text, re.M)
        if len(matches) != 1:
            raise ValueError('layer declaration '+filename)
        text = once(text, matches[0], matches[0]+'\n    input wire logic i_fc_srnn_streamed_feedforward,\n    input wire logic i_fc_srnn_group_feedforward_end,')
        if filename == tops[0]:
            text = once(text, '        .i_fc_srnn_layer_valid(i_fc_srnn_layer_valid),',
                '        .i_fc_srnn_streamed_feedforward(i_fc_srnn_streamed_feedforward),\n'
                '        .i_fc_srnn_group_feedforward_end(i_fc_srnn_group_feedforward_end),\n'
                '        .i_fc_srnn_layer_valid(i_fc_srnn_layer_valid),')
        elif filename == tops[1]:
            text = once(text, '         .i_fc_srnn_layer_valid(m2_array_owner ? m2_layer_valid :',
                '         .i_fc_srnn_streamed_feedforward((m2_array_owner || srnn_group_streamer_selected) ? 1\'b0 : i_fc_srnn_streamed_feedforward),\n'
                '         .i_fc_srnn_group_feedforward_end((m2_array_owner || srnn_group_streamer_selected) ? 1\'b0 : i_fc_srnn_group_feedforward_end),\n'
                '         .i_fc_srnn_layer_valid(m2_array_owner ? m2_layer_valid :')
        else:
            text = once(text, '.i_layer_valid(i_fc_srnn_layer_valid),',
                '.i_streamed_feedforward(i_fc_srnn_streamed_feedforward),\n'
                '            .i_group_feedforward_end(i_fc_srnn_group_feedforward_end),\n'
                '            .i_layer_valid(i_fc_srnn_layer_valid),')
        path.write_text(text, newline='\n')
        changes.append(''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
                           fromfile='A/rtl/top/'+filename, tofile='stream/rtl/top/'+filename)))
    (out/'core_changes.diff').write_text(''.join(changes))
    producer = root/'verification/experiments/nearest_neighbor/buffered_ram_r1'
    for name in ('source_scan.sv', 'source_stream.sv'):
        shutil.copy2(producer/name, out/name)
    source = text = (prior/TB).read_text()
    snippet = (Path(__file__).with_name('srnn_source_core_ingress.svh')).read_text()
    text = once(text, '    task automatic refill(input bit rec);', snippet+'\n    task automatic refill(input bit rec);')
    text = once(text, '        fixture_root = fixture;',
        '        if (!$value$plusargs("STREAMING=%d",streaming)) $fatal(1,"missing source policy");\n'
        '        if (streaming!=0 && streaming!=1) $fatal(1,"source policy range");\n'
        '        fixture_root = fixture;')
    # Replace the group driver only; profiling continues to inspect the actual
    # group wires after the mux, including FF producer backpressure.
    begin = text.index('    task automatic issue(')
    end = text.index('    endtask', begin)+len('    endtask')
    task = text[begin:end]
    task = task.replace('i_fc_srnn_group_', 'manual_group_')
    text = text[:begin]+task+text[end:]
    text = once(text, '        .i_fc_srnn_layer_valid(i_fc_srnn_layer_valid),',
        '        .i_fc_srnn_streamed_feedforward(1\'b1),\n'
        '        .i_fc_srnn_group_feedforward_end(feed_active && producer_last),\n'
        '        .i_fc_srnn_layer_valid(i_fc_srnn_layer_valid),')
    # No hidden/precomputed sparse packet input is consumed by either arm.
    text = re.sub(r'^\s*logic \[63:0\] ff_groups[^\n]*\n', '\n', text, flags=re.M)
    text = re.sub(r'^\s*logic \[(?:31|15):0\] ff_(?:offsets|counts)[^\n]*\n', '\n', text, flags=re.M)
    read = '        $readmemh({raw,"/feedforward_p4.mem"},ff_groups); $readmemh({raw,"/feedforward_offsets.mem"},ff_offsets);\n        $readmemh({raw,"/feedforward_group_counts.mem"},ff_counts);'
    if text.count(read) != 2:
        raise ValueError('raw input anchors')
    text = text.replace(read, '        $readmemh({raw,"/raw700_counts.mem"},raw_counts);')
    text = once(text, '            i_fc_srnn_feedforward_group_count = ff_counts[timestep]; i_fc_srnn_layer_valid = 1;',
        '            i_fc_srnn_feedforward_group_count = 0; i_fc_srnn_layer_valid = 1;')
    text = once(text,
        '            for (int k=int\'(ff_offsets[timestep]); k<int\'(ff_offsets[timestep+1]); k++)\n'
        '                issue(0,ff_groups[k][3:0],int\'(ff_groups[k][18:8]),int\'(ff_groups[k][29:19]),int\'(ff_groups[k][40:30]),int\'(ff_groups[k][51:41]));',
        '            feed_raw_frame();')
    text = once(text, '        $finish;',
        '        if (source_words!=102900 || source_groups!=33761 || source_tails!=588)\n'
        '            $fatal(1,"source coverage");\n'
        '        $display("SOURCE_CORE_PASS streaming=%0d learning=%0d words=%0d groups=%0d tails=%0d count_stalls=%0d group_stalls=%0d",\n'
        '            streaming,learning,source_words,source_groups,source_tails,source_count_stalls,source_group_stalls);\n'
        '        $finish;')
    (out/TB).write_text(text, newline='\n')
    (out/'source_tb_changes.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
            fromfile='accepted_representative_TB', tofile='raw_source_core_TB')))
    # Already-prepared numerical arrays remain the independent assertions.
    unchanged = [p.relative_to(prior).as_posix() for p in (prior/'multi_fixture').rglob('*') if p.is_file()]
    for rel in unchanged:
        if (prior/rel).read_bytes() != (out/rel).read_bytes():
            raise ValueError('numeric input changed '+rel)
    with (out/'sources.f').open('a') as handle:
        handle.write('"source_scan.sv"\n"source_stream.sv"\n')
    for filename in ('run_srnn_representative_inputs.py', 'srnn_source_core_ingress.svh'):
        shutil.copy2(Path(__file__).with_name(filename), out/filename)
    shutil.copy2(Path(__file__), out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_NOT_SIMULATED', scope=SCOPE,
        source_reference='srnn_representative_inputs/a_r1', numerical_files_byte_equal=len(unchanged),
        changed_core_files=4, unchanged_core_files=105, new_producers=2,
        original_eight_sequence_input_selection=True, oracle_sparse_count=False,
        recurring_state_reset_between_sequences='original logical dynamics only; no global reset/parameter reload',
        storage_baseline='accepted posedge-only frame RAM source_scan, not FF-expanded baseline'), indent=2)+'\n')
    print(json.dumps(dict(prepared=str(out), files=sum(p.is_file() for p in out.rglob('*')),
                         bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))))


def parse_log(log, arm, info, streaming):
    import run_srnn_representative_inputs as old
    old.SCOPE = SCOPE
    result = old.parse_log(log, arm, info)
    lines = re.findall(r'^SOURCE_CORE_PASS (.+)$', log, re.M)
    if len(lines) != 1:
        raise ValueError('missing source completion')
    source = dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)', lines[0]))
    if any(source.get(k)!=v for k,v in dict(streaming=streaming,learning=int(arm=='on'),words=102900,groups=33761,tails=588).items()):
        raise ValueError('source traffic')
    frames = [dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)', line))
              for line in re.findall(r'^SOURCE_CORE_FRAME (.+)$', log, re.M)]
    expected = [(s,t) for s, steps in enumerate(info['native_steps']) for t in range(steps)]
    if [(r['sequence'],r['timestep']) for r in frames] != expected:
        raise ValueError('source frame coverage/order')
    if any(r['words']!=175 or r['tails']!=1 or r['cycles']<=0 for r in frames):
        raise ValueError('source frame interval')
    if sum(r['groups'] for r in frames)!=33761:
        raise ValueError('frame group total')
    return dict(**result, source=source, source_frames=frames)


def worker():
    root=Path.cwd(); scratch=Path('/tmp/snn_'+NAME); scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE)
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage, command, timeout):
        state['status']=stage+'_running'; save(); begin=time.monotonic()
        with (root/(stage+'.log')).open('x') as handle:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            state[stage]=dict(command=command,pid=process.pid,exit=None);save()
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage].update(timed_out=True,wall_seconds=time.monotonic()-begin);save();raise
        state[stage].update(exit=code,wall_seconds=time.monotonic()-begin);save()
        if code:raise RuntimeError(stage+' exit '+str(code))
    try:
        info=json.loads((root/'multi_fixture/reference.json').read_text())
        build=json.loads((root/'build_command.json').read_text())
        build[build.index('--Mdir')+1]=str(scratch/'obj')
        execute('build',build,3600)
        text=(root/'build.log').read_text(errors='replace')
        state['build_warnings']=dict(ordinary=len(re.findall(r'^%Warning-',text,re.M)),
            structural=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',text,re.M))
        if state['build_warnings']['structural']:raise ValueError('structural warning')
        checks={}
        for streaming in (0,1):
            for arm in ('off','on'):
                name=('stream_' if streaming else 'buffered_')+arm
                execute('run_'+name,[str(scratch/'obj/Vtb_srnn_profile_measurement'),'+RAW=multi_fixture',
                        '+FIXTURE=multi_fixture','+LEARNING='+str(int(arm=='on')),'+STREAMING='+str(streaming)],7200)
                checks[name]=parse_log((root/('run_'+name+'.log')).read_text(errors='replace'),arm,info,streaming)
                state[name]=checks[name];save()
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state['status']='PASS_raw_count_buffered_streaming_A_core_four_arms'
    except Exception as error:state.update(status='FAIL',error=repr(error))
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
        for number,path in enumerate(files):
            with client.open_sftp() as sftp:
                remote=destination+'/'+path.relative_to(prepared).as_posix();upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual:
                    actual.prefetch(file_size=path.stat().st_size,max_concurrent_requests=16)
                    if actual.read()!=path.read_bytes():raise ValueError('uploaded input differs: '+str(path))
            if number%50==0:print('TRANSFER',number+1,len(files),flush=True)
        at_start=json.loads(run(client,'python3 -c '+shlex.quote('REQUIRED_DATA='+str(256*1024**2)+'\n'+probe)))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_srnn_source_core bash -lc '+shlex.quote(command))
        record=dict(launched=True,remote=destination,screen='snn_srnn_source_core',before_upload=before,before_start=at_start,
                    prepared_input_files_byte_equal=len(files),prepared_input_bytes=size)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch'))
    args=parser.parse_args();{'prepare':prepare,'worker':worker,'launch':launch}[args.action]()
