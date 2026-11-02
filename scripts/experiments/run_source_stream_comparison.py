"""Three-policy pre-LIF comparison and isolated producer synthesis cost."""
import argparse
import collections
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME = 'source_stream_compare_r1'
EXPERIMENT = Path('verification/experiments/nearest_neighbor/stream_r1')
VERILATOR = '/root/autodl-tmp/verilator_5_048_install/bin/verilator'
VIVADO = '/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado'
SOURCES = ('memory/unified_dual_role_weight_store_p4all_candidate_v1.sv',
           'tiles/unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1.sv',
           'tiles/unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv')
SCOPE = ('Count-preserving four-source literature scan adaptation, buffered local zero skip, '
         'and streaming local zero skip; same 12 original stores/accumulators and inputs. '
         'Pre-LIF only, not author RTL, whole-network speedup, production integration or routed STA.')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('TB edit anchor missing or duplicated: '+old)
    return text.replace(old, new)


def prepare():
    from run_srnn_resident_remote import ROOT, local_path
    root = Path(local_path(ROOT)); out = root/EXPERIMENT
    old = root/'verification/experiments/nearest_neighbor/scan_rtl_r1'
    if (out/'launch_worker.py').exists():
        raise FileExistsError('already prepared')
    for rel in ('source_scan.sv', 'scan_consumer.sv'):
        shutil.copy2(old/rel, out/rel)
    shutil.copytree(old/'rtl', out/'rtl')
    shutil.copytree(old/'fixture', out/'fixture')
    shutil.copy2(root/'scripts/experiments/run_fc_bitmap_mapping.py', out/'run_fc_bitmap_mapping.py')
    tb = (old/'tb_source_scan.sv').read_text()
    edits = [
        ('logic sparse=0, count_valid=0, count_ready;', 'logic sparse=0, streaming=0, count_valid=0, count_ready;\n    integer policy=0;'),
        ('source_scan producer(', 'source_scan_experiment producer(.streaming(streaming),'),
        ('integer stalls=0, memory_fires=0, accum_fires=0, result_stalls=0;',
         'integer stalls=0, memory_fires=0, accum_fires=0, result_stalls=0;\n    integer phase_cycles[0:4];\n    integer input_stalls=0, input_idle=0;'),
        ('logic finished;', 'logic finished, accepted;'),
        ('if (measuring) begin', 'if (measuring) begin\n            if (phase>4) $fatal(1,"unknown phase");\n            phase_cycles[phase]++;\n            if (count_valid && !count_ready) input_stalls++;\n            if (count_ready && !count_valid) input_idle++;'),
        ('word_index=0;', 'word_index=0; count_valid=0; input_stalls=0; input_idle=0;\n        for(int k=0;k<5;k++) phase_cycles[k]=0;'),
        ('count_valid=scenario==0 || (cycles-origin)%5!=0;',
         'if (!count_valid) count_valid=scenario==0 || (cycles-origin)%5!=0;'),
        ('if(count_valid && count_ready) word_index++;',
         'accepted=count_valid && count_ready;\n            if(accepted) word_index++;'),
        ('@(negedge clk);\n        end\n        count_valid=0;',
         '@(negedge clk);\n            if(accepted) count_valid=0;\n            if(cycles-origin>100000) $fatal(1,"input timeout");\n        end\n        count_valid=0;'),
        ('scans!=(nonzero==0 ? 0 : 175)', 'scans!=((nonzero==0 || streaming) ? 0 : 175)'),
        ('$display("SCAN_FRAME scenario=',
         'if(finish_cycle-origin!=phase_cycles[0]+phase_cycles[1]+phase_cycles[2]+phase_cycles[3]+phase_cycles[4])\n            $fatal(1,"exclusive phase sum mismatch");\n        $display("SCAN_FRAME policy=%0d load=%0d scan=%0d emit=%0d finish=%0d drain=%0d input_stalls=%0d input_idle=%0d scenario='),
        ('scenario,sparse,f,finish_cycle-origin,ingress_end-origin,finish_cycle-ingress_end,',
         'policy,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3],phase_cycles[4],input_stalls,input_idle,\n            scenario,sparse,f,finish_cycle-origin,ingress_end-origin,finish_cycle-ingress_end,'),
        ('for(int p=0;p<2;p++) begin\n                sparse=1\'(p);',
         'for(int p=0;p<3;p++) begin\n                policy=p; sparse=(p!=0); streaming=(p==2);'),
        ('SCAN_PASS frames=2368 current_checks=1065600 policies=2 scenarios=2',
         'SCAN_PASS frames=3552 current_checks=1598400 policies=3 scenarios=2'),
    ]
    for before, after in edits:
        tb = replace_once(tb, before, after)
    (out/'tb_source_scan.sv').write_text(tb)
    shutil.copy2(Path(__file__), out/'launch_worker.py')
    shutil.copy2(root/'scripts/experiments/source_stream_mapping.tcl', out/'mapping.tcl')
    (out/'clock.xdc').write_text('create_clock -name core_clk -period 15.000 [get_ports clk]\n')
    (out/'preparation.json').write_text(json.dumps(dict(scope=SCOPE,
        policies=['literature_count_scan_adaptation','buffered_local_zero_skip','streaming_local_zero_skip'],
        source='Existing accepted E10 fixture, not a new sample distribution',
        changes='New streaming producer; original stores, accumulators, weights and currents unchanged',
        input='Same ordered 175x32-bit ready/valid words; source now holds valid until accepted in all policies',
        accounting='Exclusive LOAD/SCAN/EMIT/FINISH/DRAIN sum to interval. Input backpressure is included, not added twice.',
        cost='Static specialization of each producer, xc7vx690tffg1761-3, Vivado2022.2 synthesis at15ns; no new full-core EDA',
        limitations='Streaming cannot replay a consumed input vector; downstream must accept backpressure. No LIF, recurrent update, learning or abort claim.'),indent=2)+'\n')
    files = [p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))


def parse_log(text, fixture):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b', text):
        raise ValueError('runtime error')
    rows = [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)', line)}
            for line in re.findall(r'^SCAN_FRAME (.+)$', text, re.M)]
    if [(r['scenario'],r['policy'],r['frame']) for r in rows] != [
            (s,p,f) for s in range(2) for p in range(3) for f in range(592)]:
        raise ValueError('incomplete/duplicate frame set')
    if text.count('SCAN_PASS frames=3552 current_checks=1598400 policies=3 scenarios=2') != 1:
        raise ValueError('incomplete final checks')
    with (fixture/'frames.csv').open(newline='') as f:
        reference = list(csv.DictReader(f))
    for r in rows:
        if r['checked']!=450 or r['input_words']!=175 or r['delimiters']!=1:
            raise ValueError('numeric/ingress/delimiter count')
        if r['memory']!=r['groups']+1 or r['accum']!=r['memory']:
            raise ValueError('consumer accounting')
        if r['total']!=sum(r[k] for k in ('load','scan','emit','finish','drain')) or r['total']!=r['ingress']+r['service']:
            raise ValueError('exclusive cycles do not sum')
        if r['policy']==2 and r['scans']:
            raise ValueError('streaming unexpectedly rescanned')
        if r['frame']<588:
            ref=reference[r['frame']]
            groups=int(ref['dense_issue_groups' if r['policy']==0 else 'packed_issue_groups'])
            if r['policy'] and int(ref['whole_vector_zero']): groups-=1
            if r['groups']!=groups:
                raise ValueError('group count differs from independent reference')
        if r['scenario']==0:
            if r['total']!=175+r['scans']+r['groups']+1+2 or r['group_stalls'] or r['result_stalls']:
                raise ValueError('unstalled cycle identity')
    totals=[]
    for s in range(2):
        for p in range(3):
            selected=[r for r in rows if r['scenario']==s and r['policy']==p and r['frame']<588]
            total=dict(scenario=s,policy=p,frames=588)
            total.update({k:sum(r[k] for r in selected) for k in
                ('total','load','scan','emit','finish','drain','input_words','input_stalls','input_idle',
                 'groups','delimiters','group_stalls','result_stalls','memory','accum','checked')})
            if s==1 and (not total['group_stalls'] or not total['result_stalls']):
                raise ValueError('downstream backpressure not exercised')
            if p==2 and not total['input_stalls']:
                raise ValueError('streaming input backpressure not exercised')
            totals.append(total)
    return dict(scope=SCOPE,frames=rows,real_frame_totals=totals,current_checks=1598400,preload_writes=16800)


def diagnostics(text):
    categories=collections.Counter(re.findall(r'^%Warning-([^:]+):',text,re.M))
    return dict(warning_categories=dict(sorted(categories.items())),ordinary_warnings=sum(categories.values()),
        errors=len(re.findall(r'^%Error|^ERROR:',text,re.M)),
        critical_warnings=len(re.findall(r'^CRITICAL WARNING:',text,re.M)),
        vivado_warnings=len(re.findall(r'^WARNING:',text,re.M)),
        fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b',text)),
        structural=[k for k in categories if k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')],
        oom_or_killed=bool(re.search(r'out of memory|\bOOM\b|\bKilled\b',text,re.I)),
        disk_exhaustion=bool(re.search(r'no space left on device',text,re.I)))


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    root=Path.cwd(); scratch=Path('/tmp/snn_'+NAME); scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scope=SCOPE,scratch=str(scratch))
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout,env=None):
        state['status']=stage+'_running';save();begin=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,pid=process.pid,timed_out=True,wall_seconds=time.monotonic()-begin)
                save();raise
        diag=diagnostics((root/(stage+'.log')).read_text(errors='replace'))
        state[stage]=dict(command=command,exit=code,wall_seconds=time.monotonic()-begin,diagnostics=diag);save()
        if code or diag['errors'] or diag['fatal'] or diag['structural'] or diag['critical_warnings']:
            raise RuntimeError(stage+' failed or has critical diagnostics')
    try:
        execute('build',[VERILATOR,'--binary','--timing','--assert','-j','2','-Wall','-Wno-fatal',
            '--top-module','tb_source_scan','--Mdir',str(scratch/'obj'),
            *['rtl/'+p for p in SOURCES],'source_scan.sv','source_stream.sv','scan_consumer.sv','tb_source_scan.sv'],1800)
        execute('run',[str(scratch/'obj/Vtb_source_scan'),'+FIXTURE=fixture'],1800)
        checks=parse_log((root/'run.log').read_text(),root/'fixture')
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state['real_frame_totals']=checks['real_frame_totals'];save()
        execute('mapping',[VIVADO,'-mode','batch','-nojournal','-nolog','-source','mapping.tcl',
                           '-tclargs',str(root),str(scratch)],1800,vivado_environment(VIVADO))
        if 'SOURCE_STREAM_MAPPING_COMPLETE' not in (root/'mapping.log').read_text():
            raise ValueError('mapping completion marker missing')
        state['status']='PASS_pre_LIF_streaming_comparison_and_producer_synthesis_not_routed_STA'
    except Exception as error: state.update(status='FAIL',error=repr(error))
    for path in scratch.rglob('*'):
        rel=path.relative_to(scratch)
        if path.is_file() and not set(rel.parts)&{'obj','elab_obj','.Xil','__pycache__'}:
            target=root/'scratch_originals'/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    inputs=sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    input_bytes=sum(p.stat().st_size for p in inputs)
    code='''import datetime,json,pathlib,shutil
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
if record['data_free'] < INPUT_BYTES+128*1024**2 or record['tmp_free'] < 1024**3:raise RuntimeError('insufficient small-run headroom')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
    client=connect()
    try:
        record=json.loads(run(client,'python3 -c '+shlex.quote('DEST='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nINPUT_BYTES='+str(input_bytes)+'\n'+code)))
        with client.open_sftp() as sftp:
            for path in inputs:
                remote=destination+'/'+path.relative_to(prepared).as_posix()
                upload(sftp,str(path),remote)
                with sftp.open(remote,'rb') as actual:
                    actual.prefetch(file_size=path.stat().st_size,max_concurrent_requests=16)
                    if actual.read()!=path.read_bytes():raise ValueError('uploaded input differs: '+str(path))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_source_stream bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_source_stream',prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args();{'prepare':prepare,'launch':launch,'worker':worker}[args.action]()
