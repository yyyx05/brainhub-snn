"""Prepare the final candidate's necessary nonzero-update-to-RTL-query check."""
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

NAME='scnn_turnover_tail_r1'
EXPERIMENT=Path('verification/experiments/scnn_forward_turnover/tail_r1/candidate_r1')
TB='tb_scnn_tail_visibility.sv'
PARSER='prepare_scnn_tail_testbench.py'
BANK_READS=[0]*7+[122683392,90112]
SCOPE='Final compact512 turnover candidate. Original cold B4T12+query4 necessary natural nonzero fixture, only52 Conv8 tail descriptors with explicit CPU Conv7 feature service. Actual final-top capture/shared learning/publication/invalidation/RTL classifier query. Not full raw Conv1..8, T20/Test, new throughput, DDR, arbitrary abort or STA.'

TAIL_REPORT=r'''
    longint unsigned st_query_replacements=0;
    longint unsigned st_pub_replacements [0:2]='{default:0};
    task automatic st_report;
        longint unsigned expected_reads;
        if (fg_forward_valid || st_reads!=st_responses || st_reads!=122773504 ||
            st_replacements==0 || st_query_replacements==0 || PARAM_ADDR_WIDTH!=24 || TAG_WIDTH!=16)
            $fatal(1,"turnover tail final credit/width/coverage mismatch");
        for (int bank=0; bank<9; bank++) begin
            expected_reads=bank<7 ? 0 : bank==7 ? 122683392 : 90112;
            if (st_bank_reads[bank]!=expected_reads || st_bank_responses[bank]!=expected_reads)
                $fatal(1,"turnover tail bank mismatch");
            $display("SCNN_TURNOVER_TAIL_BANK bank=%0d reads=%0d responses=%0d",bank,st_bank_reads[bank],st_bank_responses[bank]);
        end
        for (int window=0;window<3;window++) begin
            if (st_pub_replacements[window]==0) $fatal(1,"post-learning publication did not exercise turnover");
            $display("SCNN_TURNOVER_TAIL_PUBLICATION window=%0d replacements=%0d",window+1,st_pub_replacements[window]);
        end
        $display("SCNN_TURNOVER_TAIL_SERVICE_PASS reads=%0d responses=%0d replacements=%0d query_replacements=%0d slots=1 code_bits=16 address_bits=24 tag_bits=16 minimum_latency=1 replaceable=1 m2_cache_depth=512 end_cycle=%0d",
            st_reads,st_responses,st_replacements,st_query_replacements,cycle_count);
        $fflush();
    endtask
'''


def transform(text):
    from run_scnn_turnover_frame import once,MONITOR
    clear='if (fg_forward_valid && ft_o_store_forward_response_ready) fg_forward_valid <= 0;'
    request='if (ft_o_store_forward_read_valid && ft_i_store_forward_read_ready) begin'
    start=text.index('always @(posedge clk) begin : temporal_external_ddr')
    if text.index(clear,start)>text.index(request,start):raise ValueError('replacement host priority is wrong')
    text=once(text,'assign ft_i_store_forward_read_ready = !fg_forward_valid;',
        'assign ft_i_store_forward_read_ready = !fg_forward_valid || ft_o_store_forward_response_ready;')
    text=once(text,'.TOPOLOGY_MASK(TOPOLOGY_MASK),','.TOPOLOGY_MASK(TOPOLOGY_MASK),\n        .M2_CACHE_DEPTH(512),')
    for bank in (0,1):
        text=once(text,f'.g_context[0].g_enabled.u_store.g_bank[{bank}].memory[0]',
            f'.g_context[0].g_enabled.g_original.u_store.g_bank[{bank}].memory[0]')
    text=once(text,'        fg_finish();\n        $finish;','        fg_finish();\n        st_report();\n        $finish;')
    if MONITOR.count('    task automatic st_report;')!=1:raise ValueError('native monitor shape changed')
    monitor=MONITOR.split('    task automatic st_report;',1)[0]
    monitor=once(monitor,'                st_responses++; st_bank_responses[st_pending_bank]++;',
        '''                st_responses++; st_bank_responses[st_pending_bank]++;
                if (request_fire && tv_query) st_query_replacements++;
                if (request_fire && st_pending_bank==8 && tv_window_end && !tv_query)
                    st_pub_replacements[tv_window]++;''')
    return once(text,'endmodule',monitor+TAIL_REPORT+'\nendmodule')


def parse_log(log,original_parser=None):
    if original_parser is None:
        from prepare_scnn_tail_testbench import parse_log as original_parser
    checks=original_parser(log)
    def rows(marker):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)} for line in log.splitlines() if line.startswith(marker+' ')]
    service=rows('SCNN_TURNOVER_TAIL_SERVICE_PASS')
    expected=dict(reads=122773504,responses=122773504,slots=1,code_bits=16,address_bits=24,tag_bits=16,
        minimum_latency=1,replaceable=1,m2_cache_depth=512)
    if len(service)!=1 or any(service[0].get(k)!=v for k,v in expected.items()):raise ValueError('tail native service completeness')
    item=service[0]
    if not 0<item['query_replacements']<=item['replacements']<item['reads'] or item['end_cycle']!=checks['final']['cycles']:
        raise ValueError('tail turnover/query coverage')
    banks=rows('SCNN_TURNOVER_TAIL_BANK')
    if banks!=[dict(bank=i,reads=n,responses=n) for i,n in enumerate(BANK_READS)]:raise ValueError('tail bank work changed')
    pubs=rows('SCNN_TURNOVER_TAIL_PUBLICATION')
    if [x.get('window') for x in pubs]!=[1,2,3] or any(not 0<x.get('replacements',0)<22528 for x in pubs):
        raise ValueError('post-learning publication coverage')
    if sum(x['replacements'] for x in pubs)+item['query_replacements']>item['replacements']:
        raise ValueError('turnover subset accounting')
    markers=[line.split()[0] for line in log.splitlines() if line.startswith(('SCNN_TAIL_VISIBILITY_PASS ','SCNN_TURNOVER_TAIL_'))]
    if markers!=['SCNN_TAIL_VISIBILITY_PASS']+['SCNN_TURNOVER_TAIL_BANK']*9+['SCNN_TURNOVER_TAIL_PUBLICATION']*3+['SCNN_TURNOVER_TAIL_SERVICE_PASS']:
        raise ValueError('native completion ordering')
    checks.update(status='PASS_final_turnover_nonzero_Conv8_tail_RTL_query_not_full_graph_or_abort',
        native_service=item,native_banks=banks,native_publications=pubs,scope=SCOPE)
    return checks


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    base=Path(local_path(ROOT/RESULT));accepted=base/'scnn_tail_visibility_r1';source=base/'scnn_turnover_frame_r2/candidate'
    out=Path(local_path(ROOT/EXPERIMENT));out.mkdir(parents=True,exist_ok=False)
    original=(accepted/TB).read_text();changed=transform(original)
    (out/TB).write_text(changed,newline='\n')
    (out/'testbench_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='accepted_tail_r1/tb',tofile='final_turnover_tail/tb')))
    shutil.copytree(source/'rtl',out/'rtl')
    paths=sorted(p.relative_to(source).as_posix() for p in (source/'rtl').rglob('*') if p.is_file())
    if len(paths)!=109 or sum(p.endswith('.sv') for p in paths)!=108:raise ValueError('final source tree incomplete')
    dependencies=[]
    for folder in ('fixture','models','control','tail_reference'):
        shutil.copytree(accepted/folder,out/folder)
        dependencies.extend(p.relative_to(accepted).as_posix() for p in (accepted/folder).rglob('*') if p.is_file())
    for rel in ('tb/monitors/supplement_h40_activity_monitor.sv','scripts/supplement_scnn_graph_host.mk'):
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(accepted/rel,target);dependencies.append(rel)
    (out/'dependencies.json').write_text(json.dumps(sorted(dependencies),indent=2)+'\n')
    shutil.copy2(accepted/'corrected_local_parser.py',out/PARSER)
    command=json.loads((accepted/'scnn_compatibility_build.json').read_text())['command']
    (out/'accepted_build.json').write_text(json.dumps(command,indent=2)+'\n')
    (out/'sources.f').write_text(''.join('"'+rel+'"\n' for rel in paths if rel.endswith('.sv'))+'"'+TB+'"\n"tb/monitors/supplement_h40_activity_monitor.sv"\n')
    record=dict(status='PREPARED_NOT_SIMULATED',scope=SCOPE,source='scnn_turnover_frame_r2/candidate/rtl',source_files=109,
        original_test='scnn_tail_visibility_r1',dependencies=len(dependencies),build_timeout=5400,run_timeout=21600,
        service='Same replaceable latency1/code16/one-response-slot host as accepted raw frame, native addr24/tag16. Old response clear precedes replacement assignment; request/response held and credit checks added.',
        reason='First two cold natural windows change residual only; nonzero code appears only in window3. Retaining the original cold fixture avoids inventing a warm temporal/residual state. Only one final candidate, no old-RTL rerun or full raw52-frame graph.',
        boundary='Prepare locally while implementation is active. Do not upload/launch during any active EDA; no terminal audit before natural completion and full fetch.')
    (out/'reference.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py');print(str(out))


def worker():
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    config=json.loads((root/'reference.json').read_text());state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch),scope=SCOPE)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();started=time.monotonic()
        with (root/(stage+'.log')).open('w') as log:
            try:result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,wall_seconds=time.monotonic()-started,timeout_seconds=timeout);save();raise
        state[stage]=dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-started);save()
        if result.returncode:raise RuntimeError(stage+' exit '+str(result.returncode))
    try:
        command=json.loads((root/'accepted_build.json').read_text())
        command[command.index('--top-module')+1]='tb_scnn_tail_visibility'
        command[command.index('--Mdir')+1]=str(scratch/'obj');command[command.index('-f')+1]='sources.f'
        command[command.index('-MAKEFLAGS')+1]='OPT_FAST=-O1 OPT_GLOBAL=-O1 -f '+str(root/'scripts/supplement_scnn_graph_host.mk')
        execute('build',command,config['build_timeout'])
        log=(root/'build.log').read_text(errors='replace')
        state['ordinary_warnings']=len(re.findall(r'^%Warning-',log,re.M));state['structural_warnings']=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M)
        if state['structural_warnings']:raise ValueError('structural warnings')
        execute('run',[str(scratch/'obj/Vscgraph'),'+SCNN_TEMPORAL_FIXTURE=fixture','+SCNN_TEMPORAL_CONTROL=control','+TAIL_FIXTURE=tail_reference'],config['run_timeout'])
        checks=parse_log((root/'run.log').read_text(errors='replace'));(root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state.update(status=checks['status'],final=checks['final'])
    except Exception as error:state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    if not (prepared/'reference.json').is_file():raise ValueError('prepare first')
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
if p.exists() or pathlib.Path(SCRATCH).exists():raise FileExistsError('unique tail destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA; do not overlap: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p.parent).free,tmp_free=shutil.disk_usage('/tmp').free)
if record['data_free']<192*1024**2 or record['tmp_free']<2*1024**3:raise RuntimeError('insufficient tail archive/build margin')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
        record=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\n'+code)))
        inputs=[]
        with client.open_sftp() as s:
            for path in prepared.rglob('*'):
                if not path.is_file() or '__pycache__' in path.relative_to(prepared).parts:continue
                rel=path.relative_to(prepared).as_posix();upload(s,str(path),destination+'/'+rel)
                if s.open(destination+'/'+rel,'rb').read()!=path.read_bytes():raise ValueError('actual input mismatch: '+rel)
                inputs.append(rel)
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_turnover_tail bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_turnover_tail',prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','worker'));args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':launch()
    else:worker()
