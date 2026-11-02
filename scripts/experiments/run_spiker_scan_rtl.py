#!/usr/bin/env python3
"""Bounded pre-LIF literature-mechanism comparison, not author Spiker RTL."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME='spiker_scan_rtl_r1'
EXPERIMENT=Path('verification/experiments/nearest_neighbor/scan_rtl_r1')
VERILATOR='/root/autodl-tmp/verilator_5_048_install/bin/verilator'
SOURCES=('memory/unified_dual_role_weight_store_p4all_candidate_v1.sv',
         'tiles/unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1.sv',
         'tiles/unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv')
SCOPE='Four-lane count-preserving source-scan literature adaptation versus experimental online P4 packing; same actual pre-LIF consumer. Not author RTL, whole core, physical QoR, or production integration.'


def prepare():
    from run_srnn_resident_remote import ROOT,local_path
    root=Path(local_path(ROOT)); out=root/EXPERIMENT
    if (out/'launch_worker.py').exists(): raise FileExistsError('already prepared')
    for rel in SOURCES:
        dest=out/'rtl'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'verification/experiments/scnn_control_payload/candidate_r4/rtl'/rel,dest)
    shutil.copytree(root/'verification/experiments/nearest_neighbor/scan_fixture_r1',out/'fixture')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    (out/'preparation.json').write_text(json.dumps(dict(scope=SCOPE,original_consumer_files=list(SOURCES),
        ingress='175 words of four uint8 counts, same 175x32 frame buffer; no free host packing',
        scan='One block inspection cycle, followed by max(count) emissions; dense emits once even if block empty. Whole-vector zero bypass in both.',
        delimiter='One masked-zero final group in both arms, reported separately; no free last-group discovery',
        consumer='12 original H40 accumulators and 12 original 4x200x578 true-dual-port stores; two read ports per bank; 450 active destinations, 30 padding lanes',
        precision='Original signed INT8 weights, 48-bit accumulator checked against signed32 current; no LIF or recurrent state',
        tests='588 actual frames plus four count boundary cases, unstalled and deterministic stalled service',
        production_change=False),indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(prepared=str(out),files=len(files),bytes=sum(p.stat().st_size for p in files))))


def parse_log(text,fixture):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b',text): raise ValueError('runtime error')
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',s)}
          for s in re.findall(r'^SCAN_FRAME (.+)$',text,re.M)]
    if len(rows)!=2368 or text.count('SCAN_PASS frames=2368 current_checks=1065600 policies=2 scenarios=2')!=1:
        raise ValueError('incomplete frame/current set')
    expected=[(s,p,f) for s in range(2) for p in range(2) for f in range(592)]
    if [(r['scenario'],r['sparse'],r['frame']) for r in rows]!=expected: raise ValueError('frame order')
    with (fixture/'frames.csv').open(newline='') as stream:
        reference=list(csv.DictReader(stream))
    for r in rows:
        if r['checked']!=450 or r['input_words']!=175 or r['delimiters']!=1:
            raise ValueError('count/consumer boundary')
        if r['memory']!=r['groups']+1 or r['accum']!=r['memory'] or r['total']!=r['ingress']+r['service']:
            raise ValueError('pipeline accounting')
        if r['frame']<588:
            ref=reference[r['frame']]
            groups=int(ref['packed_issue_groups' if r['sparse'] else 'dense_issue_groups'])
            if r['sparse'] and int(ref['whole_vector_zero']): groups-=1
            if r['groups']!=groups: raise ValueError('packet count differs from prepared fixture')
    totals=[]
    for s in range(2):
        for p in range(2):
            selected=[r for r in rows if r['scenario']==s and r['sparse']==p and r['frame']<588]
            totals.append(dict(scenario=s,sparse=p,frames=len(selected),**{
                k:sum(r[k] for r in selected) for k in ('total','ingress','service','scans','groups','delimiters','group_stalls','result_stalls','memory','accum','checked')}))
    if any(r['group_stalls'] or r['result_stalls'] for r in totals if r['scenario']==0):
        raise ValueError('unexpected unstalled service')
    if any(r['group_stalls']==0 or r['result_stalls']==0 for r in totals if r['scenario']==1):
        raise ValueError('backpressure not exercised')
    return dict(scope=SCOPE,frames=rows,real_frame_totals=totals,current_checks=1065600,
                preload_writes=16800,boundary_frame_runs=16)


def worker():
    root=Path.cwd(); scratch=Path('/tmp/snn_'+NAME); scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scope=SCOPE,scratch=str(scratch))
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();begin=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,pid=process.pid,timed_out=True,wall_seconds=time.monotonic()-begin)
                save();raise
        state[stage]=dict(command=command,exit=code,wall_seconds=time.monotonic()-begin);save()
        if code: raise RuntimeError(stage+' exit '+str(code))
    try:
        execute('build',[VERILATOR,'--binary','--timing','--assert','-j','2','-Wall','-Wno-fatal',
            '--top-module','tb_source_scan','--Mdir',str(scratch/'obj'),
            *['rtl/'+p for p in SOURCES],'source_scan.sv','scan_consumer.sv','tb_source_scan.sv'],1800)
        build=(root/'build.log').read_text(errors='replace')
        state['warnings']={k:len(re.findall(r'^%Warning-'+k+r':',build,re.M))
            for k in sorted(set(re.findall(r'^%Warning-(\w+):',build,re.M)))}
        if any(state['warnings'].get(k) for k in ('LATCH','MULTIDRIVEN','UNOPTFLAT')):
            raise ValueError('structural warning')
        execute('run',[str(scratch/'obj/Vtb_source_scan'),'+FIXTURE=fixture'],1800)
        checks=parse_log((root/'run.log').read_text(errors='replace'),root/'fixture')
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state.update(status='PASS_pre_LIF_literature_scan_RTL_not_author_implementation',
                     real_frame_totals=checks['real_frame_totals'],current_checks=checks['current_checks'])
    except Exception as error: state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT)); destination=REMOTE+'/'+NAME
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
                    if actual.read()!=path.read_bytes():raise ValueError('uploaded input differs: '+str(path))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_spiker_scan bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_spiker_scan',prepared_input_files_byte_equal=len(inputs))
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','worker'))
    args=parser.parse_args();{'prepare':prepare,'launch':launch,'worker':worker}[args.action]()
