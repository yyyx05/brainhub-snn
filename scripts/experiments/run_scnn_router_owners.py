"""Prepare/run a focused final-router forward-path check for all three owners."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME='scnn_router_owners_r1'
EXPERIMENT=Path('verification/experiments/scnn_forward_turnover/router_owners/candidate_r1')
ROUTER='rtl/learning/unified_tp_three_client_parameter_backend_router.sv'
TB='tb_scnn_router_owners.sv'
VERILATOR='/root/autodl-tmp/verilator_5_048_install/bin/verilator'


def parse_log(log):
    if re.search(r'%Error|%Fatal|Assertion failed',log):raise ValueError('RTL diagnostic')
    def rows(marker):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)} for line in log.splitlines() if line.startswith(marker+' ')]
    final=rows('SCNN_ROUTER_OWNERS_PASS')
    if final!=[dict(owners=3,services=2,latency_profiles=3,cases=18,reads=1152,responses=1152,recovery_cases=12)]:
        raise ValueError('missing/duplicate completion')
    cases=rows('SCNN_ROUTER_OWNER_CASE');recoveries=rows('SCNN_ROUTER_OWNER_RECOVERY')
    markers=[line.split()[0] for line in log.splitlines() if line.startswith('SCNN_ROUTER_OWNER')]
    if markers!=(['SCNN_ROUTER_OWNER_CASE']*6+['SCNN_ROUTER_OWNER_RECOVERY'])*3+['SCNN_ROUTER_OWNERS_PASS']:
        raise ValueError('completion/recovery ordering')
    expected=[(owner,service,latency) for owner in (1,2,3) for service in (0,1) for latency in (1,4,12)]
    if [(r['owner'],r['service'],r['latency']) for r in cases]!=expected:raise ValueError('case order/coverage')
    for r in cases:
        if (r['reads'],r['responses'],r['error_payloads'],r['address_bits'],r['tag_bits'])!=(64,64,4,24,16):raise ValueError('incomplete data path')
        if any(v<0 for v in r.values()) or r['replacements']>=64 or bool(r['replacements'])!=bool(r['service']):raise ValueError('replacement/service mismatch')
        if r['request_stalls']==0 or r['response_stalls']==0:raise ValueError('backpressure not exercised')
    if recoveries!=[dict(owner=o,owner_switch=1,inactive_owner=1,unsolicited=1,nonowner=1,quiescent_clear=1,reissue=1) for o in (1,2,3)]:
        raise ValueError('missing owner/error/clear/reissue checks')
    return dict(status='PASS_router_three_owners_forward_path_not_final_top_recovery',cases=cases,recoveries=recoveries,
        boundary='Exact final candidate router only. Owner1/2/3 are client encodings, not full FC/SRNN/SCNN applications. Forward24/tag16/code16, independent expected payload and response-error checks, native tag rollover, same-slot services and variable latency/backpressure. Learning inputs idle. Error drain/clear/reissue is module protocol recovery, not arbitrary final-top abort, learning rollback, new throughput or STA.')


def prepare():
    from run_srnn_resident_remote import ROOT,RESULT,local_path
    source=Path(local_path(ROOT/RESULT/'scnn_turnover_frame_r2/candidate'/ROUTER))
    out=Path(local_path(ROOT/EXPERIMENT));out.mkdir(parents=True,exist_ok=False)
    (out/'rtl/learning').mkdir(parents=True)
    shutil.copy2(source,out/ROUTER)
    shutil.copy2(Path(local_path(ROOT/'verification/experiments/scnn_forward_turnover/router_owners'/TB)),out/TB)
    record=dict(status='PREPARED_NOT_SIMULATED',source='scnn_turnover_frame_r2/candidate/'+ROUTER,
        router_byte_equal=True,cases=18,owners=[1,2,3],services=['legacy_single_slot','replaceable_single_slot'],
        latencies=[1,4,12],address_bits=24,tag_bits=16,code_bits=16,
        boundary='Focused changed forward router path only. No source transformation, no learning-path or whole-top PASS. Do not launch concurrently with active Vivado.')
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'launch_worker.py')
    print(str(out).removeprefix('\\\\?\\'))


def worker():
    root=Path.cwd();scratch=Path('/tmp/snn_'+NAME);scratch.mkdir(exist_ok=False)
    state=dict(status='running',start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch))
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(stage,command,timeout):
        state['status']=stage+'_running';save();start=time.monotonic()
        with (root/(stage+'.log')).open('w') as stream:
            try:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage]=dict(command=command,exit=None,wall_seconds=time.monotonic()-start,timeout_seconds=timeout);save();raise
        state[stage]=dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-start);save()
        if result.returncode:raise RuntimeError(stage+' exit '+str(result.returncode))
    try:
        execute('build',[VERILATOR,'--binary','--timing','--assert','-j','2','-Wall','-Wno-fatal','-Wno-PINCONNECTEMPTY',
            '--top-module','tb_scnn_router_owners','--Mdir',str(scratch/'obj'),ROUTER,TB],180)
        log=(root/'build.log').read_text(errors='replace')
        state['ordinary_warnings']=len(re.findall(r'^%Warning-',log,re.M))
        state['structural_warnings']=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*',log,re.M)
        if state['structural_warnings'] or '%Error' in log:raise ValueError('build diagnostic')
        execute('run',[str(scratch/'obj/Vtb_scnn_router_owners')],30)
        checks=parse_log((root/'run.log').read_text(errors='replace'))
        (root/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        state['status']=checks['status']
    except Exception as error:state.update(status='FAIL',error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(),scratch_preserved=True);save()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    prepared=Path(local_path(ROOT/EXPERIMENT));destination=REMOTE+'/'+NAME
    if not (prepared/'preparation.json').is_file():raise ValueError('prepare first')
    client=connect()
    try:
        code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
if p.exists() or pathlib.Path(SCRATCH).exists():raise FileExistsError('unique destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA; do not overlap: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p.parent).free,tmp_free=shutil.disk_usage('/tmp').free)
if record['data_free']<64*1024**2 or record['tmp_free']<128*1024**2:raise RuntimeError('insufficient unit space')
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
        record=json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\n'+code)))
        with client.open_sftp() as s:
            for rel in (ROUTER,TB,'preparation.json','launch_worker.py'):
                upload(s,str(prepared/rel),destination+'/'+rel)
                if s.open(destination+'/'+rel,'rb').read()!=(prepared/rel).read_bytes():raise ValueError('actual byte mismatch '+rel)
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_router_owners bash -lc '+shlex.quote(command))
        record.update(launched=True,remote=destination,screen='snn_router_owners',actual_input_bytes_equal=True)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','worker'))
    action=parser.parse_args().action
    if action=='prepare':prepare()
    elif action=='launch':launch()
    else:worker()
