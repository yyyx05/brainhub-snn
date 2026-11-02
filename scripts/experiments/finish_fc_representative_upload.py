"""Finish this pre-launch upload with pipelined reads; never alter an active run."""
import datetime as dt
import json
from pathlib import Path
import shlex
import time

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
from run_fc_representative_inputs import NAME, EXPERIMENT


def main():
    prepared = Path(local_path(ROOT/EXPERIMENT))
    output = Path(local_path(ROOT/RESULT/NAME))
    destination = REMOTE+'/'+NAME
    inputs = sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    total = sum(p.stat().st_size for p in inputs)
    report = dict(start=dt.datetime.now().astimezone().isoformat(),
        reason='Original local uploader PID7352 ended with exit1 before RTL launch; sequential unprefetched readback was slow. Only transport resumed; prepared payload and worker unchanged.',
        existing_equal=[],missing_uploaded=[],partial_completed=[],files=len(inputs),bytes=total)
    client = connect()
    started = time.monotonic()
    try:
        probe = '''import pathlib,json,shutil,datetime
p=pathlib.Path(DEST)
if not (p/'preflight.json').is_file():raise ValueError('original staging preflight missing')
if (p/'status.json').exists() or pathlib.Path(SCRATCH).exists():raise ValueError('worker already started; refuse staging writes or relaunch')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise ValueError('active EDA: '+repr(active))
r=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)
if r['data_free'] < REQUIRED or r['tmp_free'] < 1024**3:raise ValueError('insufficient remaining capacity')
print(json.dumps(r))
'''
        def observe(required):
            return json.loads(run(client,'python3 -c '+shlex.quote('DEST='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nREQUIRED='+str(required)+'\n'+probe)))
        report['before_transfer'] = observe(total+128*1024**2)
        with client.open_sftp() as sftp:
            for i,path in enumerate(inputs):
                rel = path.relative_to(prepared).as_posix()
                remote = destination+'/'+rel
                data = path.read_bytes()
                try: size = sftp.stat(remote).st_size
                except FileNotFoundError: size = None
                if size is not None:
                    with sftp.open(remote,'rb') as stream:
                        stream.prefetch(size, max_concurrent_requests=16)
                        existing = stream.read()
                    if len(existing) > len(data) or existing != data[:len(existing)]:
                        raise ValueError('staged bytes not an unchanged prefix: '+rel)
                    if len(existing) == len(data):
                        report['existing_equal'].append(rel)
                        continue
                    with sftp.open(remote,'ab') as stream:
                        stream.set_pipelined(True)
                        stream.write(data[len(existing):])
                    report['partial_completed'].append(dict(path=rel,preserved_prefix_bytes=len(existing),final_bytes=len(data)))
                else:
                    upload(sftp,str(path),remote)
                    report['missing_uploaded'].append(rel)
                with sftp.open(remote,'rb') as stream:
                    stream.prefetch(len(data), max_concurrent_requests=16)
                    if stream.read() != data: raise ValueError('uploaded bytes differ: '+rel)
                if i%20 == 0: print(json.dumps(dict(checked=i+1,total=len(inputs))),flush=True)
        report['before_launch'] = observe(128*1024**2)
        command = 'cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_fc_representative bash -lc '+shlex.quote(command))
        report.update(launched=True,remote=destination,screen='snn_fc_representative',
                      all_prepared_files_byte_equal=True,end=dt.datetime.now().astimezone().isoformat(),
                      wall_seconds=time.monotonic()-started)
        (output/'transfer_resume.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'launch_observation.json').write_text(json.dumps(dict(**report['before_launch'],launched=True,
            remote=destination,screen='snn_fc_representative',prepared_input_files_byte_equal=len(inputs),input_bytes=total),indent=2)+'\n')
        print(json.dumps(dict(launched=True,files=len(inputs),bytes=total,wall_seconds=report['wall_seconds'],
            existing_equal=len(report['existing_equal']),missing_uploaded=len(report['missing_uploaded']),
            partial_completed=report['partial_completed'],before_launch=report['before_launch'])),flush=True)
    finally: client.close()


if __name__ == '__main__': main()
