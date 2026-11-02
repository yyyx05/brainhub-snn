"""Synthesize payload experiment A with the archived, unchanged turnover recipe."""
import argparse
import json
from pathlib import Path
import shlex
import shutil

NAME='scnn_cfg_payload_core_synth_r1'
SCRATCH='/tmp/snn_'+NAME
FUNCTIONAL='scnn_cfg_payload_unit_r4'
RECIPE='scnn_turnover_core_synth_r1'


def worker():
    import run_scnn_turnover_synthesis as original
    original.NAME=NAME
    original.SCRATCH=SCRATCH
    original.worker()


def launch():
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,upload,local_path
    from run_scnn_turnover_synthesis import TCL,XDC,SHELL
    local=Path(local_path(ROOT/RESULT/FUNCTIONAL))
    audit=json.loads((local/'terminal_audit.json').read_text())
    if audit['status']!='PASS_cfg_payload_real_core_archive_audit':raise ValueError('functional evidence incomplete')
    reference=Path(local_path(ROOT/RESULT/RECIPE))
    prepared=Path(local_path(ROOT/'verification/experiments/scnn_control_payload/core_synthesis_r1'))
    prepared.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),prepared/'launch_worker.py')
    shutil.copy2(ROOT/'scripts/experiments/run_scnn_turnover_synthesis.py',prepared/'run_scnn_turnover_synthesis.py')
    dest=REMOTE+'/'+NAME
    probe=f'''
import pathlib,shutil,json,datetime
base=pathlib.Path({REMOTE!r});target=base/{NAME!r};scratch=pathlib.Path({SCRATCH!r})
if target.exists() or scratch.exists():raise FileExistsError('unique synthesis destination exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA '+repr(active))
data=shutil.disk_usage(base).free;temp=shutil.disk_usage('/tmp').free
if data<512*1024**2 or temp<4*1024**3:raise RuntimeError('insufficient observed scratch/archive headroom')
target.mkdir();shutil.copytree(base/{FUNCTIONAL!r}/'rtl',target/'rtl')
for rel in {[TCL,XDC,SHELL]!r}:
 p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/{RECIPE!r}/rel,p)
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=temp,
 source={FUNCTIONAL!r},recipe={RECIPE!r},scratch=str(scratch),
 boundary='Experiment A only: thirteen payload muxes, unchanged arbitration and numeric RTL. Same full-core 108SV+ROM/512/no-model-key, 8 threads, 15ns, protection, original synth recipe. No route, bitstream, power or production adoption.')
for n in ('cpu.max','memory.max'):
 p=pathlib.Path('/sys/fs/cgroup')/n;record[n]=p.read_text().strip() if p.exists() else None
(target/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
    client=connect()
    try:
        record=json.loads(run(client,'python3 -c '+shlex.quote(probe)))
        compared=[]
        with client.open_sftp() as sftp:
            paths=[p.relative_to(local).as_posix() for p in (local/'rtl').rglob('*') if p.is_file()]
            if len(paths)!=109:raise ValueError('candidate source tree incomplete')
            for rel in paths+[TCL,XDC,SHELL]:
                expected=local/rel if rel in paths else reference/rel
                with sftp.open(dest+'/'+rel,'rb') as f:
                    if f.read()!=expected.read_bytes():raise ValueError('actual input mismatch '+rel)
                compared.append(rel)
            for filename,remote in [('launch_worker.py','worker.py'),('run_scnn_turnover_synthesis.py','run_scnn_turnover_synthesis.py')]:
                path=prepared/filename;upload(sftp,str(path),dest+'/'+remote)
                with sftp.open(dest+'/'+remote,'rb') as f:
                    if f.read()!=path.read_bytes():raise ValueError('worker mismatch '+remote)
        command='cd '+shlex.quote(dest)+' && exec python3 worker.py worker'
        run(client,'screen -dmS snn_cfg_payload_synth bash -lc '+shlex.quote(command))
        record.update(remote=dest,launched=True,screen='snn_cfg_payload_synth',actual_input_byte_equal=compared,actual_workers_byte_equal=True)
        out=Path(local_path(ROOT/RESULT/NAME));out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='actual_input_byte_equal'},indent=2))
        print('byte_equal_inputs',len(compared))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('launch','worker'))
    if parser.parse_args().action=='launch':launch()
    else:worker()
