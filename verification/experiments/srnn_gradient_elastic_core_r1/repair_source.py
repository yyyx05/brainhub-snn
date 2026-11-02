"""Create an isolated TB-only revision after the archived R1 pre-run failure."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.dont_write_bytecode=True
from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path,upload

OLD='srnn_source_core_r1'
NAME='srnn_source_core_r2'
TB='tb_srnn_profile_measurement.sv'


def prepare():
    root=Path(local_path(ROOT));prior=root/'verification/experiments'/OLD
    out=root/'verification/experiments'/NAME
    shutil.copytree(prior,out,ignore=shutil.ignore_patterns('__pycache__'))
    text=(out/TB).read_text()
    names=['valid','recurrent','source_valid']+['weight_addr'+str(i) for i in range(4)]
    for name in names:
        text,n=re.subn(r"^(    logic[^\n]*\bi_fc_srnn_group_"+name+r") = '0;",r'\1;',text,flags=re.M)
        if n!=1:raise ValueError('missing declaration '+name)
    (out/TB).write_text(text,newline='\n')
    worker=(out/'launch_worker.py').read_text()
    if worker.count("NAME = '"+OLD+"'")!=1:raise ValueError('worker name')
    worker=worker.replace("NAME = '"+OLD+"'","NAME = '"+NAME+"'")
    worker=worker.replace("Path('verification/experiments/"+OLD+"')","Path('verification/experiments/"+NAME+"')")
    (out/'launch_worker.py').write_text(worker,newline='\n')
    record=json.loads((out/'preparation.json').read_text())
    record.update(revision_of=OLD,tb_fix='Remove exactly seven declaration initializers on always_comb mux outputs; manual drivers retain initialization.',
                  original_failure_preserved=True,core_rtl_changed_from_r1=False)
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(__file__,out/'repair_source.py')
    files=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    changed=[p.relative_to(out).as_posix() for p in files if not (prior/p.relative_to(out)).is_file() or p.read_bytes()!=(prior/p.relative_to(out)).read_bytes()]
    if set(changed)!={TB,'launch_worker.py','preparation.json','repair_source.py'}:raise ValueError(changed)
    print(json.dumps(dict(prepared=str(out),files=len(files),bytes=sum(p.stat().st_size for p in files),changed=changed)))


def launch():
    root=Path(local_path(ROOT));prepared=root/'verification/experiments'/NAME;prior=root/'verification/experiments'/OLD
    evidence=json.loads((root/RESULT/OLD/'failure_archive_audit.json').read_text())
    if evidence['status']!='ARCHIVED_NOT_PASSED_pre_simulation_structural_failure':raise ValueError('R1 archive incomplete')
    files=sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    size=sum(p.stat().st_size for p in files)
    unchanged=[p.relative_to(prepared).as_posix() for p in files if (prior/p.relative_to(prepared)).is_file() and p.read_bytes()==(prior/p.relative_to(prepared)).read_bytes()]
    changed=[p for p in files if p.relative_to(prepared).as_posix() not in unchanged]
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
    destination=REMOTE+'/'+NAME;client=connect()
    try:
        before=json.loads(run(client,'python3 -c '+shlex.quote('REQUIRED_DATA='+str(size+256*1024**2)+'\n'+probe)))
        # Copy only archived, byte-verified static inputs. Do not copy old logs,
        # status, executable caches, or touch the failed experiment.
        copy='''from pathlib import Path
import shutil,filecmp,json
old=Path(OLD);new=Path(NEW)
assert not new.exists() and not Path(SCRATCH).exists()
new.mkdir()
for rel in FILES:
 src=old/rel;dst=new/rel
 assert src.is_file()
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(src,dst)
 assert filecmp.cmp(src,dst,shallow=False),rel
print(json.dumps(dict(copied_byte_equal=len(FILES))))
'''
        copied=json.loads(run(client,'python3 -c '+shlex.quote('OLD='+repr(REMOTE+'/'+OLD)+'\nNEW='+repr(destination)+'\nSCRATCH='+repr('/tmp/snn_'+NAME)+'\nFILES='+repr(unchanged)+'\n'+copy)))
        with client.open_sftp() as sftp:
            for path in changed:
                target=destination+'/'+path.relative_to(prepared).as_posix();upload(sftp,str(path),target)
                with sftp.open(target,'rb') as handle:
                    if handle.read()!=path.read_bytes():raise ValueError('changed input bytes '+str(path))
        at_start=json.loads(run(client,'python3 -c '+shlex.quote('REQUIRED_DATA='+str(256*1024**2)+'\n'+probe)))
        command='cd '+shlex.quote(destination)+' && exec python3 launch_worker.py worker'
        run(client,'screen -dmS snn_srnn_source_core_r2 bash -lc '+shlex.quote(command))
        record=dict(launched=True,remote=destination,screen='snn_srnn_source_core_r2',before_copy=before,before_start=at_start,
            prepared_input_files=len(files),prepared_input_bytes=size,copied_from_archived_r1_byte_equal=copied['copied_byte_equal'],
            uploaded_changed_files_byte_equal=len(changed),copied_verification='Local R2/R1 bytes, archived R1 inputs, remote copy shallow=False; no hashes.')
        out=root/RESULT/NAME;out.mkdir(parents=True,exist_ok=True)
        (out/'launch_observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch'))
    args=parser.parse_args();{'prepare':prepare,'launch':launch}[args.action]()
