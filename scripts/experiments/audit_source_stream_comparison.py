"""Archive the completed RTL comparison and retain its mapping failure."""
import csv
import importlib.util
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode=True
from run_srnn_resident_remote import ROOT,RESULT,REMOTE,connect,run,local_path
from run_source_stream_comparison import NAME,EXPERIMENT


def inventory(client,name,scratch):
    code='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);scratch=pathlib.Path(SCRATCH);skip={'obj','elab_obj','__pycache__','.Xil'}
def files(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files(p),scratch_noncache=files(scratch),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    return json.loads(run(client,'python3 -c '+shlex.quote('ROOT='+repr(REMOTE+'/'+name)+'\nSCRATCH='+repr(scratch)+'\n'+code)))


def main():
    root=Path(local_path(ROOT));archive=root/RESULT/NAME;prepared=root/EXPERIMENT
    if (archive/'terminal_audit.json').exists():raise FileExistsError('already audited')
    state=json.loads((archive/'status.json').read_text())
    assert state['end'] and state['status']=='FAIL'
    assert state['build']['exit']==state['run']['exit']==0 and state['mapping']['exit']==1
    client=connect()
    try: observation=inventory(client,NAME,state['scratch'])
    finally: client.close()
    assert not observation['active_eda']
    files=observation.pop('files')
    assert not list(archive.rglob('*.part'))
    assert all((archive/p).is_file() and (archive/p).stat().st_size==n for p,n in files.items())
    assert all((archive/'scratch_originals'/p).stat().st_size==n for p,n in observation['scratch_noncache'].items())
    inputs=[p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.read_bytes()==(archive/p.relative_to(prepared)).read_bytes() for p in inputs)
    original=root/'verification/experiments/nearest_neighbor/scan_rtl_r1'
    unchanged=['source_scan.sv','scan_consumer.sv']
    unchanged += [p.relative_to(original).as_posix() for folder in ('rtl','fixture') for p in (original/folder).rglob('*') if p.is_file()]
    assert all((archive/p).read_bytes()==(original/p).read_bytes() for p in unchanged)
    assert (archive/'launch_worker.py').read_bytes()==(root/'scripts/experiments/run_source_stream_comparison.py').read_bytes()
    spec=importlib.util.spec_from_file_location('actual_stream_worker',archive/'launch_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    checks=worker.parse_log((archive/'run.log').read_text(),archive/'fixture')
    assert checks==json.loads((archive/'checks.json').read_text())
    assert checks['real_frame_totals']==state['real_frame_totals']
    diag={s:worker.diagnostics((archive/(s+'.log')).read_text()) for s in ('build','run','mapping')}
    assert all(diag[s]==state[s]['diagnostics'] for s in diag)
    mapping=(archive/'mapping.log').read_text()
    assert "[Synth 8-6735] net type must be explicitly specified" in mapping
    with (archive/'frame_measurements.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks['frames'][0]));w.writeheader();w.writerows(checks['frames'])
    report=dict(status='PASS_RTL_archive_with_mapping_FAIL_preserved',accepted_rtl=True,accepted_mapping=False,
        start=state['start'],end=state['end'],raw_files=len(files),raw_bytes=sum(files.values()),all_sizes_match=True,no_part=True,
        prepared_files_byte_equal=len(inputs),prepared_bytes=sum(p.stat().st_size for p in inputs),
        original_consumer_scanner_fixture_files_byte_equal=len(unchanged),actual_worker_strict_reparse=True,
        current_checks=checks['current_checks'],real_frame_totals=checks['real_frame_totals'],
        stages={s:state[s] for s in diag},diagnostics=diag,remote_observation=observation,
        dcp_crc='not applicable: no DCP',production_changed=False,
        boundary='Original FAIL retained. RTL PASS is not synthesis success, author implementation or whole-network speedup. No cleanup.')
    (archive/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
