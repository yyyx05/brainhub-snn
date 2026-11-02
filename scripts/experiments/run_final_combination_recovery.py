#!/usr/bin/env python3
"""Reuse the legal early-abort test for the exact final physical RTL candidate."""
import json
import shlex

from run_srnn_resident_remote import REMOTE, connect, run


def main():
    name = 'final_combination_recovery_r1'
    bootstrap = '''import json,pathlib,shutil
root=pathlib.Path(ROOT)
old=root/'abort_before_learning_r1'
candidate=root/'scnn_issue_banked_synth_r1'
target=root/NAME
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator') or comm.startswith('Vtb_'):active.append((p.name,comm))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError(active)
free=shutil.disk_usage(root).free
if free<1000000000:raise RuntimeError('insufficient free space: '+str(free))
previous=json.loads((old/'status.json').read_text())
if previous['status']!='PASS_final_top_abort_before_learning_retry_first_step_pair':raise ValueError('original recovery not accepted')
target.mkdir(exist_ok=False)
# Keep the original test, models, reference and build recipe, not its RTL or outputs.
for p in old.rglob('*'):
 rel=p.relative_to(old)
 if not p.is_file() or any(x in ('obj','elab_obj','__pycache__','.Xil') for x in rel.parts):continue
 if rel.parts[0]=='rtl' or (p.suffix=='.sv' and 'verification' in rel.parts):continue
 if rel.as_posix() in ('status.json','build.log','run_off.log','run_on.log','sources.f'):continue
 dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(old/'status.json',target/'accepted_original_recovery_status.json')
shutil.copytree(candidate/'rtl',target/'rtl')
rtls=sorted((target/'rtl').rglob('*.sv'))
if len(rtls)!=107:raise ValueError('unexpected final RTL count')
for p in rtls:
 if p.read_bytes()!=(candidate/p.relative_to(target)).read_bytes():raise ValueError(p)
tb=[]
for line in (old/'sources.f').read_text().splitlines():
 rel=pathlib.PurePosixPath(line.strip().strip('"'))
 if rel.as_posix()=='tb_srnn_profile_measurement.sv' or rel.parts[0]=='tb':
  if (target/rel).read_bytes()!=(old/rel).read_bytes():raise ValueError(rel)
  tb.append(rel.as_posix())
if len(tb)!=2:raise ValueError('unexpected testbench source list')
(target/'sources.f').write_text(''.join('"'+p+'"\\n' for p in [p.relative_to(target).as_posix() for p in rtls]+tb))
record=dict(rtl_source=str(candidate/'rtl'),rtl_files=107,same_bytes=True,original_test=str(old),
 free_bytes_before=free,scope='Exact final RTL; unchanged legal START_LEARNING abort and retry of the first native sample0 step, off/on. Not arbitrary abort, SCNN rollback, full sequence or throughput.')
(target/'candidate_source.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
    code = 'ROOT=' + repr(REMOTE) + '\nNAME=' + repr(name) + '\n' + bootstrap
    client = connect()
    try:
        print(run(client, 'python3 -c ' + shlex.quote(code)))
        shell = 'cd ' + shlex.quote(REMOTE + '/' + name) + ' && exec python3 scripts/experiments/srnn_resident_server_worker.py recovery'
        screen = 'snn_resident_' + name
        run(client, 'screen -dmS ' + shlex.quote(screen) + ' bash -lc ' + shlex.quote(shell))
        print(json.dumps(dict(launched=True, remote=REMOTE + '/' + name, screen=screen)))
    finally:
        client.close()


if __name__ == '__main__':
    main()
