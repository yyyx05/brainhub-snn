#!/usr/bin/env python3
"""Prepare a one-wire candidate and run it against an unweakened resident baseline."""
import difflib
import json
from pathlib import Path
import shlex

from run_srnn_resident_remote import ROOT, REMOTE, RESULT, connect, local_path, run, upload

MODULE = 'rtl/learning/unified_srnn_m2_timestep_train_infer_coordinator.sv'
OLD = 'assign o_h40_prepare_version = o_forward_adapt_version;'
NEW = 'assign o_h40_prepare_version = o_forward_model_version;'
NAME = 'srnn_equal_storage_r1'


def transform(text):
    if text.count(OLD) != 1:
        raise ValueError('cache-version assignment not unique')
    return text.replace(OLD, NEW)


def main():
    source = Path(local_path(ROOT / RESULT / 'scnn_issue_banked_synth_r1' / MODULE))
    original = source.read_text(encoding='utf-8')
    changed = transform(original)
    folder = ROOT / 'verification/experiments/srnn_equal_storage/candidate_r1'
    target = Path(local_path(folder / MODULE))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(changed, encoding='utf-8')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile='resident_baseline/' + MODULE, tofile='model_version_candidate/' + MODULE))
    Path(local_path(folder / 'cache_key.diff')).write_text(diff, encoding='utf-8')
    bootstrap = '''import json,pathlib,shutil
root=pathlib.Path(ROOT); old=root/'srnn_specialized_r1'; final=root/'scnn_issue_banked_synth_r1'; target=root/NAME
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator') or comm.startswith('Vtb_'):active.append((p.name,comm))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError(active)
free=shutil.disk_usage(root).free
if free<1000000000:raise RuntimeError('insufficient free space '+str(free))
target.mkdir(exist_ok=False)
for policy in ('baseline','candidate'):
 t=target/policy;t.mkdir()
 for name in ('scripts','tb','results','build'):
  shutil.copytree(old/name,t/name,ignore=shutil.ignore_patterns('__pycache__','obj','elab_obj','.Xil'))
 for name in ('tb_srnn_profile_measurement.sv','srnn_profile_measurement_generated_body.svh','input_commands.json'):
  shutil.copy2(old/name,t/name)
 shutil.copy2(old/'status.json',t/'accepted_profile_status.json')
 shutil.copytree(final/'rtl',t/'rtl')
 rtls=sorted(p.relative_to(t).as_posix() for p in (t/'rtl').rglob('*.sv'))
 if len(rtls)!=107:raise ValueError('unexpected RTL count')
 sources=rtls+['tb_srnn_profile_measurement.sv','tb/monitors/supplement_h40_activity_monitor.sv']
 (t/'sources.f').write_text(''.join('"'+p+'"\\n' for p in sources))
 (t/'comparison_input.json').write_text(json.dumps(dict(policy=policy,rtl_source=str(final),input_source=str(old),free_bytes_before=free,scope='Identical dual-context stores and frozen WH1 residency. Baseline uses existing adapt-version full-Wout refresh; candidate keys only numeric cache by model version. No model, learning, service or memory changes.'),indent=2)+'\\n')
print(json.dumps(dict(free_bytes_before=free,root=str(target))))
'''
    client = connect()
    try:
        print(run(client, 'python3 -c ' + shlex.quote('ROOT=' + repr(REMOTE) + '\nNAME=' + repr(NAME) + '\n' + bootstrap)))
        with client.open_sftp() as sftp:
            for policy in ('baseline', 'candidate'):
                base = REMOTE + '/' + NAME + '/' + policy
                upload(sftp, ROOT / 'scripts/experiments/srnn_equal_storage_worker.py', base + '/scripts/experiments/srnn_equal_storage_worker.py')
            upload(sftp, target, REMOTE + '/' + NAME + '/candidate/' + MODULE)
            upload(sftp, Path(local_path(folder / 'cache_key.diff')), REMOTE + '/' + NAME + '/cache_key.diff')
        shell = 'cd ' + shlex.quote(REMOTE + '/' + NAME) + ' && exec python3 baseline/scripts/experiments/srnn_equal_storage_worker.py pair'
        run(client, 'screen -dmS snn_srnn_equal_storage bash -lc ' + shlex.quote(shell))
        print(json.dumps(dict(launched=True, root=REMOTE + '/' + NAME)))
    finally:
        client.close()


if __name__ == '__main__':
    main()
