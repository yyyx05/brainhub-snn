"""Serial paired synthesis of the actual producer-plus-core boundary."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
NAME = 'final_raw_source_synth_r1'
SCRATCH = '/tmp/snn_' + NAME
REPORT = Path('server_results/final_composition_synth')
SHELL = 'server/run_unified_h40_final_composition_v1_synth.sh'
TOP = 'unified_h40_raw_source_top_v1'


def now():
    return dt.datetime.now().astimezone().isoformat()


def active_eda():
    active = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            name = (p/'comm').read_text().strip()
            if name in ('vivado', 'verilator_bin', 'xelab', 'xsim', 'Vscgraph') or name.startswith('Vtb_'):
                active.append([p.name, name])
        except (OSError, ProcessLookupError):
            pass
    return active


def classify(directory, exit_code):
    report = directory / REPORT
    text = (report/'vivado.log').read_text(errors='replace')
    errors = re.findall(r'^ERROR:.*', text, re.M)
    critical = re.findall(r'^CRITICAL WARNING:.*', text, re.M)
    path = report/'vivado_status.txt'
    tcl = dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line) if path.exists() else {}
    expected = dict(status='pass', top=TOP, part='xc7vx690tffg1761-3', period_ns='15.000',
                    source_count='112', enable_online_learning='1', m2_cache_depth='512', black_boxes='0')
    checkpoint = report/'post_synth.dcp'
    crc, members = None, None
    if checkpoint.exists():
        with zipfile.ZipFile(checkpoint) as archive:
            crc, members = archive.testzip() is None, len(archive.infolist())
    passed = exit_code == 0 and not errors and not critical and crc is True and all(tcl.get(k) == v for k, v in expected.items())
    return dict(status='PASS_synthesis_not_routed_STA' if passed else 'FAIL_synthesis_or_checkpoint',
                exit=exit_code, tcl=tcl, errors=errors, critical_warnings=critical,
                ordinary_warnings=len(re.findall(r'^WARNING:', text, re.M)),
                checkpoint=dict(bytes=checkpoint.stat().st_size if checkpoint.exists() else None,
                                crc_pass=crc, members=members))


def archive_files(root, scratch):
    files = [p for p in scratch.rglob('*') if p.is_file() and '.Xil' not in p.relative_to(scratch).parts]
    needed = sum(p.stat().st_size for p in files if not (root/p.relative_to(scratch)).exists())
    if shutil.disk_usage(root).free < needed + 128*1024**2:
        raise RuntimeError('archive reserve insufficient; scratch retained')
    inventory = []
    for source in files:
        relative = source.relative_to(scratch)
        target = root/relative
        if source.is_symlink():
            raise ValueError('unexpected scratch symlink')
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError('scratch source changed: ' + str(relative))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size:
            raise ValueError('archive size mismatch')
        inventory.append(dict(relative=relative.as_posix(), bytes=source.stat().st_size))
    (root/'scratch_archive.json').write_text(json.dumps(inventory, indent=2)+'\n')


def worker():
    root, scratch = Path.cwd(), Path(SCRATCH)
    scratch.mkdir(exist_ok=False)
    state = dict(status='running', start=now(), scratch=SCRATCH, arms={})
    def save():
        (root/'status.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        for arm in ('baseline', 'candidate'):
            observation = dict(time=now(), active_eda=active_eda(),
                               data_free=shutil.disk_usage(root).free, tmp_free=shutil.disk_usage(scratch).free)
            if observation['active_eda'] or observation['data_free'] < 512*1024**2 or observation['tmp_free'] < 4*1024**3:
                raise RuntimeError('per-arm original synthesis reserve/idle check: ' + repr(observation))
            target = scratch/arm
            target.mkdir()
            for folder in ('rtl', 'vivado', 'server'):
                shutil.copytree(root/arm/folder, target/folder)
            env = os.environ.copy()
            env.update(TMPDIR=str(target), TMP=str(target), TEMP=str(target))
            command = ['bash', str(target/SHELL), str(target), '1']
            begin = time.monotonic()
            with (target/'worker_wrapper.log').open('x') as log:
                process = subprocess.Popen(command, cwd=target, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                state['status'] = arm + '_running'
                state['arms'][arm] = dict(pid=process.pid, exit=None, command=command, before=observation)
                save()
                try:
                    code = process.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    state['arms'][arm].update(timed_out=True, wall_seconds=time.monotonic()-begin)
                    state.update(status='TIMEOUT_PROCESS_RETAINED', end=now())
                    save()
                    return
            try:
                result = classify(target, code)
            except Exception as error:
                result = dict(status='FAIL_terminal_classification', exit=code,
                              classification_error=repr(error))
            state['arms'][arm].update(result, wall_seconds=time.monotonic()-begin, end=now())
            save()
            # Archive a natural terminal even when it failed; never start the next arm after failure.
            archive_files(root, scratch)
            if not state['arms'][arm]['status'].startswith('PASS'):
                raise RuntimeError(arm + ' synthesis not accepted')
        state['status'] = 'PASS_paired_raw_source_synthesis_not_routed_STA'
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state.update(end=now(), scratch_preserved=True, data_free_after=shutil.disk_usage(root).free,
                 tmp_free_after=shutil.disk_usage('/tmp').free)
    save()


def launch():
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    root = Path(local_path(ROOT))
    prepared = root/'verification/experiments/final_raw_source_composition_r1'
    prep = json.loads((prepared/'preparation.json').read_text())
    if prep['status'] != 'PREPARED_NOT_ELABORATED_OR_SYNTHESIZED':
        raise ValueError('unexpected preparation')
    unit = root/RESULT/'raw_source_binding_local_r1/attempt_001/status.json'
    accepted = json.loads(unit.read_text())
    if not accepted['status'].startswith('PASS'):
        raise ValueError('binding unit is not accepted')
    # The unit runner archives this exact binding alongside its tested producers.
    tested = root/RESULT/'raw_source_binding_local_r1/attempt_001/raw_source_binding.sv'
    for arm in ('baseline', 'candidate'):
        if (prepared/arm/'rtl/top/raw_source_binding.sv').read_bytes() != tested.read_bytes():
            raise ValueError('prepared/tested binding differs')
    shutil.copy2(__file__, prepared/'worker.py')
    shutil.copy2(unit, prepared/'binding_acceptance.json')
    files = sorted(p for p in prepared.rglob('*') if p.is_file())
    payload = sum(p.stat().st_size for p in files)
    destination = REMOTE+'/'+NAME
    probe = '''import datetime,json,pathlib,shutil
base=pathlib.Path(BASE);target=base/NAME
if CREATE and (target.exists() or pathlib.Path(SCRATCH).exists()):raise FileExistsError('isolated run exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  n=(p/'comm').read_text().strip()
  if n in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or n.startswith('Vtb_'):active.append([p.name,n])
 except (OSError,ProcessLookupError):pass
data=shutil.disk_usage(base).free;tmp=shutil.disk_usage('/tmp').free
if active or data < RESERVE or tmp < 4*1024**3:raise RuntimeError('idle/synthesis archive reserve: '+repr((active,data,tmp)))
if CREATE:target.mkdir()
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=tmp)))
'''
    client = connect()
    try:
        def observe(create, reserve):
            source = 'BASE='+repr(REMOTE)+'\nNAME='+repr(NAME)+'\nSCRATCH='+repr(SCRATCH)+'\nCREATE='+repr(create)+'\nRESERVE='+repr(reserve)+'\n'+probe
            return json.loads(run(client, 'python3 -c '+shlex.quote(source)))
        before = observe(True, payload+512*1024**2)
        with client.open_sftp() as sftp:
            for path in files:
                target = destination+'/'+path.relative_to(prepared).as_posix()
                upload(sftp, str(path), target)
                with sftp.open(target, 'rb') as handle:
                    if handle.read() != path.read_bytes():
                        raise ValueError('uploaded input differs: '+str(path))
        at_start = observe(False, 512*1024**2)
        command = 'cd '+shlex.quote(destination)+' && exec python3 worker.py worker'
        run(client, 'screen -dmS snn_'+NAME+' bash -lc '+shlex.quote(command))
        record = dict(launched=True, time=now(), before_copy=before, before_start=at_start,
                      prepared_files=len(files), prepared_bytes=payload, all_uploaded_bytes_equal=True,
                      remote=destination, scratch=SCRATCH, screen='snn_'+NAME,
                      boundary='Two serial Vivado synthesis processes, not concurrent. Actual static producer plus complete core. No route/power/bitstream or physical-frequency claim. Original per-arm 512MiB public/4GiB scratch and actual-archive+128MiB reserve retained.')
        output = root/RESULT/NAME
        output.mkdir(parents=True, exist_ok=True)
        (output/'launch_observation.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('launch', 'worker'))
    action = p.parse_args().action
    if action == 'launch':
        launch()
    else:
        worker()
