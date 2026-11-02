"""Implement the accepted producer-plus-core DCP pair with one shared recipe."""
import argparse
import datetime as dt
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile

sys.dont_write_bytecode = True
from run_final_raw_source_synthesis import active_eda, archive_files

NAME = 'final_raw_source_impl_r1'
SCRATCH = '/tmp/snn_' + NAME
EXPERIMENT = Path('verification/experiments/final_raw_source_implementation_r1')
SYNTHESIS = 'final_raw_source_synth_r1'
OUT = Path('server_results/final_composition_impl')
ORIGINAL = Path('vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl')
VIVADO = '/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado'
ARMS = ('baseline', 'candidate')
PUBLIC_RESERVE = 768 * 1024**2
SCRATCH_RESERVE = 6 * 1024**3


def now():
    return dt.datetime.now().astimezone().isoformat()


def implementation_tcl(original):
    from run_scnn_turnover_implementation import implementation_tcl as add_reopen
    replacements = {
        'set expected_top unified_h40_final_composition_top_v1':
            'set expected_top unified_h40_raw_source_top_v1',
        '    g_srnn_phase_bridge.u_srnn_phase_bridge\n':
            '    u_core/g_srnn_phase_bridge.u_srnn_phase_bridge\n',
        '    u_system\n': '    u_core/u_system\n',
        '    u_parameter_plane\n': '    u_core/u_parameter_plane\n',
    }
    result = add_reopen(original)
    for old, new in replacements.items():
        if result.count(old) != 1:
            raise ValueError('ambiguous wrapper adaptation: ' + old)
        result = result.replace(old, new, 1)
    clock_check = '''
if {$clock_count != 1 || [get_property PERIOD [get_clocks core_clk]] != 15.000} {
    error "unexpected inherited 15ns clock"
}
'''
    if result.count('\nopt_design\n') != 1:
        raise ValueError('ambiguous implementation entry')
    return result.replace('\nopt_design\n', clock_check + '\nopt_design\n', 1)


def prepare():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    evidence = root / RESULT
    accepted = json.loads((evidence / SYNTHESIS / 'terminal_audit.json').read_text())
    if accepted['status'] != 'PASS_paired_synthesis_archive':
        raise ValueError('accepted synthesis pair required')
    source_state = json.loads((evidence / SYNTHESIS / 'status.json').read_text())
    original_path = evidence / 'm2_compact_impl_r1' / ORIGINAL
    original = original_path.read_text()
    generated = implementation_tcl(original)
    sources = {}
    for arm in ARMS:
        relative = Path(SYNTHESIS) / arm / 'server_results/final_composition_synth/post_synth.dcp'
        source = evidence / relative
        record = source_state['arms'][arm]
        if record['exit'] != 0 or record['checkpoint']['crc_pass'] is not True:
            raise ValueError('unaccepted synthesis arm: ' + arm)
        if source.stat().st_size != record['checkpoint']['bytes']:
            raise ValueError('local accepted source size: ' + arm)
        sources[arm] = dict(relative=relative.as_posix(), bytes=source.stat().st_size)
    output = root / EXPERIMENT
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(original_path, output / 'original_impl.tcl')
    (output / 'implementation.tcl').write_text(generated, newline='\n')
    (output / 'implementation_changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), generated.splitlines(True),
        fromfile='accepted_compact_recipe', tofile='paired_raw_source_recipe')), newline='\n')
    record = dict(status='PREPARED_NOT_LAUNCHED', sources=sources,
        source_acceptance=accepted['status'], original_recipe=str(original_path.relative_to(root)),
        recipe='opt_design; place_design ExtraNetDelay_high; phys_opt_design ExploreWithHoldFix; route_design AggressiveExplore; phys_opt_design Explore',
        delta='Wrapper top and three u_core hierarchy paths; inherited 15ns check; existing checkpoint export/reopen. All optimization commands and original protections/congestion rule retained.',
        boundary='Paired internal OOC implementation of actual producer-plus-core DCPs. Host descriptors/manual suffix external. Functional cycles remain the accepted core experiment. No power/bitstream/production merge.')
    (output / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    for filename in (Path(__file__).name, 'run_final_raw_source_synthesis.py',
                     'run_scnn_turnover_implementation.py', 'run_m2_compact_route_recovery.py'):
        target = 'worker.py' if filename == Path(__file__).name else filename
        shutil.copy2(Path(__file__).with_name(filename), output / target)
    print(json.dumps(record, indent=2))


def fields(path):
    if not path.is_file():
        return {}
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def classify(directory, exit_code):
    report = directory / OUT
    logfile = report / 'vivado.log'
    log = logfile.read_text(errors='replace') if logfile.exists() else ''
    errors = re.findall(r'^ERROR:.*', log, re.M)
    critical = re.findall(r'^CRITICAL WARNING:.*', log, re.M)
    tcl = fields(report / 'vivado_status.txt')
    reopened = fields(report / 'checkpoint_reopen_status.txt')
    checkpoints = {}
    for name in ('post_place_physopt.dcp', 'post_route_physopt.dcp'):
        path = report / name
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                checkpoints[name] = dict(bytes=path.stat().st_size,
                    members=len(archive.infolist()), crc_pass=archive.testzip() is None)
    complete = (exit_code == 0 and bool(log) and not errors and
        tcl.get('status') in ('pass', 'diagnostic_complete') and
        tcl.get('top') == 'unified_h40_raw_source_top_v1' and
        tcl.get('part') == 'xc7vx690tffg1761-3' and
        reopened.get('status') == 'pass' and
        checkpoints.get('post_route_physopt.dcp', {}).get('crc_pass') is True)
    accepted = complete and tcl.get('status') == 'pass' and not critical
    return dict(status=('PASS_internal15ns_implementation' if accepted else
                        'DIAGNOSTIC_implementation' if complete else 'FAIL_implementation'),
        measurement_complete=bool(complete), exit=exit_code, tcl=tcl, reopened=reopened,
        errors=errors, critical_warnings=critical,
        ordinary_warnings=len(re.findall(r'^WARNING:', log, re.M)), checkpoints=checkpoints)


def worker():
    root, scratch = Path.cwd(), Path(SCRATCH)
    scratch.mkdir(exist_ok=False)
    prep = json.loads((root / 'preparation.json').read_text())
    state = dict(status='running', start=now(), scratch=SCRATCH, arms={})
    done, samples = threading.Event(), []

    def save():
        (root / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    def sample():
        while not done.is_set():
            samples.append(dict(time=now(), data_free=shutil.disk_usage(root).free,
                                tmp_free=shutil.disk_usage(scratch).free))
            (root / 'disk_observation.json').write_text(json.dumps(samples[-1]) + '\n')
            done.wait(5)

    save()
    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        for arm in ARMS:
            before = dict(time=now(), active_eda=active_eda(),
                data_free=shutil.disk_usage(root).free, tmp_free=shutil.disk_usage(scratch).free)
            if before['active_eda'] or before['data_free'] < PUBLIC_RESERVE or before['tmp_free'] < SCRATCH_RESERVE:
                raise RuntimeError('per-arm existing reserve/idle check: ' + repr(before))
            source = Path(prep['sources'][arm]['remote'])
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                    prep['sources'][arm]['bytes'], prep['sources'][arm]['mtime_ns']):
                raise ValueError('accepted source DCP changed: ' + arm)
            target = scratch / arm
            report = target / OUT
            report.mkdir(parents=True)
            install = str(Path(VIVADO).parents[1])
            env = os.environ.copy()
            env.update(XILINX_VIVADO=install, PATH=install + '/bin:' + env.get('PATH', ''),
                LD_LIBRARY_PATH=install + '/lib/lnx64.o/SuSE:' + env.get('LD_LIBRARY_PATH', ''),
                TMPDIR=str(target), TMP=str(target), TEMP=str(target))
            if Path('/lib/x86_64-linux-gnu/libudev.so.1').is_file():
                env['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libudev.so.1'
            command = [VIVADO, '-mode', 'batch', '-nojournal', '-nolog',
                '-source', str(root / 'implementation.tcl'), '-tclargs', str(target), str(source)]
            begin = time.monotonic()
            with (report / 'vivado.log').open('x') as log:
                process = subprocess.Popen(command, cwd=target, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                state['status'] = arm + '_running'
                state['arms'][arm] = dict(pid=process.pid, exit=None, command=command, before=before)
                save()
                try:
                    code = process.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    state['arms'][arm].update(timed_out=True, wall_seconds=time.monotonic() - begin)
                    state.update(status='TIMEOUT_PROCESS_RETAINED', end=now())
                    return
            try:
                parsed = classify(target, code)
            except Exception as error:
                parsed = dict(status='FAIL_terminal_classification', exit=code,
                    measurement_complete=False, classification_error=repr(error))
            state['arms'][arm].update(parsed, end=now(), wall_seconds=time.monotonic() - begin)
            save()
            archive_files(root, scratch)
            # Negative STA is a completed observation; a tool failure stops the pair.
            if not parsed['measurement_complete']:
                raise RuntimeError(arm + ' did not complete implementation')
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('source DCP metadata changed: ' + arm)
        state['status'] = 'COMPLETE_paired_raw_source_implementation'
    except Exception as error:
        state.update(status='FAIL', error=repr(error), end=now())
    finally:
        done.set()
        sampler.join()
        (root / 'disk_samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        state.update(end=now(), scratch_preserved=True,
            data_free_after=shutil.disk_usage(root).free, tmp_free_after=shutil.disk_usage('/tmp').free)
        save()


def launch():
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    prepared = Path(local_path(ROOT / EXPERIMENT))
    prep = json.loads((prepared / 'preparation.json').read_text())
    files = sorted(p for p in prepared.rglob('*') if p.is_file())
    payload = sum(p.stat().st_size for p in files)
    destination = REMOTE + '/' + NAME
    probe = '''import datetime,json,pathlib,shutil,zipfile
root=pathlib.Path(REMOTE);target=root/NAME;scratch=pathlib.Path(SCRATCH)
if CREATE and (target.exists() or scratch.exists()):raise FileExistsError('isolated run exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  n=(q/'comm').read_text().strip()
  if n in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or n.startswith('Vtb_'):active.append([q.name,n])
 except (OSError,ProcessLookupError):pass
data=shutil.disk_usage(root).free;tmp=shutil.disk_usage('/tmp').free
if active or data<RESERVE or tmp<SCRATCH_RESERVE:raise RuntimeError('existing idle/reserve check: '+repr((active,data,tmp)))
for arm,record in SOURCES.items():
 p=root/record['relative'];s=p.stat()
 if s.st_size!=record['bytes']:raise ValueError('source DCP size: '+arm)
 if CREATE:
  with zipfile.ZipFile(p) as z:
   if z.testzip() is not None:raise ValueError('source DCP CRC: '+arm)
  record.update(remote=str(p),mtime_ns=s.st_mtime_ns,crc_pass=True)
 elif record['mtime_ns']!=s.st_mtime_ns:raise ValueError('source DCP mtime: '+arm)
if CREATE:target.mkdir()
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=tmp,sources=SOURCES)))
'''
    client = connect()
    try:
        def observe(create, reserve, sources):
            values = dict(REMOTE=REMOTE, NAME=NAME, SCRATCH=SCRATCH, CREATE=create,
                RESERVE=reserve, SCRATCH_RESERVE=SCRATCH_RESERVE, SOURCES=sources)
            source = '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n' + probe
            return json.loads(run(client, 'python3 -c ' + shlex.quote(source)))
        before = observe(True, PUBLIC_RESERVE + payload, prep['sources'])
        prep['sources'] = before['sources']
        # Only the new preparation receives the actual read-only source metadata.
        (prepared / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
        files = sorted(p for p in prepared.rglob('*') if p.is_file())
        payload = sum(p.stat().st_size for p in files)
        with client.open_sftp() as sftp:
            for path in files:
                target = destination + '/' + path.relative_to(prepared).as_posix()
                upload(sftp, str(path), target)
                with sftp.open(target, 'rb') as stream:
                    if stream.read() != path.read_bytes():
                        raise ValueError('uploaded input differs: ' + str(path))
        at_start = observe(False, PUBLIC_RESERVE, prep['sources'])
        shell = 'cd ' + shlex.quote(destination) + ' && exec python3 worker.py worker >worker_stdout.log 2>&1'
        run(client, 'screen -dmS snn_' + NAME + ' bash -lc ' + shlex.quote(shell))
        record = dict(time=now(), launched=True, before_copy=before, before_start=at_start,
            prepared_files=len(files), prepared_bytes=payload, all_uploaded_bytes_equal=True,
            remote=destination, scratch=SCRATCH, screen='snn_' + NAME,
            boundary='Two serial internal OOC implementations from accepted DCPs. Existing 768MiB public/6GiB scratch checks and actual archive+128MiB reserve retained. Timeout retains PID; no automatic kill/restart. Source DCPs only read.')
        output = Path(local_path(ROOT / RESULT / NAME))
        output.mkdir(parents=True, exist_ok=True)
        (output / 'launch_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'worker'))
    args = parser.parse_args()
    {'prepare': prepare, 'launch': launch, 'worker': worker}[args.action]()
