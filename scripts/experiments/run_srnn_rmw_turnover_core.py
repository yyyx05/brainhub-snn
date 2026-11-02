"""One new closed-loop learning arm for the isolated RMW turnover candidate."""
import argparse
import collections
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NAME = 'srnn_rmw_turnover_core_r1'
PRIOR = 'srnn_source_core_r2'
RMW = Path('rtl/learning/unified_tp_parameter_rmw_service.sv')
SCOPE = ('New single-credit RMW retirement-turnover candidate, full closed-loop direct learning '
         'on the accepted raw-count streaming core. Same eight representative sequences/588 steps, '
         'same initial model, rules and original independent references. Only the new learning-on '
         'arm runs; accepted source R2 stream_on is the comparison. No new distribution or SGD '
         'training; no physical timing, DDR, board result or production adoption.')


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / 'verification/experiments' / NAME, root / RESULT / NAME, root / RESULT


def prepare():
    root, out, _, result = roots()
    baseline = root / 'verification/experiments' / PRIOR
    unit = root / 'verification/experiments/rmw_turnover_r1'
    accepted = json.loads((result / 'rmw_turnover_local_r1/summary.json').read_text())
    if accepted['status'] != 'PASS_isolated_direct_service_only':
        raise ValueError('local candidate not accepted')
    for p in (unit / 'original').glob('*.sv'):
        if p.read_bytes() != (baseline / 'rtl/learning' / p.name).read_bytes():
            raise ValueError('unit/core original differs: ' + p.name)
    shutil.copytree(baseline, out, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(baseline / 'launch_worker.py', out / 'source_core_parser.py')
    shutil.copy2(unit / 'candidate' / RMW.name, out / RMW)
    shutil.copy2(unit / 'changes.diff', out / 'rmw_changes.diff')
    shutil.copy2(result / 'rmw_turnover_local_r1/summary.json', out / 'local_candidate_acceptance.json')
    shutil.copy2(__file__, out / 'launch_worker.py')
    all_rtl = [p for p in (out / 'rtl').rglob('*.sv')]
    changed = [p.relative_to(out).as_posix() for p in all_rtl
               if p.read_bytes() != (baseline / p.relative_to(out)).read_bytes()]
    if changed != [RMW.as_posix()]:
        raise ValueError(('unexpected core changes', changed))
    prep = json.loads((out / 'preparation.json').read_text())
    prep.update(scope=SCOPE, baseline=PRIOR, changed_from_baseline=changed,
                new_runs=['stream_on'], baseline_not_rerun=True,
                prior_repair_source='Historical input copied unchanged, not executed here.',
                physical_result='None; added result_ready-to-request_ready path unmeasured.')
    (out / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    files = [p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(prepared=str(out), files=len(files), bytes=sum(p.stat().st_size for p in files))))


def worker():
    import source_core_parser as parser
    parser.SCOPE = SCOPE
    root = Path.cwd()
    scratch = Path('/tmp/snn_' + NAME)
    scratch.mkdir(exist_ok=False)
    state = dict(status='running', start=dt.datetime.now().astimezone().isoformat(),
                 scratch=str(scratch), scope=SCOPE)

    def save():
        (root / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    def execute(stage, command, timeout):
        state['status'] = stage + '_running'
        save()
        begin = time.monotonic()
        with (root / (stage + '.log')).open('x') as handle:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            state[stage] = dict(command=command, pid=process.pid, exit=None)
            save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage].update(timed_out=True, wall_seconds=time.monotonic() - begin)
                save()
                raise
        state[stage].update(exit=code, wall_seconds=time.monotonic() - begin)
        save()
        if code:
            raise RuntimeError(stage + ' exit ' + str(code))

    try:
        build = json.loads((root / 'build_command.json').read_text())
        build[build.index('--Mdir') + 1] = str(scratch / 'obj')
        execute('build', build, 3600)
        text = (root / 'build.log').read_text(errors='replace')
        state['build_warnings'] = dict(ordinary=len(re.findall(r'^%Warning-', text, re.M)),
            structural=re.findall(r'^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*', text, re.M))
        if state['build_warnings']['structural']:
            raise ValueError('structural warning')
        execute('run_stream_on', [str(scratch / 'obj/Vtb_srnn_profile_measurement'),
            '+RAW=multi_fixture', '+FIXTURE=multi_fixture', '+LEARNING=1', '+STREAMING=1'], 7200)
        info = json.loads((root / 'multi_fixture/reference.json').read_text())
        parsed = parser.parse_log((root / 'run_stream_on.log').read_text(errors='replace'), 'on', info, 1)
        state['stream_on'] = parsed
        (root / 'checks.json').write_text(json.dumps(dict(stream_on=parsed), indent=2) + '\n')
        state['status'] = 'PASS_streaming_core_closed_loop_direct_RMW_turnover'
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(), scratch_preserved=True)
    save()


def probe_code(required):
    return '''import datetime,json,pathlib,shutil
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage('/root/autodl-tmp').free,tmp_free=shutil.disk_usage('/tmp').free)
if active or record['data_free']<''' + str(required) + ''' or record['tmp_free']<3*1024**3:raise RuntimeError(record)
print(json.dumps(record))
'''


def launch():
    from run_srnn_resident_remote import REMOTE, connect, run, upload
    root, prepared, out, result = roots()
    prior = root / 'verification/experiments' / PRIOR
    evidence = json.loads((result / PRIOR / 'terminal_audit.json').read_text())
    if evidence['status'] != 'PASS_raw_source_A_core_comparison_terminal_audit':
        raise ValueError('baseline archive incomplete')
    files = sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    size = sum(p.stat().st_size for p in files)
    unchanged = [p.relative_to(prepared).as_posix() for p in files
                 if (prior / p.relative_to(prepared)).is_file()
                 and p.read_bytes() == (prior / p.relative_to(prepared)).read_bytes()]
    changed = [p for p in files if p.relative_to(prepared).as_posix() not in unchanged]
    destination = REMOTE + '/' + NAME
    client = connect()
    try:
        before = json.loads(run(client, 'python3 -c ' + shlex.quote(probe_code(size + 256 * 1024**2))))
        code = '''from pathlib import Path
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
        copied = json.loads(run(client, 'python3 -c ' + shlex.quote(
            'OLD=' + repr(REMOTE + '/' + PRIOR) + '\nNEW=' + repr(destination) +
            '\nSCRATCH=' + repr('/tmp/snn_' + NAME) + '\nFILES=' + repr(unchanged) + '\n' + code)))
        with client.open_sftp() as sftp:
            for path in changed:
                target = destination + '/' + path.relative_to(prepared).as_posix()
                upload(sftp, str(path), target)
                with sftp.open(target, 'rb') as handle:
                    if handle.read() != path.read_bytes():
                        raise ValueError('changed input bytes ' + str(path))
        at_start = json.loads(run(client, 'python3 -c ' + shlex.quote(probe_code(256 * 1024**2))))
        command = 'cd ' + shlex.quote(destination) + ' && exec python3 launch_worker.py worker'
        run(client, 'screen -dmS snn_' + NAME + ' bash -lc ' + shlex.quote(command))
        record = dict(launched=True, remote=destination, screen='snn_' + NAME,
            before_copy=before, before_start=at_start, prepared_input_files=len(files),
            prepared_input_bytes=size, copied_from_accepted_R2_byte_equal=copied['copied_byte_equal'],
            uploaded_changed_files_byte_equal=len(changed), changed_files=[p.relative_to(prepared).as_posix() for p in changed],
            copied_verification='Local prepared/baseline bytes; accepted baseline raw inputs; remote shallow=False. No hashes.')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'launch_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
    finally:
        client.close()


def audit():
    from run_srnn_resident_remote import REMOTE, connect, run
    root, prepared, archive, result = roots()
    report_path = archive / 'terminal_audit.json'
    if report_path.exists():
        raise FileExistsError('already audited; do not repeat')
    state = json.loads((archive / 'status.json').read_text())
    if not state.get('end'):
        raise ValueError('natural terminal state required')
    code = '''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);scratch=pathlib.Path(SCRATCH);active=[]
skip={'obj','elab_obj','__pycache__','.Xil'}
def inventory(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),scratch_noncache_files=inventory(scratch),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    client = connect()
    try:
        observation = json.loads(run(client, 'python3 -c ' + shlex.quote(
            'ROOT=' + repr(REMOTE + '/' + NAME) + '\nSCRATCH=' + repr(state['scratch']) + '\n' + code)))
    finally:
        client.close()
    if observation['active_eda'] or observation['scratch_noncache_files']:
        raise ValueError('active EDA or unfetched non-cache evidence')
    files = observation.pop('files')
    if list(archive.rglob('*.part')) or any(p.endswith('.part') for p in files):
        raise ValueError('partial archive')
    for rel, size in files.items():
        if not (archive / rel).is_file() or (archive / rel).stat().st_size != size:
            raise ValueError('raw archive size ' + rel)
    inputs = [p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in inputs:
        if p.read_bytes() != (archive / p.relative_to(prepared)).read_bytes():
            raise ValueError('input bytes ' + str(p))
    baseline = result / PRIOR
    # The accepted 109-file boundary contains 108 SV files and one MEM table.
    prior_rtl = [p for p in (baseline / 'rtl').rglob('*') if p.is_file()]
    prior_paths = {p.relative_to(baseline).as_posix() for p in prior_rtl}
    actual_paths = {p.relative_to(archive).as_posix() for p in (archive / 'rtl').rglob('*') if p.is_file()}
    if prior_paths != actual_paths:
        raise ValueError('core source path sets differ')
    changed = [p.relative_to(baseline).as_posix() for p in prior_rtl
               if p.read_bytes() != (archive / p.relative_to(baseline)).read_bytes()]
    if len(prior_rtl) != 109 or changed != [RMW.as_posix()]:
        raise ValueError('only RMW may differ from accepted R2')
    for rel in ('source_scan.sv', 'source_stream.sv', 'sources.f',
                'tb_srnn_profile_measurement.sv', 'tb/monitors/supplement_h40_activity_monitor.sv'):
        if (baseline / rel).read_bytes() != (archive / rel).read_bytes():
            raise ValueError('source producer/TB boundary ' + rel)
    unit = result / 'rmw_turnover_local_r1/sources/candidate' / RMW.name
    if unit.read_bytes() != (archive / RMW).read_bytes():
        raise ValueError('locally tested candidate differs')
    numerical = [p for p in (baseline / 'multi_fixture').rglob('*') if p.is_file()]
    for p in numerical:
        if p.read_bytes() != (archive / p.relative_to(baseline)).read_bytes():
            raise ValueError('numerical fixture changed')
    diagnostics = {}
    for stage in ('build', 'run_stream_on'):
        path = archive / (stage + '.log')
        if not path.exists():
            continue
        text = path.read_text(errors='replace')
        categories = collections.Counter(re.findall(r'^%Warning-([^:]+):', text, re.M))
        diagnostics[stage] = dict(warning_categories=dict(categories), ordinary_warnings=sum(categories.values()),
            errors=len(re.findall(r'^%Error', text, re.M)),
            fatal=bool(re.search(r'%Fatal|Assertion failed|\bFATAL\b', text)),
            structural=[k for k in categories if k in ('LATCH', 'MULTIDRIVEN', 'UNOPTFLAT')])
    report = dict(original_status=state['status'], start=state['start'], end=state['end'],
        raw_files=len(files), raw_bytes=sum(files.values()), all_raw_sizes_match=True, no_part=True,
        prepared_input_files_byte_equal=len(inputs), prepared_input_bytes=sum(p.stat().st_size for p in inputs),
        numerical_files_byte_equal=len(numerical), changed_from_source_R2=changed,
        unchanged_R2_core_files=108, local_candidate_byte_equal=True,
        core_boundary=dict(total_files=109, sv_files=108, mem_files=1),
        audit_revision='Local audit enumeration corrected from SV-only to all 109 original source files; original worker/prepared inputs/status/logs unchanged.',
        first_audit_failure_record='audit_attempt1_failure.txt',
        stages={s: state[s] for s in ('build', 'run_stream_on') if s in state},
        diagnostics=diagnostics, remote_observation=observation, dcp_crc='not applicable: no DCP', scope=SCOPE)
    if state['status'].startswith('PASS_'):
        if any(state.get(s, {}).get('exit') != 0 for s in ('build', 'run_stream_on')):
            raise ValueError('real build/run exits not zero')
        if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):
            raise ValueError('diagnostic failure')

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        load('run_srnn_representative_inputs', archive / 'run_srnn_representative_inputs.py')
        actual_worker = load('archived_turnover_worker', archive / 'launch_worker.py')
        parser = load('source_core_parser', archive / 'source_core_parser.py')
        parser.SCOPE = actual_worker.SCOPE
        log = (archive / 'run_stream_on.log').read_text()
        info = json.loads((archive / 'multi_fixture/reference.json').read_text())
        parsed = parser.parse_log(log, 'on', info, 1)
        if parsed != state['stream_on'] or parsed != json.loads((archive / 'checks.json').read_text())['stream_on']:
            raise ValueError('actual archived parser mismatch')
        for bad in (log.replace('SOURCE_CORE_PASS', 'MISSING'), log.replace('SOURCE_CORE_FRAME', 'MISSING', 1), log + '\n%Error: synthetic\n'):
            try:
                parser.parse_log(bad, 'on', info, 1)
            except ValueError:
                pass
            else:
                raise AssertionError('damaged transcript accepted')
        original = json.loads((baseline / 'checks.json').read_text())['stream_on']
        for x, y in zip(original['sequences'], parsed['sequences']):
            if any(x[k] != y[k] for k in ('bone', 'hidden', 'logits', 'store', 'code', 'residual',
                                        'commits', 'backbone_loads', 'wh1_loads', 'wout_loads')):
                raise ValueError('numerical/update/residency difference')
        ac, bc = original['total']['cycles'], parsed['total']['cycles']
        report.update(status='PASS_new_turnover_core_terminal_audit', archived_worker_strict_reparse_matches=True,
            damaged_transcripts_rejected=3, baseline_cold_cycles=ac, candidate_cold_cycles=bc,
            saved_cycles=ac - bc, cold_reduction_percent=100 * (ac - bc) / ac,
            baseline_service_cycles=original['service_cycles'], candidate_service_cycles=parsed['service_cycles'])
    else:
        report.update(status='ARCHIVED_NOT_ACCEPTED_turnover_core_failure', error=state.get('error'))
    shutil.copy2(__file__, archive / 'terminal_audit_runner.py')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'launch', 'audit'))
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker, 'launch': launch, 'audit': audit}[args.action]()
