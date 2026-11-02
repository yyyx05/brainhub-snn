"""One new reducer-retirement arm; reuse accepted static remote inputs read-only."""
import argparse
import collections
import datetime as dt
import difflib
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
NAME = 'srnn_gradient_retire_core_r1'
PRIOR = 'srnn_m2_refill_turnover_core_r1'
UNIT = 'gradient_retire_r1'
ACCEPTED_UNIT = 'gradient_retire_local_r1'
TB = 'tb_srnn_profile_measurement.sv'
CHANGES = ('rtl/learning/unified_srnn_tp_wout_gradient_reducer.sv',)
SCOPE = ('One new stream_on whole-core arm with reducer gradient retirement overlapping '
         'the next contribution in its existing accumulator. Arithmetic, finalization register, '
         'RNE, overflow, terminal drain and storage credit unchanged. Same eight representative '
         'sequences/588 steps and independent references; accepted streaming/RMW/refill baseline '
         'not rerun. Actual forward/probability/trace/reduction/update/commit closed loop, not '
         'frozen-gradient replay or SGD. No new distribution, physical timing, DDR, author-system '
         'comparison or production adoption. 15ns is nominal only.')


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / 'verification/experiments' / NAME, root / RESULT / NAME, root / RESULT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    root, out, _, result = roots()
    prior = root / 'verification/experiments' / PRIOR
    unit = root / 'verification/experiments' / UNIT
    tested = result / ACCEPTED_UNIT
    accepted = json.loads((tested / 'status.json').read_text())
    if accepted['status'] != 'PASS_reducer_retirement_overlap_local':
        raise ValueError('unit candidate not accepted')
    for rel in CHANGES:
        if (unit / 'original' / Path(rel).name).read_bytes() != (prior / rel).read_bytes():
            raise ValueError('unit original/core mismatch')
        for arm in ('original', 'candidate'):
            if (unit / arm / Path(rel).name).read_bytes() != (tested / arm / Path(rel).name).read_bytes():
                raise ValueError('unit archive/source mismatch')
    shutil.copytree(prior, out, ignore=shutil.ignore_patterns('__pycache__'))
    for rel in CHANGES:
        shutil.copy2(unit / 'candidate' / Path(rel).name, out / rel)
    shutil.copy2(unit / 'changes.diff', out / 'gradient_retire_changes.diff')
    shutil.copy2(tested / 'status.json', out / 'gradient_retire_unit_acceptance.json')
    shutil.copy2(__file__, out / 'launch_worker.py')
    core = [p for p in (out / 'rtl').rglob('*') if p.is_file()]
    changed = sorted(p.relative_to(out).as_posix() for p in core
                     if p.read_bytes() != (prior / p.relative_to(out)).read_bytes())
    if len(core) != 109 or changed != sorted(CHANGES):
        raise ValueError('unexpected one-source core boundary')
    for rel in (TB, 'sources.f', 'source_scan.sv', 'source_stream.sv'):
        if (out / rel).read_bytes() != (prior / rel).read_bytes():
            raise ValueError('driver/source changed')
    prep = dict(status='PREPARED_NOT_SIMULATED', scope=SCOPE, baseline=PRIOR,
        changed_from_baseline=changed, unchanged_core_files=108,
        core_boundary=dict(total_files=109, sv_files=108, mem_files=1),
        original_producers_byte_equal=2, original_numerical_files_byte_equal=227,
        accepted_unit=ACCEPTED_UNIT, new_runs=['stream_on'], baseline_not_rerun=True,
        tb_change='None; previous refill monitor unchanged. No new reducer overlap event counter.',
        remote_static_input_reuse='Individual absolute symlinks to accepted prior static files, used read-only; new files are independent. No old file permission or content change.',
        physical_result='No new gradient_ready-to-contribution_ready/address STA or routing.')
    (out / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    files = [p for p in out.rglob('*') if p.is_file()]
    print(json.dumps(dict(prepared=str(out), files=len(files), bytes=sum(p.stat().st_size for p in files))))


def parse_log(log, info):
    import source_core_parser as parser
    parser.SCOPE = SCOPE
    parsed = parser.parse_log(log, 'on', info, 1)
    lines = re.findall(r'^M2_REFILL_TURNOVER_PASS (.+)$', log, re.M)
    if len(lines) != 1:
        raise ValueError('missing/duplicate refill turnover completion')
    turn = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', lines[0]))
    if set(turn) != {'reads', 'responses', 'replacements', 'max_credit', 'final_credit'}:
        raise ValueError('refill turnover fields')
    if turn['reads'] != 3043840 or turn['responses'] != turn['reads'] or turn['max_credit'] != 1 or turn['final_credit'] != 0:
        raise ValueError('refill traffic/credit changed')
    if not 0 <= turn['replacements'] < turn['reads']:
        raise ValueError('refill replacement count range')
    for name in ('m2_refill_read_accepted', 'm2_refill_response_accepted'):
        counts = re.findall(r'^SRNN_PROFILE_COUNT name=' + name + r' value=(\d+)$', log, re.M)
        if len(counts) != 8 or int(counts[-1]) != turn['reads']:
            raise ValueError('refill monitor/profile mismatch')
    return dict(**parsed, refill_turnover=turn)


def worker():
    root = Path.cwd()
    scratch = Path('/tmp/snn_' + NAME)
    scratch.mkdir(exist_ok=False)
    state = dict(status='running', start=dt.datetime.now().astimezone().isoformat(), scratch=str(scratch), scope=SCOPE)

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
        parsed = parse_log((root / 'run_stream_on.log').read_text(errors='replace'), info)
        state['stream_on'] = parsed
        (root / 'checks.json').write_text(json.dumps(dict(stream_on=parsed), indent=2) + '\n')
        state['status'] = 'PASS_streaming_core_closed_loop_gradient_retirement'
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(), scratch_preserved=True)
    save()


def launch():
    from run_srnn_resident_remote import REMOTE, connect, run, upload
    root, prepared, out, result = roots()
    prior = root / 'verification/experiments' / PRIOR
    accepted = json.loads((result / PRIOR / 'terminal_audit.json').read_text())
    if accepted['status'] != 'PASS_M2_refill_turnover_core_terminal_audit':
        raise ValueError('baseline archive incomplete')
    common = load('refill_common', prepared / 'rmw_core_common.py')
    files = sorted(p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    size = sum(p.stat().st_size for p in files)
    unchanged = [p.relative_to(prepared).as_posix() for p in files
                 if (prior / p.relative_to(prepared)).is_file()
                 and p.read_bytes() == (prior / p.relative_to(prepared)).read_bytes()]
    changed = [p for p in files if p.relative_to(prepared).as_posix() not in unchanged]
    uploaded_bytes = sum(p.stat().st_size for p in changed)
    destination = REMOTE + '/' + NAME
    client = connect()
    try:
        before = json.loads(run(client, 'python3 -c ' + shlex.quote(common.probe_code(uploaded_bytes + len(unchanged) * 4096 + 256 * 1024**2))))
        code = '''from pathlib import Path
import filecmp,json
old=Path(OLD);new=Path(NEW)
assert not new.exists() and not Path(SCRATCH).exists()
new.mkdir()
observed=[]
for rel in FILES:
 src=old/rel;dst=new/rel
 assert src.is_file() and not src.is_symlink()
 assert src.resolve().is_relative_to(old.resolve())
 dst.parent.mkdir(parents=True,exist_ok=True)
 stat=src.stat()
 dst.symlink_to(src)
 assert filecmp.cmp(src,dst,shallow=False),rel
 observed.append(dict(path=rel,target=str(src),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
print(json.dumps(dict(linked_byte_equal=len(FILES),observed_inputs=observed)))
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
        at_start = json.loads(run(client, 'python3 -c ' + shlex.quote(common.probe_code(256 * 1024**2))))
        command = 'cd ' + shlex.quote(destination) + ' && exec python3 launch_worker.py worker'
        run(client, 'screen -dmS snn_' + NAME + ' bash -lc ' + shlex.quote(command))
        record = dict(launched=True, remote=destination, screen='snn_' + NAME,
            before_copy=before, before_start=at_start, prepared_input_files=len(files), prepared_input_bytes=size,
            linked_from_accepted_baseline_byte_equal=copied['linked_byte_equal'],
            linked_static_inputs=copied['observed_inputs'], uploaded_changed_bytes=uploaded_bytes,
            disk_budget='Changed payload plus 4096 bytes per link plus existing 256MiB free reserve; existing 3GiB scratch minimum unchanged.',
            uploaded_changed_files_byte_equal=len(changed), changed_files=[p.relative_to(prepared).as_posix() for p in changed],
            copied_verification='Local prepared/prior bytes and complete accepted raw archive; per-file remote links to exact prior static files with shallow=False. Full dereferenced terminal fetch required. No hashes or old file mutation.')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'launch_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k:v for k,v in record.items() if k != 'linked_static_inputs'}, indent=2))
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
links={q.relative_to(p).as_posix():dict(target=str(q.resolve()),size=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns) for q in p.rglob('*') if q.is_symlink()}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),static_links=links,scratch_noncache_files=inventory(scratch),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
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
    observed_links = observation.pop('static_links')
    launched = json.loads((archive / 'launch_observation.json').read_text())
    expected_links = {r['path']: {k:r[k] for k in ('target', 'size', 'mtime_ns')} for r in launched['linked_static_inputs']}
    if observed_links != expected_links:
        raise ValueError('static source links or original metadata changed')

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
    prior_paths = {p.relative_to(baseline).as_posix() for p in (baseline / 'rtl').rglob('*') if p.is_file()}
    actual_paths = {p.relative_to(archive).as_posix() for p in (archive / 'rtl').rglob('*') if p.is_file()}
    changed = sorted(p for p in prior_paths if (baseline / p).read_bytes() != (archive / p).read_bytes())
    if prior_paths != actual_paths or len(prior_paths) != 109 or changed != sorted(CHANGES):
        raise ValueError('one-source core boundary')
    for rel in ('source_scan.sv', 'source_stream.sv', 'sources.f', 'source_core_parser.py',
                'tb/monitors/supplement_h40_activity_monitor.sv', 'run_srnn_representative_inputs.py'):
        if (baseline / rel).read_bytes() != (archive / rel).read_bytes():
            raise ValueError('unchanged source/helper boundary ' + rel)
    accepted = json.loads((archive / 'gradient_retire_unit_acceptance.json').read_text())
    if accepted['status'] != 'PASS_reducer_retirement_overlap_local':
        raise ValueError('unit evidence status')
    unit = result / ACCEPTED_UNIT
    for rel in CHANGES:
        if (unit / 'candidate' / Path(rel).name).read_bytes() != (archive / rel).read_bytes():
            raise ValueError('local candidate differs ' + rel)
    if (baseline / TB).read_bytes() != (archive / TB).read_bytes():
        raise ValueError('original TB/monitor modified')
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
        numerical_files_byte_equal=len(numerical), changed_from_baseline=changed, unchanged_core_files=108,
        core_boundary=dict(total_files=109, sv_files=108, mem_files=1), original_tb_monitor_byte_equal=True, static_links_and_original_metadata_preserved=len(expected_links),
        stages={s: state[s] for s in ('build', 'run_stream_on') if s in state},
        diagnostics=diagnostics, remote_observation=observation, dcp_crc='not applicable: no DCP', scope=SCOPE)
    if state['status'].startswith('PASS_'):
        if any(state.get(s, {}).get('exit') != 0 for s in ('build', 'run_stream_on')):
            raise ValueError('real build/run exits not zero')
        if any(d['errors'] or d['fatal'] or d['structural'] for d in diagnostics.values()):
            raise ValueError('diagnostic failure')
        load('run_srnn_representative_inputs', archive / 'run_srnn_representative_inputs.py')
        load('source_core_parser', archive / 'source_core_parser.py')
        actual_worker = load('archived_gradient_retire_worker', archive / 'launch_worker.py')
        log = (archive / 'run_stream_on.log').read_text()
        info = json.loads((archive / 'multi_fixture/reference.json').read_text())
        parsed = actual_worker.parse_log(log, info)
        if parsed != state['stream_on'] or parsed != json.loads((archive / 'checks.json').read_text())['stream_on']:
            raise ValueError('actual archived worker/parser mismatch')
        damaged = (log.replace('M2_REFILL_TURNOVER_PASS', 'MISSING'),
            log.replace('max_credit=1 final_credit=0', 'max_credit=2 final_credit=0'),
            log.replace('SOURCE_CORE_FRAME', 'MISSING', 1), log + '\n%Error: synthetic\n')
        for bad in damaged:
            if bad == log:
                raise ValueError('negative test did not modify transcript')
            try:
                actual_worker.parse_log(bad, info)
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
        report.update(status='PASS_gradient_retirement_core_terminal_audit', archived_worker_strict_reparse_matches=True,
            damaged_transcripts_rejected=len(damaged), baseline_cold_cycles=ac, candidate_cold_cycles=bc,
            saved_cycles=ac - bc, cold_reduction_percent=100 * (ac - bc) / ac,
            baseline_service_cycles=original['service_cycles'], candidate_service_cycles=parsed['service_cycles'],
            refill_turnover=parsed['refill_turnover'])
    else:
        report.update(status='ARCHIVED_NOT_ACCEPTED_gradient_retirement_core_failure', error=state.get('error'))
    shutil.copy2(__file__, archive / 'terminal_audit_runner.py')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'launch', 'audit'))
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker, 'launch': launch, 'audit': audit}[args.action]()
