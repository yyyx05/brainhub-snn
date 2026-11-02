"""Run the preserved small M2 refill chain with server Verilator 5.048.

Local 5.032 stopped before simulation on original-arm UNOPTFLAT. This is a
compiler discrimination, not a waiver, accepted baseline rerun or whole core.
"""
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
NAME = 'm2_refill_turnover_remote_r2'
LOCAL = 'm2_refill_turnover_local_r1'
VERILATOR = '/root/autodl-tmp/verilator_5_048_install/bin/verilator'


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / 'verification/experiments' / NAME, root / RESULT / NAME, root / RESULT


def load(path):
    spec = importlib.util.spec_from_file_location('refill_local', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    root, prepared, _, result = roots()
    local = result / LOCAL
    summary = json.loads((local / 'summary.json').read_text())
    if summary['status'] != 'STOPPED_LOCAL_BASELINE_UNOPTFLAT_NO_SIMULATION':
        raise ValueError('unexpected local state')
    source = local / summary['attempt']
    shutil.copytree(source / 'sources', prepared)
    shutil.copy2(source / 'run_m2_refill_turnover_local.py', prepared / 'local_runner.py')
    shutil.copy2(local / 'summary.json', prepared / 'local_diagnostic.json')
    shutil.copy2(__file__, prepared / 'launch_worker.py')
    (prepared / 'preparation.json').write_text(json.dumps(dict(
        source=LOCAL + '/' + summary['attempt'], input_source='actual archived local attempt2',
        same_two_module_candidate=True, compiler=VERILATOR, jobs=1, memory_limit_bytes=3*1024**3,
        scope='Original/candidate small publisher/slice/router/store. No whole core or physical result.',
        structural_rule_unchanged=True, revision='R1 stopped before EDA because GNU time was absent; R2 uses direct Popen and does not measure RSS.'), indent=2) + '\n')
    print(json.dumps(dict(prepared=str(prepared))))


def worker():
    import resource
    root = Path.cwd()
    scratch = Path('/tmp/snn_' + NAME)
    scratch.mkdir(exist_ok=False)
    local = load(root / 'local_runner.py')
    state = dict(status='running', start=dt.datetime.now().astimezone().isoformat(), scratch=str(scratch))

    def save():
        (root / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))

    def execute(stage, command):
        state['status'] = stage + '_running'
        save()
        begin = time.monotonic()
        with (root / (stage + '.log')).open('x') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, preexec_fn=limit)
            state[stage] = dict(command=command, pid=process.pid, exit=None)
            save()
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                state[stage].update(timeout_process_preserved=True, wall_seconds=time.monotonic()-begin)
                save()
                raise
        state[stage].update(exit=code, wall_seconds=time.monotonic()-begin)
        save()
        if code:
            raise RuntimeError(stage + ' exit ' + str(code))

    try:
        checks = {}
        state['tool_version'] = subprocess.check_output([VERILATOR, '--version'], text=True).strip()
        for arm_id, arm in enumerate(('original', 'candidate')):
            build = [VERILATOR, '--binary', '--timing', '--assert', '-j', '1', '-Wall', '-Wno-fatal',
                '--flatten', '--output-split', '3000', '-CFLAGS', '-O1', '--top-module',
                'tb_m2_refill_turnover', '--Mdir', str(scratch / ('obj_' + arm))]
            build += [str(root / arm / (n + '.sv')) for n in local.NAMES] + [str(root / local.TB)]
            execute(arm + '_build', build)
            text = (root / (arm + '_build.log')).read_text(errors='replace')
            if re.search(r'%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)|%Error', text):
                raise ValueError('structural build diagnostic; no simulation for ' + arm)
            execute(arm + '_run', [str(scratch / ('obj_' + arm) / 'Vtb_m2_refill_turnover'), '+arm=' + str(arm_id)])
            checks[arm] = local.parse((root / (arm + '_run.log')).read_text(), arm_id)
        state['checks'] = checks
        (root / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        state['status'] = 'PASS_M2_refill_small_chain_server_5048'
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(), scratch_preserved=True)
    save()


def launch():
    from run_srnn_resident_remote import REMOTE, connect, run, upload
    from run_srnn_rmw_turnover_core import probe_code
    _, prepared, out, _ = roots()
    files = sorted(p for p in prepared.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in files)
    destination = REMOTE + '/' + NAME
    client = connect()
    try:
        before = json.loads(run(client, 'python3 -c ' + shlex.quote(probe_code(size + 256*1024**2))))
        run(client, 'python3 -c ' + shlex.quote('from pathlib import Path\np=Path(' + repr(destination) +
            ')\nassert not p.exists() and not Path(' + repr('/tmp/snn_' + NAME) + ').exists()\np.mkdir()'))
        with client.open_sftp() as sftp:
            for p in files:
                target = destination + '/' + p.relative_to(prepared).as_posix()
                upload(sftp, str(p), target)
                with sftp.open(target, 'rb') as stream:
                    if stream.read() != p.read_bytes():
                        raise ValueError('uploaded input differs ' + str(p))
        at_start = json.loads(run(client, 'python3 -c ' + shlex.quote(probe_code(256*1024**2))))
        command = 'cd ' + shlex.quote(destination) + ' && exec python3 launch_worker.py worker'
        run(client, 'screen -dmS snn_' + NAME + ' bash -lc ' + shlex.quote(command))
        out.mkdir(parents=True, exist_ok=True)
        record = dict(launched=True, remote=destination, before_upload=before, before_start=at_start,
            prepared_input_files_byte_equal=len(files), prepared_input_bytes=size)
        (out / 'launch_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
    finally:
        client.close()


def audit():
    from run_srnn_resident_remote import REMOTE, connect, run
    _, prepared, archive, _ = roots()
    if (archive / 'terminal_audit.json').exists():
        raise FileExistsError('already audited')
    state = json.loads((archive / 'status.json').read_text())
    if not state.get('end'):
        raise ValueError('natural terminal required')
    code = '''import datetime,pathlib,json,shutil
p=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);skip={'obj_original','obj_candidate','obj','elab_obj','.Xil','__pycache__'}
def inventory(p):
 return {x.relative_to(p).as_posix():x.stat().st_size for x in p.rglob('*') if x.is_file() and not set(x.relative_to(p).parts)&skip}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  c=(q/'comm').read_text().strip()
  if c in ('vivado','verilator_bin','xelab','xsim') or c.startswith('Vtb_'):active.append([q.name,c])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),scratch_noncache=inventory(s),active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free,gnu_time_exists=pathlib.Path('/usr/bin/time').exists())))
'''
    client = connect()
    try:
        record = json.loads(run(client, 'python3 -c ' + shlex.quote('ROOT=' + repr(REMOTE + '/' + NAME) +
            '\nSCRATCH=' + repr(state['scratch']) + '\n' + code)))
    finally:
        client.close()
    if record['active_eda'] or record['scratch_noncache'] or list(archive.rglob('*.part')):
        raise ValueError('active process, omitted evidence or partial transfer')
    files = record.pop('files')
    for rel, size in files.items():
        if not (archive / rel).is_file() or (archive / rel).stat().st_size != size:
            raise ValueError('raw file missing/size ' + rel)
    inputs = [p for p in prepared.rglob('*') if p.is_file()]
    for p in inputs:
        if p.read_bytes() != (archive / p.relative_to(prepared)).read_bytes():
            raise ValueError('prepared input changed ' + str(p))
    diagnostics = {}
    for stage in ('original_build', 'original_run', 'candidate_build', 'candidate_run'):
        if stage not in state:
            continue
        text = (archive / (stage + '.log')).read_text(errors='replace')
        categories = collections.Counter(re.findall(r'^%Warning-([^:]+):', text, re.M))
        diagnostics[stage] = dict(warning_categories=dict(categories), warnings=sum(categories.values()),
            errors=len(re.findall(r'^%Error', text, re.M)),
            structural=[k for k in categories if k in ('LATCH', 'MULTIDRIVEN', 'UNOPTFLAT')])
    report = dict(original_status=state['status'], raw_files=len(files), raw_bytes=sum(files.values()),
        all_raw_sizes_match=True, prepared_input_files_byte_equal=len(inputs),
        prepared_input_bytes=sum(p.stat().st_size for p in inputs), no_part=True,
        stages={k: v for k, v in state.items() if k.endswith(('_build', '_run'))},
        remote_observation=record, diagnostics=diagnostics, dcp_crc='not applicable: no DCP',
        start=state['start'], end=state['end'], tool_version=state.get('tool_version'))
    if state['status'].startswith('PASS_'):
        if any(s['exit'] != 0 for s in report['stages'].values()) or any(d['structural'] or d['errors'] for d in diagnostics.values()):
            raise ValueError('build/run diagnostic')
        local = load(archive / 'local_runner.py')
        comparisons = []
        results = {}
        for arm_id, arm in enumerate(('original', 'candidate')):
            results[arm] = local.parse((archive / (arm + '_run.log')).read_text(), arm_id)
        if results != state['checks'] or results != json.loads((archive / 'checks.json').read_text()):
            raise ValueError('actual archived parser mismatch')
        log = (archive / 'candidate_run.log').read_text()
        for bad in (log.replace('M2_REFILL_PASS', 'MISSING'), log.replace('max_credit=1', 'max_credit=2', 1),
                    log.replace('M2_REFILL_FAULT id=1 pass=1', 'MISSING'), log + '\n%Error: synthetic\n'):
            try:
                local.parse(bad, 1)
            except ValueError:
                pass
            else:
                raise ValueError('damaged transcript accepted')
        for a, b in zip(results['original'], results['candidate']):
            if any(a[k] != b[k] for k in ('mode','format','variant','count','reads','responses','fills')):
                raise ValueError('two-arm accounting mismatch')
            comparisons.append(dict(mode=a['mode'], format=a['format'], variant=a['variant'],
                original_cycles=a['cycles'], candidate_cycles=b['cycles'], saved=a['cycles']-b['cycles'],
                original=a, candidate=b))
        report.update(status='PASS_M2_refill_remote_terminal_audit', comparisons=comparisons,
            archived_parser_strict_reparse_matches=True, damaged_transcripts_rejected=4)
    else:
        report.update(status='ARCHIVED_NOT_ACCEPTED_M2_refill_remote', error=state.get('error'))
    shutil.copy2(__file__, archive / 'terminal_audit_runner.py')
    (archive / 'terminal_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch', 'worker', 'audit'))
    a = p.parse_args()
    {'prepare': prepare, 'launch': launch, 'worker': worker, 'audit': audit}[a.action]()
