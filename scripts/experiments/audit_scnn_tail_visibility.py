#!/usr/bin/env python3
"""Audit the completed tail run without changing or rerunning its raw evidence."""
import collections
import datetime as dt
import difflib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import stat

from prepare_scnn_tail_testbench import ORIGINAL, parse_log, transform
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, local_path, run


def main():
    name = 'scnn_tail_visibility_r1'
    result = Path(local_path(ROOT / RESULT / name))
    archive = Path(local_path(ROOT / RESULT))
    prepared = Path(local_path(ROOT / 'verification/experiments/scnn_final_visibility'))
    status = json.loads((result / 'status.json').read_text())
    assert status['status'] == 'FAIL' and status['error'] == 'ValueError: event ordering'
    assert status['build']['exit'] == status['run']['exit'] == 0
    assert not status['structural_warnings']
    log = (result / 'run.log').read_text()
    build = (result / 'build.log').read_text()
    assert not re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b', build + log)
    assert not re.search(r'%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT)', build)
    assert not re.search(r'Out of memory|\bOOM\b|\bKilled\b', build + log, re.I)
    warnings = dict(collections.Counter(re.findall(r'%Warning-(\w+)', build)))
    assert warnings == {'MISINDENT': 9, 'UNSIGNED': 7}
    assert len(re.findall(r'^H40_ACTIVITY tile=', log, re.M)) == 12
    checks = parse_log(log)

    old_parser = result / 'scripts/experiments/prepare_scnn_tail_testbench.py'
    spec = importlib.util.spec_from_file_location('archived_tail_parser', old_parser)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    try:
        original_module.parse_log(log)
    except ValueError as error:
        assert str(error) == 'event ordering'
    else:
        raise AssertionError('archived parser no longer reproduces the recorded failure')
    original_tb = (result / 'original_tb_scnn_real_final_b4t12.sv').read_text()
    actual_tb = (result / 'tb_scnn_tail_visibility.sv').read_text()
    assert transform(original_tb) == original_module.transform(original_tb) == actual_tb

    identities = {}
    def same_tree(label, current, prior, pattern='*'):
        files = sorted(p for p in current.rglob(pattern) if p.is_file())
        assert files, label
        for path in files:
            other = prior / path.relative_to(current)
            assert other.read_bytes() == path.read_bytes(), str(path)
        identities[label] = len(files)
    same_tree('rtl_vs_banked_synthesis', result / 'rtl', archive / 'scnn_issue_banked_synth_r1/rtl', '*.sv')
    same_tree('rtl_vs_natural_component', result / 'rtl', archive / 'scnn_natural_component_r4/rtl', '*.sv')
    assert identities['rtl_vs_banked_synthesis'] == identities['rtl_vs_natural_component'] == 107
    same_tree('fixture_vs_natural_original', result / 'fixture', archive / 'scnn_natural_component_r4/fixture')
    assert identities['fixture_vs_natural_original'] == 36
    same_tree('models_vs_accepted_sparse', result / 'models', archive / 'scnn_issue_sparse_r1/models')
    same_tree('tail_reference_vs_prepared', result / 'tail_reference', prepared / 'tail_reference_r1')
    assert (result / 'original_tb_scnn_real_final_b4t12.sv').read_bytes() == Path(local_path(ROOT / ORIGINAL)).read_bytes()
    for filename in ('tb_scnn_tail_visibility.sv', 'reference.json', 'testbench_changes.diff'):
        assert (result / filename).read_bytes() == (prepared / 'candidate_r1' / filename).read_bytes()
    for filename in ('run_scnn_tail_visibility.py', 'srnn_resident_server_worker.py', 'prepare_scnn_tail_visibility.py'):
        assert (result / 'scripts/experiments' / filename).read_bytes() == (ROOT / 'scripts/experiments' / filename).read_bytes()
    assert (result / 'scripts/supplement_scnn_graph_host.mk').read_bytes() == (archive / 'scnn_issue_sparse_r1/scripts/supplement_scnn_graph_host.mk').read_bytes()
    assert (result / 'scnn_compatibility_build.json').read_bytes() == (archive / 'scnn_issue_sparse_r1/scnn_compatibility_build.json').read_bytes()
    prior_lines = (archive / 'scnn_issue_sparse_r1/compatibility_sources.f').read_text().splitlines()
    expected_lines = [s.replace('tb_scnn_profile_measurement.sv', 'tb_scnn_tail_visibility.sv') for s in prior_lines]
    assert (result / 'compatibility_sources.f').read_text().splitlines() == expected_lines

    inventory = []
    client = connect()
    try:
        with client.open_sftp() as sftp:
            def visit(relative):
                remote = REMOTE + '/' + name + ('/' + relative if relative else '')
                for entry in sftp.listdir_attr(remote):
                    if entry.filename in ('obj', 'elab_obj', '__pycache__', '.Xil'):
                        continue
                    child = (relative + '/' if relative else '') + entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        visit(child)
                    else:
                        assert stat.S_ISREG(entry.st_mode), child
                        assert (result / child).stat().st_size == entry.st_size, child
                        inventory.append(dict(path=child, bytes=entry.st_size))
            visit('')
            assert not list(result.rglob('*.part'))
            # The inherited control values are checked against their original remote input.
            control_root = '/root/autodl-tmp/SNN_SCNN_REAL_FINAL_B4T12_R2/results/paper_evidence/supplement_20261004/rtl/scnn_real_final_graph/server_b4t12_r2/control'
            controls = [p for p in (result / 'control').rglob('*') if p.is_file()]
            for path in controls:
                with sftp.open(control_root + '/' + path.relative_to(result / 'control').as_posix(), 'rb') as stream:
                    assert stream.read() == path.read_bytes(), str(path)
            identities['control_vs_original_remote'] = len(controls)
        inspection = '''import json,pathlib,shutil,subprocess
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator'): active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError): pass
d=subprocess.run(['dmesg'],capture_output=True,text=True)
print(json.dumps(dict(active=active,free_bytes=shutil.disk_usage('/root/autodl-tmp').free,
 dmesg_readable=d.returncode==0,dmesg_error=d.stderr.strip())))'''
        remote = json.loads(run(client, 'python3 -c ' + shlex.quote(inspection)))
        assert not remote['active'], remote
    finally:
        client.close()

    current_parser = Path(__file__).with_name('prepare_scnn_tail_testbench.py').read_text()
    correction = ''.join(difflib.unified_diff(old_parser.read_text().splitlines(True), current_parser.splitlines(True),
                                            fromfile='archived_executed_parser', tofile='corrected_local_parser'))
    assert correction
    report = dict(
        checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='ACCEPTED_SCOPED_RTL_RESULT_AFTER_LOCAL_PARSER_ORDER_CORRECTION',
        original_wrapper_status=status, original_status_unchanged=True, rerun=False,
        original_checks_json_present=(result / 'checks.json').exists(),
        parser_diagnosis='Archived parser expected COMMIT before LAYER; actual TB emits LAYER, then COMMIT, then FRAME_STATUS. The corrected parser enforces that order and reproduces all original counter checks.',
        raw_files=len(inventory), raw_bytes=sum(p['bytes'] for p in inventory),
        all_raw_file_sizes_match=True, partial_files=0, byte_identity=identities,
        build_warnings=warnings, structural_warnings=0, runtime_errors=0,
        oom_or_killed_in_logs=False, tile_activity_records=12, remote_at_audit=remote,
        final=checks['final'], service=checks['service'], commits=checks['commits'], scope=checks['scope'],
        boundaries=['Conv1..7 use explicit CPU integer service input, not RTL in this run.',
                    'No full raw graph, DDR, end-to-end throughput, or abort claim.',
                    'Original wrapper FAIL is preserved; checks_reparsed.json is a separate local derived result.'],
        raw_inventory=sorted(inventory, key=lambda row: row['path']))
    (result / 'checks_reparsed.json').write_text(json.dumps(checks, indent=2) + '\n')
    (result / 'terminal_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (result / 'parser_order_correction.diff').write_text(correction)
    (result / 'corrected_local_parser.py').write_text(current_parser)
    print(json.dumps({k:v for k,v in report.items() if k != 'raw_inventory'}, indent=2))


if __name__ == '__main__':
    main()
