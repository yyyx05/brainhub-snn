"""Audit one fetched natural terminal of the paired raw-source implementation."""
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_final_raw_source_implementation import NAME, SCRATCH, EXPERIMENT, OUT, ARMS
from audit_m2_compact_implementation import resources, violations, report_checks_pass
from audit_m2_compact_synthesis import timing, hierarchy
from audit_m2_compact_route_recovery import fields, route_metrics, reopened_equal, checkpoint_record


def load_worker(path, name='executed_raw_source_impl_worker'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit():
    base = Path(local_path(ROOT / RESULT))
    archive = base / NAME
    prepared = Path(local_path(ROOT / EXPERIMENT))
    destination = archive / 'terminal_audit.json'
    if destination.exists():
        raise FileExistsError('terminal audit already exists; do not repeat')
    state = json.loads((archive / 'status.json').read_text())
    if not state.get('end') or state['status'] == 'TIMEOUT_PROCESS_RETAINED' or state['status'].endswith('_running'):
        raise ValueError('natural terminal required; retain timed-out process')
    prep = json.loads((archive / 'preparation.json').read_text())
    code = '''import datetime,json,pathlib,shutil,zipfile
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH)
def inventory(p):
 return {q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  n=(p/'comm').read_text().strip()
  if n in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or n.startswith('Vtb_'):active.append([p.name,n])
 except (OSError,ProcessLookupError):pass
sources={}
for arm,row in SOURCES.items():
 p=pathlib.Path(row['remote']);stat=p.stat()
 with zipfile.ZipFile(p) as z:crc=z.testzip() is None
 sources[arm]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,crc_pass=crc)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(r),scratch_files=inventory(s),sources=sources,active_eda=active,data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
    values = dict(ROOT=REMOTE + '/' + NAME, SCRATCH=SCRATCH, SOURCES=prep['sources'])
    client = connect()
    try:
        source = '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n' + code
        observation = json.loads(run(client, 'python3 -c ' + shlex.quote(source)))
    finally:
        client.close()
    if observation['active_eda']:
        raise ValueError('EDA remains active; defer terminal audit')
    if list(archive.rglob('*.part')) or any(rel.endswith('.part') for rel in observation['files']):
        raise ValueError('partial archive')
    for rel, size in observation['files'].items():
        path = archive / rel
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('incomplete fetched original: ' + rel)
    if any(observation['files'].get(rel) != size for rel, size in observation['scratch_files'].items()):
        raise ValueError('unarchived scratch original')
    inventory_path = archive / 'scratch_archive.json'
    saved_inventory = json.loads(inventory_path.read_text()) if inventory_path.is_file() else []
    if {row['relative']: row['bytes'] for row in saved_inventory} != observation['scratch_files']:
        raise ValueError('actual worker scratch inventory differs')
    inputs = [p for p in prepared.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    if any(p.read_bytes() != (archive / p.relative_to(prepared)).read_bytes() for p in inputs):
        raise ValueError('prepared and executed input bytes differ')
    for name in ('run_final_raw_source_synthesis', 'run_m2_compact_route_recovery',
                 'run_scnn_turnover_implementation'):
        load_worker(archive / (name + '.py'), name)
    worker = load_worker(archive / 'worker.py')
    if (archive / 'original_impl.tcl').read_bytes() != Path(local_path(ROOT / prep['original_recipe'])).read_bytes():
        raise ValueError('original recipe bytes differ')
    if worker.implementation_tcl((archive / 'original_impl.tcl').read_text()) != (archive / 'implementation.tcl').read_text():
        raise ValueError('executed Tcl transform differs from original recipe')
    source_checks = {}
    for arm, row in prep['sources'].items():
        observed = observation['sources'][arm]
        source_checks[arm] = (observed['bytes'] == row['bytes'] and
            observed['mtime_ns'] == row['mtime_ns'] and observed['crc_pass'] is True)
        local_dcp = checkpoint_record(base / row['relative'])
        source_checks[arm] &= local_dcp.get('crc_pass') is True and local_dcp.get('bytes') == row['bytes']
    if not all(source_checks.values()):
        raise ValueError('accepted source DCP metadata or CRC changed')
    arms = {}
    for arm, stage in state['arms'].items():
        if stage.get('exit') is None:
            raise ValueError('missing real exit: ' + arm)
        try:
            parsed = worker.classify(archive / arm, stage['exit'])
        except Exception as error:
            parsed = dict(status='FAIL_terminal_classification', exit=stage['exit'],
                measurement_complete=False, classification_error=repr(error))
        if any(stage.get(k) != v for k, v in parsed.items()):
            raise ValueError('actual archived worker strict reparse differs: ' + arm)
        report = archive / arm / OUT
        missing, errors = [], []

        def read(name):
            path = report / name
            return path.read_text(errors='replace') if path.is_file() else ''

        def parse(name, parser):
            text = read(name)
            if not text:
                missing.append(name)
                return None
            try:
                return parser(text)
            except (ValueError, IndexError, TypeError, KeyError, AttributeError) as error:
                errors.append(dict(report=name, error=repr(error)))
                return None

        reports = {}
        clocks_valid = {}
        for label, prefix in (('exported', ''), ('reopened', 'reopened/')):
            reports[label] = dict(
                resources=parse(prefix + 'utilization_post_route.rpt', resources),
                timing=parse(prefix + 'timing_summary_post_route.rpt', timing),
                route=parse(prefix + 'route_status_post_route.rpt', route_metrics))
            clocks = re.findall(r'^(\w+)\s+(\d+\.\d+)\s+\{([^}]+)\}\s+P\s+\{([^}]+)\}',
                read(prefix + 'clocks_post_route.rpt'), re.M)
            clocks_valid[label] = clocks == [('core_clk', '15.000', '0.000 7.500', 'i_clk')] and \
                'No valid timing exceptions found.' in read(prefix + 'exceptions_post_route.rpt')
        tcl = fields(read('vivado_status.txt'))
        before = fields(read('before_checkpoint_status.txt'))
        reopened = fields(read('checkpoint_reopen_status.txt'))
        protected = [[line for line in read('protected_primitives_' + label + '.rpt').splitlines()
            if line and '=' not in line] for label in ('pre_impl', 'post_impl')]
        traces = {}
        for label in ('pre_impl', 'post_impl'):
            text = read('trace_payload_control_' + label + '.rpt')
            controls = [line for line in text.splitlines() if line.startswith('cell=')]
            traces[label] = dict(cells=len(controls), all_gnd=bool(controls) and
                all(line.endswith('drivers=GND') for line in controls),
                zero_live_controls='trace_payload_nonconstant_control_count=0' in text)
        mapped = parse('utilization_hierarchical_post_route.rpt', hierarchy) or []
        arrays = [row['path'] for row in mapped if row['module'] == 'unified_dual_role_h40_stationary_array12_candidate_v1']
        tiles = [row['path'] for row in mapped if re.fullmatch(r'g_tile\[\d+\]\.u_tile', row['name'])]
        diagnostics = {name: parse(name + '_post_route.rpt', violations) for name in ('drc', 'methodology')}
        checkpoints = {name: checkpoint_record(report / name)
            for name in ('post_place_physopt.dcp', 'post_route_physopt.dcp')}
        timing_text = read('timing_summary_post_route.rpt')
        internal = {key: int(m[1]) if (m := re.search(r'checking ' + key + r' \((\d+)\)', timing_text)) else None
            for key in ('no_clock', 'unconstrained_internal_endpoints', 'loops', 'no_input_delay', 'no_output_delay')}
        checks = dict(tool_exit_errors_and_critical_warnings=stage['exit'] == 0 and not parsed.get('errors') and
                not parsed.get('critical_warnings'),
            tcl_acceptance=tcl.get('status') == 'pass',
            reopened_counts_timing_protection_trace=reopened_equal(before, reopened),
            delivered_dcp_crcs=all(row.get('crc_pass') is True for row in checkpoints.values()),
            exported_and_reopened_reports_equal=reports['exported'] == reports['reopened'],
            clock_15ns_no_exceptions=all(clocks_valid.values()),
            protected_primitive_set=len(protected[0]) == len(set(protected[0])) == 15 and protected[0] == protected[1],
            trace_payload_controls=all(row['cells'] == 6270 and row['all_gnd'] and row['zero_live_controls'] for row in traces.values()),
            one_array_twelve_tiles=len(arrays) == 1 and len(tiles) == 12,
            internal_constraints=all(internal[key] == 0 for key in ('no_clock', 'unconstrained_internal_endpoints', 'loops')),
            no_severe_drc_methodology=all(row is not None and all(rule['severity'] == 'Warning' for rule in row['rules']) for row in diagnostics.values()),
            reports_present_and_parsed=not missing and not errors)
        for label, row in reports.items():
            checks[label + '_setup_hold_pulse_complete_route'] = bool(row['timing'] and row['route'] and
                report_checks_pass(row['timing'], row['route']))
        worst = {key: m[1] if (m := re.search(r'^\s*' + key + r':\s*(.+)$',
            read('timing_setup_paths_post_route.rpt'), re.M)) else None
            for key in ('Source', 'Destination', 'Data Path Delay', 'Logic Levels', 'Requirement')}
        arms[arm] = dict(accepted_internal_implementation=all(value is True for value in checks.values()),
            checks=checks, reports=reports, worker=parsed, checkpoints=checkpoints, protected_primitives=protected[0],
            trace_controls=traces, array_paths=arrays, tile_paths=tiles, diagnostics=diagnostics,
            internal_constraint_counts=internal, worst_setup=worst, missing_reports=missing, parse_errors=errors)
    pair_complete = set(arms) == set(ARMS) and all(row['worker']['measurement_complete'] for row in arms.values())
    both_accepted = pair_complete and all(row['accepted_internal_implementation'] for row in arms.values())
    result = dict(status='PASS_paired_implementation_archive' if pair_complete else 'ARCHIVED_incomplete_paired_implementation',
        both_internal_implementations_accepted=both_accepted, pair_measurements_complete=pair_complete,
        source_dcp_metadata_crc_preserved=source_checks, prepared_inputs_byte_equal=len(inputs),
        prepared_bytes=sum(p.stat().st_size for p in inputs), actual_worker_strict_reparse=True,
        raw_files=len(observation['files']), raw_bytes=sum(observation['files'].values()), all_sizes_match=True,
        no_part=True, scratch_missing=[], start=state['start'], end=state['end'], arms=arms,
        remote_observation=observation,
        boundary='Paired producer-plus-core internal OOC implementation with unchanged 15ns and original recipe. Archive PASS is distinct from timing acceptance. No board frequency, power, bitstream, new functional regression or production merge.')
    if pair_complete:
        result['candidate_minus_baseline'] = {
            category: {key: arms['candidate']['reports']['exported'][category][key] -
                arms['baseline']['reports']['exported'][category][key]
                for key in arms['baseline']['reports']['exported'][category]}
            for category in ('resources', 'timing')
            if all(arms[arm]['reports']['exported'][category] for arm in ARMS)}
    shutil.copy2(__file__, archive / 'terminal_audit_runner.py')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'pair_measurements_complete',
        'both_internal_implementations_accepted', 'raw_files', 'raw_bytes', 'prepared_inputs_byte_equal')}, indent=2))


if __name__ == '__main__':
    audit()
