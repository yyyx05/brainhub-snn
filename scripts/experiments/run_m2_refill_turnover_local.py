"""Bounded real publisher/slice/router/store two-arm RTL experiment. No SSH.

prepare copies accepted inputs; run snapshots sources and runs one new attempt.
Never overwrite an attempt. A timeout leaves the process alive for inspection.
"""
import argparse
from collections import Counter
import datetime as dt
import difflib
import filecmp
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NORMAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path('\\\\?\\' + str(NORMAL_ROOT)) if sys.platform == 'win32' else NORMAL_ROOT
BASE = ROOT / 'verification/experiments/srnn_rmw_turnover_core_r1/rtl'
EXP = ROOT / 'verification/experiments/m2_refill_turnover_r1'
OUT = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization/m2_refill_turnover_local_r1'
PUB = 'unified_h40_scalar_parameter_row_publisher_v1'
SLICE = 'unified_srnn_m2_forward_learning_store_slice'
ROUTER = 'unified_tp_three_client_parameter_backend_router'
STORE = 'unified_tp_shared_forward_rmw_store'
NAMES = [PUB, 'unified_array16_weight_refill_dispatcher_v1', SLICE, ROUTER, STORE,
         'unified_tp_train_infer_version_boundary',
         'unified_srnn_tp_wout_direct_online_learning_engine',
         'unified_srnn_tp_wout_gradient_reducer', 'unified_fc_tp_gradient_rmw_path',
         'unified_fc_tp_gradient_request_adapter', 'unified_tp_q015_magnitude_scaler',
         'unified_tp_parameter_rmw_service', 'unified_tp_active_tensor_update_path',
         'unified_tp_active_tensor_context', 'unified_tp_parameter_update_adapter',
         'unified_tp_direct_residual_update', 'unified_tp_q015_magnitude_residual_update']
TB = 'tb_m2_refill_turnover.sv'


def now():
    return dt.datetime.now().astimezone().isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='ascii')


def meta(path):
    st = path.stat()
    return dict(path=str(path.relative_to(ROOT)), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def linux(path):
    s = str(path).removeprefix('\\\\?\\')
    return '/mnt/' + s[0].lower() + s[2:].replace('\\', '/')


def instance(name, label, parameters, connections):
    """Enumerate ANSI ports, including comma-grouped router ports, explicitly.

    Inactive input ports are tied to zero; unused outputs are explicitly open.
    This generates only TB wiring, never changes the real module sources.
    """
    source = (EXP / 'original' / (name + '.sv')).read_text()
    header = source.split(') (', 1)[1].split(');', 1)[0]
    header = re.sub(r'//[^\n]*', '', header)
    ports = []
    direction = None
    for part in header.split(','):
        part = part.strip()
        match = re.match(r'(input|output)\b', part)
        if match:
            direction = match[1]
        port = re.search(r'\b([io]_\w+)\s*$', part)
        if not port or direction is None:
            raise ValueError('port parse: ' + part)
        port = port[1]
        ports.append(port)
        default = "'0" if direction == 'input' else ''
        if port not in connections:
            connections[port] = default
    if set(connections) != set(ports):
        raise ValueError('unknown TB port: ' + str(set(connections) - set(ports)))
    return (name + ' #(\n    ' + ',\n    '.join('.%s(%s)' % p for p in parameters.items()) +
            '\n) ' + label + ' (\n    ' +
            ',\n    '.join('.%s(%s)' % (p, connections[p]) for p in ports) + '\n);')


def harness():
    text = (ROOT / 'scripts/experiments' / TB).read_text()
    common = dict(i_clk='clk', i_rst_n='rst_n', i_clear_error='clear_error')
    pub = dict(common, i_abort='abort_pub', i_command_valid='cmd_valid', o_command_ready='cmd_ready',
               i_target_tile='tile', i_target_bank='bank', i_target_row='row', i_scalar_base='base',
               i_scalar_stride='stride', i_scalar_count='count', i_element_format='fmt', i_version='version',
               i_array_idle='array_idle', o_forward_read_valid='p_req', i_forward_read_ready='p_ready',
               o_forward_read_addr='p_addr', o_forward_read_tag='p_tag',
               i_forward_response_valid='p_rsp', o_forward_response_ready='p_take',
               i_forward_response_tag='c_rsp_tag', i_forward_response_code='c_rsp_code',
               i_forward_response_error='c_rsp_error', o_weight_fill_valid='fill_valid',
               i_weight_fill_ready='fill_ready', o_weight_fill_tile_flat='fill_tile',
               o_weight_fill_bank='fill_bank', o_weight_fill_addr_flat='fill_addr',
               o_weight_fill_data_flat='fill_data', o_done='pub_done', o_busy='pub_busy',
               o_sticky_error='pub_error', o_published_tile='published_tile',
               o_published_bank='published_bank', o_published_row='published_row',
               o_published_version='published_version')
    text = text.replace('@@PUBLISHER@@', instance(PUB, 'publisher',
        dict(PARAM_DEPTH='512', PARAM_ADDR_WIDTH='10', TAG_WIDTH='8', VERSION_WIDTH='16'), pub))
    sl = dict(common, i_cfg_valid='cfg_valid', o_cfg_ready='cfg_ready', o_cfg_error='cfg_error',
              i_cfg_profile_id="8'd2", i_cfg_wh1_tensor_id="8'd1", i_cfg_wh1_base="10'd0",
              i_cfg_wh1_increment="16'd31", i_cfg_wh1_threshold="16'd65",
              i_cfg_wout_tensor_id="8'd2", i_cfg_wout_base="10'd128",
              i_cfg_wout_increment="16'd31", i_cfg_wout_threshold="16'd65",
              i_cfg_code_min="16'sh8000", i_cfg_code_max="16'sh7fff", i_cfg_batch_count="8'd1",
              i_cfg_hidden_count="9'd1", i_cfg_backbone_count="10'd1", i_cfg_wout_element_count="10'd40",
              i_load_valid='load_valid', o_load_ready='load_ready', i_load_addr='load_addr',
              i_load_code='load_code', i_forward_start='forward_start', o_forward_start_ready='forward_ready',
              i_forward_consumer_done_valid='forward_end', o_forward_consumer_done_ready='forward_end_ready',
              o_forward_done='forward_done', o_forward_error='forward_error',
              i_cache_read_valid='c_req', o_cache_read_ready='c_ready', i_cache_read_addr='c_addr',
              i_cache_read_tag='c_tag', o_cache_response_valid='c_rsp', i_cache_response_ready='c_take',
              o_cache_response_tag='c_rsp_tag', o_cache_response_code='c_rsp_code',
              o_cache_response_error='c_rsp_error', o_error='slice_error', o_busy='slice_busy',
              o_learning_start_ready='learning_ready', o_model_version='model_version',
              o_adapt_version='adapt_version', o_forward_model_version='forward_model_version',
              o_forward_adapt_version='forward_adapt_version',
              o_store_load_valid='s_load', i_store_load_ready='r_load_ready[1]',
              o_store_load_addr='s_load_addr', o_store_load_code='s_load_code', o_store_load_residual='s_load_resid',
              o_store_forward_read_valid='s_req', i_store_forward_read_ready='s_ready',
              o_store_forward_read_addr='s_addr', o_store_forward_read_tag='s_tag',
              i_store_forward_response_valid='s_rsp', o_store_forward_response_ready='s_take',
              i_store_forward_response_tag='s_rsp_tag', i_store_forward_response_code='s_rsp_code',
              i_store_forward_response_error='s_rsp_error', i_store_protocol_error='router_error',
              o_store_learning_read_valid='unexpected_learning_read',
              o_store_learning_write_valid='unexpected_learning_write')
    text = text.replace('@@SLICE@@', instance(SLICE, 'slice', dict(USE_INTERNAL_STORE="1'b0",
        USE_RANDOM_ACCESS_FORWARD="1'b1", ENABLE_WH1_LEARNING="1'b0", STORE_DEPTH='512',
        BATCH_MAX='2', HIDDEN_MAX='2', BACKBONE_MAX='2', PARAM_ADDR_WIDTH='10', TAG_WIDTH='8'), sl))
    rt = dict(common, i_owner_active='owner_active', i_active_owner='owner',
              i_load_valid="{1'b0,s_load,1'b0}", o_load_ready='r_load_ready',
              i_load_addr="{10'd0,s_load_addr,10'd0}", i_load_code="{16'd0,s_load_code,16'd0}",
              i_load_residual="{16'd0,s_load_resid,16'd0}",
              i_forward_read_valid="{1'b0,(s_req && request_gate),1'b0}",
              o_forward_read_ready='r_ready', i_forward_read_addr="{10'd0,router_addr,10'd0}",
              i_forward_read_tag="{8'd0,s_tag,8'd0}", o_forward_response_valid='r_rsp',
              i_forward_response_ready="{1'b0,(s_take && response_gate),1'b0}",
              o_forward_response_tag='r_rsp_tag', o_forward_response_code='r_rsp_code',
              o_forward_response_error='r_rsp_error', o_quiescent='router_quiet', o_protocol_error='router_error',
              o_store_load_valid='b_load', i_store_load_ready='b_load_ready', o_store_load_addr='b_load_addr',
              o_store_load_code='b_load_code', o_store_load_residual='b_load_resid',
              o_store_forward_read_valid='b_req', i_store_forward_read_ready='b_ready',
              o_store_forward_read_addr='b_addr', o_store_forward_read_tag='b_tag',
              i_store_forward_response_valid='b_rsp', o_store_forward_response_ready='b_take',
              i_store_forward_response_tag='b_rsp_tag', i_store_forward_response_code='b_rsp_code',
              i_store_forward_response_error='b_rsp_error', i_store_protocol_error='store_error')
    text = text.replace('@@ROUTER@@', instance(ROUTER, 'router', dict(PARAM_ADDR_WIDTH='10', TAG_WIDTH='8'), rt))
    st = dict(common, i_load_valid='b_load', o_load_ready='b_load_ready', i_load_addr='b_load_addr',
              i_load_code='b_load_code', i_load_residual='b_load_resid', i_forward_read_valid='b_req',
              o_forward_read_ready='b_ready', i_forward_read_addr='b_addr', i_forward_read_tag='b_tag',
              o_forward_response_valid='b_rsp', i_forward_response_ready='b_take',
              o_forward_response_tag='b_rsp_tag', o_forward_response_code='b_rsp_code',
              o_forward_response_error='b_rsp_error', o_protocol_error='store_error')
    text = text.replace('@@STORE@@', instance(STORE, 'shared_store',
        dict(DEPTH='512', PARAM_ADDR_WIDTH='10', TAG_WIDTH='8'), st))
    if '@@' in text:
        raise ValueError('unexpanded TB placeholder')
    return text


def prepare():
    EXP.mkdir(exist_ok=False)
    OUT.mkdir(exist_ok=False)
    for arm in ('original', 'candidate'):
        (EXP / arm).mkdir()
    paths = {p.stem: p for p in BASE.rglob('*.sv')}
    records = []
    for name in NAMES:
        src = paths[name]
        records.append(meta(src))
        for arm in ('original', 'candidate'):
            shutil.copy2(src, EXP / arm / src.name)
    dump(EXP / 'source_inputs.json', records)
    (EXP / TB).write_text(harness(), encoding='ascii', newline='\n')
    print('Prepared original and candidate source copies; candidate edits must precede run.')


def stage(directory, name, executable):
    script = directory / (name + '.sh')
    if script.exists():
        raise FileExistsError(script)
    script.write_text('#!/bin/sh\ncd ' + shlex.quote(linux(directory)) + ' || exit 125\n' +
        'exec >' + name + '.log 2>&1\nprintf "%s\\n" "$$" >' + name + '.pid\n' +
        'ulimit -v 3145728 || exit 125\n' +
        ('verilator --version\nuname -a\n' if name == 'build' else '') +
        '/usr/bin/time -v -o ' + name + '_resources.txt ' + executable + '\n' +
        'rc=$?\nprintf "%s\\n" "$rc" >' + name + '.exit\nexit "$rc"\n', encoding='ascii', newline='\n')
    state = dict(start=now(), linux_command=executable, status='running', memory_limit_kib=3145728)
    begin = time.monotonic()
    with (directory / (name + '_wsl_console.log')).open('xb') as f:
        process = subprocess.Popen(['wsl', '-e', 'sh', linux(script)], stdin=subprocess.DEVNULL,
                                   stdout=f, stderr=subprocess.STDOUT)
        state['windows_pid'] = process.pid
        dump(directory / (name + '.json'), state)
        try:
            state['wsl_exit'] = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            state.update(status='timeout_process_left_running_no_autokill', end=now())
            dump(directory / (name + '.json'), state)
            raise RuntimeError('Timeout: leave process running; inspect PID before any further execution')
    state.update(linux_pid=int((directory / (name + '.pid')).read_text()),
                 linux_exit=int((directory / (name + '.exit')).read_text()),
                 wall_seconds=time.monotonic()-begin, end=now(), status='completed')
    dump(directory / (name + '.json'), state)
    print(json.dumps(state), flush=True)
    if state['linux_exit'] != 0 or state['wsl_exit'] != 0:
        raise RuntimeError('Failed stage retained: ' + str(directory / name))
    return state


def parse(text, arm):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', text):
        raise ValueError('RTL error in transcript')
    lines = re.findall(r'^M2_REFILL_RESULT (.+)$', text, re.M)
    rows = [{k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)} for line in lines]
    if len(rows) != 36 or len(re.findall(r'^M2_REFILL_PASS$', text, re.M)) != 1:
        raise ValueError('missing cases/terminal marker')
    keys = [(r['mode'], r['format'], r['variant']) for r in rows]
    if keys != [(m, f, v) for m in range(3) for f in range(3) for v in range(4)]:
        raise ValueError('duplicate or reordered cases')
    for r in rows:
        if r['arm'] != arm or r['reads'] != r['count'] or r['responses'] != r['count'] or r['fills'] != 1 or r['max_credit'] != 1:
            raise ValueError('numeric/credit accounting')
        if r['cycles'] <= 0:
            raise ValueError('cycle count')
    faults = re.findall(r'^M2_REFILL_FAULT id=(\d+) pass=1$', text, re.M)
    if faults != [str(i) for i in range(1, 9)]:
        raise ValueError('fault scenarios')
    if not re.search(r'^M2_REFILL_CACHE held=1 owner=1 boundary=1 reset=1 pass=1$', text, re.M):
        raise ValueError('real slice/cache scenarios')
    return rows


def run(attempt):
    directory = OUT / attempt
    directory.mkdir(exist_ok=False)
    # Mechanically regenerate explicit TB pin connections before snapshotting.
    (EXP / TB).write_text(harness(), encoding='ascii', newline='\n')
    diff = ''
    changed = []
    for name in NAMES:
        a = (EXP / 'original' / (name + '.sv')).read_text()
        b = (EXP / 'candidate' / (name + '.sv')).read_text()
        if a != b:
            changed.append(name)
        diff += ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                     fromfile='original/'+name+'.sv', tofile='candidate/'+name+'.sv'))
    if set(changed) != {PUB, SLICE}:
        raise ValueError('two-module change boundary: ' + str(changed))
    (EXP / 'changes.diff').write_text(diff, encoding='ascii')
    shutil.copytree(EXP, directory / 'sources')
    shutil.copy2(Path(__file__), directory / Path(__file__).name)
    shutil.copy2(ROOT / 'scripts/experiments' / TB, directory / ('template_' + TB))
    dump(directory / 'status.json', dict(status='running', start=now()))
    all_rows, all_stages, all_warnings = {}, {}, {}
    try:
        for arm_id, arm in enumerate(('original', 'candidate')):
            target = directory / arm
            target.mkdir()
            for name in NAMES:
                shutil.copy2(directory / 'sources' / arm / (name + '.sv'), target / (name + '.sv'))
            shutil.copy2(directory / 'sources' / TB, target / TB)
            command = ('verilator --binary --timing --assert -j 1 -Wall -Wno-fatal '
                       '--flatten --output-split 3000 -CFLAGS -O1 '
                       '--top-module tb_m2_refill_turnover --Mdir obj ' +
                       ' '.join(n + '.sv' for n in NAMES) + ' ' + TB)
            build = stage(target, 'build', command)
            buildlog = (target / 'build.log').read_text()
            if re.search(r'%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)|%Error', buildlog):
                raise ValueError('structural build diagnostic, no simulation')
            result = stage(target, 'run', 'obj/Vtb_m2_refill_turnover +arm=' + str(arm_id))
            transcript = (target / 'run.log').read_text()
            all_rows[arm] = parse(transcript, arm_id)
            all_stages[arm] = dict(build=build, run=result)
            all_warnings[arm] = dict(Counter(re.findall(r'%Warning-(\w+)', buildlog)))
        damaged = 0
        for bad in (transcript.replace('M2_REFILL_PASS', 'MISSING'),
                    transcript.replace('max_credit=1', 'max_credit=2', 1),
                    transcript.replace('M2_REFILL_FAULT id=1 pass=1', 'MISSING'),
                    transcript + '\n%Error: synthetic\n'):
            try:
                parse(bad, 1)
            except ValueError:
                damaged += 1
            else:
                raise ValueError('damaged transcript accepted')
        source_inputs = json.loads((EXP / 'source_inputs.json').read_text())
        for entry in source_inputs:
            src = ROOT / entry['path']
            if meta(src) != entry or not filecmp.cmp(src, EXP / 'original' / src.name, shallow=False):
                raise ValueError('accepted original source changed')
        checked = 0
        for p in (directory / 'sources').rglob('*'):
            if p.is_file():
                if not filecmp.cmp(p, EXP / p.relative_to(directory / 'sources'), shallow=False):
                    raise ValueError('prepared input changed during execution')
                checked += 1
        for arm in ('original', 'candidate'):
            for name in NAMES + [TB.removesuffix('.sv')]:
                relative = name + '.sv'
                src = directory / 'sources' / (arm if relative != TB else '') / relative
                if not filecmp.cmp(src, directory / arm / relative, shallow=False):
                    raise ValueError('compiled source mismatch')
        comparison = []
        for old, new in zip(all_rows['original'], all_rows['candidate']):
            for key in ('mode', 'format', 'variant', 'count', 'reads', 'responses', 'fills'):
                if old[key] != new[key]:
                    raise ValueError('two-arm accounting mismatch')
            comparison.append(dict(mode=old['mode'], format=old['format'], variant=old['variant'],
                original_cycles=old['cycles'], candidate_cycles=new['cycles'],
                saved=old['cycles']-new['cycles'], original=old, candidate=new))
        summary = dict(status='PASS_m2_refill_turnover_local_two_arm', time=now(), attempt=attempt,
            comparisons=comparison, stages=all_stages, warnings=all_warnings,
            damaged_transcripts_rejected=damaged, original_source_stats_and_bytes_unchanged=True,
            archived_prepared_files_checked=checked, changed_modules=changed,
            limitations=['Small real publisher/slice/router/store chain, not whole-core.',
                'No synthesis, STA, physical timing, board, learning accuracy or workload-distribution claim.',
                'Testbench response-delay and fault injection are explicit, not production components.',
                'Per-command GNU time RSS is not total-machine memory peak. No DCP: CRC not applicable.'])
        dump(directory / 'status.json', dict(status=summary['status'], end=now()))
        dump(OUT / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        dump(directory / 'status.json', dict(status='FAIL_preserved', end=now(), error=repr(exc)))
        raise


def failure_report():
    """One final handoff of natural build outcomes; no compilation or audit rerun."""
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Summary already exists; do not repeat reporting')
    attempts = {}
    for attempt in ('attempt1', 'attempt2'):
        directory = OUT / attempt
        status = json.loads((directory / 'status.json').read_text())
        build = json.loads((directory / 'original/build.json').read_text())
        if status['status'] != 'FAIL_preserved' or build.get('linux_exit') != 0 or build.get('wsl_exit') != 0:
            raise ValueError('Expected natural build success followed by structural rejection')
        if (directory / 'original/run.json').exists() or (directory / 'candidate').exists():
            raise ValueError('Unexpected executed stage')
        log = (directory / 'original/build.log').read_text()
        warnings = Counter(re.findall(r'%Warning-(\w+)', log))
        structural = re.findall(r'^%Warning-(?:UNOPTFLAT|LATCH|MULTIDRIVEN).*$', log, re.M)
        resource_text = (directory / 'original/build_resources.txt').read_text()
        resources = {}
        for line in resource_text.splitlines():
            if ': ' in line:
                key, value = line.strip().rsplit(': ', 1)
                resources[key] = value
        attempts[attempt] = dict(status=status, build=build, warnings=dict(warnings),
            structural=structural, resources=resources,
            errors=len(re.findall(r'%Error|%Fatal', log)), simulation_started=False,
            candidate_compiled=False)
    originals = json.loads((EXP / 'source_inputs.json').read_text())
    for entry in originals:
        source = ROOT / entry['path']
        if meta(source) != entry or not filecmp.cmp(source, EXP / 'original' / source.name, shallow=False):
            raise ValueError('accepted source changed')
    paths = []
    snapshot = OUT / 'attempt2/sources'
    for source in snapshot.rglob('*'):
        if source.is_file():
            rel = source.relative_to(snapshot)
            if not filecmp.cmp(source, EXP / rel, shallow=False):
                raise ValueError('prepared source differs from last build snapshot')
            paths.append(str(rel))
    summary = dict(status='STOPPED_LOCAL_BASELINE_UNOPTFLAT_NO_SIMULATION', time=now(),
        attempt='attempt2', attempts=attempts, prepared_snapshot=str(snapshot.relative_to(ROOT)),
        candidate_paths=[str((EXP/'candidate'/(n+'.sv')).relative_to(ROOT)) for n in (PUB,SLICE)],
        diff_path=str((EXP/'changes.diff').relative_to(ROOT)),
        original_sources_stats_and_bytes_unchanged=len(originals),
        prepared_snapshot_files_equal=paths, rtl_cycles_measured=None,
        read_response_or_publication_counts_measured=None, local_process_pending=False,
        structural_rule_unchanged=True, warnings_suppressed=False,
        limitations=['No simulation ran. Candidate is untested; no functional PASS or cycle benefit.',
            'Both attempts stopped after original-arm UNOPTFLAT diagnostics, not a candidate failure.',
            'Attempt2 flatten/output splitting/host -O1 did not remove the seven original warnings.',
            'Small real-module harness is ready for a different compiler, but its checks are unvalidated.',
            'No SSH, server, Vivado, whole-core, STA, physical or model work performed.',
            'GNU time reports individual command peaks, not machine peak. No DCP: CRC N/A.'])
    dump(OUT / 'summary.json', summary)
    shutil.copy2(Path(__file__), OUT / 'handoff_runner.py')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'failure-report'))
    p.add_argument('--attempt', default='attempt1')
    a = p.parse_args()
    if not re.fullmatch(r'attempt[1-9][0-9]*', a.attempt):
        p.error('attemptN required')
    if a.command == 'prepare':
        prepare()
    elif a.command == 'failure-report':
        failure_report()
    else:
        run(a.attempt)
