"""Isolated direct RMW turnover experiment; no training or baseline rerun.

Commands: prepare, build, directed, replay, report. Run each once in order.
An unsuccessful build may be preserved and replaced by a new --attempt name.
Timeouts leave both the WSL launcher and Linux process alive for inspection.
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
EXP = ROOT / 'verification/experiments/rmw_turnover_r1'
RESULTS = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization'
OUT = RESULTS / 'rmw_turnover_local_r1'
INPUT = RESULTS / 'sgd_rmw_replay_local_r2'
NAMES = ['unified_tp_parameter_rmw_service', 'unified_tp_active_tensor_update_path',
         'unified_tp_parameter_update_adapter', 'unified_tp_active_tensor_context',
         'unified_tp_direct_residual_update', 'unified_tp_q015_magnitude_residual_update']
TBS = ['tb_rmw_turnover_replay.sv', 'tb_rmw_turnover_directed.sv']
ORIGINAL_CYCLES = [8095660, 14286543]  # User-supplied accepted direct results, not rerun.


def now():
    return dt.datetime.now().astimezone().isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='ascii')


def metadata(path):
    stat = path.stat()
    return dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('source patch anchor: ' + before)
    return text.replace(before, after)


def linux_path(path):
    normal = str(path).removeprefix('\\\\?\\')
    return '/mnt/' + normal[0].lower() + normal[2:].replace('\\', '/')


def prepare():
    OUT.mkdir(exist_ok=False)
    EXP.mkdir(exist_ok=False)
    (EXP / 'original').mkdir()
    (EXP / 'candidate').mkdir()
    records = []
    for name in NAMES:
        source = ROOT / 'rtl/learning' / (name + '.sv')
        records.append(metadata(source))
        shutil.copy2(source, EXP / 'original' / source.name)
        shutil.copy2(source, EXP / 'candidate' / source.name)
    source = EXP / 'original' / (NAMES[0] + '.sv')
    original = source.read_text(encoding='ascii')
    candidate = replace_once(original,
        '        o_req_ready = (state_q == RMW_IDLE) &&\n'
        '            !i_cfg_valid && !i_release;\n'
        '        request_fire = i_req_valid && o_req_ready;\n', '')
    candidate = replace_once(candidate,
        '        result_fire = o_result_valid && i_result_ready;',
        '        result_fire = o_result_valid && i_result_ready;\n\n'
        '        // Reuse the single request slot only when the old write retires.\n'
        '        // The replacement read remains in the following cycle.\n'
        '        o_req_ready = ((state_q == RMW_IDLE) || result_fire) &&\n'
        '            !i_cfg_valid && !i_release;\n'
        '        request_fire = i_req_valid && o_req_ready;')
    candidate = replace_once(candidate,
        '            if (result_fire)\n                state_q <= RMW_IDLE;',
        '            // Explicit retirement priority: a legal replacement owns the\n'
        '            // next cycle; rejection still drains the previous result.\n'
        '            if (result_fire)\n'
        '                state_q <= (request_fire && request_address_legal) ?\n'
        '                    RMW_READ_REQUEST : RMW_IDLE;')
    (EXP / 'candidate' / source.name).write_text(candidate, encoding='ascii', newline='\n')
    (EXP / 'changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='original/' + source.name, tofile='candidate/' + source.name)),
        encoding='ascii', newline='\n')
    for tb in TBS:
        shutil.copy2(ROOT / 'scripts/experiments' / tb, EXP / tb)
    shutil.copytree(EXP, OUT / 'sources')
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    inputs = []
    for name in ('vectors.hex', 'initial.hex'):
        src = INPUT / name
        before = metadata(src)
        shutil.copy2(src, OUT / name)
        if not filecmp.cmp(src, OUT / name, shallow=False):
            raise ValueError('input copy differs: ' + name)
        if before != metadata(src):
            raise ValueError('input changed during copy: ' + name)
        inputs.append(dict(source=before, copy=metadata(OUT / name), byte_equal=True))
    write_json(OUT / 'preparation.json', dict(time=now(), inputs=inputs, production_sources=records,
        gradients=3010560, steps=588, nonzero_requests=1017020,
        original_direct_cycles=ORIGINAL_CYCLES,
        reference='Original accepted direct fields in vectors.hex; no SGD execution or reference regeneration',
        timing='No STA measured. Added result_ready-to-req_ready path. Read acceptance follows prior write retirement by at least one clock edge.'))
    print('Prepared isolated source copies and byte-identical existing inputs.', flush=True)


def stage(command, attempt):
    directory = OUT / attempt
    if command == 'build':
        directory.mkdir(exist_ok=False)
        for path in (EXP / 'candidate').glob('*.sv'):
            shutil.copy2(path, directory / path.name)
        for tb in TBS:
            shutil.copy2(EXP / tb, directory / tb)
        shutil.copy2(Path(__file__), directory / Path(__file__).name)
        executable = ('verilator --binary --timing --assert -j 1 -Wall -Wno-fatal '
                      '--top-module tb_rmw_turnover_replay --Mdir obj ' +
                      ' '.join(NAMES[i] + '.sv' for i in range(len(NAMES))) + ' ' + ' '.join(TBS))
    else:
        build = json.loads((directory / 'build.json').read_text())
        if build.get('linux_exit') != 0:
            raise RuntimeError('No successful build in selected attempt')
        executable = 'obj/Vtb_rmw_turnover_replay +' + command
    # Link only to this experiment's unchanged copied fixture, never generate data.
    setup = ''
    if command == 'replay':
        setup = 'ln -s ../vectors.hex vectors.hex\nln -s ../initial.hex initial.hex\n'
    logname = 'build' if command == 'build' else command
    script_path = directory / (logname + '.sh')
    if script_path.exists():
        raise FileExistsError('Preserve existing stage: ' + str(script_path))
    script = ('#!/bin/sh\ncd ' + shlex.quote(linux_path(directory)) + ' || exit 125\n'
              'exec >' + logname + '.log 2>&1\n'
              'printf "%s\\n" "$$" >' + logname + '.pid\n'
              'ulimit -v 3145728 || exit 125\n' + setup)
    if command == 'build':
        script += ('verilator --version\n'
                   'ps -eo pid,ppid,rss,args > environment_processes.txt\n'
                   'uname -a\n')
    script += ('/usr/bin/time -v -o ' + logname + '_resources.txt ' + executable + '\n'
               'rc=$?\nprintf "%s\\n" "$rc" >' + logname + '.exit\nexit "$rc"\n')
    script_path.write_text(script, encoding='ascii', newline='\n')
    cmd = ['wsl', '-e', 'sh', linux_path(script_path)]
    state = dict(start=now(), command=cmd, linux_command=executable, memory_limit_kib=3145728,
                 status='running', windows_pid=None, linux_pid=None, linux_exit=None, wsl_exit=None)
    state_path = directory / (logname + '.json')
    begin = time.monotonic()
    # WSL launcher messages never share a file descriptor with the Linux log.
    with (directory / (logname + '_wsl_console.log')).open('xb') as console:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
        state['windows_pid'] = process.pid
        write_json(state_path, state)
        try:
            state['wsl_exit'] = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            state.update(status='timeout_process_left_running_no_autokill', end=now(),
                         wall_seconds=time.monotonic() - begin)
            pid_file = directory / (logname + '.pid')
            if pid_file.exists():
                state['linux_pid'] = int(pid_file.read_text().strip())
            write_json(state_path, state)
            print(json.dumps(state, indent=2), flush=True)
            return 2
    for field, suffix in (('linux_pid', '.pid'), ('linux_exit', '.exit')):
        path = directory / (logname + suffix)
        if path.exists():
            state[field] = int(path.read_text().strip())
    state.update(status='completed', end=now(), wall_seconds=time.monotonic() - begin)
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if state['linux_exit'] == state['wsl_exit'] == 0 else 1


def rows(log, prefix):
    return [{key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)}
            for line in re.findall('^' + prefix + r' (.+)$', log, re.M)]


def parse_replay(log):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', log):
        raise ValueError('RTL diagnostic')
    measured = rows(log, 'RMW_TURNOVER_RESULT')
    if [(r.get('policy'), r.get('scenario')) for r in measured] != [(0, 0), (0, 1)]:
        raise ValueError('replay scenarios')
    if len(re.findall(r'^RMW_TURNOVER_REPLAY_PASS$', log, re.M)) != 1:
        raise ValueError('replay terminal marker')
    expected = dict(words=3010560, requests=1017020, reads=1017020, responses=1017020,
                    results=1017020, zero=1993540, code_writes=481275,
                    residual_writes=1017020, saturated=0, state_checks=3015680)
    for row in measured:
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('state/write/credit counts')
        if row.get('turnovers', 0) <= 0 or row.get('cycles', 0) <= 0:
            raise ValueError('missing turnover measurement')
        if row['scenario'] == 1 and (row.get('read_stalls', 0) <= 0 or row.get('result_stalls', 0) <= 0):
            raise ValueError('missing stalls')
    return measured


def report(attempt):
    directory = OUT / attempt
    stages = {s: json.loads((directory / (s + '.json')).read_text()) for s in ('build', 'directed', 'replay')}
    if any(v.get('linux_exit') != 0 or v.get('wsl_exit') != 0 for v in stages.values()):
        raise ValueError('stage exit not zero')
    log = (directory / 'replay.log').read_text()
    measured = parse_replay(log)
    for bad in (log.replace('RMW_TURNOVER_REPLAY_PASS', 'MISSING'),
                log.replace('RMW_TURNOVER_RESULT', 'MISSING', 1),
                log.replace('reads=1017020', 'reads=0', 1), log + '\n%Error: synthetic\n'):
        try:
            parse_replay(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt transcript accepted')
    directed = (directory / 'directed.log').read_text()
    drows = rows(directed, 'RMW_DIRECTED_RESULT')
    if (len(drows) != 1 or drows[0].get('cases') != 9 or drows[0].get('max_credit') != 1 or
            len(re.findall(r'^RMW_DIRECTED_CASE ', directed, re.M)) != 9 or
            len(re.findall(r'^RMW_TURNOVER_DIRECTED_PASS$', directed, re.M)) != 1 or
            re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', directed)):
        raise ValueError('directed transcript')
    inputs = json.loads((OUT / 'preparation.json').read_text())['inputs']
    for item in inputs:
        for label in ('source', 'copy'):
            if metadata(ROOT / item[label]['path']) != item[label]:
                raise ValueError('input size/mtime changed')
    for row, old in zip(measured, ORIGINAL_CYCLES):
        row.update(original_cycles=old, saved_cycles=old - row['cycles'],
                   cycle_reduction_percent=100 * (old - row['cycles']) / old,
                   service_speedup=old / row['cycles'])
    build_log = (directory / 'build.log').read_text()
    warnings = Counter(re.findall(r'%Warning-([A-Z0-9_]+)', build_log))
    summary = dict(status='PASS_isolated_direct_service_only', time=now(), attempt=attempt,
                   replay=measured, directed=drows[0], stages=stages,
                   warnings=dict(warnings), damaged_transcripts_rejected=4,
                   inputs_size_mtime_unchanged=True,
                   limitations=['No whole-core measurement, integration acceptance, synthesis, STA, CDC or formal signoff.',
                                'Added result_ready-to-req_ready combinational path is unmeasured; no inherited A STA.',
                                'Single credit only. Previous write must commit on retirement edge; next read is in following cycle.',
                                'Reuses original direct oracle fields; no SGD, gradients, reference or accuracy regeneration.'])
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'build', 'directed', 'replay', 'report'))
    parser.add_argument('--attempt', default='build_r1')
    args = parser.parse_args()
    if not re.fullmatch(r'build_r[1-9][0-9]*', args.attempt):
        parser.error('attempt must be build_rN')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'report':
        report(args.attempt)
    else:
        return stage(args.command, args.attempt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
