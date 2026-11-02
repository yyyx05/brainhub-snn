"""Two-arm focused reducer retirement test, not a whole-core performance run."""
import collections
import datetime as dt
import difflib
import json
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NATIVE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path('\\\\?\\' + str(NATIVE_ROOT)) if sys.platform == 'win32' else NATIVE_ROOT
EXPERIMENT = ROOT / 'verification/experiments/gradient_retire_r1'
REL_OUT = Path('results/paper_evidence/supplement_20261015/performance_optimization/gradient_retire_local_r1')
OUT = ROOT / REL_OUT
MODULE = 'unified_srnn_tp_wout_gradient_reducer.sv'
BATCHES = (1, 2, 3, 8, 16, 32, 64, 255)


def rne20(x):
    q, r = divmod(abs(x), 1 << 20)
    q += r > (1 << 19) or (r == (1 << 19) and q % 2 == 1)
    return -q if x < 0 else q


def fixture():
    rng = random.Random(2026101920)
    count, overflow_outputs = 0, 0
    with (OUT / 'vectors.hex').open('x', encoding='ascii') as f:
        for batch in BATCHES:
            for addr in range(20):
                values = []
                for sample in range(batch):
                    if addr < 8:
                        targets = [-524289, -524288, -524287, 524287, 524288, 524289, 1572864, -1572864]
                        error, trace = (targets[addr] if sample == batch - 1 else 0), 1
                    elif addr == 8:
                        error, trace = -(1 << 63), -(1 << 31)
                    elif addr == 9:
                        error, trace = (1 << 63) - 1, -(1 << 31)
                    elif addr == 10:
                        error, trace = ((1 << 62) if sample % 2 else -(1 << 62)), (1 << 30)
                    elif addr == 11:
                        error, trace = 0, (1 << 31) - 1
                    else:
                        error, trace = rng.randrange(-(1 << 63), 1 << 63), rng.randrange(-(1 << 31), 1 << 31)
                    values.append((error, trace))
                accumulator = sum(e * t for e, t in values)
                assert -(1 << 127) <= accumulator < (1 << 127)
                gradient = rne20(accumulator)
                overflow = not -(1 << 63) <= gradient < (1 << 63)
                overflow_outputs += overflow
                for sample, (error, trace) in enumerate(values):
                    flags = int(addr == 19 and sample == batch - 1) | (int(overflow) << 1)
                    fields = ((addr, 24), (sample, 8), (error, 64), (trace, 32), (gradient, 64), (flags, 4))
                    f.write(''.join(f'{v & ((1 << w)-1):0{w//4}x}' for v, w in fields) + '\n')
                    count += 1
    assert count == 7620
    return dict(seed=2026101920, unique_contributions=count, unique_gradients=160,
                overflow_gradients=overflow_outputs, batch_counts=BATCHES, elements_per_case=20,
                reference='Independent Python unlimited integers, signed product/sum and ties-to-even divide 2^20; RTL low64 output and sticky overflow checked.',
                scope='Synthetic legal full-width reducer inputs, including B255; not the narrower M2 physical trace bound or a new dataset.')


def parse(text, arm):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', text):
        raise ValueError('runtime diagnostic')
    rows = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
            for line in re.findall(r'^GRADIENT_RETIRE_CASE (.*)$', text, re.M)]
    if [(x['arm'], x['batch'], x['mode']) for x in rows] != [(arm, b, s) for b in BATCHES for s in range(3)]:
        raise ValueError('missing or unordered case')
    if any(x['inputs'] != x['batch'] * 20 or x['outputs'] != 20 for x in rows):
        raise ValueError('transaction count')
    if len(re.findall(r'^GRADIENT_RETIRE_DIRECTED arm=' + str(arm) + r' cases=9$', text, re.M)) != 1:
        raise ValueError('directed cases absent')
    if len(re.findall(r'^GRADIENT_RETIRE_PASS arm=' + str(arm) + r' cases=24 directed=9 inputs=22860 outputs=480$', text, re.M)) != 1:
        raise ValueError('completion absent')
    if sum(x['output_stalls'] for x in rows) == 0:
        raise ValueError('output hold not exercised')
    if (arm == 0 and any(x['overlaps'] for x in rows)) or (arm == 1 and not any(x['overlaps'] for x in rows)):
        raise ValueError('retirement overlap coverage')
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    state = dict(start=dt.datetime.now().astimezone().isoformat(), status='preparing',
                 scope='Isolated reducer same-slot retirement overlap. Unchanged multiplication, accumulator, finalization register, rounding, overflow, terminal drain. No new credit or buffer. Synthetic local unit test, not whole-core throughput or STA.')
    def save():
        (OUT / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    protected = [ROOT / 'rtl/learning' / MODULE,
                 ROOT / 'verification/experiments/srnn_m2_refill_turnover_core_r1/rtl/learning' / MODULE]
    before = [(p, p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
    try:
        (EXPERIMENT / 'original').mkdir(exist_ok=False)
        shutil.copy2(protected[1], EXPERIMENT / 'original' / MODULE)
        shutil.copy2(Path(__file__).with_name('tb_gradient_retire.sv'), EXPERIMENT / 'tb_gradient_retire.sv')
        old = (EXPERIMENT / 'original' / MODULE).read_text()
        new = (EXPERIMENT / 'candidate' / MODULE).read_text()
        (EXPERIMENT / 'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='original/' + MODULE, tofile='candidate/' + MODULE)))
        for p in EXPERIMENT.rglob('*'):
            if p.is_file():
                dest = OUT / p.relative_to(EXPERIMENT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        shutil.copy2(__file__, OUT / 'runner.py')
        state['fixture'] = fixture()
        linux = '/mnt/' + str(NATIVE_ROOT)[0].lower() + (str(NATIVE_ROOT)[2:] + '/' + REL_OUT.as_posix()).replace('\\', '/')
        stages = [('environment', 'verilator --version')]
        for arm, name in enumerate(('original', 'candidate')):
            stages.append((name + '_build', f'/usr/bin/time -v -o {name}_build_resources.txt verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_gradient_retire --Mdir {name}_obj {name}/{MODULE} tb_gradient_retire.sv'))
            stages.append((name + '_run', f'/usr/bin/time -v -o {name}_run_resources.txt {name}_obj/Vtb_gradient_retire +ARM={arm} +VECTORS=vectors.hex'))
        for stage, command in stages:
            state['status'] = stage + '_running'; save()
            # Linux owns both descriptors; WSL notices remain in separate raw launch logs.
            cmd = ['wsl', '-e', 'sh', '-lc', 'cd ' + shlex.quote(linux) + ' && ulimit -v 3145728 && ' + command + ' > ' + stage + '.log 2>&1']
            begin = time.monotonic()
            with (OUT / (stage + '_launcher.log')).open('xb') as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state[stage] = dict(pid=process.pid, command=cmd, exit=None); save()
                try:
                    code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    state[stage]['timeout'] = True; save(); raise
            state[stage].update(exit=code, wall_seconds=time.monotonic() - begin); save()
            if code:
                raise RuntimeError(stage + ' failed')
            log = (OUT / (stage + '.log')).read_text(errors='replace')
            if stage.endswith('_build'):
                categories = collections.Counter(re.findall(r'^%Warning-([^:]+):', log, re.M))
                state[stage]['diagnostics'] = dict(categories=categories, ordinary=sum(categories.values()),
                    structural=[k for k in categories if k in ('LATCH', 'MULTIDRIVEN', 'UNOPTFLAT')],
                    errors=len(re.findall(r'^%Error', log, re.M)))
                if state[stage]['diagnostics']['structural'] or state[stage]['diagnostics']['errors']:
                    raise ValueError('build diagnostics rejected')
            elif stage.endswith('_run'):
                state[stage]['parsed'] = parse(log, int(stage.startswith('candidate')))
        paired = []
        for a, b in zip(state['original_run']['parsed'], state['candidate_run']['parsed']):
            assert (a['batch'], a['mode'], a['inputs'], a['outputs']) == (b['batch'], b['mode'], b['inputs'], b['outputs'])
            paired.append(dict(batch=a['batch'], mode=a['mode'], baseline_cycles=a['cycles'], candidate_cycles=b['cycles'],
                               saved_cycles=a['cycles'] - b['cycles'], candidate_overlaps=b['overlaps']))
        log = (OUT / 'candidate_run.log').read_text()
        for bad in (log.replace('GRADIENT_RETIRE_CASE', 'MISSING', 1),
                    log.replace('cases=9', 'cases=8', 1), log.replace('GRADIENT_RETIRE_PASS', 'MISSING'),
                    log + '\n%Error: synthetic\n'):
            try:
                parse(bad, 1)
            except ValueError:
                pass
            else:
                raise AssertionError('damaged transcript accepted')
        archived = [p for p in EXPERIMENT.rglob('*') if p.is_file()]
        if any(p.read_bytes() != (OUT / p.relative_to(EXPERIMENT)).read_bytes() for p in archived):
            raise ValueError('source archive differs')
        if any((p.stat().st_size, p.stat().st_mtime_ns) != (size, mtime) for p, size, mtime in before):
            raise ValueError('protected input changed')
        if list(OUT.rglob('*.part')):
            raise ValueError('partial evidence')
        state.update(status='PASS_reducer_retirement_overlap_local', comparisons=paired,
            damaged_transcripts_rejected=4, source_archive_items_byte_equal=len(archived), protected_inputs_stat_preserved=2,
            dcp_crc='not applicable: no DCP', counters='Per-case cycle origin reset in both arms; identical externally defined supply/backpressure schedule.')
    except Exception as error:
        state.update(status='FAIL', error=repr(error))
        raise
    finally:
        state['end'] = dt.datetime.now().astimezone().isoformat(); save()
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
