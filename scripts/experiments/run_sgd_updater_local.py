"""Isolated constant-rate SGD and original direct updater: local component RTL."""
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

sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import run_srnn_integer_optimizer_comparison as original
from supplement_srnn_magnitude_sgd import magnitude_update_tensor

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT/'verification/experiments/sgd_updater_r1'
OUT = ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_updater_local_r1'
NUMERATOR = 8854437155380585


def rne(x, shift):
    if shift <= 0:
        return x << -shift
    q, r = divmod(x, 1 << shift)
    return q + int(r > (1 << (shift-1)) or (r == (1 << (shift-1)) and q & 1))


def delta_integer(g):
    magnitude = abs(int(g))
    shift = max(0, magnitude.bit_length()-53)
    rounded = rne(magnitude, shift) << shift
    product = rounded * NUMERATOR
    shift = max(0, product.bit_length()-53)
    product = rne(product, shift) << shift
    delta = rne(product, 70)
    return -delta if g > 0 else delta


def expected(g, code, residual, direct=False):
    limit = 65 if direct else 1 << 24
    if not -limit < residual < limit:
        return code, residual, 4
    delta = (0 if g == 0 else (-31 if g > 0 else 31)) if direct else delta_integer(g)
    total = residual + delta
    dc = (1 if total >= 0 else -1) * (abs(total)//limit)
    candidate = code + dc
    clipped = not -32768 <= candidate <= 32767
    updated = min(32767, max(-32768, candidate))
    return updated, 0 if clipped else total-dc*limit, int(updated != code) | (int(clipped) << 1)


def fixture(out):
    assert (-3e-5*(1 << 14)*(1 << 24)/(1 << 40)).as_integer_ratio() == (-NUMERATOR, 1 << 70)
    rng = np.random.default_rng(20261019)
    gradients = list(rng.integers(-(1 << 63), (1 << 63)-1, 32768, dtype=np.int64).tolist())
    gradients += list(rng.integers(-(1 << 44), 1 << 44, 32768, dtype=np.int64).tolist())
    boundaries = {0, -(1 << 63), (1 << 63)-1}
    for bit in range(64):
        for offset in (-1, 0, 1):
            for sign in (-1, 1):
                g = sign*((1 << bit)+offset)
                if -(1 << 63) <= g < 1 << 63:
                    boundaries.add(g)
    for k in range(1000):
        for sign in (-1, 1):
            boundaries.add(sign*(2*k+1)*200000)
    gradients += sorted(boundaries)
    codes = rng.integers(-32768, 32768, len(gradients), dtype=np.int64)
    rd = rng.integers(-64, 65, len(gradients), dtype=np.int64)
    rs = rng.integers(-(1 << 24)+1, 1 << 24, len(gradients), dtype=np.int64)
    codes[::4] = -32768; codes[1::4] = 32767
    rd[::5] = 64; rd[1::5] = -64
    rs[::5] = (1 << 24)-1; rs[1::5] = -(1 << 24)+1
    rows = list(zip(gradients, codes.tolist(), rd.tolist(), rs.tolist()))
    for g in (0, 1, -1, 600000, -600000, 1 << 43, -(1 << 63)):
        for residual in (-(1 << 31), -16777216, 16777216, (1 << 31)-1):
            rows.append((g, 123, 0, residual))
    reference_sgd = magnitude_update_tensor(codes, rs, np.array(gradients, dtype=np.int64), 3e-5)
    reference_direct = original.integer_head.direct_update_tensor(codes, rd, np.array(gradients, dtype=np.int64), 31, 65)
    double_rounding = []
    counts = dict(valid=len(gradients), invalid_residual=len(rows)-len(gradients), zero=0,
                  sgd_saturated=0, direct_saturated=0, double_rounding_differences=0)
    with (out/'vectors.hex').open('x', encoding='ascii') as handle:
        for i, (g, code, d, s) in enumerate(rows):
            dc, dr, df = expected(g, code, d, True)
            sc, sr, sf = expected(g, code, s)
            if i < len(gradients):
                if (sc, sr) != (int(reference_sgd[0][i]), int(reference_sgd[1][i])):
                    raise ValueError(('integer SGD oracle mismatch', i))
                if (dc, dr) != (int(reference_direct[0][i]), int(reference_direct[1][i])):
                    raise ValueError(('direct oracle mismatch', i))
            counts['zero'] += g == 0
            counts['sgd_saturated'] += bool(sf & 2)
            counts['direct_saturated'] += bool(df & 2)
            naive = rne(abs(g)*NUMERATOR, 70) * (-1 if g > 0 else 1)
            exact = delta_integer(g)
            if naive != exact:
                counts['double_rounding_differences'] += 1
                if len(double_rounding) < 8:
                    double_rounding.append(dict(gradient=g, original_binary64_delta=exact, single_exact_product_delta=naive))
            fields = ((g,64),(code,16),(d,16),(s,32),(dc,16),(dr,16),(df,4),(sc,16),(sr,32),(sf,4))
            handle.write(''.join(f'{value & ((1 << width)-1):0{width//4}x}' for value,width in fields)+'\n')
    return dict(count=len(rows), independent_state_values=len(gradients)*4, counts=counts,
                double_rounding_examples=double_rounding, seed=20261019,
                scope='Synthetic scalar updates; no training replay, new distribution, RMW store, or whole-core cycles')


def parse(log, n):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', log):
        raise ValueError('runtime diagnostic')
    rows = [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', line)}
            for line in re.findall(r'^UPDATER_RESULT (.+)$', log, re.M)]
    if [(x['policy'],x['scenario'],x['count']) for x in rows] != [(p,s,n) for p in range(2) for s in range(3)]:
        raise ValueError('incomplete ordered cases')
    if log.count(f'UPDATERS_PASS policies=2 scenarios=3 count={n}') != 1:
        raise ValueError('completion missing')
    reset = re.findall(r'^RESET_FLUSH policy=(\d+) discarded=(\d+)$', log, re.M)
    if reset != [(str(p),str(1 if p == 0 else 4)) for p in range(2) for _ in range(3)]:
        raise ValueError('reset coverage')
    for row in rows:
        if row['scenario'] == 0:
            latency = 1 if row['policy'] == 0 else 4
            if row['latency_min'] != latency or row['latency_max'] != latency or row['accept_gaps'] or row['cycles'] != n+latency:
                raise ValueError('unstalled latency/II')
        elif not row['stalls']:
            raise ValueError('backpressure not exercised')
    return dict(rows=rows, scalar_results=n*6, reset_flushes=len(reset))


def main():
    OUT.mkdir(exist_ok=False)
    state = dict(start=dt.datetime.now().astimezone().isoformat(), status='preparing',
                 scope='Local isolated update arithmetic, not full SGD learning engine or full-core throughput')
    def save():
        (OUT/'status.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    inputs = [EXPERIMENT/x for x in ('fixed_sgd_update.sv','direct_update_wrapper.sv','tb_updaters.sv')]
    inputs += [ROOT/'rtl/learning/unified_tp_direct_residual_update.sv']
    original_inputs = [(p,p.stat().st_size,p.stat().st_mtime_ns) for p in inputs]
    for p in inputs:
        shutil.copy2(p, OUT/p.name)
    shutil.copy2(Path(__file__), OUT/'executed_runner.py')
    linux = '/mnt/'+str(OUT)[0].lower()+str(OUT)[2:].replace('\\','/')
    def execute(stage, command):
        state['status'] = stage+'_running'; save(); begin = time.monotonic()
        cmd = ['wsl','-e','sh','-lc','cd '+shlex.quote(linux)+' && ulimit -v 3145728 && '+command]
        with (OUT/(stage+'.log')).open('xb') as handle:
            process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                state[stage] = dict(exit=None,pid=process.pid,timeout=True,command=cmd); save(); raise
        state[stage] = dict(exit=code,wall_seconds=time.monotonic()-begin,command=cmd)
        save()
        if code:
            raise RuntimeError(stage+' failed')
    try:
        state['fixture'] = fixture(OUT); save()
        execute('environment','verilator --version')
        execute('build','/usr/bin/time -v -o build_resources.txt verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_updaters --Mdir obj unified_tp_direct_residual_update.sv direct_update_wrapper.sv fixed_sgd_update.sv tb_updaters.sv')
        execute('run','/usr/bin/time -v -o run_resources.txt obj/Vtb_updaters +VECTORS=vectors.hex +N='+str(state['fixture']['count']))
        state['checks'] = parse((OUT/'run.log').read_text(errors='replace'), state['fixture']['count'])
        for p,size,mtime in original_inputs:
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime) or p.read_bytes()!=(OUT/p.name).read_bytes():
                raise ValueError('input changed: '+str(p))
        state['status'] = 'PASS_fixed_rate_SGD_and_original_direct_component_RTL'
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
    state['end'] = dt.datetime.now().astimezone().isoformat(); save()
    print(json.dumps(state, indent=2))
    if state['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
