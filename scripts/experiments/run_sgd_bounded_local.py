"""Bounded SRNN scalar operators with matched input register boundaries."""
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import run_sgd_updater_local as old

ROOT = old.ROOT
EXPERIMENT = ROOT/'verification/experiments/sgd_bounded_r1'
OUT = ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_bounded_local_r1'


def bound_proof():
    # Monotone RNE and a fixed upper state prove invariance for either spike.
    u = 0
    trajectory = [u]
    while True:
        nxt = old.rne(u*13926, 14)+16384
        if nxt == u:
            break
        assert nxt > u and len(trajectory) < 1000
        trajectory.append(nxt)
        u = nxt
    states = np.arange(u+1, dtype=np.int64)
    from screen_srnn_m2_direct_optimizer import round_shift_even
    decay = round_shift_even(states*13926, 14)
    assert decay.min() == 0 and np.all(np.diff(decay) >= 0)
    assert (decay+16384).max() == u
    assert u == 109206
    bound = u*2**26
    assert bound == 7328690601984 and bound < 2**43
    return dict(trace_upper=u, all_one_trajectory=trajectory,
                exhaustive_trace_transitions=2*(u+1), gradient_abs_upper=bound,
                gradient_signed_bits=44, fixed_point_iterations=len(trajectory),
                assumptions=['zero initial trace', 'binary hidden spike', 'RNE14 with leak13926',
                             'B in 1..64', 'probability in 0..2**24',
                             'gradient=RNE4(sum(error_Q24*trace_Q14)); fixed denominator64'],
                derivation='abs(sum)<=64*2**24*109206; RNE divide16 gives <=109206*2**26. This is an integer bound, not an observed maximum.',
                boundary='Configured SRNN M2 protocol only; arbitrary external traces, other leaks, larger B, FC and SCNN are not covered.')


def prepare_sources():
    EXPERIMENT.mkdir(exist_ok=False)
    src = old.EXPERIMENT/'fixed_sgd_update.sv'
    text = src.read_text()
    start = text.index('    function automatic [63:0] round_gradient')
    end = text.index('    function automatic [47:0] round_delta')
    text = text[:start]+text[end:]
    replacements = {
        'module fixed_sgd_update (':'module bounded_sgd_update (',
        '[63:0]':'[43:0]', '[116:0]':'[96:0]', 'b<117':'b<97',
        "117'd1":"97'd1", "64'd1":"44'd1", 'i_gradient[63]':'i_gradient[43]',
        "integer_part={1'b0,rounded[116:70]};":"integer_part={21'd0,rounded[96:70]};",
        'magnitude_q<=round_gradient(magnitude);':'magnitude_q<=magnitude;'
    }
    for before, after in replacements.items():
        assert before in text, before
        text = text.replace(before, after)
    text = text.replace('// No floating-point IP;', '// Signed44 converts exactly to binary64. No floating-point IP;')
    (EXPERIMENT/'bounded_sgd_update.sv').write_text(text, encoding='ascii')
    for source in (old.EXPERIMENT/'direct_update_wrapper.sv', ROOT/'rtl/learning/unified_tp_direct_residual_update.sv'):
        shutil.copy2(source, EXPERIMENT/source.name)
    tb = (old.EXPERIMENT/'tb_updaters.sv').read_text()
    for before, after in {
        'module tb_updaters;':'module tb_bounded;', '[63:0] gradient':'[43:0] gradient',
        'direct_update_wrapper direct_i':'direct_registered44 direct_i',
        'fixed_sgd_update sgd_i':'sgd_registered44 sgd_i',
        '[215:0]':'[195:0]', '[215:88]':'[195:88]', '[127:0] unused_inputs':'[107:0] unused_inputs',
    }.items():
        assert before in tb, before
        tb = tb.replace(before, after)
    (EXPERIMENT/'tb_bounded.sv').write_text(tb, encoding='ascii')
    wrapper = Path(__file__).with_name('sgd_registered44.sv')
    shutil.copy2(wrapper, EXPERIMENT/wrapper.name)


def fixture():
    rng = np.random.default_rng(2026101916)
    gradients = list(rng.integers(-(1 << 43), 1 << 43, 32768, dtype=np.int64).tolist())
    bound = 7328690601984
    edges = {0, -(1 << 43), (1 << 43)-1, -bound, bound, -bound-1, bound+1}
    for bit in range(44):
        for offset in (-1, 0, 1):
            for sign in (-1, 1):
                g = sign*((1 << bit)+offset)
                if -(1 << 43) <= g < 1 << 43:
                    edges.add(g)
    # Half-integer neighborhoods test product RNE as well as final integer RNE.
    for k in (0, 1, 2, 100, 10000, 1000000, 10000000, 18000000, 21000000):
        for offset in range(-2, 3):
            for sign in (-1, 1):
                edges.add(sign*((2*k+1)*200000+offset))
    gradients += sorted(edges)
    n = len(gradients)
    codes = rng.integers(-32768, 32768, n, dtype=np.int64)
    rd = rng.integers(-64, 65, n, dtype=np.int64)
    rs = rng.integers(-(1 << 24)+1, 1 << 24, n, dtype=np.int64)
    codes[::4] = -32768; codes[1::4] = 32767
    rd[::5] = 64; rd[1::5] = -64
    rs[::5] = (1 << 24)-1; rs[1::5] = -(1 << 24)+1
    rows = list(zip(gradients, codes.tolist(), rd.tolist(), rs.tolist()))
    for g in (0, 1, -1, bound, -bound, -(1 << 43), (1 << 43)-1):
        for residual in (-(1 << 31), -16777216, 16777216, (1 << 31)-1):
            rows.append((g, 123, 0, residual))
    ref_sgd = old.magnitude_update_tensor(codes, rs, np.array(gradients, dtype=np.int64), 3e-5)
    ref_direct = old.original.integer_head.direct_update_tensor(codes, rd, np.array(gradients, dtype=np.int64), 31, 65)
    differences = []
    with (OUT/'vectors.hex').open('x', encoding='ascii') as handle:
        for i, (g, code, d, s) in enumerate(rows):
            dc, dr, df = old.expected(g, code, d, True)
            sc, sr, sf = old.expected(g, code, s)
            if i < n:
                assert (sc, sr) == (int(ref_sgd[0][i]), int(ref_sgd[1][i]))
                assert (dc, dr) == (int(ref_direct[0][i]), int(ref_direct[1][i]))
            naive = old.rne(abs(g)*old.NUMERATOR, 70)*(-1 if g > 0 else 1)
            if naive != old.delta_integer(g):
                differences.append(dict(gradient=g, exact_binary64=old.delta_integer(g), naive=naive))
            fields = ((g,44),(code,16),(d,16),(s,32),(dc,16),(dr,16),(df,4),(sc,16),(sr,32),(sf,4))
            handle.write(''.join(f'{v & ((1 << w)-1):0{w//4}x}' for v,w in fields)+'\n')
    return dict(count=len(rows), valid=n, invalid_residual=28, independent_state_values=n*4,
                full_signed44_samples=True, protocol_bound=bound, seed=2026101916,
                double_rounding_differences=len(differences), examples=differences[:8])


def decode(raw):
    if raw.startswith(b'w\x00s\x00l\x00:\x00'):
        end = raw.index(b'\r\x00\n\x00')+4
        prefix = raw[:end].decode('utf-16le')
        assert 'localhost' in prefix
        return prefix+raw[end:].decode('ascii')
    return raw.decode('ascii')


def parse(log, n):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b', log):
        raise ValueError('runtime diagnostic')
    rows = [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', line)}
            for line in re.findall(r'^UPDATER_RESULT (.+)$', log, re.M)]
    if [(x['policy'],x['scenario'],x['count']) for x in rows] != [(p,s,n) for p in range(2) for s in range(3)]:
        raise ValueError('incomplete ordered cases')
    if len(re.findall(r'^'+f'UPDATERS_PASS policies=2 scenarios=3 count={n}'+r'$',log,re.M)) != 1:
        raise ValueError('completion missing')
    reset = re.findall(r'^RESET_FLUSH policy=(\d+) discarded=(\d+)$', log, re.M)
    if reset != [(str(p),str(2 if p == 0 else 5)) for p in range(2) for _ in range(3)]:
        raise ValueError('reset coverage')
    for row in rows:
        if row['scenario'] == 0:
            latency = 2 if row['policy'] == 0 else 5
            if row['latency_min'] != latency or row['latency_max'] != latency or row['accept_gaps'] or row['cycles'] != n+latency:
                raise ValueError('unstalled latency/II')
        elif not row['stalls']:
            raise ValueError('backpressure not exercised')
    return dict(rows=rows, scalar_results=n*6, reset_flushes=len(reset))


def main():
    OUT.mkdir(exist_ok=False)
    state = dict(start=dt.datetime.now().astimezone().isoformat(), status='preparing')
    def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        state['bound'] = bound_proof()
        prepare_sources()
        sources = list(EXPERIMENT.glob('*.sv'))
        before = [(p,p.stat().st_size,p.stat().st_mtime_ns) for p in sources]
        for p in sources: shutil.copy2(p,OUT/p.name)
        for p in (Path(__file__),Path(__file__).with_name('run_sgd_updater_local.py')):
            shutil.copy2(p,OUT/p.name)
        state['fixture'] = fixture(); save()
        linux = '/mnt/'+str(OUT)[0].lower()+str(OUT)[2:].replace('\\','/')
        for stage, command in (
            ('environment','verilator --version'),
            ('build','/usr/bin/time -v -o build_resources.txt verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_bounded --Mdir obj unified_tp_direct_residual_update.sv direct_update_wrapper.sv bounded_sgd_update.sv sgd_registered44.sv tb_bounded.sv'),
            ('run','/usr/bin/time -v -o run_resources.txt obj/Vtb_bounded +VECTORS=vectors.hex +N='+str(state['fixture']['count']))):
            state['status']=stage+'_running'; save(); begin=time.monotonic()
            cmd=['wsl','-e','sh','-lc','cd '+shlex.quote(linux)+' && ulimit -v 3145728 && '+command]
            with (OUT/(stage+'.log')).open('xb') as handle:
                process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                try: code=process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    state[stage]=dict(exit=None,pid=process.pid,timeout=True,command=cmd);save();raise
            state[stage]=dict(exit=code,wall_seconds=time.monotonic()-begin,command=cmd); save()
            if code: raise RuntimeError(stage+' failed')
        log=decode((OUT/'run.log').read_bytes()); n=state['fixture']['count']
        state['checks']=parse(log,n)
        for bad in (log.replace('RESET_FLUSH policy=0 discarded=2\n','',1),
                    log.replace('policy=1 discarded=5','policy=1 discarded=4',1),
                    log.replace('UPDATERS_PASS','MISSING_PASS',1),
                    log.replace('count='+str(n),'count='+str(n-1),1),
                    log.replace('latency_min=5 latency_max=5','latency_min=4 latency_max=5',1),
                    log+'\n%Error: synthetic error\n'):
            try: parse(bad,n)
            except ValueError: pass
            else: raise AssertionError('corrupt transcript accepted')
        state['corrupt_variants_rejected']=6
        for p,size,mtime in before:
            assert (p.stat().st_size,p.stat().st_mtime_ns)==(size,mtime)
            assert p.read_bytes()==(OUT/p.name).read_bytes()
        state['input_files_byte_equal']=len(before)
        state['diagnostics']={}
        for stage in ('build','run'):
            text=(OUT/(stage+'.log')).read_bytes().decode('utf-8',errors='replace')
            state['diagnostics'][stage]=dict(warnings=len(re.findall(r'%Warning',text)),errors=len(re.findall(r'%Error|%Fatal',text)))
            assert not state['diagnostics'][stage]['errors']
        assert not list(OUT.rglob('*.part'))
        state['status']='PASS_bounded_signed44_registered_component_RTL'
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
    state['end']=dt.datetime.now().astimezone().isoformat();save()
    print(json.dumps(state,indent=2))
    if state['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()
