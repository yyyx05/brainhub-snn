"""Post-reducer gradient replay through the real single-credit RMW service.

This does not recompute training, logits, gradients, or accuracy. SGD receives
the archived direct trajectory's gradients, with its own live parameter store.
"""
import datetime as dt
import difflib
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
import run_sgd_bounded_local as bounded
import run_sgd_updater_local as scalar

ROOT = Path('\\\\?\\'+str(scalar.ROOT)) if sys.platform=='win32' else scalar.ROOT
EXP = ROOT/'verification/experiments/sgd_rmw_replay_r1'
PREP = ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_rmw_replay_local_r1'
OUT = ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_rmw_replay_local_r2'
SOURCE = ROOT/'verification/experiments/srnn_representative_inputs/a_r1'
NAMES = ['unified_tp_parameter_rmw_service', 'unified_tp_active_tensor_update_path',
         'unified_tp_parameter_update_adapter', 'unified_tp_active_tensor_context',
         'unified_tp_direct_residual_update', 'unified_tp_q015_magnitude_residual_update']


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def sources():
    EXP.mkdir(exist_ok=False)
    for name in NAMES:
        shutil.copy2(ROOT/'rtl/learning'/f'{name}.sv', EXP/f'{name}.sv')
    shutil.copy2(bounded.EXPERIMENT/'bounded_sgd_update.sv', EXP/'bounded_sgd_update.sv')
    patches = []
    for name, new in zip(NAMES[:3], ['replay_sgd_rmw', 'replay_sgd_path', 'replay_sgd_adapter']):
        original = (EXP/f'{name}.sv').read_text()
        text = original.replace(name, new)
        # The single-credit RMW owner holds this gradient until result retirement.
        text = replace_once(text, '    input  wire logic                               i_rst_n,',
                            '    input  wire logic                               i_rst_n,\n'
                            '    input  wire logic signed [43:0] i_gradient,')
        if name == NAMES[0]:
            text = text.replace(NAMES[1], 'replay_sgd_path')
            text = replace_once(text, '    rmw_state_t state_q;',
                                '    rmw_state_t state_q;\n    logic signed [43:0] gradient_q;')
            text = replace_once(text, '        .i_clk(i_clk),',
                                '        .i_gradient(gradient_q),\n        .i_clk(i_clk),')
            text = replace_once(text, "        if (!i_rst_n) begin\n            state_q <= RMW_IDLE;",
                                "        if (!i_rst_n) begin\n            gradient_q <= '0;\n            state_q <= RMW_IDLE;")
            text = replace_once(text, '                    request_direction_q <= i_req_update_direction;',
                                '                    gradient_q <= i_gradient;\n'
                                '                    request_direction_q <= i_req_update_direction;')
        elif name == NAMES[1]:
            text = text.replace(NAMES[2], 'replay_sgd_adapter')
            text = replace_once(text, '    ) parameter_update (',
                                '    ) parameter_update (\n        .i_gradient(i_gradient),')
        else:
            begin = text.index('    unified_tp_direct_residual_update #(')
            end = text.index('    unified_tp_q015_magnitude_residual_update #(')
            text = text[:begin]+'''    // Only the fixed-rate, full-range sat16 SGD mode is used here.
    // Metadata is safe because the enclosing RMW service permits one request.
    bounded_sgd_update update_leaf (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(stage_valid_q && !stage_magnitude_mode_q),
        .o_ready(direct_ready), .i_gradient(i_gradient),
        .i_code(stage_code_q), .i_residual(stage_residual_q),
        .o_valid(direct_valid), .i_ready(i_ready),
        .o_code(direct_code), .o_residual(direct_residual),
        .o_code_changed(direct_code_changed), .o_saturated(direct_saturated),
        .o_config_error(direct_config_error)
    );

'''+text[end:]
        (EXP/f'{new}.sv').write_text(text, encoding='ascii')
        patches.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                          fromfile=name+'.sv', tofile=new+'.sv'))
    (EXP/'changes.diff').write_text(''.join(patches), encoding='ascii')
    shutil.copy2(Path(__file__).with_name('tb_sgd_rmw_replay.sv'), EXP/'tb_sgd_rmw_replay.sv')


def read_hex(path, bits):
    values = np.array([int(x,16) for x in path.read_text().split()], dtype=np.uint64)
    if bits == 64:
        return values.view(np.int64)
    result = values.astype(np.int64)
    return np.where(result >= (1 << (bits-1)), result-(1 << bits), result)


def fixture():
    selected = json.loads((SOURCE/'selection.json').read_text())['selected']
    assert [s['dataset_index'] for s in selected] == [8,9,20,25,27,37,43,47]
    assert sum(s['time_steps'] for s in selected) == 588
    paths = [SOURCE/'selection.json']
    for i in range(8):
        paths += [SOURCE/f'multi_fixture/sequence_{i}'/(x+'.mem') for x in
                  ('on_gradient','on_wout_before','on_wout_after','on_residual')]
    before = [(p,p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    code_d = read_hex(paths[2],16)[:5120].copy()
    code_s = code_d.copy()
    rd = np.zeros(5120,dtype=np.int64); rs = rd.copy()
    (OUT/'initial.hex').write_text(''.join(f'{int(x)&65535:04x}\n' for x in code_d),encoding='ascii')
    count=0; zero=0; independent=0; max_gradient=0; stats=[dict(code_writes=0,residual_writes=0,saturated=0,delta_zero_nonzero_gradient=0) for _ in range(2)]
    with (OUT/'vectors.hex').open('x',encoding='ascii') as output:
        for sequence, sample in enumerate(selected):
            base=SOURCE/f'multi_fixture/sequence_{sequence}'
            gradients=read_hex(base/'on_gradient.mem',64).reshape(-1,5120)[:sample['time_steps']]
            original_before=read_hex(base/'on_wout_before.mem',16).reshape(-1,5120)
            original_after=read_hex(base/'on_wout_after.mem',16).reshape(-1,5120)
            original_residual=read_hex(base/'on_residual.mem',16).reshape(-1,5120)
            for step,g in enumerate(gradients):
                assert np.array_equal(code_d,original_before[step])
                assert np.max(np.abs(g)) <= 7328690601984
                max_gradient=max(max_gradient,int(np.max(np.abs(g))))
                nd,nrd,_=scalar.original.integer_head.direct_update_tensor(code_d,rd,g,31,65)
                ns,nrs,_=scalar.magnitude_update_tensor(code_s,rs,g,3e-5)
                assert np.array_equal(nd,original_after[step]) and np.array_equal(nrd,original_residual[step])
                for a in range(5120):
                    dc,dr,df=scalar.expected(int(g[a]),int(code_d[a]),int(rd[a]),True)
                    sc,sr,sf=scalar.expected(int(g[a]),int(code_s[a]),int(rs[a]))
                    assert (dc,dr,sc,sr)==(int(nd[a]),int(nrd[a]),int(ns[a]),int(nrs[a]))
                    dw=int(dc != code_d[a]) | (int(dr != rd[a])<<1)
                    sw=int(sc != code_s[a]) | (int(sr != rs[a])<<1)
                    for p,(c,r,f,w) in enumerate(((dc,dr,df,dw),(sc,sr,sf,sw))):
                        stats[p]['code_writes'] += w&1
                        stats[p]['residual_writes'] += (w>>1)&1
                        stats[p]['saturated'] += (f>>1)&1
                    stats[1]['delta_zero_nonzero_gradient'] += bool(g[a] and scalar.delta_integer(int(g[a]))==0)
                    fields=((int(g[a]),64),(dc,16),(dr,16),(dw,4),(sc,16),(sr,32),(sw,4))
                    output.write(''.join(f'{v&((1<<w)-1):0{w//4}x}' for v,w in fields)+'\n')
                    count+=1;zero+=g[a]==0;independent+=4
                code_d,rd,code_s,rs=nd,nrd,ns,nrs
    for p,size,mtime in before:
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(size,mtime)
    return dict(count=count,zero=int(zero),requests=count-int(zero),steps=588,elements=5120,
                independent_state_values=independent,max_gradient=max_gradient,policies=stats,
                source_files=[dict(path=str(p.relative_to(ROOT)),bytes=s,mtime_ns=m) for p,s,m in before],
                selection=selected,source_unchanged=True,
                scope='Fixed archived direct gradients, not SGD closed-loop training or new data; sequential live stores; no gradient reducer/forward pass/commit engine in measured region.')


def parse(log, fixture):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b',log):raise ValueError('RTL failure')
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in re.findall(r'^RMW_RESULT (.+)$',log,re.M)]
    if [(r['policy'],r['scenario']) for r in rows] != [(p,s) for p in range(2) for s in range(2)]:raise ValueError('cases')
    if len(re.findall(r'^RMW_REPLAY_PASS$',log,re.M))!=1:raise ValueError('terminal marker')
    for row in rows:
        p=row['policy'];expected=fixture['policies'][p]
        for key in ('code_writes','residual_writes','saturated'):
            if row[key]!=expected[key]:raise ValueError(key)
        if any(row[k]!=fixture['requests'] for k in ('requests','reads','responses','results')):raise ValueError('credit')
        if row['words']!=fixture['count'] or row['zero']!=fixture['zero']:raise ValueError('coverage')
        if not row['state_checks']==fixture['count']+5120:raise ValueError('state checks')
        if row['scenario']==1 and (not row['read_stalls'] or not row['result_stalls']):raise ValueError('stalls')
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    state=dict(start=dt.datetime.now().astimezone().isoformat(),status='preparing')
    def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        prior=json.loads((PREP/'status.json').read_text())
        assert prior['status']=='FAIL' and prior['error']=="FileNotFoundError(2, 'No such file or directory')"
        assert 'build' not in prior and 'run' not in prior
        state['fixture']=prior['fixture']
        for item in state['fixture']['source_files']:
            p=ROOT/item['path'];assert (p.stat().st_size,p.stat().st_mtime_ns)==(item['bytes'],item['mtime_ns'])
        for name in ('vectors.hex','initial.hex'):shutil.copy2(PREP/name,OUT/name)
        state['preparation_reused']='sgd_rmw_replay_local_r1; archive path failure before EDA; no reference regeneration'
        save()
        for p in EXP.iterdir():shutil.copy2(p,OUT/p.name)
        for p in (Path(__file__),Path(__file__).with_name('run_sgd_updater_local.py'),Path(__file__).with_name('run_sgd_bounded_local.py')):shutil.copy2(p,OUT/p.name)
        normal=str(OUT).removeprefix('\\\\?\\')
        linux='/mnt/'+normal[0].lower()+normal[2:].replace('\\','/')
        source_names=' '.join(p.name for p in EXP.glob('*.sv'))
        for stage,command in (
            ('environment','verilator --version'),
            ('build','/usr/bin/time -v -o build_resources.txt verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_sgd_rmw_replay --Mdir obj '+source_names),
            ('run','/usr/bin/time -v -o run_resources.txt obj/Vtb_sgd_rmw_replay')):
            state['status']=stage+'_running';save();begin=time.monotonic()
            cmd=['wsl','-e','sh','-lc','cd '+shlex.quote(linux)+' && ulimit -v 3145728 && '+command]
            with (OUT/(stage+'.log')).open('xb') as handle:
                process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state[stage]=dict(pid=process.pid,exit=None,command=cmd);save()
                code=process.wait(timeout=1800)
            state[stage].update(exit=code,wall_seconds=time.monotonic()-begin);save()
            if code:raise RuntimeError(stage+' failed')
        log=bounded.decode((OUT/'run.log').read_bytes())
        state['measurements']=parse(log,state['fixture'])
        for bad in (log.replace('RMW_REPLAY_PASS','MISSING'),log.replace('RMW_RESULT','MISSING',1),
                    log.replace('reads='+str(state['fixture']['requests']),'reads=0',1),
                    log+'\n%Error: synthetic error\n'):
            try:parse(bad,state['fixture'])
            except ValueError:pass
            else:raise AssertionError('bad transcript accepted')
        state['damaged_transcripts_rejected']=4
        for p in EXP.iterdir():assert p.read_bytes()==(OUT/p.name).read_bytes()
        state['source_files_byte_equal']=len(list(EXP.iterdir()))
        state['diagnostics']={stage:dict(warnings=len(re.findall(r'%Warning',bounded.decode((OUT/(stage+'.log')).read_bytes()))),
                                       errors=len(re.findall(r'%Error|%Fatal',bounded.decode((OUT/(stage+'.log')).read_bytes())))) for stage in ('build','run')}
        assert not list(OUT.rglob('*.part'))
        state['status']='PASS_archived_gradient_RMW_service_replay_not_closed_loop_training'
    except Exception as error:
        state.update(status='FAIL',error=repr(error))
    state['end']=dt.datetime.now().astimezone().isoformat();save()
    print(json.dumps(state,indent=2))
    if state['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()
