"""Measure only the stronger SGD service arm with an exact delta-zero filter."""
import collections
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import run_sgd_rmw_replay_local as base
import run_sgd_updater_local as scalar

ROOT=base.ROOT
EXP=ROOT/'verification/experiments/sgd_rmw_filtered_r1'
OUT=ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_rmw_filtered_local_r1'


def parse(log, fixture):
    if re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b',log):raise ValueError('runtime failure')
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in re.findall(r'^RMW_RESULT (.+)$',log,re.M)]
    if [(x['policy'],x['scenario']) for x in rows]!=[(1,0),(1,1)]:raise ValueError('cases')
    if len(re.findall(r'^RMW_FILTERED_PASS$',log,re.M))!=1:raise ValueError('marker')
    requests=fixture['requests']-fixture['policies'][1]['delta_zero_nonzero_gradient']
    for row in rows:
        if row['words']!=fixture['count'] or row['zero']!=fixture['count']-requests:raise ValueError('coverage')
        if any(row[k]!=requests for k in ('requests','reads','responses','results')):raise ValueError('credit')
        if row['state_checks']!=fixture['count']+5120:raise ValueError('state checks')
        for k in ('code_writes','residual_writes','saturated'):
            if row[k]!=fixture['policies'][1][k]:raise ValueError(k)
        if row['scenario']==0 and row['cycles']!=fixture['count']+8*requests:raise ValueError('unstalled interval')
        if row['scenario']==1 and (not row['read_stalls'] or not row['result_stalls']):raise ValueError('stalls')
    return rows


def main():
    OUT.mkdir(exist_ok=False);EXP.mkdir(exist_ok=False)
    state=dict(start=dt.datetime.now().astimezone().isoformat(),status='preparing')
    def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        prior=json.loads((base.OUT/'status.json').read_text())
        assert prior['status'].startswith('PASS_')
        fixture=prior['fixture']
        # Positive multiplication and both RNE operations are monotone; negative
        # conversion is symmetric. The adjacent endpoints establish the cutoff.
        for g in range(-66667,66668):
            assert (scalar.delta_integer(g)==0)==(abs(g)<=66666)
        state['zero_filter_proof']=dict(checked_integer_values=133335,zero_abs_max=66666,
              adjacent_values=[scalar.delta_integer(g) for g in (-66667,-66666,66666,66667)],
              argument='Positive coefficient, monotone binary64 product RNE and integer RNE, odd sign symmetry; adjacent endpoints bound the complete zero interval.',
              assumptions='Fixed 3e-5 semantics, proven signed44 gradient domain, legal carried code/residual state; no external invalid residual recovery is claimed.')
        for p in base.EXP.iterdir():
            if p.suffix=='.sv' and p.name!='tb_sgd_rmw_replay.sv':shutil.copy2(p,EXP/p.name)
        shutil.copy2(Path(__file__).with_name('replay_gradient_filter.sv'),EXP/'replay_gradient_filter.sv')
        tb=(base.EXP/'tb_sgd_rmw_replay.sv').read_text()
        tb=base.replace_once(tb,'module tb_sgd_rmw_replay;','module tb_sgd_rmw_filtered;')
        tb=base.replace_once(tb,'    // Both arms retain',
             '    wire [1:0] service_req_ready, filtered_valid, skip_word;\n    // Both arms retain')
        tb=base.replace_once(tb,'        localparam int RW=(p==0)?16:32;',
             '''        localparam int RW=(p==0)?16:32;
        replay_gradient_filter #(.SGD(p==1)) filter_i (
            .i_valid(req_valid && policy==p),.i_gradient(gradient),
            .i_service_ready(service_req_ready[p]),.o_ready(req_ready[p]),
            .o_valid(filtered_valid[p]),.o_skip(skip_word[p]));''')
        tb=base.replace_once(tb,'.i_req_valid(req_valid && policy==p),.o_req_ready(req_ready[p]),',
                                '.i_req_valid(filtered_valid[p]),.o_req_ready(service_req_ready[p]),')
        tb=base.replace_once(tb,'req_valid=(word_index<N && gradient!=0);','req_valid=(word_index<N);')
        tb=base.replace_once(tb,'if (gradient==0) begin','if (skip_word[policy]) begin')
        tb=base.replace_once(tb,'for(integer p=0;p<2;p=p+1) begin','for(integer p=1;p<2;p=p+1) begin')
        tb=base.replace_once(tb,'RMW_REPLAY_PASS','RMW_FILTERED_PASS')
        (EXP/'tb_sgd_rmw_filtered.sv').write_text(tb,encoding='ascii')
        for p in EXP.iterdir():shutil.copy2(p,OUT/p.name)
        for name in ('vectors.hex','initial.hex'):shutil.copy2(base.OUT/name,OUT/name)
        shutil.copy2(Path(__file__),OUT/'executed_runner.py')
        state['fixture_reference']=str(base.OUT.relative_to(ROOT))+'/status.json'
        state['new_training_or_reference_runs']=0
        normal=str(OUT).removeprefix('\\\\?\\');linux='/mnt/'+normal[0].lower()+normal[2:].replace('\\','/')
        names=' '.join(p.name for p in EXP.glob('*.sv'))
        for stage,command in (
            ('build','verilator --binary --timing --assert -j 1 -Wall -Wno-fatal --top-module tb_sgd_rmw_filtered --Mdir obj '+names),
            ('run','obj/Vtb_sgd_rmw_filtered')):
            state['status']=stage+'_running';save();begin=time.monotonic()
            shell='cd '+shlex.quote(linux)+' && ulimit -v 3145728 && /usr/bin/time -v -o '+stage+'_resources.txt '+command+' > '+stage+'.log 2>&1'
            with (OUT/(stage+'_launcher.log')).open('xb') as handle:
                process=subprocess.Popen(['wsl','-e','sh','-lc',shell],stdout=handle,stderr=subprocess.STDOUT)
                state[stage]=dict(pid=process.pid,exit=None,command=shell);save()
                code=process.wait(timeout=1800)
            state[stage].update(exit=code,wall_seconds=time.monotonic()-begin);save()
            if code:raise RuntimeError(stage+' failed')
        log=(OUT/'run.log').read_text();rows=parse(log,fixture)
        state['measurements']=rows
        for bad in (log.replace('RMW_FILTERED_PASS','MISSING'),log.replace('RMW_RESULT','MISSING',1),log+'\n%Error: synthetic\n'):
            try:parse(bad,fixture)
            except ValueError:pass
            else:raise AssertionError('damaged transcript accepted')
        state['damaged_transcripts_rejected']=3
        state['comparison']=[]
        for row in rows:
            direct=next(x for x in prior['measurements'] if x['policy']==0 and x['scenario']==row['scenario'])
            original=next(x for x in prior['measurements'] if x['policy']==1 and x['scenario']==row['scenario'])
            state['comparison'].append(dict(scenario=row['scenario'],direct_cycles=direct['cycles'],
                filtered_sgd_cycles=row['cycles'],unfiltered_sgd_cycles=original['cycles'],
                filtered_over_direct=row['cycles']/direct['cycles'],
                direct_reduction_percent=100*(row['cycles']-direct['cycles'])/row['cycles']))
        with (OUT/'comparison.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(state['comparison'][0]));writer.writeheader();writer.writerows(state['comparison'])
        with (OUT/'service_measurements.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        state['diagnostics']={stage:dict(warnings=dict(collections.Counter(re.findall(r'%Warning-([A-Z0-9_]+):',(OUT/(stage+'.log')).read_text()))),
                                        errors=len(re.findall(r'%Error|%Fatal',(OUT/(stage+'.log')).read_text()))) for stage in ('build','run')}
        state['resources']={}
        for stage in ('build','run'):
            raw=(OUT/(stage+'_resources.txt')).read_text()
            state['resources'][stage]=dict(max_rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',raw)[1]),swaps=int(re.search(r'Swaps: (\d+)',raw)[1]))
        for p in EXP.iterdir():assert p.read_bytes()==(OUT/p.name).read_bytes()
        for item in fixture['source_files']:
            path=ROOT/item['path'];assert (path.stat().st_size,path.stat().st_mtime_ns)==(item['bytes'],item['mtime_ns'])
        assert (OUT/'vectors.hex').read_bytes()==(base.OUT/'vectors.hex').read_bytes()
        assert (OUT/'initial.hex').read_bytes()==(base.OUT/'initial.hex').read_bytes()
        assert not list(OUT.rglob('*.part'))
        state['source_files_byte_equal']=len(list(EXP.iterdir()))
        state['status']='PASS_exact_delta_zero_filtered_RMW_replay_not_closed_loop_training'
    except Exception as error:state.update(status='FAIL',error=repr(error))
    state['end']=dt.datetime.now().astimezone().isoformat();save();print(json.dumps(state,indent=2))
    if state['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()
