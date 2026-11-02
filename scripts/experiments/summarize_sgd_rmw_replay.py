"""One offline service ledger plus a separately captured lint diagnostic pass."""
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
import run_sgd_rmw_replay_local as replay
import run_sgd_bounded_local as bounded


def main():
    out=replay.OUT
    assert not (out/'analysis.json').exists()
    state=json.loads((out/'status.json').read_text())
    assert state['status'].startswith('PASS_')
    rows=state['measurements']
    normal=str(out).removeprefix('\\\\?\\')
    linux='/mnt/'+normal[0].lower()+normal[2:].replace('\\','/')
    source_names=' '.join(p.name for p in replay.EXP.glob('*.sv'))
    # Redirect in Linux before any child is created, avoiding WSL stdout/stderr
    # file-offset interference seen in the original build log. No RTL execution.
    command=('cd '+shlex.quote(linux)+' && ulimit -v 3145728 && '
             '/usr/bin/time -v -o lint_resources.txt verilator --lint-only --timing --assert '
             '-Wall -Wno-fatal --top-module tb_sgd_rmw_replay '+source_names+
             ' > lint_diagnostics.log 2>&1')
    begin=time.monotonic()
    with (out/'lint_launcher.log').open('xb') as handle:
        proc=subprocess.Popen(['wsl','-e','sh','-lc',command],stdout=handle,stderr=subprocess.STDOUT)
        code=proc.wait(timeout=120)
    assert code==0
    diagnostic=(out/'lint_diagnostics.log').read_text()
    warning_counts=dict(collections.Counter(re.findall(r'%Warning-([A-Z0-9_]+):',diagnostic)))
    assert not re.search(r'%Error|%Fatal',diagnostic)
    with (out/'service_measurements.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparison=[]
    for scenario in range(2):
        direct=next(x for x in rows if x['policy']==0 and x['scenario']==scenario)
        sgd=next(x for x in rows if x['policy']==1 and x['scenario']==scenario)
        difference=sgd['cycles']-direct['cycles']
        if scenario==0:
            assert difference==3*state['fixture']['requests']
            assert direct['cycles']==state['fixture']['count']+5*state['fixture']['requests']
            assert sgd['cycles']==state['fixture']['count']+8*state['fixture']['requests']
        else:
            assert difference==(3*state['fixture']['requests']+
                               sgd['read_stalls']-direct['read_stalls']+
                               sgd['result_stalls']-direct['result_stalls'])
        comparison.append(dict(scenario=scenario,direct_cycles=direct['cycles'],sgd_cycles=sgd['cycles'],
                               saved_cycles=difference,sgd_over_direct=sgd['cycles']/direct['cycles'],
                               direct_reduction_percent=100*difference/sgd['cycles']))
    with (out/'comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    traffic=[]
    for p in range(2):
        row=next(x for x in rows if x['policy']==p and x['scenario']==0)
        rb=2 if p==0 else 4
        traffic.append(dict(policy=p,modeled_code_bits=16,modeled_residual_bits=rb*8,
                            store_bits=5120*(16+rb*8),logical_read_bytes=row['reads']*(2+rb),
                            code_write_events=row['code_writes'],residual_write_events=row['residual_writes'],
                            logical_write_bytes=row['code_writes']*2+row['residual_writes']*rb))
    with (out/'logical_traffic.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(traffic[0]));writer.writeheader();writer.writerows(traffic)
    resources={}
    for stage in ('build','run','lint'):
        log=(out/(stage+'_resources.txt')).read_text()
        resources[stage]=dict(max_rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',log)[1]),
                              swaps=int(re.search(r'Swaps: (\d+)',log)[1]))
    # The original numerical/source checks are not rerun. Check only new files
    # and the inputs held read-only across this supplement.
    for item in state['fixture']['source_files']:
        path=replay.ROOT/item['path']
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(item['bytes'],item['mtime_ns'])
    shutil.copy2(Path(__file__),out/'executed_summary.py')
    analysis=dict(end=dt.datetime.now().astimezone().isoformat(),new_rtl_runs=0,new_training_episodes=0,
                  lint=dict(exit=code,wall_seconds=time.monotonic()-begin,command=command,warnings=warning_counts),
                  original_build_log='Raw stdout/stderr capture has overwritten diagnostic prefix; seven remaining warning tokens are only a lower bound. Original status/log kept unchanged. Separate lint diagnostics are not claimed as recovered original bytes.',
                  unoptflat='Present in lint and surviving build log: ready vector bit dependencies 3 -> 2 -> 1 -> 0. Source-level inspection suggests vector-granularity scheduling warning, not a discovered bit-level loop; no formal/netlist loop proof is claimed.',
                  resources=resources,comparison=comparison,logical_traffic=traffic,
                  whole_network_throughput_measured=False,hardware_memory_mapping_measured=False,
                  delta_zero_nonzero_gradient=state['fixture']['policies'][1]['delta_zero_nonzero_gradient'],
                  boundary='Same nonzero-gradient request schedule; no front-end delta-zero bypass. Unequal code-update amounts. SGD uses gradients from the direct trajectory, not its own closed loop. No reducer, softmax, trace generation, forward pass, per-step release/commit, DDR, route or Fmax included.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
