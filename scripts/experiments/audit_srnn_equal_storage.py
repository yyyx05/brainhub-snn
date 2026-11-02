#!/usr/bin/env python3
"""Fetch and independently check the finished equal-storage comparison."""
import collections
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import stat
import sys

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, fetch_tree, local_path, run
from run_srnn_equal_storage import MODULE, NAME, transform
from srnn_equal_storage_worker import residency, expected_loads

sys.path.insert(0, str(ROOT / 'scripts'))
from run_srnn_profile_measurement import parse_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fetch', action='store_true', help='Recheck an already complete local download against remote sizes')
    args = parser.parse_args()
    archive = Path(local_path(ROOT / RESULT))
    result = archive / NAME
    client = connect()
    try:
        with client.open_sftp() as sftp:
            with sftp.open(REMOTE + '/' + NAME + '/status.json') as stream:
                terminal = json.load(stream)
            if terminal['status'] != 'PASS_equal_storage_comparison':
                raise RuntimeError('not a successful terminal comparison: ' + terminal['status'])
            if not args.skip_fetch:
                fetch_tree(sftp, REMOTE + '/' + NAME, ROOT / RESULT / NAME)
            inventory = []
            def visit(relative=''):
                directory = REMOTE + '/' + NAME + ('/' + relative if relative else '')
                for entry in sftp.listdir_attr(directory):
                    if entry.filename in ('obj','elab_obj','__pycache__','.Xil'):
                        continue
                    child = (relative + '/' if relative else '') + entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        visit(child)
                    else:
                        assert stat.S_ISREG(entry.st_mode)
                        assert (result / child).stat().st_size == entry.st_size, child
                        inventory.append(dict(path=child, bytes=entry.st_size))
            visit()
        code = '''import json,pathlib,shutil,subprocess
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','verilator') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
d=subprocess.run(['dmesg'],capture_output=True,text=True)
print(json.dumps(dict(active=active,free_bytes=shutil.disk_usage('/root/autodl-tmp').free,dmesg_readable=d.returncode==0,dmesg_error=d.stderr.strip())))'''
        remote = json.loads(run(client, 'python3 -c ' + shlex.quote(code)))
    finally:
        client.close()
    assert not list(result.rglob('*.part'))
    expected = json.loads((archive / 'srnn_specialized_r1/status.json').read_text())
    checks, variants, rows = {}, {}, []
    for policy in ('baseline','candidate'):
        folder = result / policy
        status = json.loads((folder / 'status.json').read_text())
        assert status == terminal['variants'][policy]['status']
        assert status['status'] == 'PASS_equal_storage_native_T73_pair_not_physical'
        assert terminal['variants'][policy]['exit'] == 0
        assert all(status[stage]['exit'] == 0 for stage in ('build','run_off','run_on'))
        build = (folder / 'build.log').read_text()
        assert not re.search(r'%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT)',build)
        rtls = sorted((folder / 'rtl').rglob('*.sv'))
        assert len(rtls) == 107
        for path in rtls:
            relative = path.relative_to(folder)
            original = archive / 'scnn_issue_banked_synth_r1' / relative
            if policy == 'candidate' and relative.as_posix() == MODULE:
                assert path.read_text() == transform(original.read_text())
            else:
                assert path.read_bytes() == original.read_bytes(), str(relative)
        inherited = []
        for path in folder.rglob('*'):
            relative = path.relative_to(folder)
            original = archive / 'srnn_specialized_r1' / relative
            if path.is_file() and relative.parts[0] in ('scripts','tb','results','build') and original.is_file():
                assert path.read_bytes() == original.read_bytes(), str(relative)
                inherited.append(relative.as_posix())
        for name in ('tb_srnn_profile_measurement.sv','srnn_profile_measurement_generated_body.svh','input_commands.json'):
            assert (folder/name).read_bytes() == (archive/'srnn_specialized_r1'/name).read_bytes()
            inherited.append(name)
        source_list = [x.strip('"') for x in (folder/'sources.f').read_text().splitlines()]
        assert source_list == [p.relative_to(folder).as_posix() for p in rtls] + ['tb_srnn_profile_measurement.sv','tb/monitors/supplement_h40_activity_monitor.sv']
        variants[policy] = dict(stages={s:status[s] for s in ('build','run_off','run_on')},
                                warnings=dict(collections.Counter(re.findall(r'%Warning-(\w+)',build))),
                                actual_rtl_files=107, unchanged_input_files=inherited)
        checks[policy] = {}
        for mode in ('off','on'):
            log = (folder / ('run_' + mode + '.log')).read_text()
            assert not re.search(r'%Error|%Fatal|Assertion failed|\bFATAL\b|\bERROR\b|Out of memory|\bOOM\b|\bKilled\b', build+log, re.I)
            assert len(re.findall(r'^H40_ACTIVITY tile=',log,re.M)) == 12
            parsed = parse_log(log, expected['sample_definition'], 15)
            parsed['residency'] = residency(log)
            parsed['simulation_wall_seconds'] = status['run_'+mode]['wall_seconds']
            assert parsed == json.loads((folder/mode/'measurements.json').read_text())
            assert parsed['residency']['wout_loads'] == expected_loads(expected['sample_definition']['reference_counters'],mode,policy)
            assert parsed['residency']['wh1_loads'] == 1 and parsed['residency']['backbone_loads'] == 2
            counters = parsed['counters']
            reads = 450*256 + 256*20*parsed['residency']['wout_loads']
            for counter in ('m2_refill_read_accepted','m2_refill_response_accepted','store_forward_read_accepted','store_forward_response_accepted'):
                assert counters[counter] == reads
            checks[policy][mode] = parsed
            rows.append(dict(policy=policy,learning=mode,cycles=parsed['sample_service_cycles_including_refill_excluding_tb_checks'],
                initial_load_cycles=parsed['initial_load_and_state_cycles'],
                wh1_loads=parsed['residency']['wh1_loads'],wout_loads=parsed['residency']['wout_loads'],
                forward_reads=reads,refill_busy_cycles=counters['m2_refill_busy_cycles'],
                code_writes=counters['code_write_accepted'],residual_writes=counters['residual_write_accepted']))
    for mode in ('off','on'):
        a,b = checks['baseline'][mode],checks['candidate'][mode]
        assert {k:v for k,v in a['terminal'].items() if k!='end'} == {k:v for k,v in b['terminal'].items() if k!='end'}
        for counter in ('parameter_load_accepted','store_load_accepted','backbone_parameter_read_accepted','backbone_parameter_response_accepted',
                        'backbone_observation_accepted','m2_hidden_accepted','m2_logit_accepted','gradient_accepted','store_learning_read_accepted',
                        'store_learning_response_accepted','store_learning_write_accepted','code_write_accepted','residual_write_accepted','commit_pulse','detach_pulse'):
            assert a['counters'][counter] == b['counters'][counter], counter
    ratios = {mode:checks['baseline'][mode]['sample_service_cycles_including_refill_excluding_tb_checks']/checks['candidate'][mode]['sample_service_cycles_including_refill_excluding_tb_checks'] for mode in ('off','on')}
    report = dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS_independent_equal_storage_audit',
        raw_files=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),all_raw_sizes_match=True,
        variants=variants,measurements=rows,baseline_over_candidate_cycle_ratio=ratios,remote_at_audit=remote,
        scope='One fixed native B1T73 sample. Same final dual-context storage, service, TB and numerical reference. Baseline is the existing resident implementation, not the thrashing pre-resident design. Candidate changes only the numeric-cache version key; no candidate physical closure or peer superiority is asserted.',
        raw_inventory=inventory)
    (result/'terminal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (result/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_inventory','variants')},indent=2))


if __name__ == '__main__':
    main()
