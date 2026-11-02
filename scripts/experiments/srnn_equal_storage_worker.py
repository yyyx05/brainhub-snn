#!/usr/bin/env python3
"""Measure two equally resident SRNN cache-version policies, serially."""
import json
from pathlib import Path
import re
import subprocess
import sys

from srnn_resident_server_worker import ROOT, save, stage, stamp


def residency(log):
    matches = re.findall(r'^RESIDENCY_PASS backbone_loads=(\d+) wh1_loads=(\d+) wout_loads=(\d+) wout_reloads=(\d+)$', log, re.M)
    if len(matches) != 1:
        raise ValueError('missing or duplicate residency result')
    return dict(zip(('backbone_loads', 'wh1_loads', 'wout_loads', 'wout_reloads'), map(int, matches[0])))


def expected_loads(reference, mode, policy):
    # A final commit has no next forward in this sample and causes no refill yet.
    return 1 + sum(row['code_writes'] > 0 or (policy == 'baseline' and row['residual_writes'] > 0)
                   for row in reference[mode][:-1])


def measure():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_srnn_profile_measurement import parse_log, write_measurements, inspect_build_log
    accepted = json.loads((ROOT / 'accepted_profile_status.json').read_text())
    policy = json.loads((ROOT / 'comparison_input.json').read_text())['policy']
    record = dict(status='starting', started_utc=stamp(), policy=policy,
                  scope='Same final RTL and native B1T73 inputs; same resident capacity, ports and service. Only M2 cache key differs. Not new physical closure.')
    try:
        build = accepted['build']['command']
        stage(record, 'build', build, 3600)
        warnings = inspect_build_log((ROOT / 'build.log').read_text(errors='replace'))
        record['build_warnings'] = warnings
        if warnings['blocking']:
            raise ValueError(warnings['blocking'])
        for mode in ('off', 'on'):
            stage(record, 'run_' + mode, accepted['run_' + mode]['command'], 1800)
            log = (ROOT / ('run_' + mode + '.log')).read_text(errors='replace')
            result = parse_log(log, accepted['sample_definition'], 15)
            if not result['status'].startswith('PASS_'):
                raise ValueError(result['status'])
            counts = residency(log)
            wanted = expected_loads(accepted['sample_definition']['reference_counters'], mode, policy)
            if counts != dict(backbone_loads=2, wh1_loads=1, wout_loads=wanted, wout_reloads=wanted-1):
                raise ValueError(('cache policy count', counts, wanted))
            result['residency'] = counts
            result['simulation_wall_seconds'] = record['run_' + mode]['wall_seconds']
            (ROOT / mode).mkdir()
            write_measurements(ROOT / mode, result)
            record[mode + '_cycles'] = result['sample_service_cycles_including_refill_excluding_tb_checks']
            record[mode + '_residency'] = counts
            save(record)
        record.update(status='PASS_equal_storage_native_T73_pair_not_physical', finished_utc=stamp())
    except Exception as error:
        record.update(status='FAIL', error=repr(error), finished_utc=stamp())
        raise
    finally:
        save(record)


def pair():
    parent = ROOT.parent
    record = dict(status='running', started_utc=stamp(), variants={})
    output = parent / 'status.json'
    try:
        for policy in ('baseline', 'candidate'):
            record['active_variant'] = policy
            output.write_text(json.dumps(record, indent=2) + '\n')
            worker = parent / policy / 'scripts/experiments/srnn_equal_storage_worker.py'
            result = subprocess.run([sys.executable, str(worker), 'run'], stdin=subprocess.DEVNULL)
            record['variants'][policy] = dict(exit=result.returncode, status=json.loads((parent / policy / 'status.json').read_text()))
            if result.returncode:
                raise RuntimeError(policy + ' failed')
        record.update(status='PASS_equal_storage_comparison', finished_utc=stamp())
    except Exception as error:
        record.update(status='FAIL', error=repr(error), finished_utc=stamp())
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    pair() if sys.argv[1] == 'pair' else measure()
