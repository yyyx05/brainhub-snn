#!/usr/bin/env python3
"""Run the selected compact core against the unchanged native T73 reference."""
import json
from pathlib import Path
import re
import sys

from srnn_resident_server_worker import ROOT, save, stage, stamp
from srnn_equal_storage_worker import residency, expected_loads


def main():
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_srnn_profile_measurement import parse_log, write_measurements, inspect_build_log
    accepted=json.loads((ROOT/'accepted_profile_status.json').read_text())
    record=dict(status='starting',started_utc=stamp(),scope='Compact M2 Context1 with explicit 512 rows; native B1T73. Not new physical closure.')
    try:
        stage(record,'build',accepted['build']['command'],3600)
        record['build_warnings']=inspect_build_log((ROOT/'build.log').read_text(errors='replace'))
        if record['build_warnings']['blocking']:
            raise ValueError(record['build_warnings']['blocking'])
        for mode in ('off','on'):
            stage(record,'run_'+mode,accepted['run_'+mode]['command'],1800)
            log=(ROOT/('run_'+mode+'.log')).read_text(errors='replace')
            result=parse_log(log,accepted['sample_definition'],15)
            if not result['status'].startswith('PASS_'):
                raise ValueError(result['status'])
            if re.findall(r'^M2_COMPACT_SELECTED depth=(\d+)$',log,re.M) != ['512']:
                raise ValueError('selected capacity not exercised')
            counts=residency(log)
            wanted=expected_loads(accepted['sample_definition']['reference_counters'],mode,'baseline')
            if counts != dict(backbone_loads=2,wh1_loads=1,wout_loads=wanted,wout_reloads=wanted-1):
                raise ValueError(('residency changed',counts,wanted))
            result['residency']=counts
            result['simulation_wall_seconds']=record['run_'+mode]['wall_seconds']
            (ROOT/mode).mkdir()
            write_measurements(ROOT/mode,result)
            record[mode+'_cycles']=result['sample_service_cycles_including_refill_excluding_tb_checks']
            record[mode+'_residency']=counts
            save(record)
        record.update(status='PASS_compact512_native_T73_pair_not_physical',finished_utc=stamp())
    except Exception as error:
        record.update(status='FAIL',error=repr(error),finished_utc=stamp())
        raise
    finally:
        save(record)


if __name__=='__main__': main()
