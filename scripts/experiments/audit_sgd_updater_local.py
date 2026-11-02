"""Decode the WSL prefix explicitly; retain original parser FAIL and raw bytes."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/paper_evidence/supplement_20261015/performance_optimization/sgd_updater_local_r1'


def transcript(raw):
    marker=b'RESET_FLUSH policy=0 discarded=1\n'
    offset=raw.index(marker)
    prefix=raw[:offset]
    if not prefix.startswith(b'w\x00s\x00l\x00:\x00') or len(prefix)%2 or not prefix.endswith(b'\r\x00\n\x00'):
        raise ValueError('unexpected WSL prefix')
    decoded=prefix.decode('utf-16le')
    if not decoded.startswith('wsl: ') or 'localhost' not in decoded:
        raise ValueError('unrecognized WSL notice')
    return decoded+raw[offset:].decode('ascii'),dict(prefix_bytes=offset,encoding='UTF-16LE prefix, ASCII simulator transcript',notice=decoded)


def main():
    target=OUT/'terminal_audit.json'
    if target.exists():
        raise FileExistsError('already accepted; no repeat audit')
    state=json.loads((OUT/'status.json').read_text())
    assert state['status']=='FAIL' and state['error']=="ValueError('reset coverage')"
    assert state['build']['exit']==state['run']['exit']==0
    spec=importlib.util.spec_from_file_location('archived_sgd_runner',OUT/'executed_runner.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    log,encoding=transcript((OUT/'run.log').read_bytes())
    n=state['fixture']['count']
    checks=runner.parse(log,n)
    for corrupted in (
        log.replace('RESET_FLUSH policy=0 discarded=1\n','',1),
        log.replace('policy=1 discarded=4','policy=1 discarded=3',1),
        log.replace('UPDATERS_PASS','MISSING_PASS',1),
        log.replace('count=67938','count=67937',1),
        log.replace('latency_min=4 latency_max=4','latency_min=3 latency_max=4',1),
        log+'\n%Error: synthetic error\n',
    ):
        try:
            runner.parse(corrupted,n)
        except ValueError:
            continue
        raise AssertionError('corrupt transcript accepted')
    inputs=[ROOT/'verification/experiments/sgd_updater_r1'/x for x in
            ('fixed_sgd_update.sv','direct_update_wrapper.sv','tb_updaters.sv')]
    inputs.append(ROOT/'rtl/learning/unified_tp_direct_residual_update.sv')
    assert all(p.read_bytes()==(OUT/p.name).read_bytes() for p in inputs)
    assert not list(OUT.rglob('*.part'))
    diagnostics={}
    for stage in ('build','run'):
        text=(OUT/(stage+'.log')).read_bytes().decode('utf-8',errors='replace')
        diagnostics[stage]=dict(warnings=len(re.findall(r'%Warning',text)),errors=len(re.findall(r'%Error|%Fatal',text)))
        assert diagnostics[stage]['errors']==0
    report=dict(status='PASS_independent_component_RTL_transcript_acceptance',time=dt.datetime.now().astimezone().isoformat(),
                original_wrapper_status='FAIL',original_failure_preserved=True,
                reason='Original UTF-8 decode left a NUL before the first RESET_FLUSH. Explicit prefix decode only; no simulation rerun or raw edits.',
                encoding=encoding,original_archived_parser_used=True,corrupt_variants_rejected=6,
                four_RTL_inputs_byte_equal=True,checks=checks,diagnostics=diagnostics,
                dcp_crc='not applicable: no DCP',scope=state['scope'])
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
