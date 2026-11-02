"""Derive the first paired routed-resource table from the accepted archive."""
import csv
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path
from run_final_raw_source_implementation import NAME, ARMS


def main():
    directory = Path(local_path(ROOT / RESULT / NAME))
    audit = json.loads((directory / 'terminal_audit.json').read_text())
    state = json.loads((directory / 'status.json').read_text())
    if audit['status'] != 'PASS_paired_implementation_archive' or not audit['pair_measurements_complete']:
        raise ValueError('complete accepted paired archive required')
    rows = []
    for arm in ARMS:
        item = audit['arms'][arm]
        report = item['reports']['exported']
        if state['arms'][arm]['exit'] != 0 or report != item['reports']['reopened']:
            raise ValueError('paired exported/reopened report mismatch: ' + arm)
        row = dict(arm=arm, stage='post_route_physopt',
            accepted_internal15ns=item['accepted_internal_implementation'],
            **report['resources'], **report['timing'],
            routable_nets=report['route']['routable'], fully_routed_nets=report['route']['fully_routed'],
            routing_errors=report['route']['errors'],
            ordinary_log_warnings=item['worker']['ordinary_warnings'],
            error_lines=len(item['worker']['errors']), critical_warning_lines=len(item['worker']['critical_warnings']),
            drc_warning_violations=item['diagnostics']['drc']['total'],
            methodology_warning_violations=item['diagnostics']['methodology']['total'],
            real_exit=state['arms'][arm]['exit'], tool_wall_seconds=state['arms'][arm]['wall_seconds'],
            source='terminal_audit.json:arms.' + arm + '.reports.exported')
        rows.append(row)
    base, candidate = rows
    delta = {key: candidate[key] - base[key] for key in
        ('lut', 'ff', 'bram36eq', 'dsp', 'wns_ns', 'whs_ns', 'wpws_ns')}
    percent = {key: 100 * delta[key] / base[key] for key in ('lut', 'ff', 'bram36eq', 'dsp')}
    samples = json.loads((directory / 'disk_samples.json').read_text())
    derived = dict(status='PASS_offline_paired_physical_summary', rows=rows,
        candidate_minus_baseline=delta, resource_change_percent=percent,
        both_internal_implementations_accepted=audit['both_internal_implementations_accepted'],
        disk_sample_count=len(samples),
        sampled_minimum_free={key: min(row[key] for row in samples) for key in ('data_free', 'tmp_free')},
        no_peak_rss_measurement=True,
        boundary='Two serial implementations of the accepted producer-plus-core DCP pair. Internal OOC 15ns only; IO delays remain absent. No board Fmax, multi-seed stability, energy, power, or new functional-cycle result. Existing SRNN cycle reduction is separate.',
        worst_setup={arm: audit['arms'][arm]['worst_setup'] for arm in ARMS})
    with (directory / 'implementation_comparison.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / 'derived_metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(derived, stream, indent=2, allow_nan=False)
        stream.write('\n')
    shutil.copy2(__file__, directory / 'summary_runner.py')
    print(json.dumps(derived, indent=2))


if __name__ == '__main__':
    main()
