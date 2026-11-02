"""Offline attribution of one new whole-core RMW turnover measurement.

Quantitative grid: absolute whole-core cost is the main panel; all eight
paired sequence reductions show heterogeneity. Python-only, 183 x 92 mm,
editable PDF/SVG and 600-dpi PNG. No exclusions, repeated-trial statistics,
physical timing claim, or new RTL execution. Existing plot style is reused;
all measurement mappings and cumulative-to-increment conversions are explicit.
"""
import collections
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from summarize_srnn_source_core import records, write_csv

ROOT = Path(__file__).resolve().parents[2]
if sys.platform == 'win32':
    ROOT = Path('\\\\?\\' + str(ROOT))
RESULT = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization'
BASE = RESULT / 'srnn_source_core_r2'
NEW = RESULT / 'srnn_rmw_turnover_core_r1'


def extract(directory, data, log_name='run_stream_on.log'):
    log = (directory / log_name).read_text()
    intervals = records(log, 'SRNN_PROFILE_INTERVAL')
    stage, hardware = collections.Counter(), collections.Counter()
    end = 0
    for row in intervals:
        start, stop, cycles = (int(row[k]) for k in ('start', 'end', 'cycles'))
        if stop - start != cycles or cycles < 0:
            raise ValueError('interval length')
        if row['kind'] == 'stage':
            if start != end:
                raise ValueError('noncontiguous stage')
            end = stop
            stage[row['phase']] += cycles
        elif row['kind'] == 'hardware':
            hardware[row['phase']] += cycles
        else:
            raise ValueError('unknown interval kind')
    if end != data['total']['cycles'] or sum(stage.values()) != end:
        raise ValueError('stage coverage')
    cumulative = collections.defaultdict(list)
    for row in records(log, 'SRNN_PROFILE_COUNT'):
        cumulative[row['name']].append(int(row['value']))
    if any(len(v) != 8 or v != sorted(v) for v in cumulative.values()):
        raise ValueError('cumulative counter series')
    counts = {k: v[-1] for k, v in cumulative.items()}
    steps = []
    sequence, last, expected_t = None, 0, 0
    for line in log.splitlines():
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if line.startswith('REPRESENTATIVE_BEGIN '):
            sequence, last, expected_t = int(fields['sequence']), 0, 0
        elif line.startswith('SRNN_SAME_DUT_STEP '):
            values = {k: int(v) for k, v in fields.items()}
            if sequence is None or values['t'] != expected_t or values['cycles'] < last:
                raise ValueError('step order or cumulative cycle count')
            steps.append(dict(sequence=sequence, timestep=values.pop('t'),
                              step_cycles=values['cycles'] - last, **values))
            last = values['cycles']
            expected_t += 1
    if len(steps) != 588 or sum(x['step_cycles'] for x in steps) != data['service_cycles']:
        raise ValueError('all service timesteps required')
    return stage, hardware, counts, steps


def comparison_rows(a, b, name):
    if a.keys() != b.keys():
        raise ValueError('counter/phase names differ')
    return [dict(name=k, baseline=a[k], candidate=b[k], saved=a[k] - b[k],
                 accounting=name) for k in a]


def main():
    audit = json.loads((NEW / 'terminal_audit.json').read_text())
    if audit['status'] != 'PASS_new_turnover_core_terminal_audit':
        raise ValueError('accepted new measurement required')
    prior = json.loads((BASE / 'checks.json').read_text())
    a = prior['stream_on']
    b = json.loads((NEW / 'checks.json').read_text())['stream_on']
    ap, ah, ac, astep = extract(BASE, a)
    bp, bh, bc, bstep = extract(NEW, b)
    phases = comparison_rows(ap, bp, 'contiguous cold-total stages')
    nested = comparison_rows(ah, bh, 'nested intervals; do not add to stage totals')
    counters = comparison_rows(ac, bc, 'last of eight cumulative snapshots, not their sum')
    seq_rows, step_rows = [], []
    if len(a['sequences']) != 8 or len(b['sequences']) != 8:
        raise ValueError('all sequences required')
    for x, y in zip(a['sequences'], b['sequences']):
        if any(x[k] != y[k] for k in ('sequence', 'dataset_index', 'steps')):
            raise ValueError('sequence pairing')
        seq_rows.append(dict(sequence=x['sequence'], dataset_index=x['dataset_index'], steps=x['steps'],
            baseline_cycles=x['cycles'], candidate_cycles=y['cycles'], saved_cycles=x['cycles'] - y['cycles'],
            reduction_percent=100 * (x['cycles'] - y['cycles']) / x['cycles']))
    for x, y in zip(astep, bstep):
        if any(x[k] != y[k] for k in x if k not in ('cycles', 'step_cycles')):
            raise ValueError('per-step state/event or pairing difference')
        step_rows.append(dict(sequence=x['sequence'], timestep=x['timestep'],
            baseline_cycles=x['step_cycles'], candidate_cycles=y['step_cycles'],
            saved_cycles=x['step_cycles'] - y['step_cycles']))
    if a['source_frames'] != b['source_frames'] or a['source'] != b['source']:
        raise ValueError('feedforward source behavior changed')
    cold_a, cold_b = a['total']['cycles'], b['total']['cycles']
    saved = cold_a - cold_b
    if sum(r['saved'] for r in phases) != saved or sum(r['saved_cycles'] for r in step_rows) != saved:
        raise ValueError('attribution does not cover complete difference')
    out = NEW / 'analysis'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('summarize_srnn_source_core.py'), out / 'summarize_srnn_source_core.py')
    summary_rows = []
    for arm, d in (('buffered_original_rmw', prior['buffered_on']), ('stream_original_rmw', a), ('stream_turnover_rmw', b)):
        total = d['total']['cycles']
        summary_rows.append(dict(arm=arm, cold_cycles=total, service_cycles=d['service_cycles'],
            initialization_cycles=d['initialization_including_reset_cycles'], logical_setup_cycles=d['state_setup_cycles'],
            cold_seconds_nominal_15ns=total * 15e-9, sequences_per_second_nominal_15ns=8 / (total * 15e-9),
            timesteps_per_second_nominal_15ns=588 / (total * 15e-9)))
    for name, rows in (('stage_comparison', phases), ('nested_hardware_comparison', nested),
                       ('cumulative_counter_comparison', counters), ('sequence_comparison', seq_rows),
                       ('timestep_comparison', step_rows), ('whole_core_summary', summary_rows)):
        write_csv(out / (name + '.csv'), rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    width_inches, height_inches = 183 / 25.4, 92 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches),
                             gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.82, wspace=0.42)
    values = [r['cold_cycles'] / 1e6 for r in summary_rows]
    axes[0].bar(range(3), values, width=0.6, color=['#8B929A', '#587792', '#A56F5C'])
    axes[0].set_xticks(range(3), ['Buffered\noriginal RMW', 'Streaming\noriginal RMW', 'Streaming\nturnover RMW'])
    axes[0].set_ylabel('Cold total (million cycles)')
    axes[0].set_ylim(0, max(values) * 1.18)
    for pos, value in enumerate(values):
        axes[0].text(pos, value + max(values) * 0.025, f'{value:.3f}', ha='center', fontsize=7)
    axes[1].plot(range(8), [r['reduction_percent'] for r in seq_rows],
                 marker='o', markersize=3, linewidth=0.8, color='#A56F5C')
    axes[1].axhline(0, color='#888888', linewidth=0.5)
    axes[1].set_xticks(range(8), [r['dataset_index'] for r in seq_rows])
    axes[1].set_xlabel('Original sequence index')
    axes[1].set_ylabel('Service cycle reduction (%)')
    axes[1].set_ylim(bottom=min(0, min(r['reduction_percent'] for r in seq_rows)))
    for ax, letter, title in zip(axes, ('a', 'b'), ('Whole-core learning-on cost', 'RMW-only paired outcomes')):
        ax.text(-0.15, 1.11, letter, transform=ax.transAxes, weight='bold', fontsize=8)
        ax.set_title(title, loc='left', fontsize=8, pad=12)
    fig.text(0.09, 0.055, '8 fixed representative sequences, 588 timesteps; accepted prior arms reused, only turnover is new.\n'
             'No exclusions, repeated trials, or physical timing result. RMW-only comparison holds streaming fixed.', fontsize=6.5)
    fig.savefig(out / 'rmw_core_contribution.pdf')
    fig.savefig(out / 'rmw_core_contribution.svg')
    fig.savefig(out / 'rmw_core_contribution.png', dpi=600)
    plt.close(fig)
    report = dict(status='PASS_offline_new_core_attribution_not_RTL',
        baseline_cold_cycles=cold_a, candidate_cold_cycles=cold_b, saved_cycles=saved,
        cold_reduction_percent=100 * saved / cold_a, cold_cycle_ratio=cold_a / cold_b,
        baseline_service_cycles=a['service_cycles'], candidate_service_cycles=b['service_cycles'],
        service_reduction_percent=100 * saved / a['service_cycles'],
        combined_vs_buffered_saved_cycles=prior['buffered_on']['total']['cycles'] - cold_b,
        combined_vs_buffered_cold_reduction_percent=100 * (prior['buffered_on']['total']['cycles'] - cold_b) / prior['buffered_on']['total']['cycles'],
        improved_steps=sum(r['saved_cycles'] > 0 for r in step_rows),
        tied_steps=sum(r['saved_cycles'] == 0 for r in step_rows),
        regressed_steps=sum(r['saved_cycles'] < 0 for r in step_rows),
        paired_sequences=len(seq_rows), paired_steps=len(step_rows), excluded_observations=0,
        phase_delta={r['name']: r['saved'] for r in phases},
        nested_hardware_delta={r['name']: r['saved'] for r in nested},
        changed_cumulative_counters=[r for r in counters if r['saved']],
        unchanged_source_frames=588, unchanged_step_state_events=588,
        statistics='One deterministic RTL run per arm, no independent repeats, CIs, significance or new-distribution claim.',
        scope='Closed-loop direct learning; not SGD, DDR, physical timing, author-system replication or production integration.',
        figure=dict(backend='python', dimensions_mm=[183, 92], editable_text=True, formats=['pdf', 'svg', 'png']))
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
