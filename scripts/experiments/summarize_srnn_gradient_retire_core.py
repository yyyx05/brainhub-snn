"""Attribute the new reducer-retirement whole-core measurement; no RTL execution.

Question: does single-source reducer retirement overlap change closed-loop
service cost while preserving per-step state/events? A quantitative grid
shows absolute cold cost and all eight paired sequence outcomes. The existing
Python workflow supplies interval extraction, not inferred timing. All 588
steps are retained. One deterministic run per arm, no independent repeats or
confidence intervals; 183 x 92 mm editable PDF/SVG and 600-dpi PNG.
"""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
from summarize_srnn_rmw_turnover_core import RESULT, comparison_rows, extract
from summarize_srnn_source_core import write_csv

BASE = RESULT / 'srnn_m2_refill_turnover_core_r1'
NEW = RESULT / 'srnn_gradient_retire_core_r1'


def main():
    audit = json.loads((NEW / 'terminal_audit.json').read_text())
    if audit['status'] != 'PASS_gradient_retirement_core_terminal_audit':
        raise ValueError('accepted new measurement required')
    a = json.loads((BASE / 'checks.json').read_text())['stream_on']
    b = json.loads((NEW / 'checks.json').read_text())['stream_on']
    prior = json.loads((RESULT / 'srnn_source_core_r2/checks.json').read_text())
    ap, ah, ac, astep = extract(BASE, a)
    bp, bh, bc, bstep = extract(NEW, b)
    phases = comparison_rows(ap, bp, 'contiguous stages covering the cold total')
    nested = comparison_rows(ah, bh, 'nested hardware intervals; not additive with stages')
    counters = comparison_rows(ac, bc, 'last of eight cumulative snapshots, not their sum')
    seq_rows, step_rows = [], []
    if len(a['sequences']) != 8 or len(b['sequences']) != 8:
        raise ValueError('all eight sequences required')
    for x, y in zip(a['sequences'], b['sequences']):
        if any(x[k] != y[k] for k in x if k != 'cycles'):
            raise ValueError('sequence identity, numeric, event or residency mismatch')
        seq_rows.append(dict(sequence=x['sequence'], dataset_index=x['dataset_index'], steps=x['steps'],
            baseline_cycles=x['cycles'], candidate_cycles=y['cycles'], saved_cycles=x['cycles'] - y['cycles'],
            reduction_percent=100 * (x['cycles'] - y['cycles']) / x['cycles']))
    for x, y in zip(astep, bstep):
        if any(x[k] != y[k] for k in x if k not in ('cycles', 'step_cycles')):
            raise ValueError('per-step state/event or pairing mismatch')
        step_rows.append(dict(sequence=x['sequence'], timestep=x['timestep'],
            baseline_cycles=x['step_cycles'], candidate_cycles=y['step_cycles'],
            saved_cycles=x['step_cycles'] - y['step_cycles']))
    if a['source_frames'] != b['source_frames'] or a['source'] != b['source']:
        raise ValueError('FF source behavior changed')
    if a['refill_turnover'] != b['refill_turnover']:
        raise ValueError('refill events changed')
    ca, cb = a['total']['cycles'], b['total']['cycles']
    saved = ca - cb
    if sum(r['saved'] for r in phases) != saved or sum(r['saved_cycles'] for r in step_rows) != saved:
        raise ValueError('incomplete whole-core attribution')
    rmw = json.loads((RESULT / 'srnn_rmw_turnover_core_r1/checks.json').read_text())['stream_on']
    summary_rows = []
    for arm, d in (('buffered_original', prior['buffered_on']), ('stream_original', prior['stream_on']),
                   ('stream_rmw_turnover', rmw), ('stream_rmw_refill_turnover', a),
                   ('stream_rmw_refill_gradient_retirement', b)):
        total = d['total']['cycles']
        summary_rows.append(dict(arm=arm, cold_cycles=total, service_cycles=d['service_cycles'],
            initialization_cycles=d['initialization_including_reset_cycles'], logical_setup_cycles=d['state_setup_cycles'],
            cold_seconds_nominal_15ns=total * 15e-9, sequences_per_second_nominal_15ns=8 / (total * 15e-9),
            timesteps_per_second_nominal_15ns=588 / (total * 15e-9)))
    out = NEW / 'analysis'
    out.mkdir(exist_ok=False)
    for name in (Path(__file__).name, 'summarize_srnn_rmw_turnover_core.py', 'summarize_srnn_source_core.py'):
        shutil.copy2(Path(__file__).with_name(name), out / name)
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
                             gridspec_kw={'width_ratios': [1.4, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.30, top=0.82, wspace=0.42)
    values = [r['cold_cycles'] / 1e6 for r in summary_rows]
    axes[0].bar(range(5), values, width=0.62, color=['#A4A9AF', '#87939F', '#677D90', '#5C9089', '#AC704E'])
    axes[0].set_xticks(range(5), ['Buffered', 'Streaming', '+ RMW\nturnover', '+ Refill\nturnover', '+ Gradient\nretirement'])
    axes[0].set_ylabel('Cold total (million cycles)')
    axes[0].set_ylim(0, max(values) * 1.16)
    for pos, value in enumerate(values):
        axes[0].text(pos, value + max(values) * 0.025, f'{value:.3f}', ha='center', fontsize=7)
    reductions = [r['reduction_percent'] for r in seq_rows]
    axes[1].plot(range(8), reductions, marker='o', markersize=3, linewidth=0.8, color='#417C78')
    axes[1].axhline(0, color='#888888', linewidth=0.5)
    axes[1].set_xticks(range(8), [r['dataset_index'] for r in seq_rows])
    axes[1].set_xlabel('Original sequence index')
    axes[1].set_ylabel('Service cycle reduction (%)')
    axes[1].set_ylim(bottom=min(0, min(reductions)))
    for ax, letter, title in zip(axes, ('a', 'b'), ('Whole-core learning-on cost', 'Reducer-only paired outcomes')):
        ax.text(-0.15, 1.11, letter, transform=ax.transAxes, weight='bold', fontsize=8)
        ax.set_title(title, loc='left', fontsize=8, pad=12)
    fig.text(0.09, 0.065, '8 fixed representative sequences, 588 timesteps; prior arms reused, only gradient retirement is new.\n'
             'No exclusions or independent repeats. Cycle counts, not measured physical throughput.', fontsize=6.5)
    fig.savefig(out / 'gradient_retire_core_contribution.pdf')
    fig.savefig(out / 'gradient_retire_core_contribution.svg')
    fig.savefig(out / 'gradient_retire_core_contribution.png', dpi=600)
    plt.close(fig)
    svg = ET.parse(out / 'gradient_retire_core_contribution.svg').getroot()
    size = [float(svg.attrib[k].removesuffix('pt')) * 25.4 / 72 for k in ('width', 'height')]
    if any(abs(x - y) > 0.001 for x, y in zip(size, (183, 92))):
        raise ValueError('export dimensions changed')
    report = dict(status='PASS_offline_new_gradient_retirement_core_attribution_not_RTL',
        baseline_cold_cycles=ca, candidate_cold_cycles=cb, saved_cycles=saved,
        cold_reduction_percent=100 * saved / ca, cold_cycle_ratio=ca / cb,
        baseline_service_cycles=a['service_cycles'], candidate_service_cycles=b['service_cycles'],
        service_reduction_percent=100 * saved / a['service_cycles'],
        combined_vs_buffered_saved_cycles=summary_rows[0]['cold_cycles'] - cb,
        combined_vs_buffered_cold_reduction_percent=100 * (summary_rows[0]['cold_cycles'] - cb) / summary_rows[0]['cold_cycles'],
        improved_steps=sum(r['saved_cycles'] > 0 for r in step_rows),
        tied_steps=sum(r['saved_cycles'] == 0 for r in step_rows),
        regressed_steps=sum(r['saved_cycles'] < 0 for r in step_rows),
        sequence_reduction_percent_range=[min(reductions), max(reductions)],
        paired_sequences=len(seq_rows), paired_steps=len(step_rows), excluded_observations=0,
        phase_delta={r['name']: r['saved'] for r in phases},
        nested_hardware_delta={r['name']: r['saved'] for r in nested},
        changed_cumulative_counters=[r for r in counters if r['saved']],
        unchanged_refill_events=b['refill_turnover'], reducer_overlap_events='not instrumented in this core run', unchanged_source_frames=588, unchanged_step_state_events=588,
        statistics='One deterministic RTL run per arm; no independent repeats, CIs, significance or new distribution.',
        scope='Closed-loop direct learning, no changed numerical policy; not SGD, DDR, physical timing or production integration.',
        figure=dict(backend='python', dimensions_mm=size, editable_text=True, formats=['pdf', 'svg', 'png'],
                    mapping='cold_cycles / 1e6; per-sequence (baseline-candidate)/baseline * 100',
                    reuse='Existing Python extraction; style-only inheritance, five actual successive architectures'))
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


