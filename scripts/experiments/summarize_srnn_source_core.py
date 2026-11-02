"""Offline attribution of the accepted raw-source whole-core comparison.

Figure: quantitative grid, Python-only, 183 x 92 mm, editable PDF/SVG plus
PNG. Panels distinguish absolute whole-core cycles from paired sequence
reductions. All eight original sequences and both learning modes are used;
there are no repeated trials, confidence intervals, or significance claims.
"""
import collections
import csv
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if sys.platform == 'win32':
    ROOT = Path('\\\\?\\' + str(ROOT))
BASE = ROOT / 'results/paper_evidence/supplement_20261015/performance_optimization/srnn_source_core_r2'


def records(log, prefix):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in re.findall(r'^' + prefix + r' (.+)$', log, re.M)]


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit = json.loads((BASE / 'terminal_audit.json').read_text())
    if audit['status'] != 'PASS_raw_source_A_core_comparison_terminal_audit':
        raise ValueError('accepted terminal archive required')
    checks = json.loads((BASE / 'checks.json').read_text())
    out = BASE / 'analysis'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    stage_rows, hardware_rows, count_rows, summary_rows = [], [], [], []
    stages, frames = {}, {}
    for arm, data in checks.items():
        log = (BASE / ('run_' + arm + '.log')).read_text()
        intervals = records(log, 'SRNN_PROFILE_INTERVAL')
        stage = [r for r in intervals if r['kind'] == 'stage']
        total = data['total']['cycles']
        end = 0
        totals = collections.Counter()
        for row in stage:
            start, stop, cycles = (int(row[k]) for k in ('start', 'end', 'cycles'))
            if start != end or stop - start != cycles or cycles < 0:
                raise ValueError('stage gap/overlap ' + arm)
            end = stop
            totals[row['phase']] += cycles
        if end != total or sum(totals.values()) != total:
            raise ValueError('stage totals ' + arm)
        stages[arm] = totals
        stage_rows += [dict(arm=arm, phase=k, cycles=v) for k, v in totals.items()]
        hardware = collections.defaultdict(list)
        for row in intervals:
            if row['kind'] != 'hardware':
                continue
            start, stop, cycles = (int(row[k]) for k in ('start', 'end', 'cycles'))
            if stop - start != cycles or cycles < 0:
                raise ValueError('hardware interval')
            hardware[row['phase']].append(cycles)
        hardware_rows += [dict(arm=arm, phase=k, intervals=len(v), cycles=sum(v),
                               accounting='nested intervals; not additive to stage totals')
                          for k, v in hardware.items()]
        # profile_finish prints cumulative counters after each sequence.
        cumulative = collections.defaultdict(list)
        for row in records(log, 'SRNN_PROFILE_COUNT'):
            cumulative[row['name']].append(int(row['value']))
        for name, values in cumulative.items():
            if len(values) != 8 or values != sorted(values):
                raise ValueError('cumulative counter series ' + name)
            count_rows.append(dict(arm=arm, name=name, value=values[-1],
                                   snapshots=8))
        frames[arm] = data['source_frames']
        ff = sum(r['cycles'] for r in frames[arm])
        summary_rows.append(dict(
            arm=arm, cold_cycles=total,
            initialization_cycles=data['initialization_including_reset_cycles'],
            logical_setup_cycles=data['state_setup_cycles'],
            service_cycles=data['service_cycles'], ff_interval_cycles=ff,
            ff_interval_is_subset_of_backbone_stage=True,
            cold_seconds_nominal_15ns=total * 15e-9,
            sequence_per_second_nominal_15ns=8 / (total * 15e-9),
            timestep_per_second_nominal_15ns=588 / (total * 15e-9),
            accepted_input_words=data['source']['words'],
            accepted_ff_groups_including_tail=data['source']['groups'],
            tails=data['source']['tails'],
            count_stalls=data['source']['count_stalls'],
            group_stalls=data['source']['group_stalls']))
    sequence_rows, frame_rows, comparison = [], [], []
    for learning in ('off', 'on'):
        a, b = (checks[p + '_' + learning] for p in ('buffered', 'stream'))
        for x, y in zip(a['sequences'], b['sequences']):
            if (x['sequence'], x['dataset_index'], x['steps']) != (y['sequence'], y['dataset_index'], y['steps']):
                raise ValueError('sequence pairing')
            sequence_rows.append(dict(learning=learning, sequence=x['sequence'],
                dataset_index=x['dataset_index'], steps=x['steps'],
                buffered_service_cycles=x['cycles'], stream_service_cycles=y['cycles'],
                saved_cycles=x['cycles'] - y['cycles'],
                reduction_percent=100 * (x['cycles'] - y['cycles']) / x['cycles']))
        paired = []
        for x, y in zip(frames['buffered_' + learning], frames['stream_' + learning]):
            if any(x[k] != y[k] for k in ('sequence', 'timestep', 'words', 'groups', 'tails')):
                raise ValueError('frame pairing')
            row = dict(learning=learning, sequence=x['sequence'], timestep=x['timestep'],
                words=x['words'], groups_including_tail=x['groups'],
                buffered_ff_cycles=x['cycles'], stream_ff_cycles=y['cycles'],
                saved_cycles=x['cycles'] - y['cycles'])
            paired.append(row)
        frame_rows += paired
        ac, bc = a['total']['cycles'], b['total']['cycles']
        ff_saved = sum(r['saved_cycles'] for r in paired)
        phase_delta = {k: stages['buffered_' + learning][k] - stages['stream_' + learning][k]
                       for k in stages['buffered_' + learning].keys() | stages['stream_' + learning].keys()}
        if sum(phase_delta.values()) != ac - bc:
            raise ValueError('cycle attribution')
        comparison.append(dict(learning=learning, cold_saved_cycles=ac - bc,
            cold_reduction_percent=100 * (ac - bc) / ac, cold_cycle_ratio=ac / bc,
            service_saved_cycles=a['service_cycles'] - b['service_cycles'],
            service_reduction_percent=100 * (a['service_cycles'] - b['service_cycles']) / a['service_cycles'],
            ff_saved_cycles=ff_saved,
            ff_reduction_percent=100 * ff_saved / sum(r['buffered_ff_cycles'] for r in paired),
            ff_cycle_ratio=sum(r['buffered_ff_cycles'] for r in paired) / sum(r['stream_ff_cycles'] for r in paired),
            improved_frames=sum(r['saved_cycles'] > 0 for r in paired),
            tied_frames=sum(r['saved_cycles'] == 0 for r in paired),
            regressed_frames=sum(r['saved_cycles'] < 0 for r in paired),
            phase_delta=phase_delta))
    if len(sequence_rows) != 16 or len(frame_rows) != 1176:
        raise ValueError('all observations required')
    for name, rows in (('stage_totals', stage_rows), ('nested_hardware_totals', hardware_rows),
                       ('cumulative_counters', count_rows), ('service_summary', summary_rows),
                       ('sequence_comparison', sequence_rows), ('frame_comparison', frame_rows)):
        write_csv(out / (name + '.csv'), rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'legend.frameon': False})
    width_inches, height_inches = 183 / 25.4, 92 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches),
                             gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.82, wspace=0.44)
    colors = ('#587792', '#A56F5C')
    for policy, label, color, dx in zip(('buffered', 'stream'), ('Buffered RAM', 'Streaming'), colors, (-0.18, 0.18)):
        values = [checks[policy + '_' + l]['total']['cycles'] / 1e6 for l in ('off', 'on')]
        axes[0].bar([x + dx for x in (0, 1)], values, width=0.34, label=label, color=color)
    axes[0].set_xticks([0, 1], ['Learning off', 'Learning on'])
    axes[0].set_ylabel('Cold total (million cycles)')
    axes[0].legend(loc='upper left', fontsize=6.5)
    axes[0].set_ylim(bottom=0)
    indices = [r['dataset_index'] for r in checks['buffered_off']['sequences']]
    for learning, label, color, marker in zip(('off', 'on'), ('Learning off', 'Learning on'), colors, ('o', 's')):
        rows = [r for r in sequence_rows if r['learning'] == learning]
        axes[1].plot(range(8), [r['reduction_percent'] for r in rows],
                     marker=marker, markersize=3, linewidth=0.8, color=color, label=label)
    axes[1].axhline(0, color='#888888', linewidth=0.5)
    axes[1].set_xticks(range(8), indices)
    axes[1].set_xlabel('Original sequence index')
    axes[1].set_ylabel('Service cycle reduction (%)')
    axes[1].legend(loc='best', fontsize=6.5)
    for ax, letter, title in zip(axes, ('a', 'b'), ('Whole-core cost', 'Paired sequence outcomes')):
        ax.text(-0.15, 1.11, letter, transform=ax.transAxes, weight='bold', fontsize=8)
        ax.set_title(title, loc='left', fontsize=8, pad=12)
    fig.text(0.09, 0.065, '8 fixed representative sequences, 588 timesteps per arm; no exclusions or repeated trials.\n'
             'Same raw inputs and core; new protocol has no physical timing result.', fontsize=6.5)
    fig.savefig(out / 'source_core_contribution.pdf')
    fig.savefig(out / 'source_core_contribution.svg')
    fig.savefig(out / 'source_core_contribution.png', dpi=600)
    plt.close(fig)
    report = dict(status='PASS_offline_attribution_not_new_RTL_run', comparisons=comparison,
        source_rows=2352, paired_frames=1176, paired_sequences=16, excluded_observations=0,
        stage_accounting='Exact contiguous stages cover cold total; hardware/FF intervals nested, not added.',
        performance='Cycle counts measured in RTL. 15ns rates nominal only; no inherited physical timing.',
        statistics='One deterministic run per arm, same 8 sequences; no CI, significance, or new distribution claim.',
        figure=dict(backend='python', dimensions_mm=[183, 92], editable_text=True,
                    formats=['pdf', 'svg', 'png']))
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
