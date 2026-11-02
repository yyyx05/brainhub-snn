"""Plot accepted SRNN analyses without rerunning simulation, synthesis or audits."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from run_srnn_resident_remote import ROOT, RESULT, local_path

CORE = "srnn_gradient_elastic_core_r1/analysis"
PHYSICAL = "gradient_elastic_impl_r1/derived_metrics.json"
OUTPUT = "paper/figures/srnn_elastic_20261021"
PRIOR = "#58759B"
ELASTIC = "#23877D"
BUFFERED = "#A4ABB3"


def load_data():
    evidence = Path(local_path(ROOT / RESULT))
    core = json.loads((evidence / CORE / "analysis.json").read_text())
    physical = json.loads((evidence / PHYSICAL).read_text())
    with (evidence / CORE / "timestep_comparison.csv").open(newline="") as handle:
        steps = [{key: int(value) for key, value in row.items()}
                 for row in csv.DictReader(handle)]
    with (evidence / CORE / "sequence_comparison.csv").open(newline="") as handle:
        sequences = list(csv.DictReader(handle))
    # Check only the figure's data mapping, not the already accepted experiment.
    assert len(steps) == core["paired_timesteps"] == 588
    assert len(sequences) == core["paired_sequences"] == 8
    assert sum(row["saved_cycles"] for row in steps) == core["saved_cycles"]
    assert all(row["baseline_cycles"] > 0 for row in steps)
    assert all(row["saved_cycles"] == row["baseline_cycles"] - row["candidate_cycles"] for row in steps)
    assert [sum(row["saved_cycles"] > 0 for row in steps),
            sum(row["saved_cycles"] == 0 for row in steps),
            sum(row["saved_cycles"] < 0 for row in steps)] == [572, 16, 0]
    for row in sequences:
        assert sum(step["sequence"] == int(row["sequence"]) for step in steps) == int(row["timesteps"])
    return core, steps, sequences, physical


def panel(ax, label, title):
    ax.text(-0.055, 1.10, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.set_title(title, loc="left", fontsize=8, pad=11)


def build_figure(core, steps, sequences, physical):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2047244094, 6.6929133858), facecolor="white")  # 183 x 170 mm
    grid = fig.add_gridspec(3, 2, left=0.10, right=0.98, bottom=0.065, top=0.94,
                           hspace=0.84, wspace=0.42, height_ratios=[1.0, 1.17, 0.92])
    a = fig.add_subplot(grid[0, :])
    panel(a, "a", "All paired timesteps: elastic reducer versus prior candidate")
    savings = [100 * row["saved_cycles"] / row["baseline_cycles"] for row in steps]
    ticks, labels, offset = [], [], 0
    for index, seq in enumerate(sequences):
        count = int(seq["timesteps"])
        if index % 2 == 0:
            a.axvspan(offset - 0.5, offset + count - 0.5, color="#F1F3F5", zorder=0)
        ticks.append(offset + (count - 1) / 2)
        labels.append(f'{seq["sequence"]} ({count})')
        offset += count
    a.scatter(range(len(steps)), savings, s=5, color=ELASTIC, linewidths=0, zorder=2)
    a.axhline(0, color="#777777", linewidth=0.6, linestyle="--", zorder=1)
    a.set(xlim=(-5, len(steps) + 4), ylim=(-0.7, max(savings) * 1.18),
          ylabel="Cycle reduction (%)", xlabel="Sequence (timesteps)", xticks=ticks, xticklabels=labels)
    a.text(0.01, 0.95, "588/588 shown: 572 lower, 16 equal, 0 higher", transform=a.transAxes,
           fontsize=6.5, va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    b = fig.add_subplot(grid[1, 0])
    panel(b, "b", "Cold protocol totals")
    totals = [core["original_buffered_cold_cycles"], core["baseline_cold_cycles"], core["candidate_cold_cycles"]]
    b.bar(range(3), [value / 1e6 for value in totals], color=[BUFFERED, PRIOR, ELASTIC], width=0.58)
    for i, value in enumerate(totals):
        b.text(i, value / 1e6 + 0.45, f"{value / 1e6:.3f}", ha="center", fontsize=7)
    b.set(ylim=(0, 31), ylabel="Cycles (million)", xticks=range(3),
          xticklabels=["Buffered\nRAM", "Prior\ncandidate", "Elastic\nreducer"])
    b.text(0.5, 0.97, f'{core["cold_reduction_percent"]:.3f}% below prior', ha="center", va="top",
           transform=b.transAxes, fontsize=6.5)
    b.text(0.5, -0.31, f'{core["combined_vs_buffered_reduction_percent"]:.3f}% below buffered RAM (direct ratio)',
           ha="center", transform=b.transAxes, fontsize=6.2)

    c = fig.add_subplot(grid[1, 1])
    panel(c, "c", "Overlapping learning counters")
    by_name = {row["name"]: row for row in core["changed_cumulative_counters"]}
    counters = [by_name["pure_learning_busy_cycles"], by_name["gradient_stalled_cycles"]]
    for key, label, color, dx in (("baseline", "Prior", PRIOR, -0.18),
                                   ("candidate", "Elastic", ELASTIC, 0.18)):
        positions = [i + dx for i in range(2)]
        values = [row[key] / 1e6 for row in counters]
        c.bar(positions, values, width=0.32, color=color, label=label)
        for x, value in zip(positions, values):
            c.text(x, value + 0.25, f"{value:.3f}", ha="center", fontsize=6.2)
    c.set(ylim=(0, 12.3), ylabel="Counter value (million cycles)", xticks=[0, 1],
          xticklabels=["Pure learning\nbusy", "Gradient\nstalled"])
    c.legend(ncol=2, fontsize=6.5, loc="upper center", handlelength=1.1, columnspacing=1.2)
    c.text(0.5, -0.31, "Stalls increase; these counters must not be added", ha="center",
           transform=c.transAxes, fontsize=6.2)

    d = fig.add_subplot(grid[2, :])
    panel(d, "d", "Routed internal OOC implementation: same device, 15 ns and recipe")
    d.axis("off")
    prior, candidate = physical["rows"]
    rows = [
        ["LUT", f'{prior["lut"]:,}', f'{candidate["lut"]:,}',
         f'+{candidate["lut"]-prior["lut"]:,} (+{physical["resource_change_percent"]["lut"]:.3f}%)'],
        ["FF", f'{prior["ff"]:,}', f'{candidate["ff"]:,}', f'{candidate["ff"]-prior["ff"]:+,}'],
        ["BRAM tile / DSP", f'{prior["bram36eq"]} / {prior["dsp"]}',
         f'{candidate["bram36eq"]} / {candidate["dsp"]}', "Unchanged"],
        ["WNS (ns)", f'{prior["wns_ns"]:+.3f}', f'{candidate["wns_ns"]:+.3f}',
         f'{1000 * (candidate["wns_ns"] - prior["wns_ns"]):+.0f} ps margin'],
        ["WHS (ns)", f'{prior["whs_ns"]:+.3f}', f'{candidate["whs_ns"]:+.3f}',
         f'{1000 * (candidate["whs_ns"] - prior["whs_ns"]):+.0f} ps margin'],
        ["Setup / hold / pulse failures", *[" / ".join(str(arm[key]) for key in
         ("setup_failing", "hold_failing", "pulse_failing")) for arm in (prior, candidate)],
         "Both internally accepted" if all(arm["accepted_internal15ns"] for arm in (prior, candidate))
         else "See acceptance record"],
    ]
    table = d.table(cellText=rows, colLabels=["Metric", "Prior", "Elastic", "Change"],
                    colWidths=[0.34, 0.18, 0.18, 0.30], cellLoc="center", bbox=[0, -0.10, 1, 1.11])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#D6DBDF")
        if row == 0:
            cell.set_facecolor("#F1F3F5")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
        if row == 0 and col in (1, 2):
            cell.set_text_props(color=PRIOR if col == 1 else ELASTIC)
    fig.text(0.10, 0.012, "One implementation per version; incomplete OOC interface constraints. Not full-system signoff.",
             fontsize=6.2)
    return fig


def main():
    core, steps, sequences, physical = load_data()
    fig = build_figure(core, steps, sequences, physical)
    out = Path(local_path(ROOT / OUTPUT))
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "srnn_elastic_evidence"
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    fig.savefig(str(stem) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            bounds = item.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
                outside.append(item.get_text())
    qa = {"backend": "Python/matplotlib", "matplotlib": matplotlib.__version__,
          "size_mm": [183, 170], "preview_dpi": 300, "tiff_dpi": 600,
          "source_steps": len(steps), "plotted_steps": len(steps), "excluded_steps": 0,
          "sequence_count": len(sequences), "simulation_rerun": False,
          "physical_rerun": False, "terminal_audit_rerun": False,
          "text_outside_canvas": outside,
          "sources": [CORE + "/analysis.json", CORE + "/timestep_comparison.csv",
                      CORE + "/sequence_comparison.csv", PHYSICAL],
          "statistics": "Deterministic paired records, not independent replicates. No CI, p-values or error bars.",
          "scope": "SRNN elastic only; SCNN row-address reuse is not combined or shown as physically accepted."}
    (out / "plot_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
