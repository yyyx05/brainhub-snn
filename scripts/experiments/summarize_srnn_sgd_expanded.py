"""Derived tables and figure for the expanded development screen; no adaptation.

Quantitative grid, Python workflow. Claim: the prespecified extended SGD grid
covers effective code updates and degradation, but finds no positive validation
gain. Panel a: all endpoint accuracies. Panel b: all absolute code-LSB steps.
183 x 88 mm, editable SVG/PDF and 600 dpi PNG; descriptive single split only.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_sgd_expanded_r1"


def main():
    with (OUT / "dev/summary.json").open(encoding="utf-8") as handle:
        dev = json.load(handle)
    with (OUT / "report/summary.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    data = sorted((r for r in dev["episodes"] if r["method"] == "magnitude_sgd"),
                  key=lambda r: r["learning_rate"])
    direct = next(r for r in dev["episodes"] if r["method"] == "direct_response")
    new = [r for r in data if r["evidence"].startswith("new_")]
    x = np.array([r["learning_rate"] for r in data])
    steps = np.array([r["counters"]["code_lsb_steps"] for r in data])
    if len(data) != 12 or len(new) != 9 or not (x > 0).all() or not (steps > 0).all():
        raise ValueError("all twelve positive log-axis observations required")
    metrics = dict(scope="derived_from_new_dev_and_reused_report_not_new_measurement",
        dev_new_adaptations=len(new), dev_reused_rates=len(data)-len(new),
        independent_probability_gradient_values=sum(s["independent_checked_values"] for r in new for s in r["support"]),
        selected_rate=dev["selection"]["selected_rate"],
        positive_dev_shifted_rates=sum(r["shifted_accuracy_improvement"] > 0 for r in data),
        dev_code_events_min_max=[min(r["counters"]["code_changes"] for r in new), max(r["counters"]["code_changes"] for r in new)],
        dev_support_timesteps=[s["timesteps"] for s in new[0]["support"]],
        report_new_adaptations=report["new_adaptation_episodes"],
        report_mean_gains={},
        evidence_boundary="One reused development split and one historical shift; shared-query support resamples; no inferential test or optimizer-wide superiority")
    for method in ("direct_response", "magnitude_sgd"):
        group = [r for r in report["episodes"] if r["method"] == method]
        metrics["report_mean_gains"][method] = dict(
            shifted_pp=float(np.mean([100*r["shifted_accuracy_improvement"] for r in group])),
            clean_pp=float(np.mean([100*r["clean_accuracy_change"] for r in group])))
    with (OUT / "derived_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
        handle.write("\n")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2047244094, 3.4645669291), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.28, wspace=0.32)
    colors = {"clean": "#236C91", "shifted": "#B35C29"}
    for view, color in colors.items():
        y = np.array([r[view+"_after"]["accuracy_percent"] for r in data])
        a.plot(x, y, color=color, lw=1, zorder=2)
        a.scatter(x[:3], y[:3], s=17, edgecolor=color, facecolor="white", zorder=4)
        a.scatter(x[3:], y[3:], s=17, color=color, zorder=4)
        a.axhline(direct[view+"_after"]["accuracy_percent"], color=color, ls="--", lw=0.8, alpha=0.7)
    a.set(ylim=(0, 103), ylabel="Development accuracy (%)", title="a   Accuracy after 128 support samples")
    a.legend(handles=[Line2D([], [], color=colors["clean"], label="Clean"),
                      Line2D([], [], color=colors["shifted"], label="Shift +16")],
             loc="lower left", ncol=2, frameon=False)
    b.plot(x, steps, color="#496A50", lw=1, zorder=2)
    b.scatter(x[:3], steps[:3], s=17, edgecolor="#496A50", facecolor="white", zorder=4)
    b.scatter(x[3:], steps[3:], s=17, color="#496A50", zorder=4)
    b.axhline(direct["counters"]["code_lsb_steps"], color="#555555", ls="--", lw=0.8)
    b.set(yscale="log", ylim=(1, 1e9), ylabel="Accumulated absolute code-LSB steps",
          title="b   Effective update magnitude")
    b.text(0.02, 0.92, "Direct rule: 451,401 LSB steps", transform=b.transAxes, fontsize=6.5)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlim(1.5e-5, 20)
        ax.set_xlabel("SGD learning rate")
        ax.set_xticks([1e-4, 1e-2, 1, 10])
        ax.minorticks_off()
        ax.tick_params(width=0.7, length=3)
        ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
    legend = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", color="#555555", label="Imported old rate"),
              Line2D([], [], marker="o", linestyle="none", color="#555555", label="New development rate"),
              Line2D([], [], linestyle="--", color="#555555", label="Imported direct-rule endpoint")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.07), ncol=3, frameon=False)
    fig.text(0.08, 0.96, "Expanded integer Wout SGD screen", fontsize=9, fontweight="bold", va="top")
    fig.text(0.08, 0.90, "One fixed checkpoint; support 128 / query 688; no positive shifted gain in the 12-rate grid", fontsize=7, va="top")
    fig.text(0.08, 0.025, "Development only. Larger rates were not selected using official-test results. Dashed lines are not SGD observations.", fontsize=6)
    fig.savefig(OUT / "development_grid.svg", facecolor="white")
    fig.savefig(OUT / "development_grid.pdf", facecolor="white")
    fig.savefig(OUT / "development_grid.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
