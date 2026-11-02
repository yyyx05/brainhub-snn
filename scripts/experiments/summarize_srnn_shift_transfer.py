"""Descriptive transfer results, prediction transitions, and figure; no rerun.

Python quantitative grid, 183x88 mm. Both panels show every paired support
resample: shifted gain and clean change. No error bars or significance claims.
"""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_shift_transfer_r1"
METHODS = ("direct_response", "magnitude_sgd")


def csv_write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (OUT / "summary.json").open(encoding="utf-8") as handle:
        source = json.load(handle)
    if source["status"] != "PASS_two_new_shift_conditions_integer_adaptation" or len(source["episodes"]) != 12:
        raise ValueError("complete two-condition result required")
    with np.load(OUT / "features/query_shift-16.npz") as data:
        labels = data["labels"]
    transitions, per_class, metrics = [], [], []
    for r in source["episodes"]:
        for view in ("clean", "shifted"):
            before = np.asarray(r["baseline_"+view]["predictions"])
            after = np.asarray(r[view+"_after"]["predictions"])
            if len(before) != 2264 or len(after) != 2264:
                raise ValueError("full official query required")
            gained = (before != labels) & (after == labels)
            lost = (before == labels) & (after != labels)
            changed = before != after
            if int(gained.sum()-lost.sum()) != r[view+"_after"]["correct"]-r["baseline_"+view]["correct"]:
                raise AssertionError("transition net does not match observed accuracy")
            transitions.append(dict(shift=r["shift"], seed=r["seed"], method=r["method"], view=view,
                wrong_to_correct=int(gained.sum()), correct_to_wrong=int(lost.sum()),
                wrong_to_other_wrong=int((changed & ~gained & ~lost).sum()), changed_predictions=int(changed.sum()),
                wrong_to_correct_indices=np.flatnonzero(gained).tolist(), correct_to_wrong_indices=np.flatnonzero(lost).tolist()))
            for label in range(20):
                mask = labels == label
                per_class.append(dict(shift=r["shift"], seed=r["seed"], method=r["method"], view=view,
                    label=label, samples=int(mask.sum()), before_correct=int(((before==labels)&mask).sum()),
                    after_correct=int(((after==labels)&mask).sum()), wrong_to_correct=int((gained&mask).sum()),
                    correct_to_wrong=int((lost&mask).sum())))
    for shift in (-16, 32):
        for method in METHODS:
            rows = [r for r in source["episodes"] if r["shift"] == shift and r["method"] == method]
            gains = [r["shifted_gain_pp"] for r in rows]
            cleans = [r["clean_change_pp"] for r in rows]
            metrics.append(dict(shift=shift, method=method, support_resamples=len(rows),
                shifted_gain_mean_pp=float(np.mean(gains)), shifted_gain_range_pp=[min(gains), max(gains)],
                clean_change_mean_pp=float(np.mean(cleans)), clean_change_range_pp=[min(cleans), max(cleans)],
                code_events=[r["counters"]["code_changes"] for r in rows],
                residual_events=[r["counters"]["residual_changes"] for r in rows],
                saturations=sum(r["counters"]["saturations"] for r in rows)))
    dropped, frozen_checks = [], 0
    for path in sorted((OUT / "features").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        raw = sum(sum(b["raw_count_mass"]) for b in meta["batches"])
        lost = sum(sum(b["dropped_count_mass"]) for b in meta["batches"])
        frozen_checks += meta["checked_frozen_logits"]
        dropped.append(dict(feature=path.stem, shift=meta["shift"], samples=len(meta["indices"]),
                            original_count_mass=raw, dropped_count_mass=lost, dropped_percent=100*lost/raw))
    derived = dict(scope="offline_descriptive_analysis_not_new_adaptation", groups=metrics,
        forward_independent_values=sum(sum(c["checked_values"].values()) for c in source["independent_forward_checks"]),
        frozen_head_logits_checked=frozen_checks,
        gradient_probability_independent_values=source["independent_probability_gradient_values"],
        record_views_evaluated=sum(x["samples"] for x in dropped), transitions=transitions,
        interpretation="Six condition/support pairs, not six independent models; no per-condition tuning, tests, or confidence intervals")
    with (OUT / "derived_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(derived, handle, indent=2, allow_nan=False)
        handle.write("\n")
    csv_write(OUT / "per_class.csv", per_class)
    csv_write(OUT / "count_loss.csv", dropped)
    csv_write(OUT / "prediction_transitions.csv", [{k:v for k,v in row.items() if not k.endswith("_indices")} for row in transitions])

    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":7, "axes.titlesize":8, "axes.labelsize":7, "legend.fontsize":6.5,
        "axes.linewidth":0.7, "axes.spines.top":False, "axes.spines.right":False,
        "svg.fonttype":"none", "pdf.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2047244094, 3.4645669291))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.80, bottom=0.28, wspace=0.32)
    palette = {"direct_response":"#267B80", "magnitude_sgd":"#757575"}
    shapes = ("o", "s", "^")
    offsets = (-0.05, 0, 0.05)
    for ax, field, title in zip(axes, ("shifted_gain_pp", "clean_change_pp"),
                               ("a   Shifted-query gain", "b   Clean-query change")):
        for group, shift in enumerate((-16, 32)):
            for i, seed in enumerate((228474, 228475, 228476)):
                pair = [next(r for r in source["episodes"] if r["shift"] == shift and r["seed"] == seed and r["method"] == method) for method in METHODS]
                xs = [group*3+offsets[i], group*3+1+offsets[i]]
                ax.plot(xs, [r[field] for r in pair], color="#BBBBBB", lw=0.65, zorder=1)
                for x, row in zip(xs, pair):
                    ax.scatter(x, row[field], marker=shapes[i], color=palette[row["method"]], s=25, zorder=3)
        ax.axhline(0, color="#444444", lw=0.65)
        ax.set_xticks([0,1,3,4], ["Direct", "SGD", "Direct", "SGD"])
        ax.set(xlim=(-0.5,4.5), ylim=(-0.12,2.1), yticks=[0,0.5,1,1.5,2], title=title,
               ylabel="Accuracy change (percentage points)")
        ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
        ax.tick_params(length=3, width=0.7)
        ax.text(0.2, -0.23, "Channel shift -16", transform=ax.transAxes, ha="center", fontsize=7)
        ax.text(0.8, -0.23, "Channel shift +32", transform=ax.transAxes, ha="center", fontsize=7)
    fig.text(0.085, 0.96, "Fixed-rule transfer to two new input conditions", fontsize=9, weight="bold", va="top")
    fig.text(0.085, 0.90, "Support 128; full query 2264; direct 31/65 and SGD 3e-5 retained without retuning", fontsize=7, va="top")
    handles = [Line2D([], [], color="#555555", linestyle="none", marker=shape,
                       markersize=4, label=f"Support seed {seed}") for shape,seed in zip(shapes,(228474,228475,228476))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53,0.045), ncol=3, frameon=False)
    fig.text(0.085,0.018,"All six paired support/condition observations shown; shared checkpoint/query, not independent training repeats.", fontsize=6)
    fig.savefig(OUT / "condition_transfer.svg", facecolor="white")
    fig.savefig(OUT / "condition_transfer.pdf", facecolor="white")
    fig.savefig(OUT / "condition_transfer.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(json.dumps({k:v for k,v in derived.items() if k != "transitions"}, indent=2))


if __name__ == "__main__":
    main()
