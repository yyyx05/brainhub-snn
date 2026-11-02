"""Derive paired application-gain/update-cost measurements from an accepted run.

This script does not run adaptation or RTL.  It reads the completed fixed-rule
transfer experiment, preserves every paired observation, and reports software
state-change counters beside clean/shifted accuracy changes.  The counters are
not hardware cycles, physical writes, energy, or an equal-update-budget proof.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_shift_transfer_r1"
OUT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_learning_cost_summary_r1"


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return round(float(value), 9)


def main():
    with (SOURCE / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 12:
        raise ValueError(f"expected 12 accepted transfer rows, got {len(source_rows)}")
    methods = {row["method"] for row in source_rows}
    if methods != {"direct_response", "magnitude_sgd"}:
        raise ValueError(f"unexpected methods: {sorted(methods)}")

    episode_rows = []
    for row in source_rows:
        code = int(row["code_events"])
        residual = int(row["residual_events"])
        total = code + residual
        gain = float(row["shifted_gain_pp"])
        clean = float(row["clean_change_pp"])
        direct_key = (row["shift"], row["seed"], "direct_response")
        episode_rows.append({
            "shift": int(row["shift"]),
            "seed": int(row["seed"]),
            "method": row["method"],
            "code_events": code,
            "residual_events": residual,
            "software_state_change_events": total,
            "final_changed_addresses": int(row["final_changed_addresses"]),
            "clean_change_pp": finite(clean),
            "shifted_gain_pp": finite(gain),
            "shifted_gain_pp_per_10000_events": finite(gain * 10000 / total) if total else None,
            "clean_change_pp_per_10000_events": finite(clean * 10000 / total) if total else None,
            "saturations": int(row["saturations"]),
            "source_row": f"{row['shift']}:{row['seed']}:{row['method']}",
        })

    by_pair = {(r["shift"], r["seed"], r["method"]): r for r in episode_rows}
    for row in episode_rows:
        direct = by_pair[(row["shift"], row["seed"], "direct_response")]
        row["event_ratio_to_direct_pair"] = finite(
            row["software_state_change_events"] / direct["software_state_change_events"]
        )

    groups = defaultdict(list)
    for row in episode_rows:
        groups[(row["shift"], row["method"])].append(row)
    group_rows = []
    for (shift, method), rows in sorted(groups.items()):
        def vals(field):
            return [float(r[field]) for r in rows]

        group_rows.append({
            "shift": shift,
            "method": method,
            "support_resamples": len(rows),
            "shifted_gain_mean_pp": finite(sum(vals("shifted_gain_pp")) / len(rows)),
            "shifted_gain_min_pp": finite(min(vals("shifted_gain_pp"))),
            "shifted_gain_max_pp": finite(max(vals("shifted_gain_pp"))),
            "clean_change_mean_pp": finite(sum(vals("clean_change_pp")) / len(rows)),
            "software_events_mean": finite(sum(vals("software_state_change_events")) / len(rows)),
            "software_events_min": min(r["software_state_change_events"] for r in rows),
            "software_events_max": max(r["software_state_change_events"] for r in rows),
            "code_events_mean": finite(sum(vals("code_events")) / len(rows)),
            "residual_events_mean": finite(sum(vals("residual_events")) / len(rows)),
            "event_ratio_to_direct_mean": finite(sum(vals("event_ratio_to_direct_pair")) / len(rows)),
            "saturations_total": sum(r["saturations"] for r in rows),
        })

    derived = {
        "status": "PASS_offline_paired_learning_cost_summary",
        "scope": "offline_derivation_from_accepted_srnn_shift_transfer",
        "source": str(SOURCE / "comparison.csv"),
        "rows": len(episode_rows),
        "groups": len(group_rows),
        "interpretation": (
            "Closed-loop software adaptation records are paired by shift and support seed. "
            "Code/residual counters describe logical state-change events only; they do not "
            "measure RTL cycles, physical writes, energy, BRAM/DDR traffic, or equal update budgets."
        ),
        "negative_result_preserved": (
            "magnitude_sgd has materially fewer code events than direct_response but near-zero "
            "shifted gain on both new shifts; the evidence does not establish a universal SGD advantage."
        ),
        "episode_rows": episode_rows,
        "group_rows": group_rows,
    }

    OUT.mkdir(parents=False, exist_ok=False)
    write_csv(OUT / "episode_cost.csv", episode_rows, list(episode_rows[0]))
    write_csv(OUT / "group_cost.csv", group_rows, list(group_rows[0]))
    with (OUT / "derived_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(derived, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    colors = {"direct_response": "#267B80", "magnitude_sgd": "#757575"}
    labels = {"direct_response": "Direct 31/65", "magnitude_sgd": "SGD 3e-5"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2047244094, 3.4645669291))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.22, wspace=0.33)
    for method in ("direct_response", "magnitude_sgd"):
        rows = [r for r in episode_rows if r["method"] == method]
        xs = [r["software_state_change_events"] for r in rows]
        axes[0].scatter(xs, [r["shifted_gain_pp"] for r in rows],
                        color=colors[method], s=26, label=labels[method], zorder=3)
        axes[1].scatter(xs, [r["clean_change_pp"] for r in rows],
                        color=colors[method], s=26, label=labels[method], zorder=3)
    for ax, title, ylabel in (
        (axes[0], "a   Shifted-query gain vs logical update events", "Shifted gain (percentage points)"),
        (axes[1], "b   Clean change vs logical update events", "Clean change (percentage points)"),
    ):
        ax.axhline(0, color="#444444", lw=0.65)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Code + residual state-change events")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
        ax.tick_params(length=3, width=0.7)
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.text(0.09, 0.965, "Application gain and measured software update cost", fontsize=9, weight="bold", va="top")
    fig.text(0.09, 0.91, "Six paired support/shift observations per method; not an equal-budget or hardware-cost comparison", fontsize=7, va="top")
    fig.text(0.09, 0.035, "Counters are logical software events; no cycle, energy, BRAM/DDR, or physical-write inference.", fontsize=6)
    fig.savefig(OUT / "learning_cost.svg", facecolor="white")
    fig.savefig(OUT / "learning_cost.pdf", facecolor="white")
    fig.savefig(OUT / "learning_cost.png", dpi=600, facecolor="white")
    plt.close(fig)

    print(json.dumps({k: derived[k] for k in ("status", "rows", "groups", "interpretation", "negative_result_preserved")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
