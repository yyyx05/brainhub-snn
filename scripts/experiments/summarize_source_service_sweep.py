"""Offline parser tests and grouped figure for the completed local service sweep."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys

from run_srnn_resident_remote import ROOT, RESULT, local_path


def csv_write(path, rows):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(local_path(ROOT))
    out = root / RESULT / "source_service_sweep_local_r1"
    spec = importlib.util.spec_from_file_location("archived_service_sweep", out / "experiment_runner.py")
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    log = (out / "run.log").read_text(errors="replace")
    checks = archived.parse_log(log, out / "fixture")
    assert checks == json.loads((out / "checks.json").read_text())
    first = re.search(r"^SCAN_FRAME .+$", log, re.M).group()
    completion = "SERVICE_SWEEP_PASS frames=7104 current_checks=3196800 policies=3 scenarios=4"
    bad_total = re.sub(r"\btotal=(\d+)", lambda m: "total=" + str(int(m[1]) + 1), first)
    mutations = {
        "missing_frame": log.replace(first, "", 1),
        "duplicate_frame": log.replace(first, first + "\n" + first, 1),
        "missing_completion": log.replace(completion, "", 1),
        "duplicate_completion": log + "\n" + completion,
        "missing_current_check": log.replace(first, first.replace("checked=450", "checked=449"), 1),
        "broken_cycle_closure": log.replace(first, bad_total, 1),
        "runtime_error": log + "\n%Error: injected parser test\n",
    }
    tests = {"complete_actual_transcript": "PASS"}
    for name, text in mutations.items():
        try:
            archived.parse_log(text, out / "fixture")
        except (ValueError, KeyError):
            tests[name] = "rejected"
        else:
            raise AssertionError(name + " was not rejected")

    preparation = json.loads((out / "preparation.json").read_text())
    original = root / "verification/experiments/nearest_neighbor/stream_r1"
    for item in preparation["unchanged_inputs"]:
        assert (original / item["path"]).read_bytes() == (out / item["path"]).read_bytes()
        assert (out / item["path"]).stat().st_size == item["bytes"]
    assert not list(out.rglob("*.part"))
    with (out / "fixture/frames.csv").open(newline="") as handle:
        fixture = list(csv.DictReader(handle))
    rows = checks["frames"]
    grouped = []
    for scenario in range(4):
        for index in sorted({int(r["dataset_index"]) for r in fixture}):
            frames = {f for f, r in enumerate(fixture) if int(r["dataset_index"]) == index}
            sums = [sum(r["total"] for r in rows if r["scenario"] == scenario
                        and r["policy"] == p and r["frame"] in frames) for p in range(3)]
            grouped.append(dict(scenario=scenario, condition=archived.SCENARIOS[scenario],
                                dataset_index=index, frames=len(frames), scan_cycles=sums[0],
                                buffered_cycles=sums[1], streaming_cycles=sums[2],
                                buffered_over_streaming=sums[1] / sums[2]))
    boundaries = []
    names = ("all_zero", "one_source_count255", "mixed_and_last_source", "all_one")
    for scenario in range(4):
        for f, name in enumerate(names, 588):
            values = [r["total"] for r in rows if r["scenario"] == scenario and r["frame"] == f]
            assert len(values) == 3
            boundaries.append(dict(scenario=scenario, boundary=name, scan_cycles=values[0],
                                   buffered_cycles=values[1], streaming_cycles=values[2],
                                   buffered_over_streaming=values[1] / values[2]))
    csv_write(out / "sequence_comparison.csv", grouped)
    csv_write(out / "boundary_comparison.csv", boundaries)
    diagnostics = {}
    for stage in ("build", "run"):
        raw = (out / (stage + ".log")).read_bytes()
        diagnostics[stage] = dict(verilator_warning_tokens=raw.count(b"%Warning"),
                                  verilator_error_tokens=raw.count(b"%Error"),
                                  wsl_prefix_present=b"w\x00s\x00l\x00:" in raw)
        assert diagnostics[stage]["verilator_error_tokens"] == 0
    summary = dict(parser_tests=tests, unchanged_inputs=len(preparation["unchanged_inputs"]),
                   unchanged_bytes=sum(r["bytes"] for r in preparation["unchanged_inputs"]),
                   diagnostics=diagnostics, comparisons=checks["comparisons"],
                   sequence_ranges=[dict(condition=archived.SCENARIOS[s],
                       minimum=min(r["buffered_over_streaming"] for r in grouped if r["scenario"] == s),
                       maximum=max(r["buffered_over_streaming"] for r in grouped if r["scenario"] == s))
                       for s in range(4)],
                   scope="Offline derivation and parser tests, not another RTL run or independent input distribution.")
    (out / "analysis_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw(out, grouped, boundaries, checks["comparisons"])
    print(json.dumps(summary, indent=2))


def draw(out, grouped, boundaries, totals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2047244, 3.7795276),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.83, wspace=0.35)
    labels = ("4-cycle\nsupply", "16-cycle\nsupply", "Burst\nconsumer", "4-cycle +\nburst")
    for ax, letter, title in zip(axes, "ab", ("Eight real sequences", "Four directed boundaries")):
        ax.set_title(title, loc="left", fontsize=9, pad=23)
        ax.text(-0.15, 1.22, letter, transform=ax.transAxes, fontsize=10, weight="bold")
        ax.axhline(1, color="0.55", lw=0.7, ls="--")
        ax.set_xticks(range(4), labels, fontsize=7)
        ax.set_ylabel("Buffered / streaming cycles")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0.95, 2.15)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    for s in range(4):
        seq = [r for r in grouped if r["scenario"] == s]
        axes[0].scatter([s + (i - 3.5) * 0.045 for i in range(8)],
                        [r["buffered_over_streaming"] for r in seq],
                        s=19, facecolor="#4f86a6", edgecolor="white", linewidth=0.35,
                        label="One complete sequence" if s == 0 else None, zorder=3)
        value = totals[s]["buffered_over_streaming"]
        axes[0].scatter(s, value, marker="D", s=24, color="#202020",
                        label="Ratio of all cycle sums" if s == 0 else None, zorder=4)
        axes[0].annotate(f"{value:.3f}", (s, value), xytext=(0, 11),
                         textcoords="offset points", ha="center", fontsize=7)
    colors = ("#727272", "#b5793f", "#607b55", "#4f86a6")
    names = ("all_zero", "one_source_count255", "mixed_and_last_source", "all_one")
    short = ("All zero", "One source, count 255", "Mixed / last source", "All one")
    for name, label, color, marker in zip(names, short, colors, ("o", "s", "^", "D")):
        data = [r["buffered_over_streaming"] for r in boundaries if r["boundary"] == name]
        axes[1].plot(range(4), data, marker=marker, ms=3.7, lw=0.8, color=color, label=label)
    axes[0].legend(loc="upper left", fontsize=6.8)
    axes[1].legend(loc="upper left", fontsize=6.5)
    fig.text(0.09, 0.10, "588 real frames, retained in 8 sequences; 4 boundaries shown separately.",
             fontsize=7)
    fig.text(0.09, 0.055, "Each condition: identical counts, weights and consumers. No IID or confidence-interval claim.",
             fontsize=7)
    fig.savefig(out / "service_sensitivity.svg")
    fig.savefig(out / "service_sensitivity.pdf")
    fig.savefig(out / "service_sensitivity.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
