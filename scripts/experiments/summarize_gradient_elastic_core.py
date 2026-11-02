"""Attribute the first elastic-reducer core measurement using existing parsers."""
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from summarize_srnn_rmw_turnover_core import RESULT, comparison_rows, extract
from summarize_srnn_source_core import write_csv

NAME = "srnn_gradient_elastic_core_r1"


def main():
    new = RESULT / NAME
    accepted = json.loads((new / "terminal_audit.json").read_text())
    if accepted["status"] != "PASS_elastic_gradient_core_archive":
        raise ValueError("accepted new core required")
    a = json.loads((new / "expected_checks.json").read_text())
    b = json.loads((new / "checks.json").read_text())["wrapper_on"]
    ap, ah, ac, astep = extract(RESULT / "final_raw_source_functional_r1", a, "run_wrapper_on.log")
    bp, bh, bc, bstep = extract(new, b, "run_wrapper_on.log")
    phases = comparison_rows(ap, bp, "contiguous cold-total stages")
    nested = comparison_rows(ah, bh, "nested hardware intervals; not additive to stages")
    counters = comparison_rows(ac, bc, "last cumulative snapshot, not snapshot sum")
    sequences, steps = [], []
    for x, y in zip(a["sequences"], b["sequences"]):
        if any(x[k] != y[k] for k in x if k != "cycles"):
            raise ValueError("sequence non-latency record changed")
        sequences.append(dict(sequence=x["sequence"], dataset_index=x["dataset_index"],
            timesteps=x["steps"], baseline_cycles=x["cycles"], candidate_cycles=y["cycles"],
            saved_cycles=x["cycles"] - y["cycles"],
            reduction_percent=100 * (x["cycles"] - y["cycles"]) / x["cycles"]))
    for x, y in zip(astep, bstep):
        if any(x[k] != y[k] for k in x if k not in ("cycles", "step_cycles")):
            raise ValueError("per-step non-latency record changed")
        steps.append(dict(sequence=x["sequence"], timestep=x["timestep"],
            baseline_cycles=x["step_cycles"], candidate_cycles=y["step_cycles"],
            saved_cycles=x["step_cycles"] - y["step_cycles"]))
    ca, cb = a["total"]["cycles"], b["total"]["cycles"]
    saved = ca - cb
    if len(sequences) != 8 or len(steps) != 588 or sum(r["saved"] for r in phases) != saved or sum(
            r["saved_cycles"] for r in steps) != saved:
        raise ValueError("incomplete paired attribution")
    raw = (new / "run_wrapper_on.log").read_text()
    marker = re.findall(r"^GRADIENT_ELASTIC_CORE_PASS (.*)$", raw, re.M)
    if len(marker) != 1:
        raise ValueError("coverage marker")
    overlap = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", marker[0])}
    ram = json.loads((RESULT / "srnn_source_core_r2/checks.json").read_text())["buffered_on"]
    original = ram["total"]["cycles"]
    report = dict(status="DERIVED_new_elastic_core_attribution",
        baseline_cold_cycles=ca, candidate_cold_cycles=cb, saved_cycles=saved,
        cold_reduction_percent=100 * saved / ca,
        original_buffered_cold_cycles=original,
        combined_vs_buffered_reduction_percent=100 * (original - cb) / original,
        improved_steps=sum(r["saved_cycles"] > 0 for r in steps),
        tied_steps=sum(r["saved_cycles"] == 0 for r in steps),
        regressed_steps=sum(r["saved_cycles"] < 0 for r in steps),
        sequence_reduction_percent_range=[min(r["reduction_percent"] for r in sequences),
                                          max(r["reduction_percent"] for r in sequences)],
        phase_delta={r["name"]: r["saved"] for r in phases},
        nested_hardware_delta={r["name"]: r["saved"] for r in nested},
        changed_cumulative_counters=[r for r in counters if r["saved"]],
        overlap=overlap, paired_sequences=8, paired_timesteps=588, excluded_observations=0,
        boundary="One new deterministic RTL run versus accepted unchanged-input prior. "
            "Numerical/state/traffic/commit assertions preserved. No new distribution, "
            "SGD/author-system comparison, physical timing or production merge; nested counters not additive.")
    out = new / "analysis"
    out.mkdir(exist_ok=False)
    for filename in (Path(__file__).name, "summarize_srnn_rmw_turnover_core.py",
                     "summarize_srnn_source_core.py"):
        shutil.copy2(Path(__file__).with_name(filename), out / filename)
    for label, rows in (("stage_comparison", phases), ("nested_hardware_comparison", nested),
                        ("cumulative_counter_comparison", counters),
                        ("sequence_comparison", sequences), ("timestep_comparison", steps)):
        write_csv(out / (label + ".csv"), rows)
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
