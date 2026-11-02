"""First paired accounting of accepted final FC measurements versus historical A."""
import csv
import json
from pathlib import Path

from run_srnn_resident_remote import ROOT, RESULT, local_path


def main():
    base = Path(local_path(ROOT / RESULT))
    out = base / "final_fc_functional_r1"
    audit = json.loads((out / "terminal_audit.json").read_text())
    if audit["status"] != "PASS_final_FC_wrapper_archive":
        raise ValueError("final FC not accepted")
    rows = []
    stage_rows = []
    stable = ("input_events", "hidden_events", "hidden_checks", "dot_checks", "q6_checks",
              "commits", "groups", "forward_reads", "learning_reads", "code_writes", "residual_writes")
    for measured in audit["measurements"]:
        batch, arm = measured["batch"], measured["arm"]
        name = f"run_{batch}_{arm}_checks.json"
        old = json.loads((base / "fc_representative_a_r1" / name).read_text())
        new = json.loads((out / name).read_text())
        if any(old["legacy"][k] != new["legacy"][k] for k in stable):
            raise ValueError("numerical/service totals differ")
        old_cold = sum(v["cycles"] for v in old["stages"].values())
        new_cold = sum(v["cycles"] for v in new["stages"].values())
        if new_cold != measured["counted_postreset_cycles"]:
            raise ValueError("audit/summary mismatch")
        rows.append(dict(batch=batch, arm=arm, samples=16, steps=99,
                         old_runtime=old["runtime_cycles"], new_runtime=new["runtime_cycles"],
                         old_postreset=old_cold, new_postreset=new_cold,
                         commits=new["legacy"]["commits"], code_writes=new["legacy"]["code_writes"],
                         residual_writes=new["legacy"]["residual_writes"]))
        for stage, value in old["stages"].items():
            stage_rows.append(dict(batch=batch, arm=arm, stage=stage, old_cycles=value["cycles"],
                                   new_cycles=new["stages"][stage]["cycles"],
                                   delta_cycles=new["stages"][stage]["cycles"] - value["cycles"]))
    totals = {}
    for arm in ("off", "on"):
        selected = [r for r in rows if r["arm"] == arm]
        total = {k: sum(r[k] for r in selected) for k in
                 ("samples", "old_runtime", "new_runtime", "old_postreset", "new_postreset",
                  "commits", "code_writes", "residual_writes")}
        for metric in ("runtime", "postreset"):
            total[metric + "_reduction_percent"] = 100 * (
                total["old_" + metric] - total["new_" + metric]) / total["old_" + metric]
        total["new_postreset_samples_per_second_15ns_projection"] = (
            total["samples"] / (total["new_postreset"] * 15e-9))
        totals[arm] = total
    for name, records in (("comparison.csv", rows), ("stage_comparison.csv", stage_rows)):
        with (out / name).open("x", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = dict(
        source="Accepted final wrapper and original A three B16T99 cold off/on pairs.",
        numerical_and_service_totals_unchanged=list(stable), totals=totals,
        scope="First local derivation, no RTL rerun. Postreset excludes six initial reset clocks; "
              "includes parameter load/refill/state/configuration. Three batches are purposive, "
              "not IID/full Test. Old A versus final combination is not a single-module ablation. "
              "No new FC code changes/accuracy/board or physical claim.")
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
