"""First physical-cost comparison for the elastic reducer; no EDA or re-audit."""
import csv
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = "gradient_elastic_impl_r1"
PRIOR = "final_raw_source_impl_r1"
RESOURCE_KEYS = ("lut", "ff", "bram36eq", "dsp")
DELTA_KEYS = RESOURCE_KEYS + ("wns_ns", "whs_ns", "wpws_ns")


def comparison(old, new):
    return {
        "candidate_minus_prior": {key: new[key] - old[key] for key in DELTA_KEYS},
        "resource_change_percent": {
            key: 100 * (new[key] - old[key]) / old[key] if old[key] else None
            for key in RESOURCE_KEYS
        },
    }


def main():
    base = Path(local_path(ROOT / RESULT))
    directory = base / NAME
    audit = json.loads((directory / "terminal_audit.json").read_text())
    state = json.loads((directory / "status.json").read_text())
    old_summary = json.loads((base / PRIOR / "derived_metrics.json").read_text())
    if audit["status"] != "PASS_elastic_implementation_archive" or not audit["candidate_measurement_complete"]:
        raise ValueError("complete newly accepted archive required")
    if old_summary["status"] != "PASS_offline_paired_physical_summary":
        raise ValueError("existing comparison summary required")
    prior = dict(next(row for row in old_summary["rows"] if row["arm"] == "candidate"))
    prior["arm"] = "prior_candidate"
    prior["source"] = PRIOR + "/derived_metrics.json:rows.candidate"
    item = audit["arms"]["candidate"]
    report = item["reports"]["exported"]
    if report != item["reports"]["reopened"]:
        raise ValueError("new report export/reopen mismatch")
    current = dict(arm="elastic_candidate", stage="post_route_physopt",
        accepted_internal15ns=item["accepted_internal_implementation"],
        **report["resources"], **report["timing"],
        routable_nets=report["route"]["routable"], fully_routed_nets=report["route"]["fully_routed"],
        routing_errors=report["route"]["errors"],
        ordinary_log_warnings=item["worker"]["ordinary_warnings"],
        error_lines=len(item["worker"]["errors"]), critical_warning_lines=len(item["worker"]["critical_warnings"]),
        drc_warning_violations=item["diagnostics"]["drc"]["total"],
        methodology_warning_violations=item["diagnostics"]["methodology"]["total"],
        real_exit=state["arms"]["candidate"]["exit"],
        tool_wall_seconds=state["arms"]["candidate"]["wall_seconds"],
        source=NAME + "/terminal_audit.json:arms.candidate.reports.exported")
    samples = json.loads((directory / "disk_samples.json").read_text())
    result = dict(status="COMPLETE_offline_elastic_physical_comparison",
        rows=[prior, current], **comparison(prior, current),
        candidate_internal15ns=current["accepted_internal15ns"],
        disk_sample_count=len(samples),
        sampled_minimum_free={key: min(row[key] for row in samples) for key in ("data_free", "tmp_free")},
        no_peak_rss_measurement=True,
        worst_setup=dict(prior=old_summary["worst_setup"]["candidate"], candidate=item["worst_setup"]),
        boundary="Same device, 15ns and physical recipe; old accepted candidate not rerun. "
            "One implementation per version, not a multi-seed significance result. "
            "OOC clock-source/partition-pin and IO-delay gaps retained. "
            "Cycle gains are separate functional evidence, not a new physical cycle measurement. "
            "No board Fmax, power, bitstream or production merge.")
    destinations = [directory / "implementation_comparison.csv", directory / "derived_metrics.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("first derived comparison already exists; do not regenerate")
    with destinations[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prior))
        writer.writeheader()
        writer.writerows(result["rows"])
    with destinations[1].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    shutil.copy2(__file__, directory / "summary_runner.py")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
