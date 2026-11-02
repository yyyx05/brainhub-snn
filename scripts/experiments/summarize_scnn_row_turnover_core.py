"""First paired latency breakdown of the accepted publisher-only SCNN run."""
import csv
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = "scnn_row_turnover_core_r1"


def paired_rows(before, after):
    if len(before) != len(after):
        raise ValueError("paired record count")
    rows = []
    for a, b in zip(before, after):
        identity = {k: v for k, v in a.items() if k != "cycles"}
        if identity != {k: v for k, v in b.items() if k != "cycles"}:
            raise ValueError("paired record identity")
        rows.append(dict(identity, baseline_cycles=a["cycles"], candidate_cycles=b["cycles"],
                         saved_cycles=a["cycles"] - b["cycles"],
                         reduction_percent=100 * (a["cycles"] - b["cycles"]) / a["cycles"]))
    return rows


def intervals(data):
    starts = data["windows"]
    result = [dict(stage="initial_setup", cycles=starts[0]["cycle"])]
    for i, item in enumerate(starts):
        end = starts[i + 1]["cycle"] if i + 1 < len(starts) else data["final"]["cycles"]
        result.append(dict(stage="window_" + str(item["window"]),
                           cycles=end - item["cycle"]))
    return result


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive = Path(local_path(ROOT / RESULT / NAME))
    accepted = json.loads((archive / "terminal_audit.json").read_text())
    if accepted["status"] != "PASS_SCNN_row_turnover_archive":
        raise ValueError("accepted new core required")
    before = json.loads((archive / "expected_checks.json").read_text())
    after = json.loads((archive / "checks.json").read_text())
    frames = paired_rows(before["frames"], after["frames"])
    layers = paired_rows(before["layers"], after["layers"])
    windows = paired_rows(intervals(before), intervals(after))
    ca, cb = before["final"]["cycles"], after["final"]["cycles"]
    saved = ca - cb
    if len(frames) != 52 or len(layers) != 52 or sum(r["saved_cycles"] for r in windows) != saved:
        raise ValueError("complete original protocol required")
    result = dict(status="DERIVED_SCNN_row_turnover_latency", baseline="final_scnn_functional_r1",
        candidate=NAME, baseline_cycles=ca, candidate_cycles=cb, saved_cycles=saved,
        reduction_percent=100 * saved / ca,
        improved_frames=sum(r["saved_cycles"] > 0 for r in frames),
        tied_frames=sum(r["saved_cycles"] == 0 for r in frames),
        regressed_frames=sum(r["saved_cycles"] < 0 for r in frames),
        frame_saved_cycles=sum(r["saved_cycles"] for r in frames),
        outside_frame_saved_cycles=saved - sum(r["saved_cycles"] for r in frames),
        layer_saved_cycles=sum(r["saved_cycles"] for r in layers),
        frame_reduction_percent_range=[min(r["reduction_percent"] for r in frames),
                                       max(r["reduction_percent"] for r in frames)],
        native_service_transactions_unchanged={k: before["native_service"][k] == after["native_service"][k]
            for k in after["native_service"] if k != "end_cycle"},
        window_intervals=windows, excluded_frames=0,
        boundary="New publisher-only candidate, no elastic SRNN reducer. Original 52 Conv8 tail "
            "descriptors with CPU Conv7 feature service, three cold windows and query4. "
            "Window intervals partition protocol time; frames/layers are nested, not additive. "
            "Not full raw graph/T20/Test, new accuracy, physical timing or author-system superiority.")
    output = archive / "analysis"
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__, output / "summary_runner.py")
    for name, rows in (("frame_comparison", frames), ("layer_comparison", layers),
                       ("window_interval_comparison", windows)):
        write_csv(output / (name + ".csv"), rows)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
