#!/usr/bin/env python3
"""Collect matched FC final-top logs without converting cycles into board claims."""
import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fixture_path(commands, root=ROOT):
    """Rebase a retrieved run's project-local fixture, retaining its original record."""
    path_type = PureWindowsPath if PureWindowsPath(commands["cwd"]).drive else PurePosixPath
    relative = path_type(commands["fixture"]).relative_to(path_type(commands["cwd"]))
    if ".." in relative.parts:
        raise ValueError("fixture escapes the recorded project root")
    return root.joinpath(*relative.parts)


def load_run(path):
    measurement = json.loads((path / "measurements.json").read_text(encoding="utf-8"))
    commands = json.loads((path / "commands.json").read_text(encoding="utf-8"))
    if measurement["status"] != "PASS" or commands["run_exit"] != 0:
        raise ValueError("run has not passed")
    reference = json.loads((fixture_path(commands) / "reference.json").read_text(encoding="utf-8"))
    result = measurement["rows"][-1]
    expected_steps = reference.get("total_steps", reference["steps"])
    if result["steps"] != expected_steps or result["hidden_checks"] != expected_steps*16*160:
        raise ValueError("incomplete run")
    for key, refkey in (("input_events", "input_events"), ("hidden_events", "hidden_spikes"),
                        ("code_writes", "code_changes"), ("residual_writes", "residual_changes")):
        if result[key] != reference["totals"][refkey]:
            raise ValueError(f"reference counter mismatch: {key}")
    return {"source": path.relative_to(ROOT).as_posix(), "cycles": result,
            "load": measurement["rows"][0], "commands": commands, "reference": reference}


def numerical_differences(on, off):
    root_on, root_off = (fixture_path(run["commands"]) for run in (on, off))
    batches = on["reference"].get("batches", 1)
    totals = {"dot": 0, "q6": 0}
    first = {}
    for batch in range(batches):
        suffix = f"batch_{batch}" if batches > 1 else ""
        for name in totals:
            lhs = np.fromiter((int(x, 16) for x in (root_on/suffix/f"{name}.mem").read_text().split()), dtype=np.int64)
            rhs = np.fromiter((int(x, 16) for x in (root_off/suffix/f"{name}.mem").read_text().split()), dtype=np.int64)
            if lhs.shape != rhs.shape:
                raise ValueError("paired prediction dimensions differ")
            indices = np.flatnonzero(lhs != rhs)
            totals[name] += len(indices)
            if len(indices) and name not in first:
                t, sample, cls = np.unravel_index(indices[0], (len(lhs)//160, 16, 10))
                first[name] = {"batch_index": batch, "t": int(t), "sample_index": int(sample), "class": int(cls)}
    return {"counts": totals, "first": first,
            "boundary": "reference differences; both corresponding streams must independently pass RTL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    on, off = (load_run(path.resolve()) for path in (args.on, args.off))
    if not on["reference"]["learning"] or off["reference"]["learning"]:
        raise ValueError("incorrect arm labels")
    for key in ("steps", "batch", "batches", "input_events", "hidden_events"):
        if on["cycles"].get(key, 1) != off["cycles"].get(key, 1):
            raise ValueError(f"unmatched workload: {key}")
    if on["commands"]["run"][0] != off["commands"]["run"][0]:
        raise ValueError("paired runs did not use the same executable")
    for key in ("support_indices", "samples", "counts_npz", "model"):
        if on["reference"].get(key) != off["reference"].get(key):
            raise ValueError(f"paired reference identity mismatch: {key}")
    samples = 16 * on["reference"].get("batches", 1)
    dense_sops = on["cycles"]["steps"] * 16 * (2312*160+160*10)
    arms = {}
    for name, run in (("on", on), ("off", off)):
        values = run["cycles"]
        event_sops = values["input_events"]*160 + values["hidden_events"]*10
        arms[name] = {"raw_evidence": run["source"], **values,
                      "initial_load_and_refill_cycles": run["load"]["cycles"],
                      "initial_state_cycles": run["load"]["init_cycles"],
                      "cycles_per_sample_in_measured_interval": values["cycles"]/samples,
                      "forward_interval_cycles": values["w0_cycles"]+values["w2_cycles"],
                      "model_event_synaptic_adds": event_sops,
                      "model_dense_synaptic_adds": dense_sops,
                      "model_input_zero_sop_fraction": 1-event_sops/dense_sops,
                      "scalar_forward_response_code_payload_bytes": values["forward_reads"]*2,
                      "learning_response_code_plus_residual_payload_bytes": values["learning_reads"]*4,
                      "write_enabled_field_bytes": (values["code_writes"]+values["residual_writes"])*2,
                      "simulation_wall_seconds": run["commands"]["run_seconds"]}
    summary = {"status": "PASS", "samples": samples, "timesteps_per_sample": on["reference"].get("batch_steps", [on["reference"]["steps"]]),
               "arms": arms, "learning_interval_cycle_delta": arms["on"]["cycles"]-arms["off"]["cycles"],
               "updated_prediction_visibility": numerical_differences(on, off),
               "final_changed_w2": on["reference"]["final_changed_w2"],
               "first_code_change": on["reference"].get("first_code_change", on["reference"].get("first_code_change_t")),
               "scope": "public-interface TB scheduler, final production composition + store, real-count FC support; not a host driver or board test",
               "cycle_boundary": "excludes first parameter load/refill and first state initialization; includes later refill and inter-batch initialization",
               "warning": "learning wait starts after last sample; it overlaps other learning work and is not pure engine cost. Refill counter includes initial refill. No frequency/board/DDR/power claim."}
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("output must remain in project")
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
