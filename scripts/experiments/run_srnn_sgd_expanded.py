"""Development-only expanded SGD screen on accepted integer hidden features.

This runs new output-head adaptations, not a backbone/RTL rerun. Old small-rate
and direct-response endpoints are imported, never recomputed. Report has no
learning-rate argument and reads the development decision before query data.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import run_srnn_integer_optimizer_comparison as base
from screen_srnn_m2_direct_optimizer import read_signed_hex
from supplement_srnn_magnitude_sgd import REPORT_SEEDS, update_regime

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/paper_evidence/supplement_20261012/srnn_integer_optimizer"
OUTPUT = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_sgd_expanded_r1"
OLD_RATES = (3e-5, 1e-4, 3e-4)
NEW_RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ALL_RATES = OLD_RATES + NEW_RATES
RULE = ("Development seed128474 only, final support128 endpoint. Exclude zero code "
        "updates, all-saturated updates, and clean loss above 1 percentage point. "
        "Maximize shifted gain, then clean change, then smaller rate. No query "
        "selection, early stopping, or automatic expansion beyond this grid.")


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path, data):
    with open(base.native_path(path), "x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def select_expanded(rows):
    candidates = [r for r in rows if r["method"] == "magnitude_sgd"]
    if sorted(r["learning_rate"] for r in candidates) != list(ALL_RATES):
        raise ValueError("expected each prespecified development rate exactly once")
    reasons, eligible = [], []
    for row in candidates:
        excluded = []
        if row["update_regime"]["zero_code_updates"]:
            excluded.append("zero_code_updates")
        if row["update_regime"]["all_saturated"]:
            excluded.append("all_saturated")
        if row["clean_accuracy_change"] < -0.01 - 1e-12:
            excluded.append("clean_loss_exceeds_1pp")
        reasons.append(dict(rate=row["learning_rate"], exclusions=excluded))
        if not excluded:
            eligible.append(row)
    winner = max(eligible, key=lambda r: (r["shifted_accuracy_improvement"],
                 r["clean_accuracy_change"], -r["learning_rate"])) if eligible else None
    return dict(status="selected" if winner else "inconclusive", rule=RULE,
                selected_rate=winner["learning_rate"] if winner else None,
                upper_grid_endpoint_selected=bool(winner and winner["learning_rate"] == ALL_RATES[-1]),
                candidates=reasons)


def peak_working_set():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    api.restype = ctypes.c_int
    if not api(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counter.PeakWorkingSetSize


class Inputs:
    def __init__(self):
        self.used = {}

    def record(self, path):
        path = Path(path)
        stat = path.stat()
        self.used[str(path.relative_to(ROOT))] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def json(self, path):
        self.record(path)
        return base.read_json(path)

    def npz(self, path):
        self.record(path)
        with np.load(base.native_path(path), allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}

    def check_unchanged(self):
        for name, before in self.used.items():
            stat = (ROOT / name).stat()
            if before != dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns):
                raise RuntimeError(f"read-only input changed: {name}")


def query(inputs, directory, tag, expected_indices):
    meta = inputs.json(directory / "features" / (tag + ".json"))
    if meta["indices"] != expected_indices:
        raise ValueError("query indices differ from saved protocol")
    data = inputs.npz(directory / "features" / (tag + ".npz"))
    if data["counts"].shape != (len(expected_indices), 256) or data["labels"].shape != (len(expected_indices),):
        raise ValueError("query feature shape mismatch")
    return data["counts"], data["labels"]


def support(inputs, directory, tag, expected_indices):
    meta = inputs.json(directory / "features" / (tag + ".json"))
    if meta["indices"] != expected_indices or meta["shift"] != 16 or len(meta["batches"]) != 2:
        raise ValueError("support differs from saved two-B64/+16 protocol")
    batches = []
    for b in range(2):
        data = inputs.npz(directory / "features" / f"{tag}_batch{b}.npz")
        if (data["spikes"].shape != (meta["batches"][b]["padded_steps"], 64, 256) or
                data["labels"].shape != (64,) or
                meta["batches"][b]["indices"] != expected_indices[64*b:64*(b+1)]):
            raise ValueError("support shape/order mismatch")
        batches.append((data["spikes"], data["labels"]))
    return batches


def episode(output, phase, seed, rate, initial, batches, clean, shifted):
    started = time.perf_counter()
    code, residual = initial.copy(), np.zeros_like(initial)
    totals, records = {}, []
    baseline_clean, baseline_shifted = base.evaluate(*clean, initial), base.evaluate(*shifted, initial)
    curve = [dict(support_samples=0, clean=baseline_clean, shifted=baseline_shifted,
                  code_changes=0, code_lsb_steps=0, residual_changes=0, saturations=0)]
    tag = f"{phase}_seed{seed}_sgd_{rate:g}"
    for b, (spikes, labels) in enumerate(batches):
        result = base.adapt(spikes, labels, code, residual, "magnitude_sgd", rate, verify=True)
        code, residual = result["codes"], result["residual"]
        for key, value in result["counters"].items():
            totals[key] = totals.get(key, 0) + value
        records.append(dict(batch=b, timesteps=len(spikes), labels=labels.tolist(),
                            counters=result["counters"], rows=result["rows"],
                            independent_checked_values=result["independent_checked_values"]))
        curve.append(dict(support_samples=(b+1)*64, clean=base.evaluate(*clean, code),
                          shifted=base.evaluate(*shifted, code), **totals))
        np.savez_compressed(base.native_path(output / f"{tag}_batch{b+1}_state.npz"),
                            initial=initial, codes=code, residual=residual)
    after = curve[-1]
    row = dict(phase=phase, seed=seed, method="magnitude_sgd", learning_rate=rate,
               evidence="new_output_head_adaptation_on_accepted_frozen_integer_features",
               baseline_clean=baseline_clean, baseline_shifted=baseline_shifted,
               clean_after=after["clean"], shifted_after=after["shifted"],
               clean_accuracy_change=(after["clean"]["correct"]-baseline_clean["correct"])/len(clean[1]),
               shifted_accuracy_improvement=(after["shifted"]["correct"]-baseline_shifted["correct"])/len(shifted[1]),
               counters=totals, update_regime=update_regime(totals, code),
               changed_code_addresses=int(np.count_nonzero(code != initial)),
               final_l1_code_delta=int(np.abs(code-initial).sum()),
               final_max_abs_code_delta=int(np.abs(code-initial).max()),
               nonzero_residual_addresses=int(np.count_nonzero(residual)),
               support=records, curve=curve, elapsed_seconds=time.perf_counter()-started)
    write_json(output / (tag + ".json"), row)
    print(f"DONE {tag} clean={after['clean']['correct']}/{len(clean[1])} "
          f"shift={after['shifted']['correct']}/{len(shifted[1])} "
          f"code_events={totals['code_changes']} lsb_steps={totals['code_lsb_steps']} "
          f"saturations={totals['saturations']}", flush=True)
    return row


def imported(row, source):
    result = dict(row)
    result.update(evidence="imported_accepted_endpoint_not_rerun", source=str(source.relative_to(ROOT)))
    return result


def write_tables(output, rows):
    comparison, curves = [], []
    for r in rows:
        comparison.append(dict(phase=r["phase"], seed=r["seed"], method=r["method"],
            rate=r["learning_rate"], evidence=r["evidence"], query=r["clean_after"]["total"],
            clean_correct=r["clean_after"]["correct"], shifted_correct=r["shifted_after"]["correct"],
            clean_change_pp=100*r["clean_accuracy_change"], shifted_gain_pp=100*r["shifted_accuracy_improvement"],
            code_events=r["counters"]["code_changes"], code_lsb_steps=r["counters"]["code_lsb_steps"],
            residual_events=r["counters"]["residual_changes"], saturations=r["counters"]["saturations"],
            final_changed_addresses=r["changed_code_addresses"]))
        for c in r.get("curve", []):
            curves.append(dict(phase=r["phase"], seed=r["seed"], rate=r["learning_rate"],
                support_samples=c["support_samples"], clean_correct=c["clean"]["correct"],
                shifted_correct=c["shifted"]["correct"], query=c["clean"]["total"],
                code_events=c["code_changes"], code_lsb_steps=c["code_lsb_steps"],
                residual_events=c["residual_changes"], saturations=c["saturations"]))
    for name, data in (("comparison.csv", comparison), ("curves.csv", curves)):
        if data:
            with open(base.native_path(output / name), "x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)


def run(phase, output):
    inputs = Inputs()
    weight_path = ROOT / "tb/data/srnn_m2_v28_wout_initial_q14.mem"
    inputs.record(weight_path)
    expected_initial = read_signed_hex(weight_path, 16, 5120).reshape(20, 256)
    protocol = dict(phase=phase, started=now(), new_rates=list(NEW_RATES),
        imported_rates=list(OLD_RATES), selection=RULE,
        source=str(OLD.relative_to(ROOT)), network="700-450rec-256-20", mask="Wout-only",
        feature_scope="Explicit reuse of accepted saved integer hidden spikes/counts; no backbone regeneration",
        forward="existing Q28 logits, Q14 trace, Q24 LUT softmax, Q40 fixed /64 gradient",
        update="unchanged magnitude_update_tensor; FP64 lr product, integer Q0.24 LSB residual, sat16",
        schedule="two B64 windows; reset trace/logits at each batch, retain code/residual between batches",
        curves="0/64/128 support samples; final endpoint selection only; no early stopping",
        rationale="Prespecified 1-3 log grid 1e-3 through 10 to cover effective updates, not only sub-LSB near-frozen rates",
        boundary="Retrospective existing +16 condition, fixed checkpoint/shared official query; not new distribution, equal historical tuning budget, RTL, or hardware speedup",
        host=dict(python=platform.python_version(), numpy=np.__version__, cpu_threads=1, device="cpu"))
    write_json(output / "protocol.json", protocol)
    rows = []
    if phase == "dev":
        directory = OLD / "dev_r1"
        old = inputs.json(directory / "summary.json")
        indices = inputs.json(directory / "indices.json")
        base.validate_dev_split(indices["support"], indices["query"])
        saved = inputs.npz(directory / "dev_seed128474_direct_response_state.npz")
        initial = saved["initial"]
        np.testing.assert_array_equal(initial, expected_initial)
        clean = query(inputs, directory, "dev_query_clean", indices["query"])
        shifted = query(inputs, directory, "dev_query_shift16", indices["query"])
        batches = support(inputs, directory, "dev_support", indices["support"])
        rows.extend(imported(r, directory / "summary.json") for r in old["episodes"])
        # The development branch never opens report features or report summaries.
        for rate in NEW_RATES:
            result = episode(output, phase, 128474, rate, initial, batches, clean, shifted)
            if (result["baseline_clean"] != rows[0]["baseline_clean"] or
                    result["baseline_shifted"] != rows[0]["baseline_shifted"]):
                raise AssertionError("development cached query baseline mismatch")
            rows.append(result)
        decision = select_expanded(rows)
        extra = dict(selection=decision)
    else:
        dev = inputs.json(output.parent / "dev/summary.json")
        if dev["status"] != "PASS_expanded_integer_SGD_dev":
            raise ValueError("completed development result required")
        decision = select_expanded(dev["episodes"])
        if decision != dev["selection"] or decision["status"] != "selected":
            raise ValueError("development screen did not select a rate")
        rate = decision["selected_rate"]
        directory = OLD / "report_r1"
        old = inputs.json(directory / "summary.json")
        measured = [r for r in old["episodes"] if r["method"] == "magnitude_sgd" and r["learning_rate"] == rate]
        if sorted(r["seed"] for r in measured) == list(REPORT_SEEDS):
            # A reused winner does not justify rerunning accepted adaptations.
            rows = [imported(r, directory / "summary.json") for r in old["episodes"]
                    if r["method"] == "direct_response" or r in measured]
            inputs.check_unchanged()
            write_tables(output, rows)
            write_json(output / "input_files.json", inputs.used)
            return dict(status="PASS_expanded_integer_SGD_report_reused", protocol=protocol,
                        episodes=rows, selected_rate=rate, development_selection=decision,
                        new_adaptation_episodes=0,
                        report_reuse_reason="Selected rate and all three accepted report episodes already exist; no query/adaptation rerun")
        clean = query(inputs, directory, "official_query_clean", list(range(2264)))
        shifted = query(inputs, directory, "official_query_shift16", list(range(2264)))
        np.testing.assert_array_equal(clean[1], shifted[1])
        for seed in REPORT_SEEDS:
            indices = inputs.json(directory / f"seed{seed}_indices.json")
            saved = inputs.npz(directory / f"report_seed{seed}_direct_response_state.npz")
            initial = saved["initial"]
            np.testing.assert_array_equal(initial, expected_initial)
            direct = next(r for r in old["episodes"] if r["seed"] == seed and r["method"] == "direct_response")
            rows.append(imported(direct, directory / "summary.json"))
            batches = support(inputs, directory, f"seed{seed}_support", indices["support"])
            result = episode(output, phase, seed, rate, initial, batches, clean, shifted)
            if (result["baseline_clean"] != direct["baseline_clean"] or
                    result["baseline_shifted"] != direct["baseline_shifted"]):
                raise AssertionError("initial cached query evaluation mismatch")
            rows.append(result)
        extra = dict(selected_rate=rate, development_selection=decision)
    np.testing.assert_array_equal(clean[1], shifted[1])
    inputs.check_unchanged()
    write_tables(output, rows)
    write_json(output / "input_files.json", inputs.used)
    return dict(status=f"PASS_expanded_integer_SGD_{phase}", protocol=protocol,
                episodes=rows, **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "report"), required=True)
    args = parser.parse_args()
    output = OUTPUT / args.phase
    Path(base.native_path(output)).mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = None
    try:
        result = run(args.phase, output)
    except Exception:
        result = dict(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended=now(), elapsed_seconds=time.perf_counter()-started,
                      process_peak_working_set_bytes=peak_working_set())
        write_json(output / "summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("episodes", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
