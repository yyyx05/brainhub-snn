"""Paired Wout update comparison on the existing full-integer SRNN path.

Development uses the saved validation split; report uses the three saved
support resamples and all official queries. No training/model/RTL changes.
Frozen hidden spikes are computed once per input view, not per optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/paper_evidence/supplement_20261004/algorithm"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(OLD / "srnn_integer_adaptation"))
import numpy as np
import integer_wout as integer_head
import build_srnn_formal_m2_fixture as reference
from screen_srnn_m2_direct_optimizer import round_shift_even
from supplement_srnn_magnitude_sgd import (
    RATES, REPORT_SEEDS, UpdateObserver, select_rate, update_regime,
)

DEFAULT_OUTPUT = ROOT / "results/paper_evidence/supplement_20261012/srnn_integer_optimizer"


def native_path(path):
    path = str(Path(path).absolute())
    return "\\\\?\\" + path if os.name == "nt" and not path.startswith("\\\\?\\") else path


def read_json(path):
    with open(native_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    Path(native_path(Path(path).parent)).mkdir(parents=True, exist_ok=True)
    with open(native_path(path), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def load_batch(source, indices, shift=0):
    from evaluate_srnn_full_integer import frame_counts
    if shift not in (0, 16) or not 1 <= len(indices) <= 64:
        raise ValueError("existing clean/+16, B<=64 protocol only")
    frames = [frame_counts(source["spikes/times"][i], source["spikes/units"][i]) for i in indices]
    lengths = [len(x) for x in frames]
    counts = np.zeros((max(lengths), len(indices), 700), dtype=np.int64)
    for b, frame in enumerate(frames):
        if shift:
            counts[:len(frame), b, shift:] = frame[:, :-shift]
        else:
            counts[:len(frame), b] = frame
    labels = np.asarray([source["labels"][i] for i in indices], dtype=np.int64)
    return counts, labels, dict(indices=list(indices), native_steps=lengths,
                               padded_steps=max(lengths), shift=shift)


def frozen_logits(spike_counts, codes):
    """Sum first only for frozen signed16 weights and binary 256-wide spikes.

    Each per-step dot is bounded by 256*32768 < 2**31, so saturation never
    changes it. T<65536 also keeps the cumulative Q28 logit inside signed64.
    """
    counts, codes = np.asarray(spike_counts), np.asarray(codes)
    if (counts.dtype.kind not in "iu" or codes.dtype.kind not in "iu" or
            counts.ndim != 2 or counts.shape[1] != 256 or codes.shape != (20, 256) or
            np.any((counts < 0) | (counts >= 65536)) or
            np.any((codes < -32768) | (codes > 32767))):
        raise ValueError("frozen-head integer bounds exceeded")
    return (counts.astype(np.int64) @ codes.astype(np.int64).T) * 16384


def adapt(spikes, labels, codes, residual, method, rate=None, verify=False):
    spikes = np.asarray(spikes)
    codes, residual = np.array(codes, dtype=np.int64), np.array(residual, dtype=np.int64)
    if (spikes.ndim != 3 or spikes.shape[2] != 256 or not 1 <= spikes.shape[1] <= 64 or
            not 1 <= len(spikes) < 65536 or not np.isin(spikes, (0, 1)).all()):
        raise ValueError("expected binary T,B,256")
    if codes.shape != (20, 256) or residual.shape != codes.shape:
        raise ValueError("expected 20x256 state")
    limit = 65 if method == "direct_response" else 1 << 24
    if np.any(np.abs(residual) >= limit) or np.any((codes < -32768) | (codes > 32767)):
        raise ValueError("invalid optimizer state")
    observer = UpdateObserver(integer_head.direct_update_tensor, method, rate)
    trace = np.zeros((spikes.shape[1], 256), dtype=np.int64)
    logits = np.zeros((spikes.shape[1], 20), dtype=np.int64)
    lut = reference.read_exp_lut()
    checked = 0
    rows = []
    for t, spike8 in enumerate(spikes):
        spike = spike8.astype(np.int64)
        logits += reference.sat32(spike @ codes.T) * 16384
        trace = reference.sat32(round_shift_even(trace * 13926, 14) + spike * 16384)
        probability, gradient = integer_head.gradient_q40(logits, trace, labels, lut)
        if verify and t in (0, 1, len(spikes)-1):
            expected_probability, _, expected_gradient = reference.wout_gradient(logits, trace, labels, lut)
            np.testing.assert_array_equal(probability, expected_probability)
            np.testing.assert_array_equal(gradient, expected_gradient)
            checked += probability.size + gradient.size
        codes, residual, stats = observer(codes, residual, gradient, 31, 65)
        rows.append(dict(timestep=t, gradient_abs_max=int(np.abs(gradient).max()), **stats))
    return dict(codes=codes, residual=residual, logits=logits, rows=rows,
                counters=observer.totals, independent_checked_values=int(checked))


def evaluate(spike_counts, labels, codes):
    prediction = frozen_logits(spike_counts, codes).argmax(axis=1)
    correct = int(np.count_nonzero(prediction == labels))
    return dict(correct=correct, total=len(labels), accuracy_percent=100*correct/len(labels),
                predictions=prediction.tolist())


def validate_dev_split(support, query):
    if (len(support) != 128 or len(query) != 688 or
            len(set(support + query)) != 816):
        raise ValueError("expected the saved disjoint support128/query688 validation split")


def audit_saved_report(output, dev_path):
    """Reanalyse already measured predictions/counters without any GPU rerun."""
    result, dev = read_json(output / "summary.json"), read_json(dev_path)
    rows = result["episodes"]
    dev_indices = read_json(dev_path.parent / "indices.json")
    support_sets = [set(read_json(output / f"seed{seed}_indices.json")["support"]) for seed in REPORT_SEEDS]
    with np.load(native_path(output / "features/official_query_clean.npz")) as data:
        labels = data["labels"]
    transitions, counters = [], []
    checked = 0
    for row in rows:
        for view in ("clean", "shifted"):
            before = np.asarray(row["baseline_"+view]["predictions"])
            after = np.asarray(row[view+"_after"]["predictions"])
            changed = np.flatnonzero(before != after)
            transitions.append(dict(seed=row["seed"], method=row["method"], view=view,
                changed_predictions=int(len(changed)),
                wrong_to_correct=int(np.count_nonzero((before != labels) & (after == labels))),
                correct_to_wrong=int(np.count_nonzero((before == labels) & (after != labels))),
                wrong_class_to_other_wrong=int(np.count_nonzero((before != labels) & (after != labels) & (before != after))),
                changes=[dict(index=int(i), label=int(labels[i]), before=int(before[i]), after=int(after[i])) for i in changed]))
        if row["method"] != "direct_response":
            continue
        historical = read_json(OLD / f"srnn_integer_adaptation/report_cuda_20261008/seed{row['seed']}.json")
        for key in ("baseline_clean", "baseline_shifted", "clean_after", "shifted_after"):
            if row[key]["predictions"] != historical[key]["predictions"]:
                raise AssertionError("historical predictions mismatch")
            checked += len(row[key]["predictions"])
        pairs = {k: [row["counters"][v],historical["counters"][k]] for k,v in
            (("rmw","nonzero_gradients"),("code_writes","code_changes"),
             ("residual_writes","residual_changes"),("saturations","saturations"),("zero_skips","zero_gradients"))}
        if any(a != b for a,b in pairs.values()):
            raise AssertionError("historical update counters mismatch")
        counters.append(dict(seed=row["seed"], current_and_historical=pairs))
    audit = dict(scope="read_only_saved_result_arithmetic_not_new_measurement", historical_predictions_checked=checked,
        historical_counter_checks=counters, prediction_transitions=transitions,
        support_pairwise_overlap=[dict(seeds=[REPORT_SEEDS[i],REPORT_SEEDS[j]], overlap=len(support_sets[i]&support_sets[j]))
                                  for i,j in ((0,1),(0,2),(1,2))],
        unique_report_support=len(set.union(*support_sets)),
        report_support_overlap_with_dev_query=[len(s&set(dev_indices["query"])) for s in support_sets],
        selected_rate=dev["selection"]["selected_rate"],
        interpretation="Historical query already exposed; retrospective control, not fresh held-out confirmation. Support sensitivity only; accuracy-flat SGD dev tie-break is not effective tuning.")
    save_json(output / "review_audit.json", audit)
    return audit


class Experiment:
    def __init__(self, output, device):
        import torch
        import evaluate_srnn_full_integer as full
        self.output, self.device, self.full = output, device, full
        self.started = time.perf_counter()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.weights = full.load_weights(device)
        self.initial = self.weights[3].T.cpu().numpy().astype(np.int64)
        # Windows NumPy needs extended paths for the restored deep fixture.
        full.ROOT = Path(native_path(ROOT))
        self.prefix = full.check_prefix(self.weights, device)
        save_json(output / "prefix_check.json", dict(status="PASS", compared_values=self.prefix))

    def features(self, h5_path, indices, shift, tag, support=False):
        import h5py
        cache = self.output / "features" / (tag + ".npz")
        Path(native_path(cache.parent)).mkdir(parents=True, exist_ok=True)
        # An output directory belongs to one execution; no automatic stale-cache reuse.
        sums, labels, batches, supports = [], [], [], []
        with h5py.File(native_path(h5_path), "r") as source:
            for first in range(0, len(indices), 64):
                counts, y, meta = load_batch(source, indices[first:first+64], shift)
                logits, captured = self.full.forward(counts, self.weights, self.device, capture=True)
                spikes = captured["m2_spike"].astype(np.uint8)
                del captured, counts
                summed = spikes.sum(axis=0, dtype=np.int64)
                np.testing.assert_array_equal(frozen_logits(summed, self.initial), logits)
                sums.append(summed)
                labels.extend(y.tolist())
                batches.append(meta)
                if support:
                    supports.append((spikes, y))
                    path = self.output / "features" / f"{tag}_batch{first//64}.npz"
                    np.savez_compressed(native_path(path), spikes=spikes, labels=y)
                print(f"FEATURES {tag} {min(first+64,len(indices))}/{len(indices)}", flush=True)
        counts = np.concatenate(sums)
        y = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(native_path(cache), counts=counts, labels=y)
        save_json(cache.with_suffix(".json"), dict(indices=indices, shift=shift, batches=batches,
                  checked_frozen_logits=len(indices)*20, source=str(h5_path.relative_to(ROOT))))
        return counts, y, supports

    def episode(self, phase, seed, support, query_clean, query_shift, method, rate=None):
        label = f"{phase}_seed{seed}_{method}" + (f"_{rate:g}" if rate else "")
        codes, residual = self.initial.copy(), np.zeros_like(self.initial)
        records, totals = [], {}
        for b, (spikes, y) in enumerate(support):
            result = adapt(spikes, y, codes, residual, method, rate, verify=True)
            codes, residual = result["codes"], result["residual"]
            records.append(dict(batch=b, timesteps=len(spikes), labels=y.tolist(),
                                counters=result["counters"], rows=result["rows"],
                                independent_checked_values=result["independent_checked_values"]))
            for key, value in result["counters"].items():
                totals[key] = totals.get(key, 0) + value
            print(f"ADAPT {label} batch={b+1}/2 codes={totals['code_changes']}", flush=True)
        baseline_clean = evaluate(*query_clean, self.initial)
        baseline_shifted = evaluate(*query_shift, self.initial)
        clean_after = evaluate(*query_clean, codes)
        shifted_after = evaluate(*query_shift, codes)
        row = dict(phase=phase, seed=seed, method=method, learning_rate=rate,
                   baseline_clean=baseline_clean, baseline_shifted=baseline_shifted,
                   clean_after=clean_after, shifted_after=shifted_after,
                   shifted_accuracy_improvement=(shifted_after["correct"]-baseline_shifted["correct"])/shifted_after["total"],
                   clean_accuracy_change=(clean_after["correct"]-baseline_clean["correct"])/clean_after["total"],
                   counters=totals, support=records, update_regime=update_regime(totals, codes),
                   changed_code_addresses=int(np.count_nonzero(codes != self.initial)),
                   nonzero_residual_addresses=int(np.count_nonzero(residual)))
        np.savez_compressed(native_path(self.output / (label + "_state.npz")),
                            initial=self.initial, codes=codes, residual=residual)
        save_json(self.output / (label + ".json"), row)
        return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "report", "audit"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-result", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(DEFAULT_OUTPUT) or output == DEFAULT_OUTPUT:
        raise ValueError("write experimental results under supplement_20261012/srnn_integer_optimizer")
    if args.phase == "audit":
        if args.dev_result is None:
            raise ValueError("audit requires saved development result")
        audit = audit_saved_report(output, args.dev_result.resolve())
        print(json.dumps({key:audit[key] for key in ("historical_predictions_checked","support_pairwise_overlap","unique_report_support")}))
        return
    output.mkdir(parents=True, exist_ok=False)
    train = ROOT / "imports/srnn/dataset_cache/SHD/shd_train.h5"
    test = ROOT / "imports/srnn/dataset_cache/SHD/shd_test.h5"
    protocol = dict(phase=args.phase, mask="Wout-only", shift_channels=16, support=128, batch=64,
                    network="700-450rec-256-20", initial="existing V28 signed16 Q2.14 Wout",
                    forward="production INT8 recurrent backbone + Q14 M2 + Q28 logit + Q24 softmax",
                    gradient="existing integer gradient_q40; fixed /64 normalization unchanged",
                    methods=["frozen", "direct_response", "magnitude_sgd"],
                    direct_rule=dict(increment=31, threshold=65), sgd_dev_rates=list(RATES),
                    sgd_arithmetic="same integer gradient/code and persistent Q0.24 LSB remainder; existing FP64 rate-product quantization, no float master/momentum",
                    claim_scope="software integer-forward Wout update control, not full TP vs STDP/BPTT, not RTL/board/energy",
                    tuning_scope="fixed historical direct rule vs three preset validation-tuned SGD rates; not equal historical search budgets",
                    logical_state_bytes=dict(direct_code_plus_residual=20480, sgd_code_plus_residual=30720),
                    logical_state_note="16-bit code + 16-bit direct residual or 32-bit SGD residual; not synthesized SGD resource results",
                    numpy_runtime_state_bytes_each=81920)
    save_json(output / "protocol.json", protocol)
    exp = Experiment(output, args.device)
    if args.phase == "dev":
        if args.dev_result is not None:
            raise ValueError("dev does not consume a selection")
        saved = read_json(OLD / "srnn_magnitude_sgd_a3/pilot_cpu_20261007/seed128474_direct_response.json")
        support_indices, query_indices = saved["support_indices"], saved["query_indices"]
        validate_dev_split(support_indices, query_indices)
        save_json(output / "indices.json", dict(support=support_indices, query=query_indices, source="saved validation episode128474"))
        _, _, support = exp.features(train, support_indices, 16, "dev_support", support=True)
        clean, y, _ = exp.features(train, query_indices, 0, "dev_query_clean")
        shifted, y_shift, _ = exp.features(train, query_indices, 16, "dev_query_shift16")
        np.testing.assert_array_equal(y, y_shift)
        rows = [exp.episode("dev", 128474, support, (clean,y), (shifted,y), "direct_response")]
        rows += [exp.episode("dev", 128474, support, (clean,y), (shifted,y), "magnitude_sgd", rate) for rate in RATES]
        selection = select_rate(rows)
    else:
        if args.dev_result is None:
            raise ValueError("report requires validation selection")
        dev = read_json(args.dev_result)
        selection = select_rate(dev["episodes"])
        if dev["protocol"]["phase"] != "dev" or selection != dev["selection"] or selection["status"] != "selected":
            raise ValueError("no eligible development-only selection")
        protocol["dev_result"] = str(args.dev_result.resolve().relative_to(ROOT))
        save_json(output / "protocol.json", protocol)
        query_indices = list(range(2264))
        clean, y, _ = exp.features(test, query_indices, 0, "official_query_clean")
        shifted, ys, _ = exp.features(test, query_indices, 16, "official_query_shift16")
        np.testing.assert_array_equal(y, ys)
        baseline = evaluate(clean, y, exp.initial)
        prior = read_json(OLD / "srnn_full_integer/official2264_cuda_20261008/result.json")
        if baseline["predictions"] != prior["predictions"] or y.tolist() != prior["labels"]:
            raise AssertionError("initial official predictions/labels differ from measured integer result")
        rows = []
        for seed in REPORT_SEEDS:
            historical = read_json(OLD / f"srnn_integer_adaptation/report_cuda_20261008/seed{seed}.json")
            indices = historical["support_indices"]
            if len(indices) != 128 or len(set(indices)) != 128:
                raise ValueError("invalid saved support")
            save_json(output / f"seed{seed}_indices.json", dict(support=indices, query=query_indices,
                      spaces="training_validation_support_vs_official_test_query"))
            _, _, support = exp.features(train, indices, 16, f"seed{seed}_support", support=True)
            direct = exp.episode("report", seed, support, (clean,y), (shifted,y), "direct_response")
            for key in ("baseline_clean", "baseline_shifted", "clean_after", "shifted_after"):
                if direct[key]["predictions"] != historical[key]["predictions"]:
                    raise AssertionError(f"existing direct-response {seed}/{key} not reproduced")
            direct["historical_query_predictions_checked"] = 4*2264
            rows.append(direct)
            rows.append(exp.episode("report", seed, support, (clean,y), (shifted,y),
                                    "magnitude_sgd", selection["selected_rate"]))
    import torch
    from supplement_srnn_backbone_numeric import peak_memory
    result = dict(status="completed_software_integer_forward_comparison", protocol=protocol,
                  selection=selection, episodes=rows, prefix_check=exp.prefix,
                  elapsed_seconds=time.perf_counter()-exp.started,
                  process_memory=peak_memory(), gpu_device=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else 0)
    if args.phase == "report":
        result["method_summary"] = {}
        for method in ("direct_response", "magnitude_sgd"):
            mr = [r for r in rows if r["method"] == method]
            gain = [100*r["shifted_accuracy_improvement"] for r in mr]
            result["method_summary"][method] = dict(n=3, shifted_gain_mean_pp=float(np.mean(gain)),
                shifted_gain_sample_sd_pp=float(np.std(gain, ddof=1)),
                clean_change_mean_pp=float(np.mean([100*r["clean_accuracy_change"] for r in mr])))
        result["paired_gain_difference_pp"] = [100*(rows[i]["shifted_accuracy_improvement"]-rows[i+1]["shifted_accuracy_improvement"]) for i in (0,2,4)]
        result["replication_scope"] = "Three support resamples of one checkpoint/shared query, not independent training seeds; no general superiority claim"
    save_json(output / "summary.json", result)
    with open(native_path(output / "comparison.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase","seed","method","rate","clean_before","clean_after","shift_before","shift_after","total","shift_gain_pp","clean_change_pp","code_change_events","residual_change_events","saturations"])
        for r in rows:
            writer.writerow([args.phase,r["seed"],r["method"],r["learning_rate"],r["baseline_clean"]["correct"],r["clean_after"]["correct"],r["baseline_shifted"]["correct"],r["shifted_after"]["correct"],r["clean_after"]["total"],100*r["shifted_accuracy_improvement"],100*r["clean_accuracy_change"],r["counters"]["code_changes"],r["counters"]["residual_changes"],r["counters"]["saturations"]])
    print("COMPLETE " + json.dumps(dict(selection=selection, summary=result.get("method_summary"), elapsed=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
