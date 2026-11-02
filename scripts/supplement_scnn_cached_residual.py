"""A2 SCNN development experiment with a shared frozen-backbone feature cache.

Cache actual support/query trajectories, not gradients. Each arm recomputes
logits, errors and gradients from its own changing classifier at every step.
No labels enter query forward execution. No original evidence is overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload"
DEFAULT_OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/scnn_cached_a2"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(38472, 1729, 65537),
                        default=[38472, 1729, 65537])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seed")
    print(json.dumps(dict(seeds=args.seeds, subject_pair=[24, 25], corruption="jitter1",
                          support_policy="original contiguous full B4, not forced class-mixed",
                          checkpoint=str(PAYLOAD / "checkpoints/parameters.pt"),
                          output=str(args.output_dir), run=args.run)), flush=True)
    if not args.run:
        return 0
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay inside the project")
    paths = [output / f"seed{seed}.json" for seed in args.seeds]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts import run_scnn_tp_selected_layer_shift_adaptation as api
    from scripts import run_scnn_tp_subject_pair_ablation as subjects_api
    from scripts.supplement_algorithm_residual import Counters, update_adapter

    torch.set_num_threads(1)
    torch.manual_seed(38472)
    np.random.seed(38472)
    device = torch.device(args.device)
    subjects, count = subjects_api.read_subjects(PAYLOAD / "data/test_264x20_metadata.json")
    labels_np = subjects_api.read_labels(PAYLOAD / "data/test_264_labels.hex", count)
    support = subjects_api.require_complete_subject(labels_np, subjects, 24)
    query = subjects_api.require_complete_subject(labels_np, subjects, 25)
    support_batches = api.full_adaptation_batches(list(api.batches(support, 4)), 4)
    # Only 41.25 MiB raw float inputs on CPU; convolution state stays minibatched.
    samples = torch.from_numpy(api.read_signed_q8_8(
        PAYLOAD / "data/test_264x20_inputs_q8_8.hex", count * 20 * 2 * 32 * 32
    ).reshape(count, 20, 2, 32, 32))
    labels = torch.from_numpy(labels_np)
    output.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def extract(model, index_batches, corruption, training):
        cache = []
        model.train(training)
        for indices in index_batches:
            x = api.transform_event_view(samples[indices].to(device), indices,
                                         dx=0, dy=0, corruption=corruption)
            target = F.one_hot(labels[indices].to(device), 11).float() if training else None
            model.reset_potential()
            rows = dict(indices=indices, features=[], traces=[], logits=[], gradients=[])
            for t in range(20):
                logits = model(x[:, t], target)[0] if training else model.inference(x[:, t])
                student = model.hs[-1].flatten(1)
                other = model.ts[-1].flatten(1) if training else student
                rows["features"].append(torch.cat((student, other)).detach().cpu())
                rows["traces"].append(model.trace[-1].flatten(1).detach().cpu())
                rows["logits"].append(logits.detach().cpu())
                if training:
                    rows["gradients"].append((torch.einsum(
                        "bi,bj->ij", model.e, model.trace[-1].flatten(1)) / len(indices)).cpu())
                model.detach_membrane_states()
            cache.append(rows)
        return cache

    @torch.no_grad()
    def replay(head, cache, *, adapt=False, verify=False):
        correct = 0
        predictions = []
        residual = torch.zeros_like(head.fc.weight, dtype=torch.int32)
        max_logit_error = max_gradient_error = 0.0
        for batch in cache:
            indices = batch["indices"]
            head.reset()
            gradient = torch.zeros_like(head.fc.weight)
            expected = labels[indices].to(device)
            target = F.one_hot(expected, 11).float() if adapt or batch["gradients"] else None
            for t, feature in enumerate(batch["features"]):
                logits = head(feature.to(device))[:len(indices)]
                if verify:
                    max_logit_error = max(max_logit_error, float(
                        (logits.cpu() - batch["logits"][t]).abs().max()))
                if target is not None:
                    error = F.softmax(logits, dim=1) - target
                    current = torch.einsum("bi,bj->ij", error,
                                           batch["traces"][t].to(device)) / len(indices)
                    if verify and batch["gradients"]:
                        max_gradient_error = max(max_gradient_error, float(
                            (current.cpu() - batch["gradients"][t]).abs().max()))
                    gradient.add_(current)
                if adapt and (t + 1) % 4 == 0:
                    api.apply_fixed_sgd_residual(
                        head.fc.weight, gradient, residual, learning_rate=5 * 2.0 ** -22,
                        weight_fractional_bits=15, residual_fractional_bits=15)
                    gradient.zero_()
            prediction = logits.argmax(1)
            correct += int((prediction == expected).sum())
            predictions.extend(prediction.cpu().tolist())
        if verify and (max_logit_error != 0 or max_gradient_error != 0):
            raise RuntimeError(f"cache replay mismatch: logits={max_logit_error}, gradients={max_gradient_error}")
        total = sum(len(batch["indices"]) for batch in cache)
        return dict(correct=correct, total=total, accuracy_percent=100 * correct / total,
                    predictions=predictions, max_logit_error=max_logit_error if verify else None,
                    max_gradient_error=max_gradient_error if verify else None,
                    verified_against_original_forward=verify,
                    final_nonzero_residuals=int(torch.count_nonzero(residual)))

    for seed, path in zip(args.seeds, paths):
        started = time.perf_counter()
        model = api.build_model(PAYLOAD / "traces_propagation-main",
                                PAYLOAD / "checkpoints/parameters.pt", batch_size=4,
                                device=device, weight_fractional_bits=15)
        initial_head = copy.deepcopy(model.network[-1])
        corruption = api.EventCorruption(temporal_jitter=1, seed=seed)
        shifted_support = extract(model, support_batches, corruption, True)
        clean_query = extract(model, list(api.batches(query, 4)), None, False)
        shifted_query = extract(model, list(api.batches(query, 4)), corruption, False)
        del model
        extraction_seconds = time.perf_counter() - started
        verification = replay(copy.deepcopy(initial_head), shifted_support, verify=True)
        clean = replay(copy.deepcopy(initial_head), clean_query, verify=True)
        shifted = replay(copy.deepcopy(initial_head), shifted_query, verify=True)
        rows = []
        for arm in ("on", "drop"):
            head = copy.deepcopy(initial_head)
            initial_weight = head.fc.weight.detach().clone()
            counters = Counters()
            with update_adapter("scnn", arm, counters):
                support_metric = replay(head, shifted_support, adapt=True)
            after_clean = replay(head, clean_query)
            after_shifted = replay(head, shifted_query)
            rows.append(dict(
                residual_arm=arm, baseline_clean=clean, baseline_shifted=shifted,
                clean_after=after_clean, shifted_after=after_shifted,
                shifted_gain_points=after_shifted["accuracy_percent"] - shifted["accuracy_percent"],
                clean_retention_points=after_clean["accuracy_percent"] - clean["accuracy_percent"],
                final_changed_codes=int(torch.count_nonzero(head.fc.weight != initial_weight)),
                final_nonzero_residuals=support_metric["final_nonzero_residuals"],
                counters=counters.values,
            ))
        result = dict(
            status="pass", topology="SCNN", seed=seed, subject_pair=[24, 25],
            model="VGG9 Q0.15 classifier-only", device=str(device),
            checkpoint=str((PAYLOAD / "checkpoints/parameters.pt").relative_to(ROOT)),
            configuration=dict(batch_size=4, window_timesteps=4, total_timesteps=20,
                               temporal_jitter=1, learning_rate=5 * 2.0 ** -22,
                               weight_fractional_bits=15, residual_fractional_bits=15,
                               force_class_mixed_batches=False),
            support_indices=support, used_support_batches=support_batches, query_indices=query,
            cache_verification=dict(support=verification, clean=clean, shifted=shifted),
            cache_extraction_seconds=extraction_seconds,
            elapsed_seconds_wall=time.perf_counter() - started, arms=rows,
            claim_boundary="Hybrid software on one development subject pair, three corruption seeds. Frozen full-backbone features and traces are cached; each arm recomputes classifier logits, errors and gradients. Not full bit-exact or RTL dataset accuracy.",
            counter_boundary="Address events per update, not distinct addresses or physical writes. retained_residual_changes excludes temporary remainders discarded by the drop arm.",
        )
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"SCNN_CACHED_A2_PASS seed={seed} seconds={result['elapsed_seconds_wall']:.3f} "
              + " ".join(f"{r['residual_arm']}:gain={r['shifted_gain_points']:.3f},writes={r['counters']['code_changes']}" for r in rows), flush=True)
        del initial_head, shifted_support, clean_query, shifted_query, head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
