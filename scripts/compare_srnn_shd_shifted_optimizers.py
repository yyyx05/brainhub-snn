#!/usr/bin/env python3
"""Compare Adam and hardware-facing direct updates on a held-out SHD shift."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from screen_srnn_m2_direct_optimizer import direct_update_tensor
from screen_srnn_shd_frequency_shift import FrequencyShiftDataset


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]
BATCH = 64
Q14_SCALE = 1 << 14
Q40_SCALE = 1 << 40

TRAINABLE_MASKS: dict[str, tuple[bool, bool]] = {
    "off": (False, False),
    "wout": (False, True),
    "wh1": (True, False),
    "wh1_wout": (True, True),
}


def compact(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": metric["samples"],
        "correct": metric["correct"],
        "accuracy": metric["accuracy"],
        "evaluation_ce": metric["evaluation_ce"],
        "class_coverage": metric["class_coverage"],
    }


def fresh_model(handoff: Path, config: dict[str, Any], trainer, base):
    model, _, optimizers, schedulers = trainer.prepare(config)
    checkpoint = handoff / "tp_exp" / "runs" / "formal_m2" / "checkpoints" / "best_checkpoint.pt"
    base.restore_checkpoint(checkpoint, model, optimizers, schedulers)
    return model, optimizers


def move_model_and_optimizers(model, optimizers, device: torch.device) -> None:
    model.to(device)
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)


@torch.no_grad()
def evaluate_on_device(
    model,
    dataset,
    collate_fn,
    query_indices: list[int],
    base,
    output_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batches = [
        query_indices[start : start + BATCH]
        for start in range(0, len(query_indices), BATCH)
    ]
    predictions: list[int] = []
    total = 0
    correct = 0
    ce_sum = 0.0
    for data, labels in base.indexed_loader(dataset, collate_fn, batches):
        data = data.to(device)
        labels = labels.to(device)
        model.reset_potential()
        output = None
        for step in range(data.shape[1]):
            output = model.inference(data[:, step, :].float())
        if output is None:
            continue
        prediction = output.argmax(1)
        total += int(labels.numel())
        correct += int((prediction == labels).sum())
        ce_sum += float(F.cross_entropy(output, labels.long(), reduction="sum"))
        predictions.extend(int(value) for value in prediction.cpu().tolist())
    return {
        "samples": total,
        "correct": correct,
        "accuracy": float(correct / total) if total else 0.0,
        "evaluation_ce": float(ce_sum / total) if total else 0.0,
        "class_coverage": len(set(predictions)),
        "output_size": output_size,
    }


def evaluate_pair(
    model,
    clean_dataset,
    shifted_dataset,
    collate_fn,
    query_indices: list[int],
    base,
    output_size: int,
    device: torch.device,
) -> dict[str, Any]:
    clean = evaluate_on_device(
        model, clean_dataset, collate_fn, query_indices, base, output_size, device
    )
    shifted = evaluate_on_device(
        model, shifted_dataset, collate_fn, query_indices, base, output_size, device
    )
    return {"clean": compact(clean), "shifted": compact(shifted)}


def direct_adapt_batch(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
    state: dict[str, np.ndarray],
    candidate: dict[str, int],
    trainable_mask: str,
) -> dict[str, Any]:
    arch = importlib.import_module("architecture_search")
    model.train()
    model.reset_potential()
    target = F.one_hot(labels.long(), int(config["output_size"])).float()
    stats = {name: 0 for name in (
        "rmw", "code_writes", "residual_writes", "saturations", "zero_skips"
    )}
    prediction = None
    tp_losses: list[float] = []
    ce_losses: list[float] = []
    wh1_weight = model.network[1].fc.weight
    wout_weight = model.network[-1].fc.weight
    wh1_enabled, wout_enabled = TRAINABLE_MASKS[trainable_mask]

    for timestep in range(int(data.shape[1])):
        wh1_weight.grad = None
        wout_weight.grad = None
        output, _ = model(data[:, timestep, :].float(), target)
        tp_loss = arch.full_local_loss(model, 1)
        wh1_step = {name: 0 for name in stats}
        wout_step = {name: 0 for name in stats}
        if wh1_enabled:
            wh1_gradient = torch.autograd.grad(
                tp_loss, wh1_weight, retain_graph=True
            )[0]
            wh1_q40 = torch.round(
                wh1_gradient.detach() * Q40_SCALE
            ).to(torch.int64)
            state["wh1_code"], state["wh1_residual"], wh1_step = (
                direct_update_tensor(
                    state["wh1_code"],
                    state["wh1_residual"],
                    wh1_q40.cpu().numpy(),
                    candidate["wh1_increment"],
                    candidate["wh1_threshold"],
                )
            )
        if wout_enabled:
            wout_gradient = (
                torch.einsum("bi,bj->ij", model.e, model.trace[-1]) /
                int(labels.numel())
            )
            wout_q40 = torch.round(
                wout_gradient.detach() * Q40_SCALE
            ).to(torch.int64)
            state["wout_code"], state["wout_residual"], wout_step = (
                direct_update_tensor(
                    state["wout_code"],
                    state["wout_residual"],
                    wout_q40.cpu().numpy(),
                    candidate["wout_increment"],
                    candidate["wout_threshold"],
                )
            )
        with torch.no_grad():
            if wh1_enabled:
                wh1_weight.copy_(
                    torch.from_numpy(
                        state["wh1_code"].astype(np.float32) / Q14_SCALE
                    ).to(wh1_weight.device)
                )
            if wout_enabled:
                wout_weight.copy_(
                    torch.from_numpy(
                        state["wout_code"].astype(np.float32) / Q14_SCALE
                    ).to(wout_weight.device)
                )
        for name in stats:
            stats[name] += wh1_step[name] + wout_step[name]
        model.detach_membrane_states()
        prediction = output.detach().argmax(1)
        tp_losses.append(float(tp_loss.detach()))
        ce_losses.append(float(F.cross_entropy(output, labels.long()).detach()))

    assert prediction is not None
    return {
        "samples": int(labels.numel()),
        "time_steps": int(data.shape[1]),
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).float().mean()),
        "tp_loss": float(np.mean(tp_losses)),
        "ce_loss": float(np.mean(ce_losses)),
        "traffic": stats,
    }


def adam_adapt_batch(
    model,
    optimizers: dict[str, torch.optim.Optimizer],
    data: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
    trainable_mask: str,
) -> dict[str, Any]:
    arch = importlib.import_module("architecture_search")
    wh1_enabled, wout_enabled = TRAINABLE_MASKS[trainable_mask]
    model.train()
    model.reset_potential()
    target = F.one_hot(labels.long(), int(config["output_size"])).float()
    prediction = None
    tp_losses: list[float] = []
    ce_losses: list[float] = []
    for timestep in range(int(data.shape[1])):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        output, _ = model(data[:, timestep, :].float(), target)
        tp_loss = arch.full_local_loss(model, 1)
        if wh1_enabled:
            model.network[1].fc.weight.grad = torch.autograd.grad(
                tp_loss,
                model.network[1].fc.weight,
                retain_graph=True,
            )[0]
            optimizers["tp_hidden_1"].step()
        if wout_enabled:
            model.network[-1].fc.weight.grad = (
                torch.einsum("bi,bj->ij", model.e, model.trace[-1]) /
                int(labels.numel())
            )
            optimizers["output"].step()
        model.detach_membrane_states()
        prediction = output.detach().argmax(1)
        tp_losses.append(float(tp_loss.detach()))
        ce_losses.append(float(F.cross_entropy(output, labels.long()).detach()))
    assert prediction is not None
    return {
        "samples": int(labels.numel()),
        "time_steps": int(data.shape[1]),
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).float().mean()),
        "tp_loss": float(np.mean(tp_losses)),
        "ce_loss": float(np.mean(ce_losses)),
    }


def direct_initial_state(model) -> dict[str, np.ndarray]:
    wh1 = torch.round(model.network[1].fc.weight.detach() * Q14_SCALE)
    wout = torch.round(model.network[-1].fc.weight.detach() * Q14_SCALE)
    wh1_code = wh1.clamp(-32768, 32767).to(torch.int64).cpu().numpy()
    wout_code = wout.clamp(-32768, 32767).to(torch.int64).cpu().numpy()
    with torch.no_grad():
        model.network[1].fc.weight.copy_(
            torch.from_numpy(wh1_code.astype(np.float32) / Q14_SCALE).to(
                model.network[1].fc.weight.device
            )
        )
        model.network[-1].fc.weight.copy_(
            torch.from_numpy(wout_code.astype(np.float32) / Q14_SCALE).to(
                model.network[-1].fc.weight.device
            )
        )
    return {
        "wh1_code": wh1_code,
        "wh1_residual": np.zeros_like(wh1_code),
        "wout_code": wout_code,
        "wout_residual": np.zeros_like(wout_code),
    }


def accumulate_traffic(total: dict[str, int], step: dict[str, int]) -> None:
    for name, value in step.items():
        total[name] += value


def summarize_curve(
    name: str,
    evaluations: list[dict[str, Any]],
    support_metrics: list[dict[str, Any]],
    traffic: dict[str, int] | None,
    configuration: dict[str, int] | None,
    trainable_mask: str,
    parameter_deltas: dict[str, dict[str, float | int]],
    mutable_parameters: int,
) -> dict[str, Any]:
    shifted = [item["shifted"]["accuracy"] for item in evaluations]
    clean = [item["clean"]["accuracy"] for item in evaluations]
    return {
        "optimizer": name,
        "trainable_mask": trainable_mask,
        "configuration": configuration,
        "mutable_parameters": mutable_parameters,
        "q2_14_code_plus_residual_bytes": mutable_parameters * 4,
        "parameter_deltas": parameter_deltas,
        "evaluations": [
            {"after_support_batches": index, **value}
            for index, value in enumerate(evaluations)
        ],
        "support_metrics": support_metrics,
        "shifted_accuracy_improvement": shifted[-1] - shifted[0],
        "clean_accuracy_change": clean[-1] - clean[0],
        "recovery_curve_mean": float(np.mean(shifted)),
        "traffic": traffic,
    }


def parameter_deltas(
    model,
    wh1_before: torch.Tensor,
    wout_before: torch.Tensor,
    trainable_mask: str,
) -> dict[str, dict[str, float | int]]:
    wh1_enabled, wout_enabled = TRAINABLE_MASKS[trainable_mask]
    result: dict[str, dict[str, float | int]] = {}
    for name, current, before, enabled in (
        ("wh1", model.network[1].fc.weight, wh1_before, wh1_enabled),
        ("wout", model.network[-1].fc.weight, wout_before, wout_enabled),
    ):
        delta = current.detach() - before
        result[name] = {
            "selected": int(enabled),
            "nonzero": int(torch.count_nonzero(delta).item()),
            "l2": float(torch.linalg.vector_norm(delta).item()),
        }
        if not enabled and result[name]["nonzero"] != 0:
            raise RuntimeError(f"disabled tensor changed under {trainable_mask}: {name}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument("--shift-channels", type=int)
    parser.add_argument(
        "--episode-seed",
        type=int,
        help="reshuffle the validation split for a reproducible support episode",
    )
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="execution device; cpu preserves the historical development runs",
    )
    parser.add_argument(
        "--dataset-partition",
        choices=("validation", "official_test", "train_support_official_test"),
        default="validation",
        help=(
            "validation preserves the development protocol; official_test creates "
            "a deterministic, disjoint support/query episode from the SHD test set; "
            "train_support_official_test takes labelled support from the "
            "training-validation index set and evaluates the complete test set"
        ),
    )
    parser.add_argument(
        "--trainable-masks",
        nargs="+",
        choices=tuple(TRAINABLE_MASKS),
        default=list(TRAINABLE_MASKS),
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=("preserved_adam", "direct_response", "direct_retention"),
        default=("preserved_adam", "direct_response", "direct_retention"),
        help="run only predeclared optimizer profiles; report wrappers use one locked profile",
    )
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    python_dir = args.handoff / "tp_exp" / "python"
    sys.path.insert(0, str(python_dir))
    import architecture_search as arch
    import m1_m2_train as base
    import multilayer_tp_train as trainer

    severity = json.loads(
        (
            ROOT / "results" / "learning_reference" /
            "srnn_shd_frequency_shift_severity_screen.json"
        ).read_text(encoding="ascii")
    )
    support_indices = [int(value) for value in severity["support_indices"]]
    query_indices = [int(value) for value in severity["query_indices"]]
    shift_channels = (
        int(severity["selected_shift_channels"])
        if args.shift_channels is None
        else args.shift_channels
    )
    config = json.loads(
        (args.handoff / "tp_exp" / "configs" / "B64_FORMAL_SEED_A.json").read_text(
            encoding="utf-8"
        )
    )
    config["dataset_root_relative"] = str(args.dataset_root.resolve())
    train_dataset, test_dataset, collate_fn = trainer.datasets(config)
    if args.dataset_partition == "validation":
        if args.episode_seed is not None:
            validation_path = args.handoff / config["validation_indices_relative"]
            order = np.asarray(base.load_indices(validation_path), dtype=np.int64)
            np.random.default_rng(args.episode_seed).shuffle(order)
            if args.support_size <= 0 or args.support_size >= order.size:
                raise ValueError("support size must leave a nonempty validation query")
            support_indices = order[: args.support_size].tolist()
            query_indices = order[args.support_size :].tolist()
        support_clean_dataset = train_dataset
        query_clean_dataset = train_dataset
    elif args.dataset_partition == "official_test":
        if args.episode_seed is None:
            raise ValueError("official_test report requires an explicit episode seed")
        order = np.arange(len(test_dataset), dtype=np.int64)
        np.random.default_rng(args.episode_seed).shuffle(order)
        if args.support_size <= 0 or args.support_size >= order.size:
            raise ValueError("support size must leave a nonempty official-test query")
        support_indices = order[: args.support_size].tolist()
        query_indices = order[args.support_size :].tolist()
        support_clean_dataset = test_dataset
        query_clean_dataset = test_dataset
    else:
        if args.episode_seed is None:
            raise ValueError(
                "train_support_official_test requires an explicit episode seed"
            )
        validation_path = args.handoff / config["validation_indices_relative"]
        order = np.asarray(base.load_indices(validation_path), dtype=np.int64)
        np.random.default_rng(args.episode_seed).shuffle(order)
        if args.support_size <= 0 or args.support_size > order.size:
            raise ValueError("support size exceeds the training-validation index set")
        support_indices = order[: args.support_size].tolist()
        query_indices = list(range(len(test_dataset)))
        support_clean_dataset = train_dataset
        query_clean_dataset = test_dataset
    if len(support_indices) % BATCH:
        raise RuntimeError("support split must contain complete B64 windows")
    support_batches = [
        support_indices[start : start + BATCH]
        for start in range(0, len(support_indices), BATCH)
    ]
    support_shifted_dataset = FrequencyShiftDataset(
        support_clean_dataset, shift_channels
    )
    query_shifted_dataset = FrequencyShiftDataset(
        query_clean_dataset, shift_channels
    )
    output_size = int(config["output_size"])

    results: list[dict[str, Any]] = []

    if "preserved_adam" in args.optimizers:
        for trainable_mask in args.trainable_masks:
            adam_model, adam_optimizers = fresh_model(
                args.handoff, config, trainer, base
            )
            move_model_and_optimizers(adam_model, adam_optimizers, device)
            wh1_before = adam_model.network[1].fc.weight.detach().clone()
            wout_before = adam_model.network[-1].fc.weight.detach().clone()
            adam_evaluations = [evaluate_pair(
                adam_model, query_clean_dataset, query_shifted_dataset, collate_fn,
                query_indices, base, output_size, device
            )]
            adam_support: list[dict[str, Any]] = []
            for batch in support_batches:
                data, labels = next(iter(
                    base.indexed_loader(
                        support_shifted_dataset, collate_fn, [batch]
                    )
            ))
                data = data.to(device)
                labels = labels.to(device)
                adam_support.append(adam_adapt_batch(
                    adam_model,
                    adam_optimizers,
                    data,
                    labels,
                    config,
                    trainable_mask,
                ))
                adam_evaluations.append(evaluate_pair(
                    adam_model, query_clean_dataset, query_shifted_dataset, collate_fn,
                    query_indices, base, output_size, device
                ))
            wh1_enabled, wout_enabled = TRAINABLE_MASKS[trainable_mask]
            mutable_parameters = (
                (adam_model.network[1].fc.weight.numel() if wh1_enabled else 0) +
                (adam_model.network[-1].fc.weight.numel() if wout_enabled else 0)
            )
            results.append(summarize_curve(
                "preserved_adam",
                adam_evaluations,
                adam_support,
                None,
                None,
                trainable_mask,
                parameter_deltas(
                    adam_model, wh1_before, wout_before, trainable_mask
                ),
                mutable_parameters,
            ))

    candidates = [
        {
            "name": "direct_response",
            "wh1_increment": 25,
            "wh1_threshold": 244,
            "wout_increment": 31,
            "wout_threshold": 65,
        },
        {
            "name": "direct_retention",
            "wh1_increment": 25,
            "wh1_threshold": 488,
            "wout_increment": 31,
            "wout_threshold": 65,
        },
    ]
    for candidate in candidates:
        if candidate["name"] not in args.optimizers:
            continue
        for trainable_mask in args.trainable_masks:
            model, _ = fresh_model(args.handoff, config, trainer, base)
            model.to(device)
            direct_state = direct_initial_state(model)
            wh1_before = model.network[1].fc.weight.detach().clone()
            wout_before = model.network[-1].fc.weight.detach().clone()
            evaluations = [evaluate_pair(
                model, query_clean_dataset, query_shifted_dataset, collate_fn,
                query_indices, base, output_size, device
            )]
            support_metrics: list[dict[str, Any]] = []
            traffic = {name: 0 for name in (
                "rmw", "code_writes", "residual_writes", "saturations", "zero_skips"
            )}
            for batch in support_batches:
                data, labels = next(iter(
                    base.indexed_loader(
                        support_shifted_dataset, collate_fn, [batch]
                    )
                ))
                data = data.to(device)
                labels = labels.to(device)
                metric = direct_adapt_batch(
                    model,
                    data,
                    labels,
                    config,
                    direct_state,
                    candidate,
                    trainable_mask,
                )
                support_metrics.append(metric)
                accumulate_traffic(traffic, metric["traffic"])
                evaluations.append(evaluate_pair(
                    model, query_clean_dataset, query_shifted_dataset, collate_fn,
                    query_indices, base, output_size, device
                ))
            configuration = {
                key: value for key, value in candidate.items() if key != "name"
            }
            wh1_enabled, wout_enabled = TRAINABLE_MASKS[trainable_mask]
            mutable_parameters = (
                (model.network[1].fc.weight.numel() if wh1_enabled else 0) +
                (model.network[-1].fc.weight.numel() if wout_enabled else 0)
            )
            results.append(summarize_curve(
                candidate["name"],
                evaluations,
                support_metrics,
                traffic,
                configuration,
                trainable_mask,
                parameter_deltas(
                    model, wh1_before, wout_before, trainable_mask
                ),
                mutable_parameters,
            ))

    result = {
        "status": "pass",
        "protocol": {
            "dataset": (
                "SHD official test episodic support/query split"
                if args.dataset_partition == "official_test"
                else (
                    "SHD training-validation support and complete official-test query"
                    if args.dataset_partition == "train_support_official_test"
                    else "SHD train validation split only"
                )
            ),
            "dataset_partition": args.dataset_partition,
            "shift": (
                f"frequency channels {shift_channels:+d}, zero fill, no wrap"
            ),
            "shift_channels": shift_channels,
            "support_samples": len(support_indices),
            "support_batches": len(support_batches),
            "query_samples": len(query_indices),
            "episode_seed": args.episode_seed,
            "device": str(device),
            "support_indices": support_indices,
            "support_query_overlap": (
                0
                if args.dataset_partition == "train_support_official_test"
                else len(set(support_indices) & set(query_indices))
            ),
            "support_query_index_spaces": (
                "training_validation_vs_official_test"
                if args.dataset_partition == "train_support_official_test"
                else args.dataset_partition
            ),
            "supervision": "labels available to TP target branch and Wout CE",
            "official_test_used_for_selection": False,
        },
        "claim_boundary": (
            "Locked-profile episodic report on the official SHD test set. "
            if args.dataset_partition == "official_test"
            else (
                "Locked-profile cross-partition evaluation with labelled support "
                "from the training-validation index set and query over the complete "
                "official SHD test set. No report-stage profile selection occurs. "
                if args.dataset_partition == "train_support_official_test"
                else "Algorithm screen on a deterministic synthetic frequency shift. "
            )
        ) + (
            "Direct updates quantize trainable weights to Q2.14 and gradient signs "
            "to Q40, while preserved Adam retains checkpoint FP32/master state. "
            "This is not RTL evidence or a natural-domain benchmark."
        ),
        "optimizers": results,
    }
    output = args.output or (
        ROOT / "results" / "learning_reference" /
        "srnn_shd_shifted_optimizer_comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    compact_result = ", ".join(
        f"{item['optimizer']}={item['evaluations'][-1]['shifted']['correct']}/"
        f"{item['evaluations'][-1]['shifted']['samples']}"
        for item in results
    )
    print(f"SRNN_SHD_SHIFTED_OPTIMIZER_COMPARISON_PASS {compact_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
