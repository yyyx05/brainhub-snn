#!/usr/bin/env python3
"""Run a deterministic supervised N-MNIST spatial-shift adaptation pilot.

The experiment adapts an H400 TP model on labeled training samples shifted two
pixels to the right, then evaluates held-out shifted and clean test samples.
It is an algorithm-selection pilot, not fixed-point or RTL evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fc_tp_h400_b16_reference import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_TRACES_ROOT,
    OUTPUT_DIM,
)
from scripts.screen_fc_tp_update_candidates import (  # noqa: E402
    Candidate,
    _delta_stats,
    _state_cost,
    build_model,
    build_optimizers,
)


DEFAULT_OUTPUT = (
    ROOT / "results" / "learning_reference" / "fc_tp_h400_supervised_shift_pilot.json"
)
ADAPT_CANDIDATES = (
    Candidate("adam_ref_lr1e-4", "adam", 1e-4),
    Candidate("sgd_momentum_lr1e-3", "sgd_momentum", 1e-3, 0.9),
    Candidate("sgd_momentum_lr3e-3", "sgd_momentum", 3e-3, 0.9),
    Candidate("sgd_momentum_lr1e-2", "sgd_momentum", 1e-2, 0.9),
    Candidate(
        "sgd_momentum_w0_1e-3_w2_1e-4",
        "sgd_momentum",
        1e-3,
        0.9,
        output_learning_rate=1e-4,
    ),
    Candidate(
        "sgd_momentum_w0_2e-3_w2_1e-4",
        "sgd_momentum",
        2e-3,
        0.9,
        output_learning_rate=1e-4,
    ),
    Candidate("direct_sign_step_2m15", "direct_sign", 2.0**-15),
    Candidate("direct_sign_step_2m14", "direct_sign", 2.0**-14),
)


def interleaved_class_indices(
    targets: Iterable[int],
    *,
    samples_per_class: int,
    start_replica: int = 0,
) -> list[int]:
    by_class: dict[int, list[int]] = {class_id: [] for class_id in range(OUTPUT_DIM)}
    required = start_replica + samples_per_class
    for index, label_value in enumerate(targets):
        label = int(label_value)
        if label in by_class and len(by_class[label]) < required:
            by_class[label].append(index)
        if all(len(indices) >= required for indices in by_class.values()):
            break
    if any(len(indices) < required for indices in by_class.values()):
        raise ValueError("dataset does not contain enough samples for every class")
    return [
        by_class[class_id][replica]
        for replica in range(start_replica, required)
        for class_id in range(OUTPUT_DIM)
    ]


def shift_nmnist_frames(
    frames: torch.Tensor,
    *,
    dx: int = 2,
    dy: int = 0,
) -> torch.Tensor:
    if frames.ndim != 3 or frames.shape[-1] != 2 * 34 * 34:
        raise ValueError(f"expected [B,T,2312] N-MNIST frames, got {tuple(frames.shape)}")
    if abs(dx) >= 34 or abs(dy) >= 34:
        raise ValueError("shift must be smaller than the 34x34 sensor")
    image = frames.reshape(frames.shape[0], frames.shape[1], 2, 34, 34)
    shifted = torch.zeros_like(image)

    src_x0 = max(0, -dx)
    src_x1 = min(34, 34 - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(34, 34 + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(34, 34 - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(34, 34 + dy)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = image[
        ..., src_y0:src_y1, src_x0:src_x1
    ]
    return shifted.reshape_as(frames)


def load_dataset_batches(
    *,
    traces_root: Path,
    data_root: Path,
    train: bool,
    indices: list[int],
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic  # pylint: disable=import-outside-toplevel
    import tonic.transforms as tonic_transforms  # pylint: disable=import-outside-toplevel
    from torch.utils.data import DataLoader, Subset  # pylint: disable=import-outside-toplevel

    transform = tonic_transforms.Compose([
        tonic_transforms.ToFrame(
            sensor_size=tonic.datasets.NMNIST.sensor_size,
            time_window=1000,
        ),
        lambda frames: frames.reshape(frames.shape[0], -1),
    ])
    dataset = tonic.datasets.NMNIST(
        save_to=str(data_root),
        train=train,
        first_saccade_only=True,
        transform=transform,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=0,
    )
    return [
        (frames.to(dtype=torch.float32), labels.to(dtype=torch.long))
        for frames, labels in loader
    ]


def evaluate(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    shifted: bool,
    shift_dx: int = 2,
    shift_dy: int = 0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    correct = 0
    total = 0
    predictions: list[int] = []
    labels_all: list[int] = []
    logits_l2 = 0.0
    for frames, labels in batches:
        frames = frames.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        inputs = (
            shift_nmnist_frames(frames, dx=shift_dx, dy=shift_dy)
            if shifted
            else frames
        )
        model.reset_potential()
        logits = None
        with torch.no_grad():
            for timestep in range(inputs.shape[1]):
                logits = model.inference(inputs[:, timestep, :])
        if logits is None:
            raise RuntimeError("empty evaluation sequence")
        predicted = logits.argmax(dim=1)
        correct += int((predicted == labels).sum().item())
        total += int(labels.numel())
        predictions.extend(int(value) for value in predicted.cpu().tolist())
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        logits_l2 += float(torch.linalg.vector_norm(logits).item())
    return {
        "correct": correct,
        "total": total,
        "accuracy_percent": 100.0 * correct / total,
        "mean_batch_logits_l2": logits_l2 / len(batches),
        "predictions": predictions,
        "labels": labels_all,
    }


def adapt_candidate(
    candidate: Candidate,
    *,
    traces_root: Path,
    checkpoint_root: Path,
    adapt_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor]],
    baseline_clean: dict[str, Any],
    baseline_shifted: dict[str, Any],
    shift_dx: int = 2,
    shift_dy: int = 0,
) -> dict[str, Any]:
    model = build_model(
        traces_root,
        checkpoint_root,
        batch_size=int(adapt_batches[0][1].numel()),
    )
    w0 = model.network[0].fc.weight
    w2 = model.network[1].fc.weight
    target_projector = model.target_propagator.fc.weight
    w0_initial = w0.detach().clone()
    w2_initial = w2.detach().clone()
    target_projector_initial = target_projector.detach().clone()
    optimizers = build_optimizers(model, candidate)

    update_calls = 0
    w0_active_timesteps = 0
    w2_active_timesteps = 0
    started_at = time.perf_counter()
    for frames, labels in adapt_batches:
        shifted_frames = shift_nmnist_frames(frames, dx=shift_dx, dy=shift_dy)
        target = F.one_hot(labels, OUTPUT_DIM).to(dtype=torch.float32)
        model.reset_potential()
        for timestep in range(shifted_frames.shape[1]):
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            model(shifted_frames[:, timestep, :], target)
            model.update()
            if w0.grad is None or w2.grad is None:
                raise RuntimeError("TP update did not generate both gradients")
            w0_active_timesteps += int(torch.count_nonzero(w0.grad).item() > 0)
            w2_active_timesteps += int(torch.count_nonzero(w2.grad).item() > 0)
            for optimizer in optimizers.values():
                optimizer.step()
            update_calls += 1
            model.detach_membrane_states()
    adaptation_seconds = time.perf_counter() - started_at

    if not torch.equal(target_projector_initial, target_projector.detach()):
        raise RuntimeError("train_s=false adaptation changed target projector S")
    clean_after = evaluate(model, test_batches, shifted=False)
    shifted_after = evaluate(
        model,
        test_batches,
        shifted=True,
        shift_dx=shift_dx,
        shift_dy=shift_dy,
    )
    return {
        "name": candidate.name,
        "mode": candidate.mode,
        "learning_rate_or_step": candidate.learning_rate,
        "output_learning_rate_or_step": (
            candidate.output_learning_rate
            if candidate.output_learning_rate is not None
            else candidate.learning_rate
        ),
        "momentum": candidate.momentum,
        "adaptation_batches": len(adapt_batches),
        "adaptation_samples": sum(int(labels.numel()) for _, labels in adapt_batches),
        "optimizer_update_calls": update_calls,
        "w0_active_timesteps": w0_active_timesteps,
        "w2_active_timesteps": w2_active_timesteps,
        "w0_delta": _delta_stats(w0, w0_initial),
        "w2_delta": _delta_stats(w2, w2_initial),
        "state_cost": _state_cost(candidate),
        "adaptation_seconds_cpu": adaptation_seconds,
        "clean_after": clean_after,
        "shifted_after": shifted_after,
        "shifted_adaptation_gain_points": (
            shifted_after["accuracy_percent"] - baseline_shifted["accuracy_percent"]
        ),
        "clean_retention_change_points": (
            clean_after["accuracy_percent"] - baseline_clean["accuracy_percent"]
        ),
    }


def run_pilot(
    *,
    traces_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    output: Path,
    batch_size: int = 16,
    adapt_samples_per_class: int = 16,
    test_samples_per_class: int = 10,
) -> dict[str, Any]:
    torch.manual_seed(38472)
    torch.use_deterministic_algorithms(True)
    if batch_size < OUTPUT_DIM:
        raise ValueError("batch_size must include at least one sample per class")

    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic  # pylint: disable=import-outside-toplevel

    train_dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=True)
    test_dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=False)
    adapt_indices = interleaved_class_indices(
        train_dataset.targets,
        samples_per_class=adapt_samples_per_class,
        start_replica=2,
    )
    test_indices = interleaved_class_indices(
        test_dataset.targets,
        samples_per_class=test_samples_per_class,
    )
    adapt_batches = load_dataset_batches(
        traces_root=traces_root,
        data_root=data_root,
        train=True,
        indices=adapt_indices,
        batch_size=batch_size,
    )
    test_batches = load_dataset_batches(
        traces_root=traces_root,
        data_root=data_root,
        train=False,
        indices=test_indices,
        batch_size=batch_size,
    )

    baseline_model = build_model(traces_root, checkpoint_root, batch_size=batch_size)
    baseline_clean = evaluate(baseline_model, test_batches, shifted=False)
    baseline_shifted = evaluate(baseline_model, test_batches, shifted=True)
    candidates = [
        adapt_candidate(
            candidate,
            traces_root=traces_root,
            checkpoint_root=checkpoint_root,
            adapt_batches=adapt_batches,
            test_batches=test_batches,
            baseline_clean=baseline_clean,
            baseline_shifted=baseline_shifted,
            shift_dx=2,
            shift_dy=0,
        )
        for candidate in ADAPT_CANDIDATES
    ]
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Deterministic supervised spatial-shift adaptation pilot on a small balanced "
            "N-MNIST subset. It is FP32 software evidence for candidate selection, not full-dataset, "
            "fixed-point, RTL, convergence, or statistical-significance evidence."
        ),
        "shift": {"dx_pixels": 2, "dy_pixels": 0, "zero_fill": True},
        "source": {
            "adapt_samples_per_class": adapt_samples_per_class,
            "adapt_samples": len(adapt_indices),
            "held_out_test_samples_per_class": test_samples_per_class,
            "held_out_test_samples": len(test_indices),
            "batch_size": batch_size,
            "supervision": "one-hot labels are supplied for adaptation only",
        },
        "baseline_clean": baseline_clean,
        "baseline_shifted": baseline_shifted,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-samples-per-class", type=int, default=16)
    parser.add_argument("--test-samples-per-class", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pilot(
        traces_root=args.traces_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        output=args.output,
        batch_size=args.batch_size,
        adapt_samples_per_class=args.adapt_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
