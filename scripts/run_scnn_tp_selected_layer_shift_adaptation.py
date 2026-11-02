#!/usr/bin/env python3
"""Evaluate trainable-layer masks for bounded TP adaptation on DVS-Gesture.

The experiment starts from the effective Q8.8 weights consumed by the RTL,
uses a disjoint support/query split, and compares candidate trainable masks from
classifier-only through an all-layer software upper bound.  It is a software
experiment for adaptation/retention trends; fixed-point transaction equivalence
remains covered by the dedicated RTL regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES_ROOT = Path(r"D:\Desktop\traces_propagation-main")
DEFAULT_CHECKPOINT = Path(
    r"D:\Desktop\LIFLayerCNN1.4.1"
    r"\TP_DVSGESTURE_seed38472_best_epoch128\parameters.pt"
)
DEFAULT_INPUTS = Path(
    r"D:\Desktop\LIFLayerCNN1.4.1\datasets\DVSGesture\hardware"
    r"\test_128x20_inputs_q8_8.hex"
)
DEFAULT_LABELS = Path(
    r"D:\Desktop\LIFLayerCNN1.4.1\datasets\DVSGesture\hardware"
    r"\test_128_labels.hex"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "learning_reference"
    / "scnn_tp_selected_layer_shift_adaptation.json"
)

SAMPLES = 128
TIMESTEPS = 20
CHANNELS = 2
HEIGHT = 32
WIDTH = 32
CLASSES = 11
Q8_8_SCALE = 256.0
WEIGHT_CODE_BITS = 16

TRAINABLE_MASKS: dict[str, tuple[tuple[int, ...], bool]] = {
    "off": ((), False),
    "classifier": ((), True),
    "conv8": ((7,), False),
    "conv8_classifier": ((7,), True),
    "conv7_conv8_classifier": ((6, 7), True),
    "all_layers": (tuple(range(8)), True),
}


def read_signed_q8_8(path: Path, expected: int) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        raw = np.fromiter(
            (int(line, 16) for line in stream if line.strip()),
            dtype=np.uint16,
            count=expected,
        )
    if raw.size != expected:
        raise ValueError(f"{path}: read {raw.size} words, expected {expected}")
    return raw.view(np.int16).astype(np.float32) / Q8_8_SCALE


def read_labels(path: Path, expected: int) -> np.ndarray:
    labels = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()),
        dtype=np.int64,
        count=expected,
    )
    if labels.size != expected or np.any((labels < 0) | (labels >= CLASSES)):
        raise ValueError(f"{path}: invalid DVS-Gesture labels")
    return labels


def effective_weight(weight: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, weight.ndim))
    fan_in = int(np.prod(weight.shape[1:]))
    mean = weight.mean(dim=reduce_dims, keepdim=True)
    var = weight.var(dim=reduce_dims, keepdim=True, unbiased=False)
    return 1.8 * (weight - mean) / torch.sqrt(var * fan_in + 1.0e-5)


def quantize_weight(value: torch.Tensor, *, fractional_bits: int) -> torch.Tensor:
    scale = float(1 << fractional_bits)
    return (
        torch.clamp(
            torch.round(value * scale),
            -(1 << (WEIGHT_CODE_BITS - 1)),
            (1 << (WEIGHT_CODE_BITS - 1)) - 1,
        )
        / scale
    )


def build_model(
    traces_root: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    weight_fractional_bits: int = 8,
) -> torch.nn.Module:
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    from models.tp_cnn import TP_CNN  # pylint: disable=import-outside-toplevel

    # norm_type="none" is intentional: the checkpoint's trainable weights are
    # converted once to the effective, standardized Q8.8 values used by RTL.
    params = {
        "l_leak_t": 250.0 / 256.0,
        "l_leak_m": 135.0 / 256.0,
        "l_vth": 1.0,
        "l_rst_type": "soft",
        "l_out_leak_m": 1.0,
        "surrogate_type": "1",
        "surrogate_scale": 1.0,
        "vgg_variant": 9,
        "norm_type": "none",
        "train_s": False,
    }
    model = TP_CNN(
        input_h=HEIGHT,
        input_w=WIDTH,
        input_channels=CHANNELS,
        output_size=CLASSES,
        batch_size=batch_size,
        snn_params=params,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(checkpoint):
        missing = sorted(set(named_parameters) - set(checkpoint))
        extra = sorted(set(checkpoint) - set(named_parameters))
        raise ValueError(f"checkpoint mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            source = checkpoint[name].to(device=device, dtype=parameter.dtype)
            parameter.copy_(quantize_weight(
                effective_weight(source),
                fractional_bits=weight_fractional_bits,
            ))
    return model


def zero_filled_shift(x: torch.Tensor, *, dx: int, dy: int) -> torch.Tensor:
    shifted = torch.zeros_like(x)
    source_x0 = max(0, -dx)
    source_x1 = min(WIDTH, WIDTH - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(HEIGHT, HEIGHT - dy)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return shifted
    dest_x0 = source_x0 + dx
    dest_x1 = source_x1 + dx
    dest_y0 = source_y0 + dy
    dest_y1 = source_y1 + dy
    shifted[..., dest_y0:dest_y1, dest_x0:dest_x1] = x[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


@dataclass(frozen=True)
class EventCorruption:
    """Deterministic event-domain stress applied before spatial translation."""

    drop_probability: float = 0.0
    polarity_flip_probability: float = 0.0
    temporal_jitter: int = 0
    seed: int = 38472

    def validate(self) -> None:
        for name, probability in (
            ("drop_probability", self.drop_probability),
            ("polarity_flip_probability", self.polarity_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.temporal_jitter < TIMESTEPS:
            raise ValueError(
                f"temporal_jitter must be in [0, {TIMESTEPS - 1}]"
            )

    @property
    def enabled(self) -> bool:
        return (
            self.drop_probability > 0.0
            or self.polarity_flip_probability > 0.0
            or self.temporal_jitter > 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "event_drop_probability": self.drop_probability,
            "polarity_flip_probability": self.polarity_flip_probability,
            "temporal_jitter": self.temporal_jitter,
            "corruption_seed": self.seed,
        }


def _sample_generator(seed: int, sample_index: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    mixed_seed = (int(seed) + int(sample_index) * 0x9E3779B1) & 0x7FFF_FFFF_FFFF_FFFF
    generator.manual_seed(mixed_seed)
    return generator


def apply_event_corruption(
    x: torch.Tensor,
    sample_indices: Iterable[int],
    *,
    corruption: EventCorruption,
) -> torch.Tensor:
    """Apply reproducible event loss, polarity flips and per-event time jitter.

    Randomness is keyed by the archive sample index, so changing evaluation
    batch size or candidate ordering cannot change the corrupted event stream.
    Temporal collisions are saturated to the largest input event magnitude.
    """

    corruption.validate()
    indices = [int(value) for value in sample_indices]
    if x.ndim != 5 or tuple(x.shape[1:]) != (
        TIMESTEPS,
        CHANNELS,
        HEIGHT,
        WIDTH,
    ):
        raise ValueError(
            "event tensor must have shape "
            f"[B, {TIMESTEPS}, {CHANNELS}, {HEIGHT}, {WIDTH}]"
        )
    if len(indices) != x.shape[0]:
        raise ValueError("sample index count must match event tensor batch size")
    if not corruption.enabled:
        return x
    if torch.any(x < 0):
        raise ValueError("event corruptions require nonnegative polarity channels")

    corrupted_samples: list[torch.Tensor] = []
    for sample, sample_index in zip(x, indices, strict=True):
        generator = _sample_generator(corruption.seed, sample_index)
        transformed = sample

        if corruption.drop_probability > 0.0:
            keep = torch.rand(
                sample.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ) >= corruption.drop_probability
            transformed = transformed * keep.to(
                device=sample.device,
                dtype=sample.dtype,
            )

        if corruption.polarity_flip_probability > 0.0:
            flip = torch.rand(
                (TIMESTEPS, 1, HEIGHT, WIDTH),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ) < corruption.polarity_flip_probability
            flip = flip.to(device=sample.device)
            transformed = torch.where(
                flip,
                transformed[:, [1, 0], :, :],
                transformed,
            )

        if corruption.temporal_jitter > 0:
            active = torch.nonzero(transformed != 0, as_tuple=False)
            if active.numel() != 0:
                offsets = torch.randint(
                    -corruption.temporal_jitter,
                    corruption.temporal_jitter + 1,
                    (active.shape[0],),
                    generator=generator,
                    dtype=torch.int64,
                    device="cpu",
                ).to(active.device)
                target_time = torch.clamp(
                    active[:, 0] + offsets,
                    min=0,
                    max=TIMESTEPS - 1,
                )
                flat_target = (
                    (
                        (target_time * CHANNELS + active[:, 1]) * HEIGHT
                        + active[:, 2]
                    )
                    * WIDTH
                    + active[:, 3]
                )
                jittered = torch.zeros_like(transformed).flatten()
                event_values = transformed[
                    active[:, 0],
                    active[:, 1],
                    active[:, 2],
                    active[:, 3],
                ]
                jittered.scatter_add_(0, flat_target, event_values)
                max_event = transformed.max()
                transformed = jittered.reshape_as(transformed).clamp(
                    min=0,
                    max=max_event,
                )

        corrupted_samples.append(transformed)
    return torch.stack(corrupted_samples)


def transform_event_view(
    x: torch.Tensor,
    sample_indices: Iterable[int],
    *,
    dx: int,
    dy: int,
    corruption: EventCorruption | None,
) -> torch.Tensor:
    if corruption is not None:
        x = apply_event_corruption(
            x,
            sample_indices,
            corruption=corruption,
        )
    if dx or dy:
        x = zero_filled_shift(x, dx=dx, dy=dy)
    return x


def batches(indices: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    values = list(indices)
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


def full_adaptation_batches(
    candidate_batches: Iterable[list[int]], batch_size: int
) -> list[list[int]]:
    """Keep only batches matching the hardware normalization factor exactly."""

    result = [batch for batch in candidate_batches if len(batch) == batch_size]
    if not result:
        raise ValueError(
            "full-batch adaptation requested but no complete adaptation batch exists"
        )
    return result


def tp_pairwise_batches(
    indices: Iterable[int],
    labels: torch.Tensor,
    batch_size: int,
) -> list[list[int]]:
    """Build deterministic class-mixed batches for TP's pairwise trace loss.

    The convolutional TP objective compares samples inside a batch.  A
    singleton batch produces a 1x1 similarity matrix and an identically zero
    gradient, so silently accepting batch_size=1 invalidates any convolutional
    mask ablation.  Round-robin class ordering also prevents archive order from
    accidentally creating same-class batches.
    """

    values = list(indices)
    if batch_size < 2:
        raise ValueError(
            "convolutional TP adaptation requires batch_size >= 2 because "
            "its loss is defined over pairwise sample similarities"
        )
    if len(values) < 2:
        raise ValueError("convolutional TP adaptation requires at least two samples")

    labels_cpu = labels.detach().cpu().numpy()
    by_class: dict[int, list[int]] = {}
    for index in values:
        by_class.setdefault(int(labels_cpu[index]), []).append(index)
    if len(by_class) < 2:
        raise ValueError(
            "convolutional TP adaptation requires at least two target classes"
        )

    ordered: list[int] = []
    class_ids = sorted(by_class)
    while any(by_class[class_id] for class_id in class_ids):
        for class_id in class_ids:
            if by_class[class_id]:
                ordered.append(by_class[class_id].pop(0))

    result = list(batches(ordered, batch_size))
    if len(result) > 1 and len(result[-1]) == 1:
        if len(result[-2]) > 2:
            result[-1].insert(0, result[-2].pop())
        else:
            result[-2].extend(result[-1])
            result.pop()

    for index_batch in result:
        batch_classes = {int(labels_cpu[index]) for index in index_batch}
        if len(index_batch) < 2 or len(batch_classes) < 2:
            raise ValueError(
                "unable to construct a class-mixed TP batch; increase the "
                "support set or adaptation batch size"
            )
    return result


def split_support_selection_report(
    labels: np.ndarray,
    support_per_class: int,
    selection_per_class: int,
    *,
    episode_seed: int | None,
) -> tuple[list[int], list[int], list[int]]:
    """Build disjoint class-balanced adaptation, selection and report sets.

    A missing episode seed preserves the historical first-occurrence support
    split.  Supplying a seed shuffles each class independently and enables
    reproducible multi-episode experiments without changing sample membership.
    """

    if support_per_class < 0 or selection_per_class < 0:
        raise ValueError("per-class partition sizes must be nonnegative")
    support: list[int] = []
    selection: list[int] = []
    report: list[int] = []
    rng = np.random.default_rng(episode_seed) if episode_seed is not None else None
    for label in range(CLASSES):
        class_indices = np.flatnonzero(labels == label)
        if rng is not None:
            class_indices = rng.permutation(class_indices)
        required = support_per_class + selection_per_class + 1
        if class_indices.size < required:
            raise ValueError(
                f"class {label} has {class_indices.size} samples, needs at least "
                f"{required} for support/selection/nonempty-report"
            )
        support_end = support_per_class
        selection_end = support_end + selection_per_class
        support.extend(int(value) for value in class_indices[:support_end])
        selection.extend(
            int(value) for value in class_indices[support_end:selection_end]
        )
        report.extend(int(value) for value in class_indices[selection_end:])
    return support, selection, report


def evaluate(
    model: torch.nn.Module,
    samples: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    *,
    batch_size: int,
    dx: int,
    dy: int,
    corruption: EventCorruption | None = None,
) -> dict[str, object]:
    model.eval()
    correct = 0
    predictions: list[int] = []
    with torch.no_grad():
        for index_batch in batches(indices, batch_size):
            model.reset_potential()
            x = transform_event_view(
                samples[index_batch],
                index_batch,
                dx=dx,
                dy=dy,
                corruption=corruption,
            )
            logits = None
            for timestep in range(TIMESTEPS):
                logits = model.inference(x[:, timestep])
            assert logits is not None
            prediction = logits.argmax(dim=1)
            expected = labels[index_batch]
            correct += int((prediction == expected).sum().item())
            predictions.extend(int(value) for value in prediction.cpu())
    return {
        "correct": correct,
        "total": len(indices),
        "accuracy_percent": 100.0 * correct / len(indices),
        "predictions": predictions,
    }


@dataclass
class DirectResidualState:
    votes: dict[str, torch.Tensor]
    fixed_residuals: dict[str, torch.Tensor]
    code_writes: dict[str, int]
    code_lsb_steps: dict[str, int]
    gradient_active_windows: dict[str, int]
    gradient_nonzero_peak: dict[str, int]
    gradient_abs_max: dict[str, float]
    vote_abs_max: dict[str, int]
    fixed_increment_abs_max: dict[str, int]
    fixed_residual_abs_max: dict[str, int]


def apply_direct_residual(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    votes: torch.Tensor,
    *,
    threshold: int,
    weight_fractional_bits: int = 8,
) -> int:
    votes.add_(-torch.sign(gradient).to(torch.int16))
    return commit_direct_residual(
        parameter,
        votes,
        threshold=threshold,
        weight_fractional_bits=weight_fractional_bits,
    )


def commit_direct_residual(
    parameter: torch.Tensor,
    votes: torch.Tensor,
    *,
    threshold: int,
    weight_fractional_bits: int,
) -> int:
    positive = votes >= threshold
    negative = votes <= -threshold
    writes = int(positive.sum().item() + negative.sum().item())
    if writes:
        with torch.no_grad():
            delta = positive.to(parameter.dtype) - negative.to(parameter.dtype)
            scale = float(1 << weight_fractional_bits)
            parameter.add_(delta / scale)
            parameter.clamp_(
                -(1 << (WEIGHT_CODE_BITS - 1)) / scale,
                ((1 << (WEIGHT_CODE_BITS - 1)) - 1) / scale,
            )
        votes.sub_(positive.to(torch.int16) * threshold)
        votes.add_(negative.to(torch.int16) * threshold)
    return writes


def apply_fixed_sgd_residual(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    residual: torch.Tensor,
    *,
    learning_rate: float,
    weight_fractional_bits: int,
    residual_fractional_bits: int,
) -> tuple[int, int, int, int]:
    """Apply magnitude-aware SGD with integer residual and committed codes.

    The forward path observes only signed 16-bit weight codes.  Sub-code SGD
    deltas are accumulated in ``residual`` at ``residual_fractional_bits`` per
    code LSB; each threshold crossing commits an integer code delta.  This is
    the software contract for a shift/scale plus residual-RMW datapath, not a
    floating-point optimizer hidden behind a quantized checkpoint.
    """

    if learning_rate <= 0.0:
        raise ValueError("fixed SGD learning rate must be positive")
    if not 1 <= weight_fractional_bits < WEIGHT_CODE_BITS:
        raise ValueError("weight fractional bits must be in [1, 15]")
    if not 1 <= residual_fractional_bits <= 15:
        raise ValueError("fixed SGD residual fractional bits must be in [1, 15]")
    if residual.dtype != torch.int32:
        raise ValueError("fixed SGD residual storage must use torch.int32")

    code_scale = 1 << weight_fractional_bits
    residual_scale = 1 << residual_fractional_bits
    scaled_delta = torch.round(
        -gradient.to(torch.float64)
        * learning_rate
        * float(code_scale * residual_scale)
    ).to(torch.int64)
    total = residual.to(torch.int64) + scaled_delta
    code_delta = torch.sign(total) * torch.div(
        total.abs(), residual_scale, rounding_mode="floor"
    )
    residual_next = total - code_delta * residual_scale

    code = torch.round(parameter.detach().to(torch.float64) * code_scale).to(
        torch.int64
    )
    code_min = -(1 << (WEIGHT_CODE_BITS - 1))
    code_max = (1 << (WEIGHT_CODE_BITS - 1)) - 1
    code_next = torch.clamp(code + code_delta, code_min, code_max)
    saturated = code_next != code + code_delta
    if torch.any(saturated):
        residual_next = torch.where(
            saturated, torch.zeros_like(residual_next), residual_next
        )

    changed = code_next != code
    address_writes = int(changed.sum().item())
    lsb_steps = int((code_next - code).abs().sum().item())
    increment_abs_max = int(scaled_delta.abs().max().item())
    residual_abs_max = int(residual_next.abs().max().item())
    residual.copy_(residual_next.to(torch.int32))
    with torch.no_grad():
        parameter.copy_((code_next.to(parameter.dtype) / float(code_scale)))
    return address_writes, lsb_steps, increment_abs_max, residual_abs_max


def adapt_selected_layers(
    model: torch.nn.Module,
    samples: torch.Tensor,
    labels: torch.Tensor,
    support_indices: list[int],
    *,
    batch_size: int,
    dx: int,
    dy: int,
    conv_vote_threshold: int,
    classifier_vote_threshold: int,
    window_timesteps: int,
    trainable_mask: str,
    classifier_row_mask: str,
    classifier_sample_gate: str,
    commit_scope: str,
    support_domain: str,
    update_rule: str,
    optimizer_lr: float,
    weight_fractional_bits: int = 8,
    fixed_sgd_residual_bits: int = 15,
    full_adaptation_batches_only: bool = False,
    force_class_mixed_batches: bool = False,
    corruption: EventCorruption | None = None,
) -> dict[str, object]:
    model.train()
    conv_indices, classifier_enabled = TRAINABLE_MASKS[trainable_mask]
    selected: dict[str, torch.Tensor] = {
        f"conv{index + 1}": model.network[index].fc.weight
        for index in conv_indices
    }
    if classifier_enabled:
        selected["classifier"] = model.network[8].fc.weight
    state = DirectResidualState(
        votes={
            name: torch.zeros_like(weight, dtype=torch.int16)
            for name, weight in selected.items()
        },
        fixed_residuals={
            name: torch.zeros_like(weight, dtype=torch.int32)
            for name, weight in selected.items()
        },
        code_writes={name: 0 for name in selected},
        code_lsb_steps={name: 0 for name in selected},
        gradient_active_windows={name: 0 for name in selected},
        gradient_nonzero_peak={name: 0 for name in selected},
        gradient_abs_max={name: 0.0 for name in selected},
        vote_abs_max={name: 0 for name in selected},
        fixed_increment_abs_max={name: 0 for name in selected},
        fixed_residual_abs_max={name: 0 for name in selected},
    )
    optimizer = None
    if update_rule == "adam":
        optimizer = torch.optim.Adam(tuple(selected.values()), lr=optimizer_lr)
    elif update_rule == "sgd":
        optimizer = torch.optim.SGD(tuple(selected.values()), lr=optimizer_lr)
    elif update_rule not in ("direct", "fixed_sgd"):
        raise ValueError(f"unsupported update rule: {update_rule}")
    before = {
        f"conv{index + 1}": model.network[index].fc.weight.detach().clone()
        for index in range(8)
    }
    before["classifier"] = model.network[8].fc.weight.detach().clone()
    windows = 0
    classifier_active_sample_steps = 0
    optimizer_steps = 0
    support_batches: list[
        tuple[list[int], int, int, EventCorruption | None]
    ] = []
    if conv_indices or force_class_mixed_batches:
        adaptation_batches = tp_pairwise_batches(
            support_indices, labels, batch_size
        )
    else:
        adaptation_batches = list(batches(support_indices, batch_size))
    if full_adaptation_batches_only:
        adaptation_batches = full_adaptation_batches(
            adaptation_batches, batch_size
        )
    adaptation_samples_used = sum(len(index_batch) for index_batch in adaptation_batches)
    support_batch_class_counts: list[int] = []
    for index_batch in adaptation_batches:
        support_batch_class_counts.append(
            int(torch.unique(labels[index_batch]).numel())
        )
        support_batches.append((index_batch, dx, dy, corruption))
        if support_domain == "mixed":
            support_batches.append((index_batch, 0, 0, None))
        elif support_domain != "shifted":
            raise ValueError(f"unsupported support domain: {support_domain}")
    for index_batch, view_dx, view_dy, view_corruption in support_batches:
        model.reset_potential()
        x = transform_event_view(
            samples[index_batch],
            index_batch,
            dx=view_dx,
            dy=view_dy,
            corruption=view_corruption,
        )
        target = F.one_hot(labels[index_batch], CLASSES).to(samples.dtype)
        gradient_accum = {
            name: torch.zeros_like(weight) for name, weight in selected.items()
        }
        for timestep in range(TIMESTEPS):
            logits, _ = model(x[:, timestep], target)
            for index in conv_indices:
                name = f"conv{index + 1}"
                conv_loss, _, _ = model.loss(
                    model.trace[index + 1],
                    model.trace_target[index + 1],
                    model.trace_target[index],
                )
                (conv_gradient,) = torch.autograd.grad(
                    conv_loss,
                    selected[name],
                    retain_graph=True,
                )
                gradient_accum[name].add_(conv_gradient.detach())
            if classifier_enabled:
                classifier_error = model.e
                if classifier_sample_gate == "misclassified":
                    sample_enable = logits.argmax(1) != labels[index_batch]
                    classifier_error = classifier_error * sample_enable[:, None]
                    classifier_active_sample_steps += int(sample_enable.sum().item())
                elif classifier_sample_gate == "all":
                    classifier_active_sample_steps += len(index_batch)
                else:
                    raise ValueError(
                        f"unsupported classifier sample gate: {classifier_sample_gate}"
                    )
                if classifier_row_mask == "target":
                    classifier_error = classifier_error * target
                elif classifier_row_mask == "target_competitor":
                    competitor = logits.masked_fill(target.bool(), float("-inf")).argmax(1)
                    row_enable = target + F.one_hot(competitor, CLASSES).to(target.dtype)
                    classifier_error = classifier_error * row_enable
                elif classifier_row_mask != "all":
                    raise ValueError(
                        f"unsupported classifier row mask: {classifier_row_mask}"
                    )
                classifier_gradient = torch.einsum(
                    "bi,bj->ij", classifier_error, model.trace[-1].flatten(1)
                ) / len(index_batch)
                gradient_accum["classifier"].add_(classifier_gradient.detach())
            model.detach_membrane_states()
            window_last = (timestep + 1) % window_timesteps == 0 or (
                timestep + 1 == TIMESTEPS
            )
            if window_last:
                for name, weight in selected.items():
                    nonzero_gradient = int(
                        torch.count_nonzero(gradient_accum[name]).item()
                    )
                    if nonzero_gradient:
                        state.gradient_active_windows[name] += 1
                        state.gradient_nonzero_peak[name] = max(
                            state.gradient_nonzero_peak[name], nonzero_gradient
                        )
                        state.gradient_abs_max[name] = max(
                            state.gradient_abs_max[name],
                            float(gradient_accum[name].abs().max().item()),
                        )
                    threshold = (
                        classifier_vote_threshold
                        if name == "classifier"
                        else conv_vote_threshold
                    )
                    if update_rule == "direct":
                        state.votes[name].add_(
                            -torch.sign(gradient_accum[name]).to(torch.int16)
                        )
                        state.vote_abs_max[name] = max(
                            state.vote_abs_max[name],
                            int(state.votes[name].abs().max().item()),
                        )
                        if commit_scope == "window":
                            state.code_writes[name] += commit_direct_residual(
                                weight,
                                state.votes[name],
                                threshold=threshold,
                                weight_fractional_bits=weight_fractional_bits,
                            )
                        elif commit_scope != "support":
                            raise ValueError(f"unsupported commit scope: {commit_scope}")
                    elif update_rule == "fixed_sgd":
                        (
                            address_writes,
                            lsb_steps,
                            increment_abs_max,
                            residual_abs_max,
                        ) = apply_fixed_sgd_residual(
                            weight,
                            gradient_accum[name],
                            state.fixed_residuals[name],
                            learning_rate=optimizer_lr,
                            weight_fractional_bits=weight_fractional_bits,
                            residual_fractional_bits=fixed_sgd_residual_bits,
                        )
                        state.code_writes[name] += address_writes
                        state.code_lsb_steps[name] += lsb_steps
                        state.fixed_increment_abs_max[name] = max(
                            state.fixed_increment_abs_max[name],
                            increment_abs_max,
                        )
                        state.fixed_residual_abs_max[name] = max(
                            state.fixed_residual_abs_max[name], residual_abs_max
                        )
                    else:
                        weight.grad = gradient_accum[name].clone()
                if optimizer is not None:
                    assert optimizer is not None
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                for name in selected:
                    gradient_accum[name].zero_()
                windows += 1
    if update_rule == "direct" and commit_scope == "support":
        for name, weight in selected.items():
            threshold = (
                classifier_vote_threshold
                if name == "classifier"
                else conv_vote_threshold
            )
            state.code_writes[name] += commit_direct_residual(
                weight,
                state.votes[name],
                threshold=threshold,
                weight_fractional_bits=weight_fractional_bits,
            )
    deltas: dict[str, dict[str, object]] = {}
    for name, initial in before.items():
        index = 8 if name == "classifier" else int(name.removeprefix("conv")) - 1
        weight = model.network[index].fc.weight
        delta = weight.detach() - initial
        deltas[name] = {
            "selected": name in selected,
            "nonzero": int(torch.count_nonzero(delta).item()),
            "l2": float(torch.linalg.vector_norm(delta).item()),
        }
        if name not in selected and deltas[name]["nonzero"] != 0:
            raise RuntimeError(f"disabled tensor changed under {trainable_mask}: {name}")
    updated_tensors = [
        name for name in selected if int(deltas[name]["nonzero"]) != 0
    ]
    gradient_inactive_tensors = [
        name
        for name in selected
        if state.gradient_active_windows[name] == 0
    ]
    uncommitted_tensors = [
        name
        for name in selected
        if state.gradient_active_windows[name] != 0
        and int(deltas[name]["nonzero"]) == 0
    ]
    mutable_parameters = sum(weight.numel() for weight in selected.values())
    return {
        "trainable_mask": trainable_mask,
        "classifier_row_mask": classifier_row_mask,
        "classifier_sample_gate": classifier_sample_gate,
        "commit_scope": commit_scope,
        "support_domain": support_domain,
        "update_rule": update_rule,
        "optimizer_lr": optimizer_lr if update_rule != "direct" else None,
        "weight_fractional_bits": weight_fractional_bits,
        "fixed_sgd_residual_bits": (
            fixed_sgd_residual_bits if update_rule == "fixed_sgd" else None
        ),
        "optimizer_steps": optimizer_steps,
        "classifier_active_sample_steps": classifier_active_sample_steps,
        "support_samples": len(support_indices),
        "adaptation_samples_used": adaptation_samples_used,
        "adaptation_samples_dropped": len(support_indices) - adaptation_samples_used,
        "full_adaptation_batches_only": full_adaptation_batches_only,
        "force_class_mixed_batches": force_class_mixed_batches,
        "adaptation_batch_sizes": [
            len(index_batch) for index_batch in adaptation_batches
        ],
        "adaptation_batch_class_counts": support_batch_class_counts,
        "windows": windows,
        "mutable_parameters": mutable_parameters,
        "code_plus_residual_bytes": mutable_parameters * 4,
        # Compatibility field for existing evidence readers. The storage is
        # still one 16-bit code plus one 16-bit residual per mutable weight.
        "q8_8_code_plus_residual_bytes": mutable_parameters * 4,
        "code_writes": state.code_writes,
        "code_lsb_steps": state.code_lsb_steps,
        "gradient_active_windows": state.gradient_active_windows,
        "gradient_nonzero_peak": state.gradient_nonzero_peak,
        "gradient_abs_max": state.gradient_abs_max,
        "vote_abs_max": state.vote_abs_max,
        "fixed_increment_abs_max": state.fixed_increment_abs_max,
        "fixed_residual_abs_max": state.fixed_residual_abs_max,
        "nonzero_residuals": {
            name: int(
                (
                    state.fixed_residuals[name]
                    if update_rule == "fixed_sgd"
                    else votes
                )
                .ne(0)
                .sum()
                .item()
            )
            for name, votes in state.votes.items()
        },
        "parameter_deltas": deltas,
        "updated_tensors": updated_tensors,
        "gradient_inactive_tensors": gradient_inactive_tensors,
        "uncommitted_tensors": uncommitted_tensors,
        "mask_realized": len(updated_tensors) == len(selected),
    }


def compact(result: dict[str, object]) -> dict[str, object]:
    return {key: result[key] for key in ("correct", "total", "accuracy_percent")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--selection-per-class", type=int, default=0)
    parser.add_argument(
        "--episode-seed",
        type=int,
        help=(
            "shuffle each class reproducibly before partitioning; omit to "
            "preserve the historical first-occurrence split"
        ),
    )
    parser.add_argument("--shift-x", type=int, default=2)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--event-drop-probability", type=float, default=0.0)
    parser.add_argument("--polarity-flip-probability", type=float, default=0.0)
    parser.add_argument("--temporal-jitter", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=38472)
    parser.add_argument("--vote-threshold", type=int, default=4)
    parser.add_argument("--conv-vote-threshold", type=int)
    parser.add_argument("--classifier-vote-threshold", type=int)
    parser.add_argument("--window-timesteps", type=int, default=2)
    parser.add_argument(
        "--classifier-row-mask",
        choices=("all", "target", "target_competitor"),
        default="all",
        help="limit classifier updates to selected output rows",
    )
    parser.add_argument(
        "--classifier-sample-gate",
        choices=("all", "misclassified"),
        default="all",
        help="optionally suppress classifier gradients for currently correct samples",
    )
    parser.add_argument(
        "--commit-scope",
        choices=("window", "support"),
        default="window",
        help="commit threshold crossings per timestep window or once after support",
    )
    parser.add_argument(
        "--support-domain",
        choices=("shifted", "mixed"),
        default="shifted",
        help="adapt on shifted support only or replay paired shifted and clean views",
    )
    parser.add_argument(
        "--update-rule",
        choices=("direct", "fixed_sgd", "sgd", "adam"),
        default="direct",
        help=(
            "direct is sign-only hardware-facing; fixed_sgd preserves numeric "
            "gradient magnitude in an integer residual; SGD diagnoses float "
            "fixed-scale updates; Adam is a software feasibility upper bound"
        ),
    )
    parser.add_argument("--optimizer-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--fixed-sgd-residual-bits",
        type=int,
        default=15,
        help="fractional residual bits per weight-code LSB for fixed_sgd",
    )
    parser.add_argument(
        "--full-adaptation-batches-only",
        action="store_true",
        help=(
            "drop an incomplete adaptation tail so every gradient uses the "
            "configured power-of-two batch normalization"
        ),
    )
    parser.add_argument(
        "--weight-fractional-bits",
        type=int,
        default=8,
        help="fractional bits in the signed 16-bit trainable-weight code",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("report", "selection", "both"),
        default="report",
        help=(
            "evaluate masks on the held-out report set, the hyperparameter "
            "selection set, or both; threshold selection should use selection"
        ),
    )
    parser.add_argument(
        "--trainable-masks",
        nargs="+",
        choices=tuple(TRAINABLE_MASKS),
        default=list(TRAINABLE_MASKS),
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    conv_vote_threshold = (
        args.conv_vote_threshold
        if args.conv_vote_threshold is not None
        else args.vote_threshold
    )
    classifier_vote_threshold = (
        args.classifier_vote_threshold
        if args.classifier_vote_threshold is not None
        else args.vote_threshold
    )
    if (
        conv_vote_threshold <= 0
        or classifier_vote_threshold <= 0
        or args.window_timesteps <= 0
    ):
        raise ValueError("vote threshold and timestep window must be positive")
    if args.optimizer_lr <= 0:
        raise ValueError("optimizer learning rate must be positive")
    if not 1 <= args.weight_fractional_bits < WEIGHT_CODE_BITS:
        raise ValueError("weight fractional bits must be in [1, 15]")
    if not 1 <= args.fixed_sgd_residual_bits <= 15:
        raise ValueError("fixed SGD residual bits must be in [1, 15]")
    corruption = EventCorruption(
        drop_probability=args.event_drop_probability,
        polarity_flip_probability=args.polarity_flip_probability,
        temporal_jitter=args.temporal_jitter,
        seed=args.corruption_seed,
    )
    corruption.validate()
    torch.manual_seed(38472)
    np.random.seed(38472)
    device = torch.device(args.device)
    words = read_signed_q8_8(
        args.inputs, SAMPLES * TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    )
    samples = torch.from_numpy(
        words.reshape(SAMPLES, TIMESTEPS, CHANNELS, HEIGHT, WIDTH)
    ).to(device)
    labels_np = read_labels(args.labels, SAMPLES)
    labels = torch.from_numpy(labels_np).to(device)
    support, selection, report = split_support_selection_report(
        labels_np,
        args.support_per_class,
        args.selection_per_class,
        episode_seed=args.episode_seed,
    )
    if args.evaluation_split in ("selection", "both") and not selection:
        raise ValueError(
            "evaluation-split requires --selection-per-class greater than zero"
        )
    baseline_model = build_model(
        args.traces_root,
        args.checkpoint,
        batch_size=max(args.batch_size, args.eval_batch_size),
        device=device,
        weight_fractional_bits=args.weight_fractional_bits,
    )

    result: dict[str, object] = {
        "status": "baseline_pass" if args.baseline_only else "pass",
        "claim_boundary": (
            "Deterministic software trend experiment starting from effective "
            "Q8.8 RTL weights and comparing explicit trainable masks. It is "
            "not bit-exact RTL inference, a natural-domain benchmark, or a "
            "statistical-significance result."
        ),
        "configuration": {
            "samples": SAMPLES,
            "timesteps": TIMESTEPS,
            "support_per_class": args.support_per_class,
            "support_samples": len(support),
            "selection_per_class": args.selection_per_class,
            "selection_samples": len(selection),
            "report_samples": len(report),
            "episode_seed": args.episode_seed,
            "evaluation_split": args.evaluation_split,
            "shift_x": args.shift_x,
            "shift_y": args.shift_y,
            **corruption.as_dict(),
            "trainable_masks": args.trainable_masks,
            "weight_format": (
                f"effective signed Q{WEIGHT_CODE_BITS - 1 - args.weight_fractional_bits}."
                f"{args.weight_fractional_bits}"
            ),
            "weight_code_bits": WEIGHT_CODE_BITS,
            "weight_fractional_bits": args.weight_fractional_bits,
            "adapt_batch": args.batch_size,
            "window_timesteps": args.window_timesteps,
            "vote_increment": 1,
            "conv_vote_threshold": conv_vote_threshold,
            "classifier_vote_threshold": classifier_vote_threshold,
            "classifier_row_mask": args.classifier_row_mask,
            "classifier_sample_gate": args.classifier_sample_gate,
            "commit_scope": args.commit_scope,
            "support_domain": args.support_domain,
            "update_rule": args.update_rule,
            "optimizer_lr": (
                args.optimizer_lr if args.update_rule != "direct" else None
            ),
            "fixed_sgd_residual_bits": (
                args.fixed_sgd_residual_bits
                if args.update_rule == "fixed_sgd"
                else None
            ),
            "full_adaptation_batches_only": args.full_adaptation_batches_only,
        },
    }
    if args.evaluation_split in ("report", "both"):
        result["baseline_clean"] = compact(evaluate(
            baseline_model,
            samples,
            labels,
            report,
            batch_size=args.eval_batch_size,
            dx=0,
            dy=0,
        ))
        result["baseline_shifted"] = compact(evaluate(
            baseline_model,
            samples,
            labels,
            report,
            batch_size=args.eval_batch_size,
            dx=args.shift_x,
            dy=args.shift_y,
            corruption=corruption,
        ))
    if args.evaluation_split in ("selection", "both"):
        result["selection_baseline_clean"] = compact(evaluate(
            baseline_model,
            samples,
            labels,
            selection,
            batch_size=args.eval_batch_size,
            dx=0,
            dy=0,
        ))
        result["selection_baseline_shifted"] = compact(evaluate(
            baseline_model,
            samples,
            labels,
            selection,
            batch_size=args.eval_batch_size,
            dx=args.shift_x,
            dy=args.shift_y,
            corruption=corruption,
        ))
    if not args.baseline_only:
        masks: list[dict[str, object]] = []
        for trainable_mask in args.trainable_masks:
            model = build_model(
                args.traces_root,
                args.checkpoint,
                batch_size=max(args.batch_size, args.eval_batch_size),
                device=device,
                weight_fractional_bits=args.weight_fractional_bits,
            )
            update = adapt_selected_layers(
                model,
                samples,
                labels,
                support,
                batch_size=args.batch_size,
                dx=args.shift_x,
                dy=args.shift_y,
                conv_vote_threshold=conv_vote_threshold,
                classifier_vote_threshold=classifier_vote_threshold,
                window_timesteps=args.window_timesteps,
                trainable_mask=trainable_mask,
                classifier_row_mask=args.classifier_row_mask,
                classifier_sample_gate=args.classifier_sample_gate,
                commit_scope=args.commit_scope,
                support_domain=args.support_domain,
                update_rule=args.update_rule,
                optimizer_lr=args.optimizer_lr,
                weight_fractional_bits=args.weight_fractional_bits,
                fixed_sgd_residual_bits=args.fixed_sgd_residual_bits,
                full_adaptation_batches_only=args.full_adaptation_batches_only,
                corruption=corruption,
            )
            mask_result: dict[str, object] = {
                "trainable_mask": trainable_mask,
                "update": update,
            }
            if args.evaluation_split in ("report", "both"):
                clean_after = evaluate(
                    model,
                    samples,
                    labels,
                    report,
                    batch_size=args.eval_batch_size,
                    dx=0,
                    dy=0,
                )
                shifted_after = evaluate(
                    model,
                    samples,
                    labels,
                    report,
                    batch_size=args.eval_batch_size,
                    dx=args.shift_x,
                    dy=args.shift_y,
                    corruption=corruption,
                )
                baseline_clean = result["baseline_clean"]
                baseline_shifted = result["baseline_shifted"]
                assert isinstance(baseline_clean, dict)
                assert isinstance(baseline_shifted, dict)
                mask_result.update({
                    "clean_after": compact(clean_after),
                    "shifted_after": compact(shifted_after),
                    "shifted_gain_points": (
                        shifted_after["accuracy_percent"]
                        - baseline_shifted["accuracy_percent"]
                    ),
                    "clean_retention_points": (
                        clean_after["accuracy_percent"]
                        - baseline_clean["accuracy_percent"]
                    ),
                })
            if args.evaluation_split in ("selection", "both"):
                selection_clean_after = evaluate(
                    model,
                    samples,
                    labels,
                    selection,
                    batch_size=args.eval_batch_size,
                    dx=0,
                    dy=0,
                )
                selection_shifted_after = evaluate(
                    model,
                    samples,
                    labels,
                    selection,
                    batch_size=args.eval_batch_size,
                    dx=args.shift_x,
                    dy=args.shift_y,
                    corruption=corruption,
                )
                selection_baseline_clean = result["selection_baseline_clean"]
                selection_baseline_shifted = result["selection_baseline_shifted"]
                assert isinstance(selection_baseline_clean, dict)
                assert isinstance(selection_baseline_shifted, dict)
                mask_result.update({
                    "selection_clean_after": compact(selection_clean_after),
                    "selection_shifted_after": compact(selection_shifted_after),
                    "selection_shifted_gain_points": (
                        selection_shifted_after["accuracy_percent"]
                        - selection_baseline_shifted["accuracy_percent"]
                    ),
                    "selection_clean_retention_points": (
                        selection_clean_after["accuracy_percent"]
                        - selection_baseline_clean["accuracy_percent"]
                    ),
                })
            masks.append(mask_result)
        result["masks"] = masks

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
