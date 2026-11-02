#!/usr/bin/env python3
"""Replay two formal SHD M2 batches with preserved Adam and direct residual.

The preserved Adam lane is checked against the handoff's timestep predictions
and final optimizer state. The direct lane starts from the same V28 forward
weights and runs continuously through V28 and V29. This is a training-stream
screen, not an official-test retention or adaptation-accuracy result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]
BATCH = 64
BACKBONE = 450
HIDDEN = 256
CLASSES = 20
LEAK_Q14 = 13_926
SPIKE_Q14 = 16_384
THRESHOLD_Q14 = 8_192
PI2_Q20 = 10_349_030
WH1_LR_Q32 = 53_687
WOUT_LR_Q32 = 124_929
EXP_TABLE_Q20 = np.rint(
    np.exp(-np.arange(129, dtype=np.float64) / 16.0) * (1 << 20)
).astype(np.int64)


@dataclass
class DynamicState:
    membrane: np.ndarray
    trace: np.ndarray
    output_logit: np.ndarray


@dataclass
class DirectState:
    wh1_code: np.ndarray
    wh1_residual: np.ndarray
    wout_code: np.ndarray
    wout_residual: np.ndarray


def read_signed_hex(path: Path, bits: int, expected: int | None = None) -> np.ndarray:
    raw = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()),
        dtype=np.uint64,
    )
    if expected is not None and raw.size != expected:
        raise RuntimeError(f"{path.name}: expected {expected} values, got {raw.size}")
    sign = np.uint64(1 << (bits - 1))
    signed = raw.astype(object)
    signed[raw >= sign] -= 1 << bits
    return signed.astype(np.int64)


def round_shift_even(values: np.ndarray | int, shift: int) -> np.ndarray:
    array = np.asarray(values)
    if shift == 0:
        return array.astype(np.int64)
    use_object = array.dtype == object
    work = array.astype(object) if use_object else array.astype(np.int64, copy=False)
    negative = work < 0
    magnitude = np.where(negative, -work, work)
    divisor = 1 << shift
    quotient = magnitude // divisor
    remainder = magnitude % divisor
    half = divisor >> 1
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded = quotient + increment.astype(np.int64)
    result = np.where(negative, -rounded, rounded)
    return np.asarray(result, dtype=np.int64)


def round_div_even_unsigned(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    quotient = numerator // denominator
    remainder = numerator % denominator
    increment = (remainder * 2 > denominator) | (
        (remainder * 2 == denominator) & ((quotient & 1) != 0)
    )
    return np.asarray(quotient + increment.astype(np.int64), dtype=np.int64)


def softmax_q24(values_q12: np.ndarray) -> np.ndarray:
    shifted = values_q12 - values_q12.max(axis=1, keepdims=True)
    magnitude = -shifted
    index = magnitude >> 8
    remainder = magnitude & 255
    index += ((remainder > 128) | ((remainder == 128) & ((index & 1) != 0)))
    index = np.clip(index, 0, 128)
    exponent = EXP_TABLE_Q20[index]
    denominator = exponent.sum(axis=1, keepdims=True)
    numerator = exponent << 24
    return round_div_even_unsigned(numerator, denominator)


def surrogate_factor_q24(membrane_pre_q14: np.ndarray) -> np.ndarray:
    delta = membrane_pre_q14.astype(object) - THRESHOLD_Q14
    product = delta * delta * PI2_Q20
    term_q28 = round_shift_even(product, 20).astype(object)
    denominator = term_q28 + (1 << 28)
    numerator = np.full(denominator.shape, 1 << 52, dtype=object)
    factor = round_div_even_unsigned(numerator, denominator)
    return np.minimum(factor, 1 << 24).astype(np.int64)


def target_probability_q24(target_trace_q14: np.ndarray) -> np.ndarray:
    difference = (
        target_trace_q14[:, None, :].astype(np.int64)
        - target_trace_q14[None, :, :].astype(np.int64)
    )
    distance_squared = np.sum(difference * difference, axis=2, dtype=np.int64)
    distance_q12 = -np.fromiter(
        ((math.isqrt(int(value)) + 2) >> 2 for value in distance_squared.reshape(-1)),
        dtype=np.int64,
        count=BATCH * BATCH,
    ).reshape(BATCH, BATCH)
    return softmax_q24(distance_q12)


def forward_and_gradients(
    backbone_q14: np.ndarray,
    target_trace_q14: np.ndarray,
    labels: np.ndarray,
    wh1_code_q14: np.ndarray,
    wout_code_q14: np.ndarray,
    state: DynamicState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, DynamicState]:
    current_q14 = round_shift_even(backbone_q14 @ wh1_code_q14.T, 14)
    membrane_pre_q14 = (
        round_shift_even(state.membrane * LEAK_Q14, 14) + current_q14
    )
    spike = (membrane_pre_q14 >= THRESHOLD_Q14).astype(np.int64)
    membrane = membrane_pre_q14 - spike * THRESHOLD_Q14
    trace = round_shift_even(state.trace * LEAK_Q14, 14) + spike * SPIKE_Q14

    similarity_q28 = trace[:BATCH] @ trace[BATCH:].T
    probability_q24 = softmax_q24(round_shift_even(similarity_q28, 16))
    target_q24 = target_probability_q24(target_trace_q14)
    pair_error_q40 = (probability_q24 - target_q24) << 10
    input_delta_q40 = round_shift_even(pair_error_q40 @ trace[BATCH:], 14)
    target_delta_q40 = round_shift_even(pair_error_q40.T @ trace[:BATCH], 14)
    incoming_q40 = np.concatenate([input_delta_q40, target_delta_q40], axis=0)
    factor_q24 = surrogate_factor_q24(membrane_pre_q14)
    local_delta_q40 = round_shift_even(
        incoming_q40.astype(object) * factor_q24.astype(object), 24
    )
    wh1_gradient_q40 = round_shift_even(local_delta_q40.T @ backbone_q14, 14)

    output_logit_q28 = (
        state.output_logit + (spike[:BATCH] * SPIKE_Q14) @ wout_code_q14.T
    )
    output_probability_q24 = softmax_q24(round_shift_even(output_logit_q28, 16))
    output_error_q40 = output_probability_q24.copy()
    output_error_q40[np.arange(BATCH), labels] -= 1 << 24
    output_error_q40 <<= 16
    wout_gradient_q40 = round_shift_even(output_error_q40.T @ trace[:BATCH], 20)

    prediction = np.argmax(output_logit_q28, axis=1).astype(np.int64)
    tp_probability = probability_q24 / float(1 << 24)
    tp_target = target_q24 / float(1 << 24)
    output_probability = output_probability_q24 / float(1 << 24)
    tp_loss = float(
        -np.sum(tp_target * np.log(np.maximum(tp_probability, 2.0**-24))) / BATCH
    )
    ce_loss = float(
        -np.mean(
            np.log(np.maximum(output_probability[np.arange(BATCH), labels], 2.0**-24))
        )
    )
    return (
        wh1_gradient_q40,
        wout_gradient_q40,
        prediction,
        tp_loss,
        ce_loss,
        DynamicState(membrane, trace, output_logit_q28),
    )


def direct_update_tensor(
    code: np.ndarray,
    residual: np.ndarray,
    gradient_q40: np.ndarray,
    increment: int,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    nonzero = gradient_q40 != 0
    direction = np.where(gradient_q40 > 0, -1, np.where(gradient_q40 < 0, 1, 0))
    normalized = residual + direction * increment
    delta = np.zeros_like(code)
    for _ in range(2):
        positive = normalized >= threshold
        negative = normalized <= -threshold
        normalized = np.where(positive, normalized - threshold, normalized)
        normalized = np.where(negative, normalized + threshold, normalized)
        delta += positive.astype(np.int64) - negative.astype(np.int64)

    candidate = code + delta
    high = candidate > 32767
    low = candidate < -32768
    saturated = high | low
    updated_code = np.clip(candidate, -32768, 32767).astype(np.int64)
    updated_residual = np.where(saturated, 0, normalized).astype(np.int64)
    updated_code = np.where(nonzero, updated_code, code)
    updated_residual = np.where(nonzero, updated_residual, residual)
    return updated_code, updated_residual, {
        "rmw": int(np.count_nonzero(nonzero)),
        "code_writes": int(np.count_nonzero(updated_code != code)),
        "residual_writes": int(np.count_nonzero(updated_residual != residual)),
        "saturations": int(np.count_nonzero(saturated & nonzero)),
        "zero_skips": int(nonzero.size - np.count_nonzero(nonzero)),
    }


def empty_dynamic_state() -> DynamicState:
    return DynamicState(
        np.zeros((2 * BATCH, HIDDEN), dtype=np.int64),
        np.zeros((2 * BATCH, HIDDEN), dtype=np.int64),
        np.zeros((BATCH, CLASSES), dtype=np.int64),
    )


def load_timestep(vector_dir: Path, prefix: str, timestep: int) -> tuple[np.ndarray, np.ndarray]:
    backbone = read_signed_hex(
        vector_dir / f"{prefix}_t{timestep:03d}_backbone_q14.mem",
        32,
        2 * BATCH * BACKBONE,
    ).reshape(2 * BATCH, BACKBONE)
    target = read_signed_hex(
        vector_dir / f"{prefix}_t{timestep:03d}_target_q14.mem",
        32,
        BATCH * BACKBONE,
    ).reshape(BATCH, BACKBONE)
    return backbone, target


def add_stats(total: dict[str, int], step: dict[str, int]) -> None:
    for name, value in step.items():
        total[name] = total.get(name, 0) + value


def compare_exact(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    if mismatch.size:
        index = int(mismatch[0])
        raise RuntimeError(
            f"{name}: first mismatch at {index}: "
            f"actual={actual.reshape(-1)[index]} expected={expected.reshape(-1)[index]} "
            f"total={mismatch.size}"
        )


def evaluate_official_retention(
    handoff: Path,
    dataset_root: Path,
    direct: DirectState,
) -> dict[str, Any]:
    fixed = importlib.import_module("m2_fixedpoint_forward")
    trainer = importlib.import_module("multilayer_tp_train")
    base = importlib.import_module("m1_m2_train")
    config = json.loads(
        (handoff / "tp_exp" / "configs" / "B64_FORMAL_SEED_A.json").read_text(
            encoding="utf-8"
        )
    )
    config["dataset_root_relative"] = str(dataset_root.resolve())
    model, _, optimizers, schedulers = trainer.prepare(config)
    checkpoint = handoff / "tp_exp" / "runs" / "formal_m2" / "checkpoints" / "best_checkpoint.pt"
    base.restore_checkpoint(checkpoint, model, optimizers, schedulers)
    with torch.no_grad():
        model.network[1].fc.weight.copy_(
            torch.from_numpy(direct.wh1_code.astype(np.float32) / float(1 << 14))
        )
        model.network[2].fc.weight.copy_(
            torch.from_numpy(direct.wout_code.astype(np.float32) / float(1 << 14))
        )
    _, test_dataset, collate_fn = trainer.datasets(config)
    direct_result = fixed.fixed_inference(model, test_dataset, collate_fn, BATCH)
    baseline = json.loads(
        (
            handoff
            / "results"
            / "fixedpoint_rtl_stage11"
            / "fixedpoint_forward_results.json"
        ).read_text(encoding="utf-8")
    )["fixed"]
    return {
        "status": "pass",
        "dataset_root": str(dataset_root.resolve()),
        "samples": direct_result["samples"],
        "baseline_fixed_correct": baseline["correct"],
        "baseline_fixed_accuracy": baseline["accuracy"],
        "direct_after_two_batches_correct": direct_result["correct"],
        "direct_after_two_batches_accuracy": direct_result["accuracy"],
        "correct_delta": direct_result["correct"] - baseline["correct"],
        "accuracy_delta": direct_result["accuracy"] - baseline["accuracy"],
        "weight_saturation_count": direct_result["weight_saturation_count"],
        "state_saturation_count": direct_result["state_saturation_count"],
        "claim_boundary": (
            "Official SHD test retention after exactly two formal online-training "
            "batches. It is not shifted-domain adaptation evidence."
        ),
    }


def run_screen(
    handoff: Path,
    dataset_root: Path | None,
    wh1_increment: int,
    wh1_threshold: int,
    wout_increment: int,
    wout_threshold: int,
) -> dict[str, Any]:
    vector_dir = handoff / "tp_exp" / "vectors" / "m2_tp"
    python_dir = handoff / "tp_exp" / "python"
    sys.path.insert(0, str(python_dir))
    adam = importlib.import_module("m2_adam_fixedpoint")

    wh1_master = read_signed_hex(
        vector_dir / "V28_wh_weight_initial_q40.mem", 64, HIDDEN * BACKBONE
    ).reshape(HIDDEN, BACKBONE)
    wh1_m = read_signed_hex(
        vector_dir / "V28_wh_m_initial_q40.mem", 64, HIDDEN * BACKBONE
    ).reshape(HIDDEN, BACKBONE)
    wh1_v = read_signed_hex(
        vector_dir / "V28_wh_v_initial_q60.mem", 64, HIDDEN * BACKBONE
    ).reshape(HIDDEN, BACKBONE)
    wout_master = read_signed_hex(
        vector_dir / "V28_wo_weight_initial_q40.mem", 64, CLASSES * HIDDEN
    ).reshape(CLASSES, HIDDEN)
    wout_m = read_signed_hex(
        vector_dir / "V28_wo_m_initial_q40.mem", 64, CLASSES * HIDDEN
    ).reshape(CLASSES, HIDDEN)
    wout_v = read_signed_hex(
        vector_dir / "V28_wo_v_initial_q60.mem", 64, CLASSES * HIDDEN
    ).reshape(CLASSES, HIDDEN)

    direct = DirectState(
        read_signed_hex(
            vector_dir / "V28_wh_q14_initial.mem", 16, HIDDEN * BACKBONE
        ).reshape(HIDDEN, BACKBONE),
        np.zeros((HIDDEN, BACKBONE), dtype=np.int64),
        read_signed_hex(
            vector_dir / "V28_wo_q14_initial.mem", 16, CLASSES * HIDDEN
        ).reshape(CLASSES, HIDDEN),
        np.zeros((CLASSES, HIDDEN), dtype=np.int64),
    )
    compare_exact(
        "V28 WH1 forward quantization",
        direct.wh1_code,
        np.clip(round_shift_even(wh1_master, 26), -32768, 32767),
    )
    compare_exact(
        "V28 Wout forward quantization",
        direct.wout_code,
        np.clip(round_shift_even(wout_master, 26), -32768, 32767),
    )

    direct_stats = {name: 0 for name in (
        "rmw", "code_writes", "residual_writes", "saturations", "zero_skips"
    )}
    batch_rows: list[dict[str, Any]] = []
    optimizer_step = 212_457

    for prefix in ("V28", "V29"):
        timesteps = int((vector_dir / f"{prefix}_actual_t.mem").read_text().strip(), 16)
        labels = read_signed_hex(vector_dir / f"{prefix}_labels.mem", 8, BATCH)
        expected_predictions = read_signed_hex(
            vector_dir / f"{prefix}_expected_predictions.mem",
            8,
            timesteps * BATCH,
        ).reshape(timesteps, BATCH)
        adam_dynamic = empty_dynamic_state()
        direct_dynamic = empty_dynamic_state()
        direct_correct_all = 0
        adam_correct_all = 0
        direct_losses_tp: list[float] = []
        direct_losses_ce: list[float] = []
        adam_losses_tp: list[float] = []
        adam_losses_ce: list[float] = []
        first_direct_prediction_divergence: int | None = None

        for timestep in range(timesteps):
            backbone, target = load_timestep(vector_dir, prefix, timestep)
            adam_wh1_code = np.clip(round_shift_even(wh1_master, 26), -32768, 32767)
            adam_wout_code = np.clip(round_shift_even(wout_master, 26), -32768, 32767)
            (
                adam_wh1_gradient,
                adam_wout_gradient,
                adam_prediction,
                adam_tp_loss,
                adam_ce_loss,
                adam_dynamic,
            ) = forward_and_gradients(
                backbone,
                target,
                labels,
                adam_wh1_code,
                adam_wout_code,
                adam_dynamic,
            )
            compare_exact(
                f"{prefix} Adam prediction timestep {timestep}",
                adam_prediction,
                expected_predictions[timestep],
            )
            optimizer_step += 1
            wh1_master, wh1_m, wh1_v, _, _ = adam.fixed_step(
                wh1_master,
                wh1_m,
                wh1_v,
                adam_wh1_gradient.astype(np.float64) / (1 << 40),
                WH1_LR_Q32 / float(1 << 32),
                optimizer_step,
            )
            wout_master, wout_m, wout_v, _, _ = adam.fixed_step(
                wout_master,
                wout_m,
                wout_v,
                adam_wout_gradient.astype(np.float64) / (1 << 40),
                WOUT_LR_Q32 / float(1 << 32),
                optimizer_step,
            )

            (
                direct_wh1_gradient,
                direct_wout_gradient,
                direct_prediction,
                direct_tp_loss,
                direct_ce_loss,
                direct_dynamic,
            ) = forward_and_gradients(
                backbone,
                target,
                labels,
                direct.wh1_code,
                direct.wout_code,
                direct_dynamic,
            )
            direct.wh1_code, direct.wh1_residual, wh1_step = direct_update_tensor(
                direct.wh1_code,
                direct.wh1_residual,
                direct_wh1_gradient,
                increment=wh1_increment,
                threshold=wh1_threshold,
            )
            direct.wout_code, direct.wout_residual, wout_step = direct_update_tensor(
                direct.wout_code,
                direct.wout_residual,
                direct_wout_gradient,
                increment=wout_increment,
                threshold=wout_threshold,
            )
            add_stats(direct_stats, wh1_step)
            add_stats(direct_stats, wout_step)

            if (
                first_direct_prediction_divergence is None
                and np.any(direct_prediction != adam_prediction)
            ):
                first_direct_prediction_divergence = timestep
            adam_correct_all += int(np.count_nonzero(adam_prediction == labels))
            direct_correct_all += int(np.count_nonzero(direct_prediction == labels))
            adam_losses_tp.append(adam_tp_loss)
            adam_losses_ce.append(adam_ce_loss)
            direct_losses_tp.append(direct_tp_loss)
            direct_losses_ce.append(direct_ce_loss)

        expected_final = {
            "wh_weight": read_signed_hex(
                vector_dir / f"{prefix}_wh_weight_final_q40.mem", 64
            ).reshape(HIDDEN, BACKBONE),
            "wh_m": read_signed_hex(
                vector_dir / f"{prefix}_wh_m_final_q40.mem", 64
            ).reshape(HIDDEN, BACKBONE),
            "wh_v": read_signed_hex(
                vector_dir / f"{prefix}_wh_v_final_q60.mem", 64
            ).reshape(HIDDEN, BACKBONE),
            "wo_weight": read_signed_hex(
                vector_dir / f"{prefix}_wo_weight_final_q40.mem", 64
            ).reshape(CLASSES, HIDDEN),
            "wo_m": read_signed_hex(
                vector_dir / f"{prefix}_wo_m_final_q40.mem", 64
            ).reshape(CLASSES, HIDDEN),
            "wo_v": read_signed_hex(
                vector_dir / f"{prefix}_wo_v_final_q60.mem", 64
            ).reshape(CLASSES, HIDDEN),
        }
        compare_exact(f"{prefix} Adam WH1 weight", wh1_master, expected_final["wh_weight"])
        compare_exact(f"{prefix} Adam WH1 m", wh1_m, expected_final["wh_m"])
        compare_exact(f"{prefix} Adam WH1 v", wh1_v, expected_final["wh_v"])
        compare_exact(f"{prefix} Adam Wout weight", wout_master, expected_final["wo_weight"])
        compare_exact(f"{prefix} Adam Wout m", wout_m, expected_final["wo_m"])
        compare_exact(f"{prefix} Adam Wout v", wout_v, expected_final["wo_v"])

        adam_forward_wh1 = np.clip(round_shift_even(wh1_master, 26), -32768, 32767)
        adam_forward_wout = np.clip(round_shift_even(wout_master, 26), -32768, 32767)
        batch_rows.append({
            "prefix": prefix,
            "timesteps": timesteps,
            "samples_per_timestep": BATCH,
            "adam_vector_reproduction": "bit_exact",
            "adam_all_timestep_accuracy": adam_correct_all / (timesteps * BATCH),
            "direct_all_timestep_accuracy": direct_correct_all / (timesteps * BATCH),
            "adam_final_timestep_accuracy": float(np.mean(adam_prediction == labels)),
            "direct_final_timestep_accuracy": float(np.mean(direct_prediction == labels)),
            "adam_mean_tp_loss": float(np.mean(adam_losses_tp)),
            "direct_mean_tp_loss": float(np.mean(direct_losses_tp)),
            "adam_mean_ce_loss": float(np.mean(adam_losses_ce)),
            "direct_mean_ce_loss": float(np.mean(direct_losses_ce)),
            "first_direct_prediction_divergence_from_adam": first_direct_prediction_divergence,
            "direct_vs_adam_forward_code": {
                "wh1_mismatch_count": int(np.count_nonzero(direct.wh1_code != adam_forward_wh1)),
                "wh1_max_abs_lsb": int(np.max(np.abs(direct.wh1_code - adam_forward_wh1))),
                "wout_mismatch_count": int(np.count_nonzero(direct.wout_code != adam_forward_wout)),
                "wout_max_abs_lsb": int(np.max(np.abs(direct.wout_code - adam_forward_wout))),
            },
        })

        if prefix == "V28":
            compare_exact(
                "V29 WH1 initial continuity",
                wh1_master,
                read_signed_hex(
                    vector_dir / "V29_wh_weight_initial_q40.mem", 64
                ).reshape(HIDDEN, BACKBONE),
            )
            compare_exact(
                "V29 Wout initial continuity",
                wout_master,
                read_signed_hex(
                    vector_dir / "V29_wo_weight_initial_q40.mem", 64
                ).reshape(CLASSES, HIDDEN),
            )

    result = {
        "status": "pass",
        "screen": "two consecutive formal B64 training-stream batches",
        "claim_boundary": (
            "Preserved Adam is reproduced bit-exact for V28/V29. Direct-residual "
            "uses its own evolving states and gradients from the same frozen-backbone "
            "inputs. Training-stream metrics do not establish official-test retention "
            "or shifted-domain adaptation accuracy."
        ),
        "direct_configuration": {
            "wh1": {"increment": wh1_increment, "threshold": wh1_threshold},
            "wout": {"increment": wout_increment, "threshold": wout_threshold},
            "code": "signed Q2.14",
            "residual_bits": 16,
        },
        "preserved_adam_learning_rate_q32": {
            "wh1": WH1_LR_Q32,
            "wout": WOUT_LR_Q32,
        },
        "final_optimizer_step": optimizer_step,
        "direct_traffic": direct_stats,
        "direct_residual_range": {
            "wh1_min": int(direct.wh1_residual.min()),
            "wh1_max": int(direct.wh1_residual.max()),
            "wout_min": int(direct.wout_residual.min()),
            "wout_max": int(direct.wout_residual.max()),
        },
        "batches": batch_rows,
    }
    if dataset_root is not None:
        result["official_test_retention"] = evaluate_official_retention(
            handoff, dataset_root, direct
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--wh1-increment", type=int, default=25)
    parser.add_argument("--wh1-threshold", type=int, default=61)
    parser.add_argument("--wout-increment", type=int, default=77)
    parser.add_argument("--wout-threshold", type=int, default=47)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name, increment, threshold in (
        ("wh1", args.wh1_increment, args.wh1_threshold),
        ("wout", args.wout_increment, args.wout_threshold),
    ):
        if threshold <= 0 or increment < 0 or increment >= 2 * threshold:
            raise ValueError(
                f"illegal {name} direct configuration: "
                f"increment={increment} threshold={threshold}"
            )
    result = run_screen(
        args.handoff,
        None if args.skip_official else args.dataset_root,
        args.wh1_increment,
        args.wh1_threshold,
        args.wout_increment,
        args.wout_threshold,
    )
    output = args.output or (
        ROOT / "results" / "learning_reference" /
        "srnn_m2_direct_vs_adam_two_batch_screen.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    v28, v29 = result["batches"]
    retention = result.get("official_test_retention")
    retention_text = (
        " official="
        f"{retention['direct_after_two_batches_correct']}/{retention['samples']}"
        if retention is not None
        else ""
    )
    print(
        "SRNN_M2_DIRECT_VS_ADAM_TWO_BATCH_SCREEN_PASS "
        f"v28_direct={v28['direct_final_timestep_accuracy']:.6f} "
        f"v29_direct={v29['direct_final_timestep_accuracy']:.6f} "
        f"rmw={result['direct_traffic']['rmw']} "
        f"saturations={result['direct_traffic']['saturations']}"
        f"{retention_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
