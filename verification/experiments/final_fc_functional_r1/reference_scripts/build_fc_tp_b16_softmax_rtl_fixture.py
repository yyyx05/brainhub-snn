#!/usr/bin/env python3
"""Build real FC B16 rows for the shared Q6/Q0.16 softmax RTL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "learning_reference" / "fc_tp_h400_b16_first_active_reference.npz"
SCREEN = ROOT / "results" / "learning_reference" / "fc_tp_b16_softmax_frontend_screen.json"
OUTPUT = ROOT / "results" / "learning_reference" / "fc_tp_b16_softmax_rtl_fixture.json"
DESCRIPTOR_MEM = ROOT / "tb" / "data" / "fc_tp_b16_softmax_row_descriptors.mem"
TOKEN_MEM = ROOT / "tb" / "data" / "fc_tp_b16_softmax_tokens.mem"
OUTPUT_DESCRIPTOR_MEM = ROOT / "tb" / "data" / "fc_tp_b16_output_error_descriptors.mem"
OUTPUT_TOKEN_MEM = ROOT / "tb" / "data" / "fc_tp_b16_output_error_tokens.mem"
W2_TRACE_MEM = ROOT / "tb" / "data" / "fc_tp_b16_w2_scheduler_hidden_traces.mem"
W2_ERROR_MEM = ROOT / "tb" / "data" / "fc_tp_b16_w2_scheduler_errors.mem"
W2_EXPECTED_MEM = ROOT / "tb" / "data" / "fc_tp_b16_w2_scheduler_expected.mem"

BATCH = 16
OUTPUTS = 10
INPUT_FRAC = 6
LUT_FRAC = 3
PROB_FRAC = 16
PROB_ONE = 1 << PROB_FRAC
EXP_CUTOFF = 8


def _twos(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _exp_table() -> np.ndarray:
    indices = np.arange((EXP_CUTOFF << LUT_FRAC) + 1)
    return np.rint(np.exp(-indices / (1 << LUT_FRAC)) * PROB_ONE).astype(np.int64)


def _quantized_row(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value_q = np.rint(values * (1 << INPUT_FRAC)).astype(np.int64)
    if np.any(value_q < -32768) or np.any(value_q > 32767):
        raise AssertionError("Q6 softmax input exceeds signed 16-bit range")
    delta_q = value_q.max() - value_q
    lut_index = np.clip((delta_q + 4) >> 3, 0, EXP_CUTOFF << LUT_FRAC)
    exponent = _exp_table()[lut_index]
    denominator = int(exponent.sum())
    probability_q = (exponent * PROB_ONE + denominator // 2) // denominator
    return value_q, probability_q.astype(np.int64)


def build() -> dict[str, object]:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    selected = screen["selected"]
    if selected["input_fractional_bits"] != INPUT_FRAC:
        raise AssertionError("screened softmax input precision changed")
    if selected["lut_fractional_bits"] != LUT_FRAC:
        raise AssertionError("screened softmax LUT precision changed")
    if selected["lut_entries"] != 65:
        raise AssertionError("screened softmax LUT depth changed")

    reference = np.load(SOURCE)
    timesteps = reference["screen_timesteps"].astype(np.int64)
    hidden_trace = reference["screen_hidden_trace"].astype(np.float32)
    target_hidden_trace = reference["screen_target_hidden_trace"].astype(np.float32)
    target_input_trace = reference["screen_target_input_trace"].astype(np.float32)
    logits = reference["screen_logits"].astype(np.float32)
    labels = reference["labels"].astype(np.int64)
    if timesteps.tolist() != list(range(8)):
        raise AssertionError(f"expected timestep 0..7, found {timesteps.tolist()}")

    descriptor_lines: list[str] = []
    token_lines: list[str] = []
    rows: list[dict[str, int | str]] = []
    output_descriptor_lines: list[str] = []
    output_token_lines: list[str] = []
    row_sum_min = 1 << 30
    row_sum_max = 0

    def append_rows(values: np.ndarray, kind: int, name: str, timestep: int) -> None:
        nonlocal row_sum_min, row_sum_max
        for row_index, row in enumerate(values):
            value_q, probability_q = _quantized_row(row)
            row_length = int(row.shape[0])
            descriptor = row_length
            descriptor |= kind << 5
            descriptor |= timestep << 7
            descriptor |= row_index << 11
            descriptor_lines.append(f"{descriptor:08x}")
            probability_sum = int(probability_q.sum())
            row_sum_min = min(row_sum_min, probability_sum)
            row_sum_max = max(row_sum_max, probability_sum)
            rows.append({
                "kind": name,
                "timestep": timestep,
                "row_index": row_index,
                "row_length": row_length,
                "probability_sum_q016": probability_sum,
            })
            for input_q, expected_q in zip(value_q, probability_q):
                token = _twos(int(input_q), 16)
                token |= int(expected_q) << 16
                token_lines.append(f"{token:016x}")

    for window_index, timestep_value in enumerate(timesteps):
        student_similarity = hidden_trace[window_index] @ target_hidden_trace[window_index].T
        trace_delta = (
            target_input_trace[window_index][:, None, :] -
            target_input_trace[window_index][None, :, :] + 1e-9
        )
        target_negative_distance = -np.sqrt(np.square(trace_delta).sum(axis=-1))
        append_rows(student_similarity, 0, "student_similarity", int(timestep_value))
        append_rows(target_negative_distance, 1, "target_negative_distance", int(timestep_value))
        append_rows(logits[window_index], 2, "output_logits", int(timestep_value))
        for sample_index, row in enumerate(logits[window_index]):
            value_q, probability_q = _quantized_row(row)
            label = int(labels[sample_index])
            error_q = probability_q.copy()
            error_q[label] -= PROB_ONE
            descriptor = OUTPUTS
            descriptor |= label << 5
            descriptor |= sample_index << 9
            descriptor |= int(timestep_value) << 13
            output_descriptor_lines.append(f"{descriptor:08x}")
            for input_q, expected_error_q in zip(value_q, error_q):
                token = _twos(int(input_q), 16)
                token |= _twos(int(expected_error_q), 18) << 16
                output_token_lines.append(f"{token:016x}")

    expected_rows = len(timesteps) * BATCH * 3
    expected_tokens = len(timesteps) * BATCH * (BATCH + BATCH + OUTPUTS)
    if len(rows) != expected_rows or len(token_lines) != expected_tokens:
        raise AssertionError(
            f"fixture size mismatch rows={len(rows)} tokens={len(token_lines)}"
        )

    DESCRIPTOR_MEM.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DESCRIPTOR_MEM.write_text("\n".join(descriptor_lines) + "\n", encoding="ascii")
    TOKEN_MEM.write_text("\n".join(token_lines) + "\n", encoding="ascii")
    OUTPUT_DESCRIPTOR_MEM.write_text(
        "\n".join(output_descriptor_lines) + "\n", encoding="ascii"
    )
    OUTPUT_TOKEN_MEM.write_text("\n".join(output_token_lines) + "\n", encoding="ascii")

    final_hidden_trace_q = np.clip(
        np.rint(reference["hidden_trace"].astype(np.float32) * PROB_ONE),
        0,
        65535,
    ).astype(np.int64)
    final_logits = reference["logits_before"].astype(np.float32)
    screened_final_trace_q = np.clip(
        np.rint(hidden_trace[-1] * PROB_ONE), 0, 65535
    ).astype(np.int64)
    if not np.array_equal(final_hidden_trace_q, screened_final_trace_q):
        raise AssertionError("final hidden trace differs from screened timestep 7")
    w2_trace_lines: list[str] = []
    for sample_index in range(BATCH):
        for hidden_index in range(final_hidden_trace_q.shape[1]):
            trace_record = int(final_hidden_trace_q[sample_index, hidden_index])
            trace_record |= hidden_index << 16
            trace_record |= sample_index << 25
            w2_trace_lines.append(f"{trace_record:08x}")
    w2_error_lines: list[str] = []
    final_error_q = np.zeros((BATCH, OUTPUTS), dtype=np.int64)
    for sample_index, row in enumerate(final_logits):
        _, probability_q = _quantized_row(row)
        final_error_q[sample_index] = probability_q
        final_error_q[sample_index, labels[sample_index]] -= PROB_ONE
        for output_index, error_value in enumerate(final_error_q[sample_index]):
            error_record = _twos(int(error_value), 18)
            error_record |= output_index << 18
            error_record |= sample_index << 22
            w2_error_lines.append(f"{error_record:08x}")
    w2_expected_lines: list[str] = []
    w2_expected_count = 0
    for hidden_index in range(final_hidden_trace_q.shape[1]):
        for output_index in range(OUTPUTS):
            for sample_index in range(BATCH):
                trace_value = int(final_hidden_trace_q[sample_index, hidden_index])
                error_value = int(final_error_q[sample_index, output_index])
                if trace_value == 0 or error_value == 0:
                    continue
                contribution = hidden_index * OUTPUTS + output_index
                contribution |= trace_value << 24
                contribution |= _twos(int(error_value), 18) << 40
                w2_expected_lines.append(f"{contribution:016x}")
                w2_expected_count += 1
    W2_TRACE_MEM.write_text("\n".join(w2_trace_lines) + "\n", encoding="ascii")
    W2_ERROR_MEM.write_text("\n".join(w2_error_lines) + "\n", encoding="ascii")
    W2_EXPECTED_MEM.write_text("\n".join(w2_expected_lines) + "\n", encoding="ascii")
    summary: dict[str, object] = {
        "status": "pass",
        "claim_boundary": (
            "Real FC H400/B16 timestep 0..7 quantized rows and exact integer "
            "probabilities for RTL verification. This fixture does not by itself "
            "prove the surrounding TP matrix engines or parameter adaptation."
        ),
        "input_format": "signed Q6 in 16 bits",
        "probability_format": "unsigned Q0.16 in 17 bits",
        "lut_fractional_bits": LUT_FRAC,
        "lut_entries": 65,
        "row_count": len(rows),
        "token_count": len(token_lines),
        "output_error_row_count": len(output_descriptor_lines),
        "output_error_token_count": len(output_token_lines),
        "w2_scheduler_hidden_trace_count": len(w2_trace_lines),
        "w2_scheduler_error_count": len(w2_error_lines),
        "w2_scheduler_nonzero_contribution_count": w2_expected_count,
        "kind_counts": {
            "student_similarity": len(timesteps) * BATCH,
            "target_negative_distance": len(timesteps) * BATCH,
            "output_logits": len(timesteps) * BATCH,
        },
        "probability_sum_q016_min": row_sum_min,
        "probability_sum_q016_max": row_sum_max,
        "descriptor_format": (
            "bits[4:0]=row_length, [6:5]=kind, [10:7]=timestep, "
            "[14:11]=row_index, upper bits reserved"
        ),
        "token_format": (
            "bits[15:0]=signed input_q6, [32:16]=expected probability_q016, "
            "upper bits reserved"
        ),
        "rows": rows,
        "source": str(SOURCE),
        "descriptor_mem": str(DESCRIPTOR_MEM),
        "token_mem": str(TOKEN_MEM),
        "output_error_descriptor_mem": str(OUTPUT_DESCRIPTOR_MEM),
        "output_error_token_mem": str(OUTPUT_TOKEN_MEM),
        "w2_scheduler_hidden_trace_mem": str(W2_TRACE_MEM),
        "w2_scheduler_error_mem": str(W2_ERROR_MEM),
        "w2_scheduler_expected_mem": str(W2_EXPECTED_MEM),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = build()
    print(
        "FC_TP_B16_SOFTMAX_RTL_FIXTURE_PASS "
        f"rows={summary['row_count']} tokens={summary['token_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
