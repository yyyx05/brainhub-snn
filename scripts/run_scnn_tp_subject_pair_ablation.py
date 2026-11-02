#!/usr/bin/env python3
"""Run subject-disjoint DVS-Gesture TP mask ablations on full recordings.

The pretrained checkpoint is never modified in place.  Each candidate starts
from the same effective Q8.8 checkpoint, adapts on one shifted support subject,
and is evaluated on a disjoint query subject in both clean and shifted views.
Development and report subject pairs are deliberately separate so the report
partition can remain unread until an operating point has been selected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scnn_tp_selected_layer_shift_adaptation import (  # noqa: E402
    CHANNELS,
    CLASSES,
    HEIGHT,
    Q8_8_SCALE,
    TIMESTEPS,
    TRAINABLE_MASKS,
    WEIGHT_CODE_BITS,
    WIDTH,
    EventCorruption,
    adapt_selected_layers,
    build_model,
    compact,
    evaluate,
)


SUBJECT_PATTERN = re.compile(r"(?:^|/)user(\d+)_[^/]+/\d+\.npy$")
DEFAULT_DEV_PAIRS = ((24, 25), (25, 24), (26, 27), (27, 26))
DEFAULT_REPORT_PAIRS = ((28, 29), (29, 28))


def parse_pairs(values: list[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in values:
        fields = value.split(":", maxsplit=1)
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                f"invalid subject pair {value!r}; expected SUPPORT:QUERY"
            )
        support, query = (int(field) for field in fields)
        if support == query:
            raise argparse.ArgumentTypeError("support and query subjects must differ")
        pairs.append((support, query))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("development", "report"),
        default="development",
    )
    parser.add_argument(
        "--subject-pairs",
        nargs="+",
        help="Explicit SUPPORT:QUERY pairs; defaults depend on --partition.",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        choices=tuple(TRAINABLE_MASKS),
        default=["classifier"],
    )
    parser.add_argument("--thresholds", nargs="+", type=int, default=[32, 40])
    parser.add_argument(
        "--conv-thresholds",
        nargs="+",
        type=int,
        help=(
            "Conv vote thresholds. When omitted, --thresholds supplies both "
            "Conv and classifier thresholds with legacy paired semantics."
        ),
    )
    parser.add_argument(
        "--classifier-thresholds",
        nargs="+",
        type=int,
        help=(
            "Classifier vote thresholds. Supplying either topology-specific "
            "list enables an independent Conv/classifier threshold grid."
        ),
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[2])
    parser.add_argument("--shift-x", type=int, default=8)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--event-drop-probability", type=float, default=0.0)
    parser.add_argument("--polarity-flip-probability", type=float, default=0.0)
    parser.add_argument("--temporal-jitter", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=38472)
    parser.add_argument(
        "--adapt-batch-size",
        type=int,
        default=4,
        help=(
            "Adaptation samples per batch. Convolutional TP masks require at "
            "least two class-mixed samples; classifier-only masks may use one."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--classifier-row-mask",
        choices=("all", "target", "target_competitor"),
        default="all",
    )
    parser.add_argument(
        "--classifier-sample-gate",
        choices=("all", "misclassified"),
        default="all",
    )
    parser.add_argument(
        "--commit-scope", choices=("window", "support"), default="window"
    )
    parser.add_argument(
        "--support-domain", choices=("shifted", "mixed"), default="shifted"
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
        "--fixed-sgd-residual-bits", type=int, default=15
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
        "--force-class-mixed-batches",
        action="store_true",
        help=(
            "use the TP class-mixed batch schedule for every mask so matched "
            "classifier and convolutional candidates consume identical support"
        ),
    )
    parser.add_argument(
        "--weight-fractional-bits",
        type=int,
        default=8,
        help="fractional bits in the signed 16-bit trainable-weight code",
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=38472)
    return parser.parse_args()


def read_hex_words(path: Path, expected: int) -> np.ndarray:
    raw = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()),
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
        raise ValueError(f"{path}: invalid labels")
    return labels


def read_subjects(metadata_path: Path) -> tuple[np.ndarray, int]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("samples", [])
    subjects: list[int] = []
    for record in records:
        member = str(record["archive_member"]).replace("\\", "/")
        match = SUBJECT_PATTERN.search(member)
        if match is None:
            raise ValueError(f"cannot parse subject from archive member {member!r}")
        subjects.append(int(match.group(1)))
    if not subjects:
        raise ValueError(f"{metadata_path}: no sample records")
    return np.asarray(subjects, dtype=np.int64), len(records)


def class_counts(labels: np.ndarray, indices: list[int]) -> list[int]:
    return np.bincount(labels[indices], minlength=CLASSES).astype(int).tolist()


def require_complete_subject(
    labels: np.ndarray, subjects: np.ndarray, subject: int
) -> list[int]:
    indices = np.flatnonzero(subjects == subject).astype(int).tolist()
    counts = class_counts(labels, indices)
    if any(count == 0 for count in counts):
        raise ValueError(f"subject {subject} has incomplete class counts {counts}")
    return indices


def write_incremental(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def summarize_candidates(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        for candidate in episode.get("candidates", []):
            key = (
                str(candidate["mask"]),
                int(candidate["conv_vote_threshold"]),
                int(candidate["classifier_vote_threshold"]),
                int(candidate["window_timesteps"]),
            )
            grouped.setdefault(key, []).append(candidate)

    rows: list[dict[str, Any]] = []
    for (
        mask,
        conv_threshold,
        classifier_threshold,
        window,
    ), values in sorted(grouped.items()):
        shifted = np.asarray(
            [float(value["shifted_gain_points"]) for value in values]
        )
        retention = np.asarray(
            [float(value["clean_retention_points"]) for value in values]
        )
        writes = np.asarray([int(value["code_writes"]) for value in values])
        realized = np.asarray(
            [bool(value["update"]["mask_realized"]) for value in values],
            dtype=bool,
        )
        feasible = (shifted > 0.0) & (retention >= -1.0) & realized
        rows.append(
            {
                "mask": mask,
                "conv_vote_threshold": conv_threshold,
                "classifier_vote_threshold": classifier_threshold,
                "window_timesteps": window,
                "adapt_batch_size": int(values[0]["adapt_batch_size"]),
                "classifier_row_mask": str(
                    values[0]["update"]["classifier_row_mask"]
                ),
                "classifier_sample_gate": str(
                    values[0]["update"]["classifier_sample_gate"]
                ),
                "commit_scope": str(values[0]["update"]["commit_scope"]),
                "support_domain": str(values[0]["update"]["support_domain"]),
                "update_rule": str(values[0]["update"]["update_rule"]),
                "episodes": len(values),
                "shifted_gain_mean": float(shifted.mean()),
                "shifted_gain_std": float(shifted.std(ddof=0)),
                "clean_retention_mean": float(retention.mean()),
                "clean_retention_std": float(retention.std(ddof=0)),
                "code_writes_mean": float(writes.mean()),
                "mask_realized_episodes": int(realized.sum()),
                "all_episodes_mask_realized": bool(realized.all()),
                "feasible_episodes": int(feasible.sum()),
                "all_episodes_feasible": bool(feasible.all()),
                "mutable_parameters": int(values[0]["mutable_parameters"]),
                "mutable_state_bytes": int(values[0]["mutable_state_bytes"]),
            }
        )
    return rows


def candidate_threshold_pairs(
    mask: str,
    *,
    legacy_thresholds: list[int],
    conv_thresholds: list[int] | None,
    classifier_thresholds: list[int] | None,
) -> list[tuple[int, int]]:
    """Return only threshold combinations relevant to the selected mask.

    With neither topology-specific list supplied, preserve the historical
    paired sweep exactly: ``threshold`` is applied to both tensor classes.
    Once either list is supplied, combined masks use the Cartesian product,
    while single-class masks vary only their active threshold.
    """

    if conv_thresholds is None and classifier_thresholds is None:
        return [(threshold, threshold) for threshold in legacy_thresholds]

    conv_values = conv_thresholds or legacy_thresholds
    classifier_values = classifier_thresholds or legacy_thresholds
    conv_indices, classifier_enabled = TRAINABLE_MASKS[mask]
    if not conv_indices and not classifier_enabled:
        return [(conv_values[0], classifier_values[0])]
    if not conv_indices:
        return [(conv_values[0], threshold) for threshold in classifier_values]
    if not classifier_enabled:
        return [(threshold, classifier_values[0]) for threshold in conv_values]
    return [
        (conv_threshold, classifier_threshold)
        for conv_threshold in conv_values
        for classifier_threshold in classifier_values
    ]


def main() -> int:
    args = parse_args()
    if args.adapt_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
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
    conv_mask_requested = any(TRAINABLE_MASKS[mask][0] for mask in args.masks)
    if conv_mask_requested and args.adapt_batch_size < 2:
        raise ValueError(
            "convolutional TP masks require --adapt-batch-size >= 2; "
            "batch size 1 degenerates the pairwise trace loss to zero"
        )
    threshold_values = [
        *args.thresholds,
        *(args.conv_thresholds or []),
        *(args.classifier_thresholds or []),
    ]
    if any(value <= 0 for value in (*threshold_values, *args.windows)):
        raise ValueError("thresholds and windows must be positive")
    if args.partition == "report" and args.subject_pairs is None:
        pairs = list(DEFAULT_REPORT_PAIRS)
    elif args.subject_pairs is None:
        pairs = list(DEFAULT_DEV_PAIRS)
    else:
        pairs = parse_pairs(args.subject_pairs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    subjects, sample_count = read_subjects(args.metadata)
    labels_np = read_labels(args.labels, sample_count)
    if subjects.size != labels_np.size:
        raise ValueError("metadata and label sample counts differ")
    expected_words = sample_count * TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    words = read_hex_words(args.inputs, expected_words)
    samples = torch.from_numpy(
        words.reshape(sample_count, TIMESTEPS, CHANNELS, HEIGHT, WIDTH)
    ).to(device)
    labels = torch.from_numpy(labels_np).to(device)

    result: dict[str, Any] = {
        "status": "running",
        "claim_boundary": (
            "Subject-disjoint mask selection from the full official DVS-Gesture "
            "test recordings. Development pairs may select an operating point; "
            "report pairs must remain unread until that point is locked. The "
            "configured spatial and event-domain corruptions are bounded "
            "adaptation stresses, not a natural-domain benchmark."
        ),
        "partition": args.partition,
        "configuration": {
            "sample_count": sample_count,
            "subject_pairs": [list(pair) for pair in pairs],
            "masks": args.masks,
            "thresholds": args.thresholds,
            "conv_thresholds": args.conv_thresholds,
            "classifier_thresholds": args.classifier_thresholds,
            "windows": args.windows,
            "shift_x": args.shift_x,
            "shift_y": args.shift_y,
            **corruption.as_dict(),
            "adapt_batch_size": args.adapt_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_format": (
                f"effective signed Q{WEIGHT_CODE_BITS - 1 - args.weight_fractional_bits}."
                f"{args.weight_fractional_bits}"
            ),
            "weight_code_bits": WEIGHT_CODE_BITS,
            "weight_fractional_bits": args.weight_fractional_bits,
            "direct_update_semantics": (
                "signed residual voting, one configured weight-code LSB per "
                "committed write"
            ),
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
            "force_class_mixed_batches": args.force_class_mixed_batches,
        },
        "episodes": [],
        "summary": [],
    }
    write_incremental(args.output, result)

    model_batch_size = max(args.adapt_batch_size, args.eval_batch_size)
    for support_subject, query_subject in pairs:
        support_indices = require_complete_subject(
            labels_np, subjects, support_subject
        )
        query_indices = require_complete_subject(labels_np, subjects, query_subject)
        baseline_model = build_model(
            args.traces_root,
            args.checkpoint,
            batch_size=model_batch_size,
            device=device,
            weight_fractional_bits=args.weight_fractional_bits,
        )
        baseline_clean = evaluate(
            baseline_model,
            samples,
            labels,
            query_indices,
            batch_size=args.eval_batch_size,
            dx=0,
            dy=0,
        )
        baseline_shifted = evaluate(
            baseline_model,
            samples,
            labels,
            query_indices,
            batch_size=args.eval_batch_size,
            dx=args.shift_x,
            dy=args.shift_y,
            corruption=corruption,
        )
        episode: dict[str, Any] = {
            "support_subject": support_subject,
            "query_subject": query_subject,
            "support_samples": len(support_indices),
            "query_samples": len(query_indices),
            "support_class_counts": class_counts(labels_np, support_indices),
            "query_class_counts": class_counts(labels_np, query_indices),
            "baseline_clean": compact(baseline_clean),
            "baseline_shifted": compact(baseline_shifted),
            "candidates": [],
        }
        result["episodes"].append(episode)
        write_incremental(args.output, result)
        print(
            "SCNN_SUBJECT_BASELINE"
            f" support={support_subject} query={query_subject}"
            f" clean={baseline_clean['accuracy_percent']:.3f}"
            f" shifted={baseline_shifted['accuracy_percent']:.3f}",
            flush=True,
        )
        if args.baseline_only:
            continue

        for mask in args.masks:
            threshold_pairs = candidate_threshold_pairs(
                mask,
                legacy_thresholds=args.thresholds,
                conv_thresholds=args.conv_thresholds,
                classifier_thresholds=args.classifier_thresholds,
            )
            for conv_threshold, classifier_threshold in threshold_pairs:
                for window in args.windows:
                    model = build_model(
                        args.traces_root,
                        args.checkpoint,
                        batch_size=model_batch_size,
                        device=device,
                        weight_fractional_bits=args.weight_fractional_bits,
                    )
                    update = adapt_selected_layers(
                        model,
                        samples,
                        labels,
                        support_indices,
                        batch_size=args.adapt_batch_size,
                        dx=args.shift_x,
                        dy=args.shift_y,
                        conv_vote_threshold=conv_threshold,
                        classifier_vote_threshold=classifier_threshold,
                        window_timesteps=window,
                        trainable_mask=mask,
                        classifier_row_mask=args.classifier_row_mask,
                        classifier_sample_gate=args.classifier_sample_gate,
                        commit_scope=args.commit_scope,
                        support_domain=args.support_domain,
                        update_rule=args.update_rule,
                        optimizer_lr=args.optimizer_lr,
                        weight_fractional_bits=args.weight_fractional_bits,
                        fixed_sgd_residual_bits=args.fixed_sgd_residual_bits,
                        full_adaptation_batches_only=(
                            args.full_adaptation_batches_only
                        ),
                        force_class_mixed_batches=args.force_class_mixed_batches,
                        corruption=corruption,
                    )
                    clean_after = evaluate(
                        model,
                        samples,
                        labels,
                        query_indices,
                        batch_size=args.eval_batch_size,
                        dx=0,
                        dy=0,
                    )
                    shifted_after = evaluate(
                        model,
                        samples,
                        labels,
                        query_indices,
                        batch_size=args.eval_batch_size,
                        dx=args.shift_x,
                        dy=args.shift_y,
                        corruption=corruption,
                    )
                    candidate = {
                        "mask": mask,
                        "threshold": (
                            conv_threshold
                            if conv_threshold == classifier_threshold
                            else None
                        ),
                        "conv_vote_threshold": conv_threshold,
                        "classifier_vote_threshold": classifier_threshold,
                        "window_timesteps": window,
                        "adapt_batch_size": args.adapt_batch_size,
                        "mutable_parameters": int(update["mutable_parameters"]),
                        "mutable_state_bytes": int(
                            update["q8_8_code_plus_residual_bytes"]
                        ),
                        "code_writes": int(sum(update["code_writes"].values())),
                        "mask_realized": bool(update["mask_realized"]),
                        "update": update,
                        "clean_after": compact(clean_after),
                        "shifted_after": compact(shifted_after),
                        "shifted_gain_points": float(
                            shifted_after["accuracy_percent"]
                            - baseline_shifted["accuracy_percent"]
                        ),
                        "clean_retention_points": float(
                            clean_after["accuracy_percent"]
                            - baseline_clean["accuracy_percent"]
                        ),
                    }
                    episode["candidates"].append(candidate)
                    result["summary"] = summarize_candidates(result["episodes"])
                    write_incremental(args.output, result)
                    print(
                        "SCNN_SUBJECT_CANDIDATE"
                        f" support={support_subject} query={query_subject}"
                        f" mask={mask} conv_threshold={conv_threshold}"
                        f" classifier_threshold={classifier_threshold}"
                        f" window={window}"
                        f" gain={candidate['shifted_gain_points']:.3f}"
                        f" retention={candidate['clean_retention_points']:.3f}"
                        f" writes={candidate['code_writes']}"
                        f" rule={args.update_rule}"
                        f" realized={int(candidate['mask_realized'])}",
                        flush=True,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["summary"] = summarize_candidates(result["episodes"])
    result["status"] = "baseline_pass" if args.baseline_only else "pass"
    write_incremental(args.output, result)
    print(f"SCNN_SUBJECT_PAIR_ABLATION_{result['status'].upper()} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
