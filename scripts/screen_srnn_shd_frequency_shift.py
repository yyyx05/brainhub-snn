#!/usr/bin/env python3
"""Select a held-out SHD frequency-channel shift severity on validation query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]


class FrequencyShiftDataset(Dataset):
    def __init__(self, source: Dataset, channels: int):
        self.source = source
        self.channels = channels

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        frames, label = self.source[index]
        shifted = np.zeros_like(frames)
        if self.channels > 0:
            shifted[..., self.channels :] = frames[..., : -self.channels]
        elif self.channels < 0:
            shifted[..., : self.channels] = frames[..., -self.channels :]
        else:
            shifted[...] = frames
        return shifted, label


def compact(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": metric["samples"],
        "correct": metric["correct"],
        "accuracy": metric["accuracy"],
        "evaluation_ce": metric["evaluation_ce"],
        "class_coverage": metric["class_coverage"],
        "per_class": metric["per_class"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--shift", type=int, action="append", dest="shifts")
    args = parser.parse_args()
    shifts = args.shifts or [4, 8, 12, 16, 24, 32]

    python_dir = args.handoff / "tp_exp" / "python"
    sys.path.insert(0, str(python_dir))
    import m1_m2_train as base
    import multilayer_tp_train as trainer

    config = json.loads(
        (args.handoff / "tp_exp" / "configs" / "B64_FORMAL_SEED_A.json").read_text(
            encoding="utf-8"
        )
    )
    config["dataset_root_relative"] = str(args.dataset_root.resolve())
    train_dataset, _, collate_fn = trainer.datasets(config)
    validation_path = args.handoff / config["validation_indices_relative"]
    validation_indices = base.load_indices(validation_path)
    order = np.asarray(validation_indices, dtype=np.int64)
    np.random.default_rng(int(config["seed"]) + 90_001).shuffle(order)
    if args.support_size <= 0 or args.support_size >= len(order):
        raise ValueError("support-size must leave a nonempty query split")
    support_indices = order[: args.support_size].tolist()
    query_indices = order[args.support_size :].tolist()

    model, _, optimizers, schedulers = trainer.prepare(config)
    checkpoint = (
        args.handoff / "tp_exp" / "runs" / "formal_m2" / "checkpoints" /
        "best_checkpoint.pt"
    )
    base.restore_checkpoint(checkpoint, model, optimizers, schedulers)
    clean = base.evaluate(
        model, train_dataset, collate_fn, query_indices, 64, int(config["output_size"])
    )
    shifted_rows: list[dict[str, Any]] = []
    for channels in shifts:
        shifted_dataset = FrequencyShiftDataset(train_dataset, channels)
        metric = base.evaluate(
            model,
            shifted_dataset,
            collate_fn,
            query_indices,
            64,
            int(config["output_size"]),
        )
        shifted_rows.append({
            "channels": channels,
            **compact(metric),
            "accuracy_drop_from_clean": clean["accuracy"] - metric["accuracy"],
        })

    eligible = [
        row for row in shifted_rows
        if 0.10 <= row["accuracy_drop_from_clean"] <= 0.35
        and row["class_coverage"] >= 15
    ]
    selected = min(
        eligible or shifted_rows,
        key=lambda row: (
            abs(row["accuracy_drop_from_clean"] - 0.20),
            -row["class_coverage"],
            row["channels"],
        ),
    )
    result = {
        "status": "pass",
        "protocol": "held-out validation frequency-channel zero-fill shift",
        "seed": int(config["seed"]) + 90_001,
        "support_size": len(support_indices),
        "query_size": len(query_indices),
        "support_indices": support_indices,
        "query_indices": query_indices,
        "clean_query": compact(clean),
        "shift_candidates": shifted_rows,
        "selected_shift_channels": selected["channels"],
        "selection_rule": (
            "Closest to a 20 percentage-point clean-to-shifted query drop among "
            "candidates dropping 10..35 points and retaining at least 15 predicted "
            "classes. Validation only; official test is excluded."
        ),
    }
    output = (
        ROOT / "results" / "learning_reference" /
        "srnn_shd_frequency_shift_severity_screen.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(
        "SRNN_SHD_FREQUENCY_SHIFT_SCREEN_PASS "
        f"clean={clean['correct']}/{clean['samples']} "
        f"selected={selected['channels']} "
        f"shifted={selected['correct']}/{selected['samples']} "
        f"drop={selected['accuracy_drop_from_clean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
