"""Compare FP32 and M2 Q2.14 effective weights with identical FP32 inference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path("D:/Desktop/SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_weight_only_a1"
WEIGHT_KEYS = ("network.1.fc.weight", "network.2.fc.weight")


def q14_effective(weight):
    rounded = torch.round(weight * 16384.0)
    code = rounded.clamp(-32768, 32767)
    effective = code / 16384.0
    return effective, {
        "elements": weight.numel(),
        "clipped_elements": int(torch.count_nonzero(rounded != code).item()),
        "changed_elements": int(torch.count_nonzero(weight != effective).item()),
        "max_absolute_error": float((weight - effective).abs().max().item()),
        "mean_absolute_error": float((weight - effective).abs().mean().item()),
    }


@torch.no_grad()
def apply_arm(model, arm):
    if arm not in ("fp32", "m2_q2_14"):
        raise ValueError(arm)
    parameters = dict(model.named_parameters())
    conversions = {}
    for name in WEIGHT_KEYS:
        if name not in parameters:
            raise ValueError(f"missing M2 parameter: {name}")
        effective, conversions[name] = q14_effective(parameters[name])
        if arm == "m2_q2_14":
            parameters[name].copy_(effective)
    return conversions


@torch.no_grad()
def evaluate(model, dataset, collate_fn, base, indices, device, arm):
    model.eval()
    batches = [indices[start:start + 64] for start in range(0, len(indices), 64)]
    predictions, labels_all, timestep_counts = [], [], []
    started = time.perf_counter()
    for batch_index, (data, labels) in enumerate(base.indexed_loader(dataset, collate_fn, batches)):
        data = data.to(device)
        model.reset_potential()
        output = None
        for step in range(data.shape[1]):
            output = model.inference(data[:, step, :].float())
        if output is None or not torch.isfinite(output).all():
            raise ValueError("empty or nonfinite output")
        predictions.extend(output.argmax(1).cpu().tolist())
        labels_all.extend(labels.tolist())
        timestep_counts.append(int(data.shape[1]))
        if batch_index % 8 == 0 or batch_index + 1 == len(batches):
            print(f"SRNN_WEIGHT_A1 {arm} batch={batch_index + 1}/{len(batches)} "
                  f"samples={len(labels_all)}", flush=True)
    if len(labels_all) != len(indices):
        raise ValueError("evaluation did not cover all requested indices")
    correct = sum(a == b for a, b in zip(predictions, labels_all))
    return dict(samples=len(indices), correct=correct,
                accuracy_percent=100.0 * correct / len(indices),
                predictions=predictions, labels=labels_all,
                padded_timesteps_per_batch=timestep_counts,
                elapsed_seconds_wall=time.perf_counter() - started)


def compare_arms(fp, quant):
    if fp["labels"] != quant["labels"] or fp["samples"] != quant["samples"]:
        raise ValueError("arms do not evaluate the same ordered samples")
    if fp["padded_timesteps_per_batch"] != quant["padded_timesteps_per_batch"]:
        raise ValueError("arms have different temporal padding")
    rows = list(zip(fp["predictions"], quant["predictions"], fp["labels"]))
    return dict(
        accuracy_change_points=quant["accuracy_percent"] - fp["accuracy_percent"],
        predictions_changed=sum(a != b for a, b, _ in rows),
        correct_to_wrong=sum(a == y and b != y for a, b, y in rows),
        wrong_to_correct=sum(a != y and b == y for a, b, y in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=HANDOFF)
    parser.add_argument("--data-root", type=Path, default=Path("D:/datasets"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("Plan: same checkpoint, all SHD official-test indices, B64, FP32 "
              "versus M2 Wh1/Wout Q2.14; no adaptation or parameter selection.")
        return 0
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("choose a new output directory inside the project")
    for path in (args.data_root / "SHD/shd_train.h5", args.data_root / "SHD/shd_test.h5"):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0, str(args.handoff / "tp_exp/python"))
    import m1_m2_train as base
    import multilayer_tp_train as trainer
    from compare_srnn_shd_shifted_optimizers import fresh_model

    config = json.loads((args.handoff / "tp_exp/configs/B64_FORMAL_SEED_A.json").read_text(encoding="utf-8"))
    config["dataset_root_relative"] = str(args.data_root.resolve())
    _, dataset, collate_fn = trainer.datasets(config)
    indices = list(range(len(dataset)))
    if len(indices) != 2264:
        raise ValueError(f"unexpected official-test size: {len(indices)}")
    output.mkdir(parents=True)
    arms = {}
    conversions = {}
    for arm in ("fp32", "m2_q2_14"):
        model, optimizers = fresh_model(args.handoff, config, trainer, base)
        del optimizers
        conversions[arm] = apply_arm(model, arm)
        model.to(device)
        arms[arm] = evaluate(model, dataset, collate_fn, base, indices, device, arm)
        (output / f"{arm}.json").write_text(json.dumps(arms[arm], indent=2) + "\n", encoding="utf-8")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(
        status="completed", device=str(device),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
        checkpoint=str(args.handoff / "tp_exp/runs/formal_m2/checkpoints/best_checkpoint.pt"),
        config=config, dataset="SHD", split="official_test", sample_indices=indices,
        batch_size=64, arms=arms, quantization=conversions,
        comparison=compare_arms(arms["fp32"], arms["m2_q2_14"]),
        scope="Weight-only ablation for M2 Wh1/Wout. Backbone, neuron states, traces "
              "and computation remain FP32. No online adaptation, full-integer or RTL accuracy claim.",
    )
    (output / "clean_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparison": result["comparison"],
                      "correct": {k: v["correct"] for k, v in arms.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
