"""Bounded A2 preparation; print commands unless --run or --worker is given.

Reuse existing model, split, forward, gradient and code-update APIs. The drop
arm discards the uncommitted remainder after EVERY optimizer step, including
within temporal windows/batches. This is not a float-SGD or RTL experiment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload"
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/a2"
SEEDS = {"fc": (38472, 38473, 38474), "srnn": (128474, 128475, 128476),
         "scnn": (38472, 1729, 65537)}
DEVELOPMENT = ROOT / "results/paper_evidence/algorithm/mask_ablation_20260919"


def historical_path(topology, seed):
    if topology == "fc":
        return DEVELOPMENT / (
            "fc_h160_w2_int8_magnitude_rtl_uq6_10_lr2m14_selection/raw/"
            f"seed{seed}_xp2_yp0.json")
    if topology == "srnn":
        return DEVELOPMENT / f"srnn_wout_episode_seed{seed}_shift16.json"
    raise ValueError(topology)


def reproduction_error(topology, seed, result):
    """Compare measured on-arm outputs, not a new baseline or pass/fail gate."""
    reference_path = historical_path(topology, seed)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    collection, mask, optimizer = (
        ("masks", "w2", "int8_magnitude_residual_6.10352e-05") if topology == "fc"
        else ("optimizers", "wout", "direct_response"))

    def select(report):
        rows = [row for row in report[collection]
                if row["trainable_mask"] == mask and row["optimizer"] == optimizer]
        if len(rows) != 1:
            raise ValueError("expected one final-profile development row")
        return rows[0]

    expected, actual = select(reference), select(result)
    differences, missing, accuracy_errors, prediction_errors = [], [], [], {}
    compared = 0

    def compare(old, new, name):
        nonlocal compared
        if isinstance(old, dict):
            if not isinstance(new, dict):
                missing.append(name)
                return
            for key, value in old.items():
                if key == "adaptation_seconds_wall":
                    continue
                path = f"{name}.{key}" if name else key
                if key not in new:
                    missing.append(path)
                else:
                    compare(value, new[key], path)
        elif isinstance(old, list):
            if not isinstance(new, list) or len(new) != len(old):
                missing.append(name + ".length")
                return
            if name.endswith(("predictions", "labels")):
                mismatches = sum(a != b for a, b in zip(old, new))
                compared += len(old)
                if name.endswith("predictions"):
                    prediction_errors[name] = mismatches
                if mismatches:
                    differences.append(dict(field=name, mismatched_entries=mismatches))
            else:
                for index, (a, b) in enumerate(zip(old, new)):
                    compare(a, b, f"{name}[{index}]")
        else:
            compared += 1
            numeric = (isinstance(old, (int, float)) and not isinstance(old, bool)
                       and isinstance(new, (int, float)) and not isinstance(new, bool))
            error = abs(new - old) if numeric else None
            if numeric and (not math.isfinite(new) or not math.isfinite(old)):
                missing.append(name + ".nonfinite")
                return
            if name.endswith("accuracy_percent") or name.endswith(".accuracy"):
                if error is not None:
                    accuracy_errors.append(error * (100 if name.endswith(".accuracy") else 1))
            if old != new:
                differences.append(dict(field=name, reference=old, current=new,
                                        absolute_error=error))

    compare(expected, actual, "row")
    identity = ("episode_seed", "query_split", "support_samples", "query_samples")
    if topology == "srnn":
        compare(reference["protocol"], result["protocol"], "protocol")
    else:
        compare({key: reference[key] for key in identity}, result, "protocol")
    return dict(
        reference=str(reference_path.relative_to(ROOT)),
        status="incomplete" if missing else ("differences" if differences else "exact_match"),
        compared_scalar_fields=compared, missing_fields=missing,
        max_accuracy_error_points=max(accuracy_errors, default=None),
        prediction_mismatches=prediction_errors or None,
        differing_fields=differences,
        scope="Serialized metrics/counters/parameter deltas only; no historical weight arrays or full logits saved. SRNN historical rows have no per-sample predictions or split indices.",
        excluded_fields=["adaptation_seconds_wall"],
    )


def check_input_paths(args):
    """Check only named inputs before importing torch; never discover alternatives."""
    required = [args.traces_root] if args.topology in ("fc", "scnn") else []
    if args.topology == "fc":
        required += [args.data_root / "NMNIST/Train" / str(label) for label in range(10)]
    elif args.topology == "srnn":
        required += [args.handoff / relative for relative in (
            "tp_exp/configs/B64_FORMAL_SEED_A.json",
            "tp_exp/runs/formal_m2/checkpoints/best_checkpoint.pt",
            "tp_exp/python/architecture_search.py", "tp_exp/python/m1_m2_train.py",
            "tp_exp/python/multilayer_tp_train.py")]
        required += [args.data_root / "SHD" / name for name in ("shd_train.h5", "shd_test.h5")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing declared inputs: " + ", ".join(missing))
    if args.topology == "srnn":
        config = json.loads((args.handoff / "tp_exp/configs/B64_FORMAL_SEED_A.json").read_text(encoding="utf-8"))
        index = args.handoff / config["validation_indices_relative"]
        if not index.is_file():
            raise FileNotFoundError(index)


class Counters:
    def __init__(self):
        self.values = dict(update_calls=0, nonzero_gradients=0,
                           nonzero_increments=0, code_changes=0,
                           code_lsb_steps=0, saturations=0,
                           generated_nonzero_residuals=0,
                           retained_residual_changes=0,
                           discarded_nonzero_residuals=0)

    def add(self, **values):
        for key, value in values.items():
            self.values[key] += int(value)


@contextmanager
def update_adapter(topology, arm, counters):
    """Process-local patch only; restore the original API even on exceptions."""
    import numpy as np
    import torch

    drop = arm == "drop"
    if arm not in ("on", "drop"):
        raise ValueError(arm)

    if topology == "fc":
        module = importlib.import_module("scripts.run_fc_tp_h160_mask_ablation")
        original = module.Int8MagnitudeResidualOptimizer

        class ObservedOptimizer(original):
            @torch.no_grad()
            def step(self, raw_gradient):
                before_codes = [x.clone() for x in self.codes]
                before_residuals = [x.clone() for x in self.residuals]
                # Diagnostics use the same integer ties-to-even scaler.
                product = raw_gradient.to(torch.int64) * self.raw_gradient_multiplier
                magnitude = product.abs()
                quotient = magnitude >> self.raw_gradient_shift
                remainder = magnitude & ((1 << self.raw_gradient_shift) - 1)
                half = 1 << (self.raw_gradient_shift - 1)
                rounded = quotient + ((remainder > half) | (
                    (remainder == half) & ((quotient & 1) != 0))).to(torch.int64)
                delta = torch.where(product < 0, rounded, -rounded)
                super().step(raw_gradient)
                for parameter, code, residual, old_code, old_residual in zip(
                    self.parameters, self.codes, self.residuals,
                    before_codes, before_residuals,
                ):
                    if parameter.grad is None:
                        continue
                    total = old_residual.to(torch.int64) + delta
                    code_delta = torch.sign(total) * torch.div(
                        total.abs(), 1 << self.residual_fractional_bits,
                        rounding_mode="floor")
                    unclamped = old_code.to(torch.int64) + code_delta
                    generated = torch.count_nonzero(residual).item()
                    if drop:
                        residual.zero_()
                    counters.add(
                        update_calls=1,
                        nonzero_gradients=torch.count_nonzero(raw_gradient).item(),
                        nonzero_increments=torch.count_nonzero(delta).item(),
                        code_changes=torch.count_nonzero(code != old_code).item(),
                        code_lsb_steps=(code.to(torch.int64) - old_code).abs().sum().item(),
                        saturations=torch.count_nonzero(
                            (unclamped < self.code_min) | (unclamped > self.code_max)).item(),
                        generated_nonzero_residuals=generated,
                        retained_residual_changes=torch.count_nonzero(residual != old_residual).item(),
                        discarded_nonzero_residuals=generated if drop else 0,
                    )

        with patch.object(module, "Int8MagnitudeResidualOptimizer", ObservedOptimizer):
            yield module
    elif topology == "srnn":
        module = importlib.import_module("compare_srnn_shd_shifted_optimizers")
        original = module.direct_update_tensor

        def observed(code, residual, gradient, increment, threshold):
            updated, remainder, stats = original(code, residual, gradient, increment, threshold)
            generated = np.count_nonzero(remainder)
            retained = np.zeros_like(remainder) if drop else remainder
            counters.add(
                update_calls=1, nonzero_gradients=np.count_nonzero(gradient),
                nonzero_increments=np.count_nonzero(gradient),
                code_changes=np.count_nonzero(updated != code),
                code_lsb_steps=np.abs(updated - code).sum(),
                saturations=stats["saturations"],
                generated_nonzero_residuals=generated,
                retained_residual_changes=np.count_nonzero(retained != residual),
                discarded_nonzero_residuals=generated if drop else 0,
            )
            return updated, retained, stats

        with patch.object(module, "direct_update_tensor", observed):
            yield module
    elif topology == "scnn":
        module = importlib.import_module("scripts.run_scnn_tp_selected_layer_shift_adaptation")
        original = module.apply_fixed_sgd_residual

        def observed(parameter, gradient, residual, **kwargs):
            old = residual.clone()
            code_scale = 1 << kwargs["weight_fractional_bits"]
            residual_scale = 1 << kwargs["residual_fractional_bits"]
            code = torch.round(parameter.detach().to(torch.float64) * code_scale).to(torch.int64)
            delta = torch.round(-gradient.to(torch.float64) * kwargs["learning_rate"]
                                * float(code_scale * residual_scale)).to(torch.int64)
            total = old.to(torch.int64) + delta
            code_delta = torch.sign(total) * torch.div(total.abs(), residual_scale,
                                                       rounding_mode="floor")
            result = original(parameter, gradient, residual, **kwargs)
            generated = torch.count_nonzero(residual).item()
            if drop:
                residual.zero_()
            counters.add(
                update_calls=1, nonzero_gradients=torch.count_nonzero(gradient).item(),
                nonzero_increments=torch.count_nonzero(delta).item(),
                code_changes=result[0], code_lsb_steps=result[1],
                saturations=torch.count_nonzero(
                    (code + code_delta < -32768) | (code + code_delta > 32767)).item(),
                generated_nonzero_residuals=generated,
                retained_residual_changes=torch.count_nonzero(residual != old).item(),
                discarded_nonzero_residuals=generated if drop else 0,
            )
            return result

        with patch.object(module, "apply_fixed_sgd_residual", observed):
            yield module
    else:
        raise ValueError(topology)


def fc_episode(args, module):
    import torch
    sys.path.insert(0, str(args.traces_root))
    import tonic

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # Do not instantiate the official-test dataset in this development run.
    with patch.object(tonic.datasets.NMNIST, "download", side_effect=RuntimeError(
            "External dataset is read-only; automatic download/extraction is disabled")):
        dataset = tonic.datasets.NMNIST(save_to=str(args.data_root), train=True)
        support, query = module.class_balanced_episode_indices(
            dataset.targets, support_per_class=16, query_per_class=10, seed=args.seed)
        batches = [module.load_dataset_batches(
            traces_root=args.traces_root, data_root=args.data_root, train=True,
            indices=indices, batch_size=16) for indices in (support, query)]
    candidate = module.Candidate("int8_magnitude_residual_6.10352e-05",
                                 "int8_magnitude_residual", 2.0 ** -14, 0.9)
    result = module.adapt_mask(
        candidate, "w2", traces_root=args.traces_root, weights_path=module.DEFAULT_WEIGHTS,
        adapt_batches=batches[0], test_batches=batches[1], baseline_clean={},
        baseline_shifted={}, shift_dx=2, shift_dy=0, device=torch.device(args.device),
        vote_increment=1, vote_threshold=1, source_trace_bits=16,
        source_trace_format="rtl_uq6_10")
    return dict(status="pass", episode_seed=args.seed, query_split="train",
                device=args.device, network="2312-160-10",
                shift=dict(dx_pixels=2, dy_pixels=0, zero_fill=True),
                source_trace_bits=16, source_trace_format="rtl_uq6_10",
                support_indices=support, query_indices=query, support_query_overlap=0,
                support_samples=len(support), query_samples=len(query), masks=[result])


def legacy_episode(args, output):
    if args.topology == "srnn":
        module = importlib.import_module("compare_srnn_shd_shifted_optimizers")
        argv = ["--handoff", str(args.handoff), "--dataset-root", str(args.data_root),
                "--episode-seed", str(args.seed), "--support-size", "128",
                "--shift-channels", "16", "--dataset-partition", "validation",
                "--trainable-masks", "wout", "--optimizers", "direct_response"]
    else:
        module = importlib.import_module("scripts.run_scnn_tp_subject_pair_ablation")
        # Stream hex words through the existing reader, not read_text/splitlines.
        source = importlib.import_module("scripts.run_scnn_tp_selected_layer_shift_adaptation")
        argv = ["--traces-root", str(args.traces_root),
                "--checkpoint", str(PAYLOAD / "checkpoints/parameters.pt"),
                "--inputs", str(PAYLOAD / "data/test_264x20_inputs_q8_8.hex"),
                "--labels", str(PAYLOAD / "data/test_264_labels.hex"),
                "--metadata", str(PAYLOAD / "data/test_264x20_metadata.json"),
                "--partition", "development", "--subject-pairs", "24:25",
                "--masks", "classifier", "--conv-thresholds", "1",
                "--classifier-thresholds", "1", "--windows", "4",
                "--shift-x", "0", "--shift-y", "0", "--temporal-jitter", "1",
                "--corruption-seed", str(args.seed), "--adapt-batch-size", "4",
                "--eval-batch-size", "4", "--full-adaptation-batches-only",
                "--support-domain", "shifted", "--weight-fractional-bits", "15",
                "--update-rule", "fixed_sgd", "--optimizer-lr", str(5 * 2.0 ** -22),
                "--fixed-sgd-residual-bits", "15"]
    argv += ["--device", args.device, "--output", str(output)]
    with patch.object(sys, "argv", [module.__file__, *argv]):
        if args.topology == "scnn":
            with patch.object(module, "read_hex_words", source.read_signed_q8_8):
                module.main()
        else:
            module.main()
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=tuple(SEEDS), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--traces-root", type=Path, default=PAYLOAD / "traces_propagation-main")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="default: CPU for historical SRNN development; CUDA otherwise")
    parser.add_argument("--run", action="store_true", help="run six jobs sequentially")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("on", "drop"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.device = args.device or ("cpu" if args.topology == "srnn" else "cuda")
    if args.topology in ("fc", "srnn") and args.data_root is None:
        parser.error("FC/SRNN require the existing --data-root; no download is requested")
    if args.topology == "srnn" and args.handoff is None:
        parser.error("SRNN requires the existing --handoff containing formal_m2")
    if args.worker and (args.seed not in SEEDS[args.topology] or args.arm is None):
        parser.error("worker requires a pre-existing development seed and an arm")
    if not args.worker and (args.seed is not None or args.arm is not None):
        parser.error("seed/arm are worker-only; the prepared experiment has three seeds")
    return args


def commands(args):
    for seed in SEEDS[args.topology]:
        for arm in ("on", "drop"):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                       "--topology", args.topology, "--seed", str(seed), "--arm", arm,
                       "--device", args.device, "--output-dir", str(args.output_dir),
                       "--traces-root", str(args.traces_root)]
            for flag, path in (("--data-root", args.data_root), ("--handoff", args.handoff)):
                if path is not None:
                    command += [flag, str(path)]
            yield command


def main(argv=None):
    args = parse_args(argv)
    if not args.worker:
        for command in commands(args):
            print(subprocess.list2cmdline(command), flush=True)
            if args.run:
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
                subprocess.run(command, check=True, cwd=ROOT, env=env)
        return 0

    output = args.output_dir.resolve() / args.topology / f"seed{args.seed}_residual_{args.arm}.json"
    if not output.is_relative_to(ROOT):
        raise ValueError("supplement outputs must stay inside the project")
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new output directory: {output}")
    check_input_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    counters = Counters()
    started = time.perf_counter()
    with update_adapter(args.topology, args.arm, counters) as module:
        result = fc_episode(args, module) if args.topology == "fc" else legacy_episode(args, output)
    result["supplement_a2"] = dict(
        topology=args.topology, seed=args.seed, residual_arm=args.arm,
        status="completed", elapsed_seconds_wall=time.perf_counter() - started,
        counters=counters.values,
        counter_boundary="Per-update address events, not distinct final addresses or physical RTL writes",
        legacy_residual_counters="Original optimizer diagnostics count generated residuals before drop; use supplement counters for retained state",
        claim_boundary="Hybrid software residual mechanism ablation, not full bit-exact execution or A3 float SGD",
        input_paths=dict(traces_root=str(args.traces_root), data_root=str(args.data_root),
                         handoff=str(args.handoff)),
    )
    if args.topology in ("fc", "srnn") and args.arm == "on":
        result["supplement_a2"]["historical_on_reproduction"] = reproduction_error(
            args.topology, args.seed, result)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SUPPLEMENT_A2_COMPLETE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
