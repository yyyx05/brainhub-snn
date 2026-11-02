#!/usr/bin/env python3
"""A3 Wout-only magnitude-SGD control. Default is a read-only dry run.

Pilot: validation seed128474, support128/query688, +16 zero-fill shift,
three preset SGD rates (3e-5, 1e-4, 3e-4) and one direct_response control.
Report: read the pilot decision, then pair both methods for seeds
228474/228475/228476, training-validation support128/official-test query2264.
No report-stage learning-rate option is provided.

Reuse compare_srnn_shd_shifted_optimizers for checkpoint restoration, Q2.14
initialization, forward/trace gradient, timestep detach and evaluation. A
process-local, scoped update-function substitution leaves that file unchanged.
Both arms receive exactly round_even((e.T @ trace / B) * 2**40). SGD preserves
its magnitude; direct_response uses only its sign, with increment31/threshold65.
Neither is BPTT. Forward/state arithmetic remains FP32, not bit-exact RTL.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# Imported formal-package sources must remain read-only, including bytecode.
sys.dont_write_bytecode = True
RATES = (3e-5, 1e-4, 3e-4)
DEV_SEED = 128474
REPORT_SEEDS = (228474, 228475, 228476)
BATCH = 64
SUPPORT = 128
SHIFT = 16
RESIDUAL_BITS = 24
RESIDUAL_SCALE = 1 << RESIDUAL_BITS
CODE_SCALE = 1 << 14
GRADIENT_SCALE = 1 << 40
METHODS = ("direct_response", "magnitude_sgd")
CANDIDATE = dict(wh1_increment=25, wh1_threshold=244,
                 wout_increment=31, wout_threshold=65)
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_magnitude_sgd_a3"
SELECTION_RULE = (
    "Validation only. Exclude zero code-change events, all attempted code changes "
    "clipped by saturation (or all final codes at rails), and clean loss >1 pp. "
    "Maximize final shifted gain; ties: larger clean change, then smaller rate. "
    "Keep negative outcomes. If no candidate qualifies, select no rate, report "
    "why the screen is inconclusive, and request a decision before expansion. "
    "No automatic extra rates or official-query selection."
)


def magnitude_update_tensor(code, residual, gradient_q40, learning_rate):
    """Q2.14 code + signed Q0.24 code-LSB remainder, no master or momentum.

    d = RNE(-lr * (g_q40/2**40) * 2**14 * 2**24)
    s = residual + d; dc = trunc_toward_zero(s / 2**24)
    code' = sat16(code + dc); residual' = s - dc*2**24 (zero if clipped).
    Float64 is used only to quantize the rate product; all carried state and
    crossing arithmetic are integer. Zero gradient preserves both states.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    code = np.asarray(code, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.int64)
    gradient_q40 = np.asarray(gradient_q40)
    if code.shape != residual.shape or code.shape != gradient_q40.shape:
        raise ValueError("code, residual and gradient shapes must match")
    if np.any((code < -32768) | (code > 32767)):
        raise ValueError("code is outside signed16")
    if np.any((residual <= -RESIDUAL_SCALE) | (residual >= RESIDUAL_SCALE)):
        raise ValueError("residual must be a sub-LSB remainder")
    if not np.issubdtype(gradient_q40.dtype, np.integer):
        raise ValueError("gradient must use the existing integer Q40 representation")
    delta_float = np.rint(gradient_q40.astype(np.float64) *
                          (-learning_rate * CODE_SCALE * RESIDUAL_SCALE / GRADIENT_SCALE))
    # Leaves headroom for addition, abs(), and the signed16 candidate code.
    if not np.isfinite(delta_float).all() or np.any(np.abs(delta_float) >= 2**62):
        raise OverflowError("SGD rate product exceeds signed64 intermediate capacity")
    delta = delta_float.astype(np.int64)
    total = residual + delta
    code_delta = np.sign(total) * (np.abs(total) // RESIDUAL_SCALE)
    candidate = code + code_delta
    clipped = (candidate < -32768) | (candidate > 32767)
    updated = np.clip(candidate, -32768, 32767).astype(np.int64)
    remainder = np.where(clipped, 0, total - code_delta * RESIDUAL_SCALE)
    nonzero = gradient_q40 != 0
    return updated, remainder.astype(np.int64), {
        "rmw": int(np.count_nonzero(nonzero)),
        "code_writes": int(np.count_nonzero(updated != code)),
        "residual_writes": int(np.count_nonzero(remainder != residual)),
        "saturations": int(np.count_nonzero(clipped)),
        "zero_skips": int(np.count_nonzero(~nonzero)),
        "proposed_code_changes": int(np.count_nonzero(code_delta)),
        "nonzero_increments": int(np.count_nonzero(delta)),
    }


class UpdateObserver:
    """Small shared diagnostic wrapper; not a second adaptation/model loop."""

    def __init__(self, original, method, rate=None):
        if method not in METHODS:
            raise ValueError(method)
        self.original, self.method, self.rate = original, method, rate
        self.totals = {name: 0 for name in (
            "update_calls", "nonzero_gradients", "zero_gradients", "nonzero_increments",
            "proposed_code_changes", "code_changes", "code_lsb_steps",
            "residual_changes", "saturations", "updates_without_code_changes")}
        self.gradient_abs_max = 0.0
        self.gradient_abs_sum = 0.0

    def __call__(self, code, residual, gradient, increment, threshold):
        if self.method == "direct_response":
            updated, remainder, stats = self.original(
                code, residual, gradient, increment, threshold)
            # Diagnostics only; the original updater remains authoritative.
            votes = residual - np.sign(gradient) * increment
            proposed = int(np.count_nonzero(np.abs(votes) >= threshold))
            nonzero_increments = int(np.count_nonzero(gradient))
        else:
            updated, remainder, stats = magnitude_update_tensor(
                code, residual, gradient, self.rate)
            proposed = stats["proposed_code_changes"]
            nonzero_increments = stats["nonzero_increments"]
        values = dict(
            update_calls=1, nonzero_gradients=stats["rmw"],
            zero_gradients=stats["zero_skips"], nonzero_increments=nonzero_increments,
            proposed_code_changes=proposed, code_changes=stats["code_writes"],
            code_lsb_steps=int(np.abs(updated - code).sum()),
            residual_changes=stats["residual_writes"], saturations=stats["saturations"],
            updates_without_code_changes=int(stats["code_writes"] == 0))
        for key, value in values.items():
            self.totals[key] += value
        magnitude = np.abs(gradient.astype(np.float64)) / GRADIENT_SCALE
        self.gradient_abs_max = max(self.gradient_abs_max, float(magnitude.max()))
        self.gradient_abs_sum += float(magnitude.sum())
        return updated, remainder, stats


@contextmanager
def update_adapter(module, method, rate=None):
    observer = UpdateObserver(module.direct_update_tensor, method, rate)
    with patch.object(module, "direct_update_tensor", observer):
        yield observer


def episode_indices(validation_indices, seed, phase, test_size=2264):
    order = np.asarray(validation_indices, dtype=np.int64).copy()
    if len(order) != 816 or len(set(order.tolist())) != len(order):
        raise ValueError("expected the formal 816 unique validation indices")
    np.random.default_rng(seed).shuffle(order)
    if phase not in ("pilot", "report"):
        raise ValueError(phase)
    if phase == "report" and test_size != 2264:
        raise ValueError("report requires all 2264 official-test samples")
    query = order[SUPPORT:].tolist() if phase == "pilot" else list(range(test_size))
    return order[:SUPPORT].tolist(), query


def update_regime(counters, final_code):
    all_saturated = (
        counters["proposed_code_changes"] > 0 and
        counters["saturations"] == counters["proposed_code_changes"]
    ) or bool(np.all((final_code == -32768) | (final_code == 32767)))
    return {
        "zero_code_updates": counters["code_changes"] == 0,
        "all_saturated": all_saturated,
        "saturation_fraction_of_proposals": (
            counters["saturations"] / counters["proposed_code_changes"]
            if counters["proposed_code_changes"] else 0.0),
        "final_bound_codes": int(np.count_nonzero(
            (final_code == -32768) | (final_code == 32767))),
    }


def select_rate(rows):
    candidates = [row for row in rows if row["method"] == "magnitude_sgd"]
    if len(candidates) != len(RATES) or sorted(r["learning_rate"] for r in candidates) != list(RATES):
        raise ValueError("pilot must contain exactly the three preset SGD rates")
    eligible, reasons = [], []
    for row in candidates:
        excluded = []
        if row["update_regime"]["zero_code_updates"]:
            excluded.append("zero_code_updates")
        if row["update_regime"]["all_saturated"]:
            excluded.append("all_saturated")
        if row["clean_accuracy_change"] < -0.01 - 1e-12:
            excluded.append("clean_loss_exceeds_1pp")
        reasons.append(dict(learning_rate=row["learning_rate"], exclusions=excluded))
        if not excluded:
            eligible.append(row)
    winner = max(eligible, key=lambda r: (r["shifted_accuracy_improvement"],
                  r["clean_accuracy_change"], -r["learning_rate"])) if eligible else None
    return dict(
        status="selected" if winner else "inconclusive",
        selected_rate=winner["learning_rate"] if winner else None,
        selected_gain_positive=winner["shifted_accuracy_improvement"] > 0 if winner else None,
        candidates=reasons, rule=SELECTION_RULE,
        next_action="Run paired report without retuning" if winner else
        "No rate selected; discuss expanded development screen, never use official query to tune")


def protocol(args):
    return dict(
        phase=args.phase, network="700-450rec-256-20", trainable_mask="wout",
        wout_shape=[20, 256], frozen="W0, R, Wh1 and all other parameters",
        phase_seeds=[DEV_SEED] if args.phase == "pilot" else list(REPORT_SEEDS),
        dataset_partition="validation" if args.phase == "pilot" else "train_support_official_test",
        support_samples=SUPPORT, query_samples=688 if args.phase == "pilot" else 2264,
        batch_size=BATCH, shift_channels=SHIFT, shift="zero fill; no wrap",
        device=args.device, threads=args.threads, rates=list(RATES),
        rate_rationale="0.3x, 1x, 3x formal config learning_rate_output=1e-4; not Adam-equivalent",
        gradient="existing FP32 einsum(e,trace)/B -> RNE Q40; both arms use same conversion",
        initialization="fresh formal_m2 checkpoint per arm; Wh1 and Wout RNE/sat16 Q2.14",
        temporal="reset once per B64; update Wout each timestep; detach not clear; residual crosses B64 windows",
        vote_rule="-sign(g_q40)*31 votes; threshold65 -> code crossing; 31/65 is NOT an SGD lr",
        magnitude_rule="RNE(-lr*g_q40*2**(14+24-40)); integer residual; trunc-toward-zero crossings; sat16; clear on clip",
        residual_precision="SGD signed Q0.24 in code-LSB units, weight resolution 2**-38; int32 capacity",
        state_bytes=dict(parameters=5120, code_signed16=10240,
                         vote_signed16_residual=10240, vote_total=20480,
                         sgd_signed32_residual=20480, sgd_total=30720,
                         reference_int64_code_plus_int64_residual=81920,
                         effective_fp32_wout_model_copy=20480,
                         optimizer_moments=0, master_weights=0),
        preprocessing="formal dataset_init: tonic ToFrame(time_window=10000), reshape, PadTensors(batch_first=True)",
        query_selection="official test never used for rate selection",
        pilot_loader_note="original dataset factory may open official-test metadata; pilot never reads/evaluates its samples",
        selection_rule=SELECTION_RULE,
        counter_boundary="software per-address update events, not unique final addresses or physical RTL writes",
        claim_boundary="fixed-checkpoint support resampling, one +16 shift, hybrid FP32 forward; no RTL/bit-exact or universal superiority claim")


def load_context(args):
    import torch
    import tonic

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; no automatic device substitution")
    sys.path.insert(0, str(args.handoff / "tp_exp/python"))
    module = importlib.import_module("compare_srnn_shd_shifted_optimizers")
    base = importlib.import_module("m1_m2_train")
    trainer = importlib.import_module("multilayer_tp_train")
    config_path = args.handoff / "tp_exp/configs/B64_FORMAL_SEED_A.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = dict(input_size=700, backbone_hidden_size=450,
                    added_hidden_size=256, output_size=20, batch_size=64)
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("formal 700-450rec-256-20/B64 config required")
    if config["learning_rate_output"] != 1e-4:
        raise ValueError("preset rates were specified around formal output lr=1e-4")
    config["threads"] = args.threads
    config["dataset_root_relative"] = str(args.dataset_root.resolve())
    with patch.object(tonic.datasets.SHD, "download", side_effect=RuntimeError(
            "External data is read-only; automatic download/extraction is disabled")):
        train, test, collate = trainer.datasets(config)
    validation = base.load_indices(args.handoff / config["validation_indices_relative"])
    return SimpleNamespace(module=module, base=base, trainer=trainer, config=config,
                           train=train, test=test, collate=collate, validation=validation,
                           device=torch.device(args.device), torch=torch)


def parameter_changes(model, before):
    result = {}
    for name, current in model.named_parameters():
        delta = current.detach().cpu() - before[name]
        count = int((delta != 0).sum())
        result[name] = dict(nonzero=count, l2=float(delta.square().sum().sqrt()))
        if name != "network.2.fc.weight" and count:
            raise RuntimeError(f"frozen parameter changed: {name}")
    return result


def run_episode(args, ctx, seed, method, rate):
    start = time.perf_counter()
    module, base, torch = ctx.module, ctx.base, ctx.torch
    support, query = episode_indices(ctx.validation, seed, args.phase, len(ctx.test))
    model, optimizers = module.fresh_model(args.handoff, ctx.config, ctx.trainer, base)
    del optimizers  # Restored for exact loader reuse, but never used by either arm.
    model.to(ctx.device)
    state = module.direct_initial_state(model)
    del state["wh1_code"], state["wh1_residual"]
    initial_code = state["wout_code"].copy()
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    query_data = ctx.train if args.phase == "pilot" else ctx.test
    support_shifted = module.FrequencyShiftDataset(ctx.train, SHIFT)
    query_shifted = module.FrequencyShiftDataset(query_data, SHIFT)
    timings = dict(initialization=time.perf_counter() - start, evaluation=0.0,
                   support_loading=0.0, adaptation=0.0)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize(ctx.device)

    def evaluate():
        sync()
        tick = time.perf_counter()
        value = module.evaluate_pair(model, query_data, query_shifted, ctx.collate,
                                     query, base, ctx.config["output_size"], ctx.device)
        sync()
        timings["evaluation"] += time.perf_counter() - tick
        return value

    evaluations = [dict(after_support_batches=0, **evaluate())]
    batches = []
    with update_adapter(module, method, rate) as observer:
        for offset in range(0, SUPPORT, BATCH):
            tick = time.perf_counter()
            data, labels = next(iter(base.indexed_loader(
                support_shifted, ctx.collate, [support[offset:offset + BATCH]])))
            data, labels = data.to(ctx.device), labels.to(ctx.device)
            sync()
            timings["support_loading"] += time.perf_counter() - tick
            previous = observer.totals.copy()
            tick = time.perf_counter()
            metric = module.direct_adapt_batch(model, data, labels, ctx.config,
                                               state, CANDIDATE, "wout")
            sync()
            elapsed = time.perf_counter() - tick
            timings["adaptation"] += elapsed
            metric.update(after_support_batches=1 + offset // BATCH, elapsed_seconds=elapsed,
                          counters={k: v - previous[k] for k, v in observer.totals.items()})
            batches.append(metric)
            evaluations.append(dict(after_support_batches=len(batches), **evaluate()))
    del data, labels
    changes = parameter_changes(model, before)
    code_delta = state["wout_code"] - initial_code
    remainder = state["wout_residual"]
    counters = observer.totals
    timings["total"] = time.perf_counter() - start
    result = dict(
        status="completed", phase=args.phase, seed=seed, method=method, learning_rate=rate,
        device=args.device, support_indices=support, query_indices=query,
        support_query_index_spaces="validation" if args.phase == "pilot" else "training_validation_vs_official_test",
        support_query_overlap=len(set(support) & set(query)) if args.phase == "pilot" else 0,
        evaluations=evaluations, support_metrics=batches,
        clean_accuracy_change=evaluations[-1]["clean"]["accuracy"] - evaluations[0]["clean"]["accuracy"],
        shifted_accuracy_improvement=evaluations[-1]["shifted"]["accuracy"] - evaluations[0]["shifted"]["accuracy"],
        counters=counters, gradient_abs_max=observer.gradient_abs_max,
        gradient_abs_sum=observer.gradient_abs_sum, parameter_deltas=changes,
        weights=dict(final_changed_addresses=int(np.count_nonzero(code_delta)),
                     final_l1_code_delta=int(np.abs(code_delta).sum()),
                     final_max_abs_code_delta=int(np.abs(code_delta).max()),
                     initial_code_min=int(initial_code.min()), initial_code_max=int(initial_code.max()),
                     final_code_min=int(state["wout_code"].min()), final_code_max=int(state["wout_code"].max())),
        residual=dict(dtype=str(remainder.dtype), nonzero=int(np.count_nonzero(remainder)),
                      min=int(remainder.min()), max=int(remainder.max()),
                      unit="1/2**24 code LSB" if method == "magnitude_sgd" else "integer vote",
                      sum_abs=int(np.abs(remainder).sum())),
        update_regime=update_regime(counters, state["wout_code"]), elapsed_seconds=timings)
    return result


def report_rate(pilot, args):
    if (pilot.get("status") != "completed" or pilot.get("phase") != "pilot" or
            pilot["protocol"]["dataset_partition"] != "validation"):
        raise ValueError("report needs a validation pilot, not official-query results")
    decision = select_rate(pilot["episodes"])
    if decision != pilot["selection"] or decision["status"] != "selected":
        raise ValueError("pilot has no valid selected rate; discuss development expansion")
    for name in ("device", "threads"):
        if pilot["protocol"][name] != getattr(args, name):
            raise ValueError(f"keep pilot/report {name} identical")
    for name in ("handoff", "dataset_root"):
        if Path(pilot["inputs"][name]).resolve() != getattr(args, name).resolve():
            raise ValueError(f"keep pilot/report {name} identical")
    return decision["selected_rate"]


def paired_summary(rows):
    paired = []
    for seed in REPORT_SEEDS:
        arms = {r["method"]: r for r in rows if r["seed"] == seed}
        direct, sgd = (arms[name] for name in METHODS)
        for field in ("support_indices", "query_indices", "device"):
            if direct[field] != sgd[field]:
                raise ValueError(f"unpaired {field} for seed {seed}")
        if direct["evaluations"][0] != sgd["evaluations"][0]:
            raise ValueError(f"unequal initial evaluation for seed {seed}")
        paired.append(dict(seed=seed,
            shifted_gain_difference_sgd_minus_direct_pp=100 * (
                sgd["shifted_accuracy_improvement"] - direct["shifted_accuracy_improvement"]),
            clean_change_difference_sgd_minus_direct_pp=100 * (
                sgd["clean_accuracy_change"] - direct["clean_accuracy_change"])))
    aggregate = {}
    for method in METHODS:
        items = [r for r in rows if r["method"] == method]
        gains = [100 * r["shifted_accuracy_improvement"] for r in items]
        clean = [100 * r["clean_accuracy_change"] for r in items]
        aggregate[method] = dict(episodes=len(items), positive_episodes=sum(g > 0 for g in gains),
            shifted_gain_mean_pp=float(np.mean(gains)), shifted_gain_sample_sd_pp=float(np.std(gains, ddof=1)),
            clean_change_mean_pp=float(np.mean(clean)), clean_change_sample_sd_pp=float(np.std(clean, ddof=1)))
    return dict(pairs=paired, methods=aggregate,
                interpretation="Three support resamplings of one checkpoint, not independent pretraining seeds")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def render_readme(result):
    lines = ["# SRNN Wout-only 幅值 SGD 同条件对照", "",
        f"执行状态：{result['status']}。",
        f"阶段：{result['phase']}；设备：{result['protocol']['device']}；线程：{result['protocol']['threads']}。",
        "固定正式 checkpoint，700-450rec-256-20；只更新 5120 个 Wout，冻结其他参数。",
        "Wh1/Wout 初始有效权重为 Q2.14；前向、膜电位、trace 仍为 FP32，不是整网定点/RTL 精度。", "",
        "两臂复用原 direct_adapt_batch，每 B64 重置一次状态，每 timestep 更新并 detach；两窗口间保留残差。",
        "同一 e.T@trace/B 经 Q40 舍入后，31/65 规则仅取符号累计票数，SGD 保留幅值乘 lr。31/65 不是学习率。",
        "SGD 为 signed Q0.24 code-LSB 残差，int32 容量；Q2.14 code+残差 30720 B，票数臂 20480 B。",
        "Python 参考两臂均使用 int64 code/int64 residual 共 81920 B，另有有效 FP32 Wout 副本 20480 B；无持久 Adam 状态。", "",
        "开发率预设 3e-5/1e-4/3e-4；validation seed128474，support128/query688。",
        "选择排除零 code 更新、所有拟议更新被饱和裁剪或最终全 rail、clean 损失超过 1 个百分点的率。",
        "剩余候选按末次 shifted 增益、clean 变化、较小率排序；负增益原样保留。无有效率则不选，讨论扩展，不自动追加或用官方 query 调率。",
        "报告使用 228474/228475/228476，+16 零填充偏移，training-validation support128 / official-test query2264。", "",
        "| seed | 方法 | lr | clean 前/后 | shift 前/后 | code 变化事件 | 最终不同地址 | 饱和事件 | 秒 |",
        "|---|---|---|---|---|---|---|---|---|"]
    for row in result["episodes"]:
        first, last = row["evaluations"][0], row["evaluations"][-1]
        lines.append(f"| {row['seed']} | {row['method']} | {row['learning_rate']} | "
            f"{first['clean']['correct']}/{last['clean']['correct']} | "
            f"{first['shifted']['correct']}/{last['shifted']['correct']} | "
            f"{row['counters']['code_changes']} | {row['weights']['final_changed_addresses']} | "
            f"{row['counters']['saturations']} | {row['elapsed_seconds']['total']:.2f} |")
    lines += ["", "表内准确率列为正确样本数；分母见本阶段 query_samples。完整四列准确率、每窗口曲线、残差、所有冻结参数变化与耗时见 summary.json 和逐 episode JSON。",
              "counters 是软件地址事件，不是唯一权重数、RTL 写入或周期；主机秒数包括诊断与 query 评估，不能当硬件时延。",
              "三个 seed 是固定 checkpoint 的 support 重抽样；仅测试 +16，不作普遍优势声明。", ""]
    if "selection" in result:
        lines += [f"开发选择：{result['selection']['status']}；lr={result['selection']['selected_rate']}。", ""]
    if "paired" in result:
        lines += ["## 配对汇总", "", "```json", json.dumps(result["paired"], indent=2), "```", ""]
    lines += ["## 复现命令", "", "```powershell", result["command"], "```", "",
              "环境、正式包与数据路径记录于 summary.json；未修改外部包或现有 compare runner。"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("pilot", "report"), default="pilot")
    parser.add_argument("--handoff", type=Path, default=ROOT.parent / "SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/datasets"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="explicitly execute; otherwise print protocol and command only")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.phase == "report" and args.pilot_result is None:
        parser.error("report requires --pilot-result from validation selection")
    if args.phase == "pilot" and args.pilot_result is not None:
        parser.error("pilot does not consume a previous selection")
    if args.output_dir is None:
        args.output_dir = OUTPUT / (args.phase + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not args.output_dir.resolve().is_relative_to(ROOT):
        parser.error("outputs must remain inside the project")
    return args


def execution_command(args):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--phase", args.phase,
               "--handoff", str(args.handoff), "--dataset-root", str(args.dataset_root),
               "--device", args.device, "--threads", str(args.threads),
               "--output-dir", str(args.output_dir.resolve())]
    if args.pilot_result:
        command += ["--pilot-result", str(args.pilot_result.resolve())]
    return ("& " if os.name == "nt" else "") + subprocess.list2cmdline([*command, "--run"])


def main(argv=None):
    args = parse_args(argv)
    plan = protocol(args)
    pilot, rate = None, None
    if args.phase == "report":
        pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
        rate = report_rate(pilot, args)
    jobs = ([(DEV_SEED, "direct_response", None)] +
            [(DEV_SEED, "magnitude_sgd", r) for r in RATES] if args.phase == "pilot" else
            [(seed, method, rate if method == "magnitude_sgd" else None)
             for seed in REPORT_SEEDS for method in METHODS])
    if not args.run:
        print(json.dumps(dict(status="dry_run", protocol=plan, jobs=jobs,
                              command=execution_command(args)), indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(status="running", phase=args.phase, protocol=plan, episodes=[],
                  inputs=dict(handoff=str(args.handoff.resolve()), dataset_root=str(args.dataset_root.resolve())),
                  command=execution_command(args), started_utc=datetime.now(timezone.utc).isoformat())
    try:
        ctx = load_context(args)
        result["environment"] = dict(python=platform.python_version(), numpy=np.__version__,
            torch=ctx.torch.__version__, tonic=importlib.metadata.version("tonic"),
            cuda=ctx.torch.version.cuda, platform=platform.platform(),
            historical_note="formal package recorded Python3.12/PyTorch2.7.1; do not assume bitwise cross-version parity")
        result["formal_config"] = ctx.config
        result["reused_functions"] = ["fresh_model", "direct_initial_state", "direct_adapt_batch",
                                       "evaluate_pair", "FrequencyShiftDataset", "base.indexed_loader"]
        for seed, method, learning_rate in jobs:
            print(f"A3_START phase={args.phase} seed={seed} method={method} lr={learning_rate}", flush=True)
            row = run_episode(args, ctx, seed, method, learning_rate)
            peers = [r for r in result["episodes"] if r["seed"] == seed]
            if peers and any(row[key] != peers[0][key] for key in (
                    "support_indices", "query_indices", "device")):
                raise ValueError(f"unpaired inputs for seed {seed}")
            if peers and row["evaluations"][0] != peers[0]["evaluations"][0]:
                raise ValueError(f"unequal initial evaluation for seed {seed}")
            result["episodes"].append(row)
            tag = method if learning_rate is None else f"{method}_{learning_rate:g}"
            write_json(args.output_dir / f"seed{seed}_{tag}.json", row)
            print(f"A3_DONE seed={seed} method={method} gain_pp={100 * row['shifted_accuracy_improvement']:.6f} "
                  f"seconds={row['elapsed_seconds']['total']:.2f}", flush=True)
        if args.phase == "pilot":
            result["selection"] = select_rate(result["episodes"])
        else:
            result["pilot_result"] = str(args.pilot_result.resolve())
            result["selected_rate"] = rate
            result["paired"] = paired_summary(result["episodes"])
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds_wall"] = time.perf_counter() - started
        write_json(args.output_dir / "summary.json", result)
        with (args.output_dir / "README.md").open("x", encoding="utf-8") as handle:
            handle.write(render_readme(result))
    print(f"A3_COMPLETE {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
