#!/usr/bin/env python3
"""Evaluate the saved FC query100 with initial and support160 integer weights."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_fc_formal_test_numeric import (
    C, F, H, OVERFLOWS, check_scalar, integer_step, metrics_for, validate_counts, wrap,
)
from scripts.build_fc_w2_numeric_reference import MODEL, load_model

ALGORITHM = ROOT / "results/paper_evidence/supplement_20261004/algorithm"
OUTPUT = ALGORITHM / "fc_integer_episode_query"
EPISODE = ALGORITHM / "a2/fc/seed38472_residual_on.json"
SUPPORT = ALGORITHM / "fc_final_counts_reference/support160_on"
ARMS = ("initial_clean", "final_clean", "initial_shifted", "final_shifted")
OLD_KEYS = ("baseline_clean", "clean_after", "baseline_shifted", "shifted_after")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_mem16(path):
    with path.open(encoding="utf-8") as stream:
        values = np.fromiter((int(line.strip(), 16) for line in stream), dtype=np.int64)
    if np.any((values < 0) | (values > 65535)):
        raise ValueError(f"not a 16-bit mem: {path}")
    return wrap(values, 16)


def load_episode_weights(support_dir, episode, model):
    reference = read_json(support_dir / "reference.json")
    if (reference["status"] != "PASS" or reference["learning"] is not True
            or reference["support_indices"] != episode["support_indices"]
            or reference["batches"] != 10):
        raise ValueError("support reference does not identify the saved episode")
    initial_w0 = model["W0"].astype(np.int64)
    previous_w2 = model["W2"].astype(np.int64)
    for batch in range(10):
        directory = support_dir / f"batch_{batch}"
        parameters = read_mem16(directory / "parameters.mem")
        if parameters.size != F * H + H * C:
            raise ValueError("unexpected parameter memory length")
        np.testing.assert_array_equal(parameters[:F * H].reshape(F, H), initial_w0)
        np.testing.assert_array_equal(parameters[F * H:].reshape(H, C), previous_w2)
        with np.load(directory / "final_state.npz", allow_pickle=False) as state:
            w2, residual = state["w2"].copy(), state["residual"].copy()
        for values in (w2, residual):
            if values.shape != (H, C) or values.dtype.kind not in "iu":
                raise ValueError("expected integer [160,10] final state")
        if np.any((w2 < -127) | (w2 > 127)) or np.any(np.abs(residual) >= 32768):
            raise ValueError("final state outside the production code/residual range")
        previous_w2 = w2
    changed = int(np.count_nonzero(w2 != model["W2"]))
    if changed != reference["final_changed_w2"]:
        raise ValueError("final changed-address count differs from reference")
    for name, expected in (("w2_after", w2), ("residual_after", residual)):
        values = read_mem16(support_dir / "batch_9" / f"{name}.mem")
        if values.size != reference["batch_steps"][9] * H * C:
            raise ValueError("unexpected final batch state stream length")
        np.testing.assert_array_equal(values[-H * C:].reshape(H, C), expected)
    code_events = sum(row["code_changes"] for row in reference["records"])
    if code_events != reference["totals"]["code_changes"]:
        raise ValueError("support code event summary differs from its per-step records")
    return w2, residual, {
        "initial_model": str(MODEL), "final_state": str(support_dir / "batch_9/final_state.npz"),
        "support_reference": str(support_dir / "reference.json"),
        "code_change_events": code_events, "final_changed_addresses": changed,
        "w0_parameter_batches_compared": 10, "w0_code_comparisons": 10 * F * H,
        "w0_mismatches": 0, "w2_batch_input_chain_mismatches": 0,
        "final_state_vs_last_mem_mismatches": 0,
        "residual_use": "只记录来源；query不学习，不把residual加到W2前向权重",
    }


def load_query_batch(data_root, traces_root, indices):
    if not 1 <= len(indices) <= 16:
        raise ValueError("query batch must contain 1..16 saved indices")
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic
    from scripts.run_fc_tp_supervised_shift_adaptation import load_dataset_batches
    records = []
    original = tonic.datasets.NMNIST.__getitem__

    def observe(dataset, index):
        frames, label = original(dataset, index)
        validate_counts(frames[None])
        records.append({"dataset_index": int(index), "label": int(label), "split": "train",
                        "valid_timesteps": len(frames),
                        "rawfilename": Path(dataset.data[index]).resolve().relative_to(data_root.resolve()).as_posix()})
        return frames, label

    with patch.object(tonic.datasets.NMNIST, "download", side_effect=RuntimeError(
            "read-only data; download/extraction disabled")), \
            patch.object(tonic.datasets.NMNIST, "__getitem__", observe):
        batches = load_dataset_batches(traces_root=traces_root, data_root=data_root, train=True,
                                       indices=indices, batch_size=16)
    if len(batches) != 1 or [row["dataset_index"] for row in records] != indices:
        raise ValueError("formal loader changed query order")
    frames, labels = batches[0]
    if frames.shape[0] != len(indices) or frames.shape[1] != max(row["valid_timesteps"] for row in records):
        raise ValueError("batch must use only its own longest real T")
    np.testing.assert_array_equal(labels.numpy(), [row["label"] for row in records])
    return frames, labels.numpy(), records


def integer_forward(counts, model, capture=()):
    counts = validate_counts(counts)
    batch, steps, _ = counts.shape
    w0, w2 = model["W0"].astype(np.int64), model["W2"].astype(np.int64)
    scale = round(float(model["W2_scale"]) * 64 * 65536)
    voltage = np.zeros((batch, H), dtype=np.int64)
    trace = np.zeros_like(voltage)
    logits = np.zeros((batch, C), dtype=np.int64)
    overflow = {name: np.zeros(batch, dtype=np.int64) for name in OVERFLOWS}
    spikes = np.zeros(batch, dtype=np.int64)
    history = {int(sample): [] for sample in capture}
    for t in range(steps):
        raw = np.zeros((batch, H), dtype=np.int64)
        for sample, row in enumerate(counts[:, t].astype(np.int64)):
            active = np.flatnonzero(row)
            raw[sample] = row[active] @ w0[active]
        state = integer_step(raw, w2, voltage, trace, logits, scale)
        voltage, trace, logits = (state[key] for key in ("voltage", "trace", "logits"))
        spikes += state["spike"].sum(axis=1)
        for name in OVERFLOWS:
            overflow[name] += state["overflow"][name]
        for sample in history:
            history[sample].append({name: state[name][sample].copy() for name in
                                   ("pre", "voltage", "trace", "spike", "dot", "logits", "q6")})
    return dict(raw24=logits, q6=state["q6"], spikes=spikes, history=history, overflow=overflow)


def directed_checks(model):
    # B4/T3 tests variable batch/time, repeated counts, >49 events and empty padding.
    frames = np.zeros((4, 3, F), dtype=np.uint32)
    frames[0, 0, 17] = 3
    frames[1, 0, :70] = 1
    frames[2, 1, 500:570] = 2
    custom = {**model, "W2": np.zeros((H, C), dtype=np.int64)}
    for weights in (model, custom):
        result = integer_forward(frames, weights, capture=range(4))
        for sample in range(4):
            check_scalar(frames[sample], weights, result["history"][sample])
            if len(result["history"][sample]) != 3:
                raise AssertionError("unrequested time padding")
    if not np.all(result["raw24"] == 0) or int(result["spikes"].sum()) == 0:
        raise AssertionError("explicit W2 argument was not used")
    return {"status": "PASS", "cases": "B4/T3、重复counts、>49事件、空帧、零W2注入、8个合成样本标量核对"}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--support-dir", type=Path, default=SUPPORT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        parser.error("output must remain within algorithm/fc_integer_episode_query")
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("summary.json", "predictions.csv", "query_logits.npz")):
        raise FileExistsError("use a fresh output directory; preserve existing evidence")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    episode = read_json(EPISODE)
    query, support = episode["query_indices"], episode["support_indices"]
    if (episode["episode_seed"] != 38472 or episode["query_split"] != "train"
            or len(query) != 100 or len(set(query)) != 100
            or len(support) != 160 or len(set(support)) != 160 or set(query) & set(support)):
        raise ValueError("expected the unchanged disjoint seed38472 support160/query100")
    old_row = next(row for row in episode["masks"] if row["trainable_mask"] == "w2")
    provenance = read_json(ALGORITHM / "fc_formal_counts/summary.json")
    data_root, traces_root = Path(provenance["data_root"]), Path(provenance["traces_root"])
    initial = load_model()
    final_w2, residual, weights_source = load_episode_weights(args.support_dir.resolve(), episode, initial)
    final = {**initial, "W2": final_w2}
    frozen_w0, frozen_initial, frozen_final = initial["W0"].copy(), initial["W2"].copy(), final_w2.copy()
    checks = directed_checks(initial)
    from scripts.run_fc_tp_supervised_shift_adaptation import shift_nmnist_frames
    logits = {name: [] for name in ARMS}
    q6s = {name: [] for name in ARMS}
    overflow = {name: {key: 0 for key in OVERFLOWS} for name in ARMS}
    rows, batch_records, scalar_checks = [], [], []
    for start in range(0, 100, 16):
        tick = time.perf_counter()
        indices = query[start:start + 16]
        frames, labels, identities = load_query_batch(data_root, traces_root, indices)
        for key in OLD_KEYS:
            np.testing.assert_array_equal(labels, old_row[key]["labels"][start:start + len(indices)])
        capture = list(range(len(indices))) if len(indices) < 16 else [0]
        views = {"clean": frames, "shifted": shift_nmnist_frames(frames, dx=2, dy=0)}
        results, event_counts, repeats = {}, {}, {}
        for view, tensor in views.items():
            counts = tensor.numpy()
            validate_counts(counts)
            if start == 0:
                with np.load(ALGORITHM / f"fc_formal_counts/query_{view}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(counts, saved["dense_counts"])
                    np.testing.assert_array_equal(indices, saved["dataset_indices"])
            if view == "shifted":
                source = frames.numpy().reshape(len(indices), frames.shape[1], 2, 34, 34)
                expected = np.zeros_like(source)
                expected[..., 2:] = source[..., :-2]
                np.testing.assert_array_equal(counts, expected.reshape(counts.shape))
            event_counts[view] = counts.sum(axis=(1, 2), dtype=np.int64)
            repeats[view] = event_counts[view] - np.count_nonzero(counts, axis=(1, 2))
            for version, model in (("initial", initial), ("final", final)):
                name = f"{version}_{view}"
                result = integer_forward(counts, model, capture)
                results[name] = result
                logits[name].append(result["raw24"])
                q6s[name].append(result["q6"])
                for key in OVERFLOWS:
                    overflow[name][key] += int(result["overflow"][key].sum())
                for local in capture:
                    checked = check_scalar(counts[local], model, result["history"][local])
                    scalar_checks.append({"arm": name, "query_position": start + local,
                                          "dataset_index": indices[local], **checked})
            np.testing.assert_array_equal(results[f"initial_{view}"]["spikes"], results[f"final_{view}"]["spikes"])
        for local, identity in enumerate(identities):
            row = {"query_position": start + local, "batch_index": start // 16,
                   "sample_in_batch": local, **identity, "batch_timesteps": int(frames.shape[1])}
            for view in views:
                row[f"{view}_events"] = int(event_counts[view][local])
                row[f"{view}_repeated_extra_events"] = int(repeats[view][local])
            for name in ARMS:
                prediction = int(results[name]["raw24"][local].argmax())
                row[name + "_raw_prediction"] = prediction
                row[name + "_q6_prediction"] = int(results[name]["q6"][local].argmax())
                row[name + "_correct"] = int(prediction == labels[local])
            rows.append(row)
        record = {"batch_index": start // 16, "query_start": start, "samples": len(indices),
                  "timesteps": int(frames.shape[1]), "valid_timesteps": [r["valid_timesteps"] for r in identities],
                  "seconds": time.perf_counter() - tick}
        batch_records.append(record)
        print(json.dumps(record), flush=True)
    np.testing.assert_array_equal(initial["W0"], frozen_w0)
    np.testing.assert_array_equal(final["W0"], frozen_w0)
    np.testing.assert_array_equal(initial["W2"], frozen_initial)
    np.testing.assert_array_equal(final["W2"], frozen_final)
    logits = {name: np.concatenate(values) for name, values in logits.items()}
    q6s = {name: np.concatenate(values) for name, values in q6s.items()}
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    metrics = {name: metrics_for(labels, logits[name]) for name in ARMS}
    q6_metrics = {name: metrics_for(labels, q6s[name]) for name in ARMS}
    changes = {}
    for view in ("clean", "shifted"):
        before, after = (logits[f"{version}_{view}"].argmax(1) for version in ("initial", "final"))
        positions = np.flatnonzero(before != after)
        changes[view] = {"prediction_changes": len(positions), "query_positions": positions.tolist(),
                         "dataset_indices": [query[i] for i in positions],
                         "wrong_to_correct": int(np.count_nonzero((before != labels) & (after == labels))),
                         "correct_to_wrong": int(np.count_nonzero((before == labels) & (after != labels))),
                         "accuracy_gain_points": metrics[f"final_{view}"]["accuracy_percent"] - metrics[f"initial_{view}"]["accuracy_percent"]}
    parallel = {}
    for name, key in zip(ARMS, OLD_KEYS):
        saved = old_row[key]
        np.testing.assert_equal(saved["correct"], int(np.count_nonzero(np.array(saved["predictions"]) == labels)))
        parallel[name] = {"saved_a2_correct": saved["correct"], "integer_correct": metrics[name]["correct"],
                          "prediction_differences": int(np.count_nonzero(logits[name].argmax(1) != saved["predictions"]))}
    changed_weights = [{"hidden": int(h), "class": int(c), "initial_code": int(frozen_initial[h, c]),
                        "final_code": int(frozen_final[h, c]), "delta_code": int(frozen_final[h, c] - frozen_initial[h, c]),
                        "final_residual": int(residual[h, c])}
                       for h, c in zip(*np.nonzero(frozen_final != frozen_initial))]
    write_csv(out / "predictions.csv", rows)
    write_csv(out / "weight_changes.csv", changed_weights)
    np.savez_compressed(out / "query_logits.npz", dataset_indices=np.array(query), labels=labels,
                        valid_timesteps=np.array([row["valid_timesteps"] for row in rows]),
                        batch_timesteps=np.array([row["batch_timesteps"] for row in rows]),
                        **{name + "_raw24": values for name, values in logits.items()},
                        **{name + "_q6": values for name, values in q6s.items()})
    np.savez_compressed(out / "evaluated_codes.npz", W0=frozen_w0, W2_initial=frozen_initial,
                        W2_final=frozen_final, residual_final=residual)
    import tonic
    report = {
        "status": "PASS", "scope": "软件整数query100评估，不是全query RTL，也不是官方Test10000",
        "command": [sys.executable, "-B", *sys.argv], "episode": str(EPISODE), "seed": 38472,
        "query_split": "train", "query_indices": query, "support_query_overlap": 0,
        "data_root": str(data_root), "traces_root": str(traces_root),
        "device": "cpu", "threads": 1, "tf32": False,
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "tonic": tonic.__version__},
        "protocol": "原first-saccade<100ms/1ms counts；原序六批B16及尾批B4，各batch仅pad至本批最长真实T；每批零状态、无学习",
        "shift": {"dx_pixels": 2, "dy_pixels": 0, "zero_fill": True},
        "numeric": "复用fulltestA1 integer_step：W0 shift6/wrap18，leak32113，threshold1024，trace64225/UQ6.10，INT8 W2 raw24累计；Q6 RNE乘47303；argmax平分取最小类别",
        "weight_source": weights_source, "batches": batch_records,
        "metrics_raw24": metrics, "metrics_q6": q6_metrics, "changes": changes,
        "raw24_q6_prediction_differences": {name: int(np.count_nonzero(logits[name].argmax(1) != q6s[name].argmax(1))) for name in ARMS},
        "overflow_counts": overflow, "saved_a2_parallel": parallel,
        "saved_a2_boundary": "旧A2没有保存final权重；即使事件数、地址数或预测吻合，也不声称两个final W2逐位一致",
        "validation": {"directed_checks": checks, "scalar_sequences": len(scalar_checks),
                       "scalar_values": sum(row["compared_values"] for row in scalar_checks), "scalar_mismatches": 0,
                       "selection": "各完整B16首样本及尾批全部4样本，四格完整本批时间轴",
                       "first16_vs_exported_counts": "clean/shifted逐点相同",
                       "shift_vs_independent_zero_fill": "全部100样本逐点相同",
                       "w0_and_w2_unchanged_during_query": True},
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", report)
    write_json(out / "scalar_validation.json", scalar_checks)
    lines = ["# FC 整数续学后的 query100", "", "仅软件整数query，不是全query RTL，也不是官方Test10000。未训练或再次更新权重。", "",
             "原seed38472保存的Train query100，原顺序，与support160无交集。原Tonic first-saccade<100ms、1ms保重数counts，",
             "无49事件裁剪、不二值化；六批B16及尾批B4，每批仅处理本批最长真实T，不额外补时间。+2 shift沿x方向零填充。", "",
             "| 条件 | 初始W2正确数/100 | 整数final W2正确数/100 | 变化百分点 |", "|---|---:|---:|---:|"]
    for view in ("clean", "shifted"):
        lines.append(f"| {view} | {metrics['initial_' + view]['correct']} | {metrics['final_' + view]['correct']} | {changes[view]['accuracy_gain_points']:+.2f} |")
    lines += ["", "主结果为raw24 argmax；Q6读出的四格结果及并列情况另存summary.json。", 
              f"每批T：{[row['timesteps'] for row in batch_records]}；尾批4样本有效T：{batch_records[-1]['valid_timesteps']}。", "",
              "初始code来自原load_model()的INT5 W0/INT8 W2；final W2直接读", 
              "`../fc_final_counts_reference/support160_on/batch_9/final_state.npz` 的 `w2`，不应用residual、不加学习率。",
              f"support整数参考累计{weights_source['code_change_events']}次code改变，最终{weights_source['final_changed_addresses']}个地址不同。",
              "十批parameters.mem中的369920个W0 code逐点等于初始W0；W2跨批参数链及末步mem与final_state逐点一致。",
              "本次四格共用同一W0，前向前后W0与两份W2均未改变。code来源及逐地址变化见evaluated_codes.npz、weight_changes.csv。", "",
              "原A2保存结果并列：shifted 75→76、clean 96→95；具体源JSON字段及预测差异见summary.json。",
              "旧A2未保存final权重，不因事件数、地址数、准确率或预测吻合而声称final W2逐位一致。", "",
              f"定向检查通过；四格{len(scalar_checks)}条完整样本时间序列、{report['validation']['scalar_values']}个整数值与标量参考零失配。",
              "核查各B16首样本及尾批全部4样本；首16样本与既有clean/shifted counts导出逐点一致；+2 shift另独立核对。",
              f"四格越界/饱和总事件：{sum(sum(values.values()) for values in overflow.values())}。",
              f"CPU单线程、无GPU/TF32；程序内部墙钟{report['wall_seconds']:.3f}秒。外部计时与采样RSS见run/resources.json。", "",
              "predictions.csv含原ID、rawfilename、label、有效T/批T及四格raw/Q6预测；query_logits.npz保存全部最终raw24/Q6。",
              "summary.json含配置、混淆矩阵、前后预测变化及旧A2并列结果；scalar_validation.json记录逐样本核查。", "",
              "复现（已有结果需用本目录下新的输出子目录）：", "", "```powershell",
              "python -B scripts/evaluate_fc_integer_episode_query.py --output-dir results/paper_evidence/supplement_20261004/algorithm/fc_integer_episode_query/rerun",
              "```", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "correct": {name: row["correct"] for name, row in metrics.items()},
                      "wall_seconds": report["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
