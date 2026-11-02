#!/usr/bin/env python3
"""Summarize the historical small SRNN harness, not final-model performance."""

from __future__ import annotations

import csv
import argparse
import json
import re
from pathlib import Path


MARKER = "UNIFIED_SNN_CORE_FINAL_V1_SRNN_M2_ONLINE_PASS"


def parse_fields(line: str) -> dict[str, int | str]:
    fields: dict[str, int | str] = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", line):
        fields[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return fields


def read_marker(path: Path) -> tuple[str, dict[str, int | str]]:
    matches = [
        line
        for line in path.read_text(encoding="ascii", errors="strict").splitlines()
        if line.startswith(MARKER + " ")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one PASS marker, found {len(matches)}")
    return matches[0], parse_fields(matches[0])


def require_int(fields: dict[str, int | str], key: str) -> int:
    value = fields.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"missing integer field {key}: {fields}")
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New result directory; preserve historical exports")
    args = parser.parse_args()
    log_dir = root / "results" / "learning_reference" / "actual_srnn_m2_online"
    logs = [log_dir / "run1.log", log_dir / "run2.log"]
    markers = [read_marker(path) for path in logs]
    if markers[0][0] != markers[1][0]:
        raise RuntimeError("SRNN M2 PASS markers are not deterministic")

    fields = markers[0][1]
    train_cycles = require_int(fields, "train_cycles")
    inference_cycles = require_int(fields, "inference_cycles")
    incremental_cycles = require_int(fields, "incremental_learning_cycles")
    if train_cycles - inference_cycles != incremental_cycles:
        raise RuntimeError("incremental learning cycle arithmetic is inconsistent")
    if require_int(fields, "changed_after_training") <= 0:
        raise RuntimeError("SRNN M2 evidence contains no changed parameter")
    if require_int(fields, "learning_off_mismatch") != 0:
        raise RuntimeError("updated-weight learning-off replay mismatch")

    summary = {
        "schema": "srnn_m2_historical_online_cycles_v2",
        "status": "historical_small_harness_rtl_simulated",
        "topology": "SRNN",
        "selected_mask": None,
        "mask_note": "Raw logs do not separate Wh1/Wout writes; current legacy binding inherits ENABLE_WH1_LEARNING=1, not proof of Wout-only",
        "workload": "two real SHD backbone inputs x two captured timesteps; synthetic 450-2-2 M2 weights and targets",
        "m2_shape": [450, 2, 2],
        "final_v2_performance": False,
        "runs": 2,
        "deterministic": True,
        "train_cycles": train_cycles,
        "learning_off_cycles": inference_cycles,
        "train_minus_updated_replay_cycles": incremental_cycles,
        "changed_scan_events": require_int(fields, "changed_after_training"),
        "distinct_changed_addresses": None,
        "replay_code_scan_mismatches": require_int(fields, "learning_off_mismatch"),
        "commits": require_int(fields, "m2_commits"),
        "model_version": require_int(fields, "model_version"),
        "adapt_version": require_int(fields, "adapt_version"),
        "evidence_label": "Verilator RTL simulated",
        "not_evidence_for": [
            "final shared-array primary model performance",
            "Wout-only training mask",
            "450-256-20 M2 dataset accuracy",
            "isolated learning engine latency",
            "independent initial-state learning-off control",
            "complete support=128 adaptation episode latency",
            "post-route functional timing",
            "board-measured throughput",
            "cross-topology speed ranking",
        ],
        "source_logs": [str(path.relative_to(root)).replace("\\", "/") for path in logs],
        "marker": markers[0][0],
    }

    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(root):
        parser.error("output must stay inside the project")
    if out_dir == root / "results/paper_evidence/rtl/srnn_m2_current_cycles_20260928":
        parser.error("preserve the historical report; choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    with (out_dir / "summary.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topology",
                "selected_mask",
                "workload",
                "train_cycles",
                "learning_off_cycles",
                "train_minus_updated_replay_cycles",
                "changed_scan_events",
                "runs",
                "evidence_label",
            ],
        )
        writer.writeheader()
        writer.writerow({key: summary[key] for key in writer.fieldnames})

    readme = f"""# 历史 SRNN 小模型周期：口径修正

2026-10-05。原始日志和旧报告未覆盖。本次只重新解析已有日志，没有执行 RTL 仿真。

两次日志一致：训推事务 {train_cycles:,} cycles，更新后 learning-off 重放
{inference_cycles:,} cycles，差值 {incremental_cycles:,} cycles。
该差值不是隔离的学习引擎延迟，重放也不是同初始权重的独立关闭学习对照。

真实 SHD 输入仅驱动旧骨干；M2 是 450→2→2，权重、目标和标签为构造值。
旧汇总硬编码的 Wout-only 已移除：日志未分列 Wh1/Wout 写，现存旧 binding
继承 ENABLE_WH1_LEARNING=1，不能认证当时的 Wout-only 掩码。
changed_after_training 是两次扫描累计的变化事件，不是去重地址数；
learning_off_mismatch 是 code 扫描比较，不是独立 golden logit 一致率。

此证据不支持最终共享 12×H40 路径、450→256→20 主模型或 support=128
端到端性能，也不支持板级吞吐与后仿真结论。NA 字段保留为 null/空格，不能当零。

原始来源：

- `results/learning_reference/actual_srnn_m2_online/run1.log`
- `results/learning_reference/actual_srnn_m2_online/run2.log`
- `tb/tb_unified_snn_core_final_v1_srnn_m2_online.sv`
- `scripts/run_verilator_unified_snn_core_final_v1_srnn_m2_online.ps1`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        "SRNN_M2_HISTORICAL_CYCLES_PARSE_PASS "
        f"runs=2 train={train_cycles} inference={inference_cycles} "
        f"incremental={incremental_cycles}"
    )


if __name__ == "__main__":
    main()
