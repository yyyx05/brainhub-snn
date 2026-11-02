"""Summarize measured SCNN classifier traffic, preserving its bounded scope."""

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASES = ["参数装载", "初始 cache 发布", "前向及学习观察捕获",
          "学习及发布等待", "更新后重算", "尾部检查"]
MARKER = "SCNN_Q015_REAL_B4T4_FORWARD_LEARN_RECOMPUTE_PASS"


def parse_rows(text, prefix):
    return [dict((key, int(value)) for key, value in
                 re.findall(r"(\w+)=(-?\d+)", line))
            for line in text.splitlines() if line.startswith(prefix + " ")]


def validate(text, contract, expected_nonzero):
    if MARKER not in text or "%Error" in text or "Assertion failed" in text:
        raise ValueError("the functional test did not pass")
    phases = parse_rows(text, "SUPPLEMENT_PHASE")
    frames = parse_rows(text, "SUPPLEMENT_FRAME")
    if [row["phase"] for row in phases] != list(range(6)):
        raise ValueError("missing or duplicate phase metrics")
    b, t, f, c = (contract[k] for k in ("batch", "timesteps", "features", "classes"))
    keys = [(r["updated"], r["sample"], r["timestep"]) for r in frames]
    expected = {(u, sample, step) for u in range(2) for sample in range(b) for step in range(t)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("frame coverage differs from the declared fixture")
    for phase in (2, 4):
        row = phases[phase]
        if (row["feature_values"], row["nonzero_features"], row["scalar_outputs"],
                row["logit_rows"], row["p4_masked_terms"]) != (
                b*t*f, expected_nonzero, b*t*c, b*t, expected_nonzero*c):
            raise ValueError(f"independent fixture/count reconciliation failed in phase {phase}")
    if sum(r["parameter_loads"] for r in phases) != f*c:
        raise ValueError("parameter loads differ from tensor size")
    if sum(r["code_writes"] for r in phases) != contract["changed_classifier_codes"]:
        raise ValueError("code writes differ from reference")
    if phases[4]["code_writes"] or phases[4]["residual_writes"]:
        raise ValueError("learning-disabled recomputation wrote parameters")
    passed = parse_rows(text, MARKER)[0]
    if sum(r["p4_commands"] for r in phases) != passed["p4_commands"]:
        raise ValueError("P4 counter disagrees with existing testbench counter")
    if any(r["input_to_logit_cycles"] <= 0 for r in frames):
        raise ValueError("invalid frame latency")
    return phases, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    fixture = ROOT / "results/paper_evidence/rtl/scnn_q015_real_b4t4_forward_learn_recompute_20260928/fixture"
    contract = json.loads((fixture / "contract.json").read_text(encoding="utf-8"))
    features = ROOT / "results/paper_evidence/rtl/scnn_q015_dataset_parity/b4t4_update_fixture/features_q8_8.hex"
    values = [int(word, 16) for word in features.read_text().split()]
    b, t, f, c = (contract[k] for k in ("batch", "timesteps", "features", "classes"))
    if len(values) != b*t*f:
        raise ValueError("feature fixture has the wrong shape")
    nonzero = sum(value != 0 for value in values)
    phases, frames = validate((args.result_dir / "verilator.log").read_text(encoding="utf-8-sig"),
                              contract, nonzero)
    for row in phases:
        row["name"] = PHASES[row["phase"]]
        row["projected_time_ms_at_15ns"] = row["cycles"] * 15 / 1e6
        row["feature_port_bytes"] = row["feature_waves"] * 16
        row["observation_capture_port_bytes"] = row["capture_waves"] * 16
        row["parameter_load_payload_bytes"] = row["parameter_loads"] * 4
        row["cache_fill_port_bytes"] = row["cache_fill_words"] * 25
        row["rmw_read_payload_bytes"] = row["rmw_reads"] * 4
        row["rmw_enabled_write_bytes"] = (row["code_writes"] + row["residual_writes"]) * 2
        row["publisher_read_payload_bytes"] = row["publisher_reads"] * 2
    result = {
        "status": "PASS", "scope": contract["scope"],
        "source_tb": "tb/tb_unified_scnn_q015_real_b4t4_forward_learn_recompute_v1.sv",
        "batch": b, "timesteps": t, "features": f, "classes": c,
        "sim_clock_period_ns": 10, "projection_clock_period_ns": 15,
        "dense_model_terms_per_forward": b*t*f*c,
        "nonzero_model_terms_per_forward": nonzero*c,
        "zero_feature_fraction": 1-nonzero/len(values),
        "phase_counters": phases, "frame_latencies": frames,
        "fixture": str(fixture.relative_to(ROOT)),
        "limits": ["classifier boundary, not raw-frame full-network accuracy",
                   "TB-driven phase cycles include stimulus gaps and capture",
                   "array busy is not PE arithmetic utilization",
                   "port payload bytes omit address/control; different boundaries must not be summed as DDR traffic",
                   "updated-weight replay is not an independent same-initial-weight learning-off control",
                   "15ns time projection is not a board measurement"],
    }
    (args.result_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for filename, rows in (("phases.csv", phases), ("frames.csv", frames)):
        with (args.result_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# SCNN 真实 classifier B4×T4 周期与数据搬运", "", "状态：PASS。保留原测试全部数值断言；新增计数与输入 fixture 独立核对。", "",
             "这是从 11 个历史窗口的 warm state 开始执行第 12 个窗口；不是从初始预训练权重运行完整在线数据集。", "",
             "| 阶段 | 周期 | 按 15 ns 换算/ms | P4 命令 | RMW 读 | code 写 | residual 写 | cache 填充字 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in phases:
        lines.append(f"| {row['name']} | {row['cycles']} | {row['projected_time_ms_at_15ns']:.6f} | {row['p4_commands']} | {row['rmw_reads']} | {row['code_writes']} | {row['residual_writes']} | {row['cache_fill_words']} |")
    lines += ["", f"每次前向逻辑稠密项 {b*t*f*c}，非零项 {nonzero*c}；零特征比例 {1-nonzero/len(values):.4%}。这是输入稀疏度/操作数统计，不是相对稠密硬件实测加速比。",
              "", "## 使用边界", "",
              "- `frames.csv` 从 scheduler 首个特征握手到 logit 接受，逐样本逐时间步记录；不含 Conv1…Conv8。",
              "- 阶段计数含 TB 注入间隔，前向阶段含观察捕获；不是单独的最优硬件吞吐。", 
              "- `array_busy_cycles` 是阵列占用，不等于 480 个 PE 的实际运算利用率。", 
              "- 端口字节仅算 payload：特征波 128 bit、cache word 200 bit、RMW code/residual 各 16 bit。各接口边界不相加冒充外部 DDR 带宽。", 
              "- 15 ns 来自既有 OOC 核心工作点；本 TB 时钟 10 ns，报告以周期为权威。", 
              "- 更新后重算不是独立初始化的 learning-off 对照，不据此计算公平学习开关加速比。", 
              "", "复现命令与进程树资源采样见 `resources.json`，原始输出见 `verilator.log`。"]
    (args.result_dir / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "phase_cycles": [r["cycles"] for r in phases], "frames": len(frames)}))


if __name__ == "__main__":
    main()
