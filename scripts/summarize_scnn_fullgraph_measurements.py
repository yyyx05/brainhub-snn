"""Parse observed full-graph T2 cycles; never substitute classifier-only timing."""

import argparse
import csv
import json
import re
from pathlib import Path

from summarize_scnn_classifier_measurements import parse_rows


def summarize(text):
    marker = "SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS"
    if marker not in text or "%Error" in text or "Assertion failed" in text:
        raise ValueError("full-graph reference checks did not pass")
    layers = parse_rows(text, "SUPPLEMENT_FULLGRAPH_LAYER")
    steps = parse_rows(text, "SUPPLEMENT_FULLGRAPH_TIMESTEP")
    initialization = parse_rows(text, "SUPPLEMENT_FULLGRAPH_INITIALIZATION")
    if [(r["timestep"], r["layer"]) for r in layers] != [(t, l) for t in range(2) for l in range(1, 9)]:
        raise ValueError("expected exactly two sequential eight-layer passes")
    if [r["timestep"] for r in steps] != [0, 1] or len(initialization) != 1:
        raise ValueError("missing timestep or initialization measurements")
    existing_checks = [line for line in text.splitlines() if line.startswith(
        "SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_TIMESTEP_PASS ")]
    if len(existing_checks) != 2:
        raise ValueError("missing existing per-timestep reference PASS")
    previous = [0] * 8
    for t, line in enumerate(existing_checks):
        counts = [int(v) for v in re.search(r"requests=([\d+]+)", line)[1].split("+")]
        if len(counts) != 8:
            raise ValueError("incomplete convolution request list")
        for l in range(8):
            row = layers[t*8+l]
            if row["requests"] != counts[l]-previous[l] or row["responses"] != row["requests"]:
                raise ValueError("new monitor does not reconcile with original request counters")
        previous = counts
    for row in layers:
        row["stage"] = "Conv8+pool+classifier" if row["layer"] == 8 else f"Conv{row['layer']}+optional_pool"
        row["projected_ms_at_15ns"] = row["cycles"] * 15 / 1e6
        row["feature_mask_response_bytes"] = row["responses"] * 20
        row["feature_update_port_bytes"] = row["feature_update_words"] * 8
        row["conv_weight_fill_port_bytes"] = row["weight_fill_words"] * 25
        row["state_spill_restore_port_bytes"] = (row["state_read_words"]+row["state_write_words"])*40
        row["trace_spill_restore_port_bytes"] = (row["trace_read_words"]+row["trace_write_words"])*40
    for row in steps:
        measured = sum(r["cycles"] for r in layers if r["timestep"] == row["timestep"])
        row["driver_gap_cycles"] = row["cycles"] - measured
        if row["driver_gap_cycles"] < 0 or row["classifier_logits"] != row["timestep"]+1:
            raise ValueError("invalid timestep accounting")
    return {"status": "PASS", "sample": 0, "timesteps": 2,
            "initialization": initialization[0], "layers": layers, "steps": steps,
            "total_conv_feature_requests": sum(r["requests"] for r in layers),
            "forward_cycles_excluding_initialization": sum(r["cycles"] for r in steps),
            "scope": "One real DVS-Gesture sample, first two timesteps, learning disabled, reference-parity full graph; external feature and context/state service modeled by TB, not full dataset or board",
            "projection_clock_period_ns": 15}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = summarize((args.result_dir / "verilator_real_conv1_to_classifier_t2_same_array.log").read_text(encoding="utf-8-sig"))
    (args.result_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    with (args.result_dir / "layers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["layers"][0]))
        writer.writeheader()
        writer.writerows(result["layers"])
    cycles = result["forward_cycles_excluding_initialization"]
    lines = ["# SCNN 原始帧到 classifier：T2 全图测量", "", "状态：PASS。保留原有逐层数值、池化、状态/trace 恢复和累计 logit 检查；新增只读计数。", "",
             "输入为既有真实 DVS-Gesture sample 0 的前两个时间步。不是完整记录长度或完整测试集，learning-off。", "",
             "| 时间步 | 层段 | 周期 | 15 ns 换算/ms | 特征请求 | 卷积权重填充字 | 状态读/写字 | trace 读/写字 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["layers"]:
        lines.append(f"| {r['timestep']} | {r['stage']} | {r['cycles']} | {r['projected_ms_at_15ns']:.6f} | {r['requests']} | {r['weight_fill_words']} | {r['state_read_words']}/{r['state_write_words']} | {r['trace_read_words']}/{r['trace_write_words']} |")
    lines += ["", f"两个时间步完整前向窗口：{cycles:,} 周期，按 15 ns 换算 {cycles*15/1e6:.6f} ms；初始化单列为 {result['initialization']['cycles']:,} 周期。",
              f"八层卷积特征请求合计 {result['total_conv_feature_requests']:,}。以每层原始计数相加核对，不能遗漏 Conv8。", "",
              "## 边界与复现", "",
              "- 层段从调用配置启动到结果/释放完成，含 TB 提供的上下文装载与状态维护等待；不是纯 PE 计算延迟。",
              "- 最后一段包含 Conv8、池化和 classifier，没有将 classifier 周期重复相加。",
              "- 特征 response 是四个 40-bit mask；有效 mask 位属于 spike/value-bit 微操作，不直接称 Q8.8 MAC。",
              "- `weight_fill_words` 是本 TB 的卷积装载接口；classifier 初始发布经过另一通路，不能由此宣称统计了所有权重流量。",
              "- 状态/trace 每字 320 bit，卷积权重填充每字 200 bit，feature update 每字 64 bit；这些是接口有效载荷，不是 DDR/MIG 实测。",
              "- 15 ns 为既有 OOC 核心工作点投影；仿真 TB 的 10 ns 时钟不构成 100 MHz 物理时序声明。",
              "- 一个样本的两步窗口不构成稳态完整样本吞吐测量。原始完整记录长度、整网在线更新仍待补。",
              "", "命令与主机资源见 `resources.json`；原始日志包含所有计数和功能 PASS。"]
    (args.result_dir / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "total_conv_feature_requests", "forward_cycles_excluding_initialization")}))


if __name__ == "__main__":
    main()
