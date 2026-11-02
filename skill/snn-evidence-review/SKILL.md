---
name: snn-evidence-review
description: Review the FC/SRNN/SCNN fixed-profile SNN project's source closure, profile compatibility and evidence claims when preparing reproduction or competition materials.
---

# SNN固定画像与证据检查

优先读取项目`README.md`、`rtl/final_sources.f`、`software/autonomous_reconfiguration/catalog/hardware_profiles.json`及所评价版本的结果说明。

## 源码与配置

- 生产源由清单指定；`verification/experiments/`存在同名或派生模块，不能递归合并为一个生产设计。
- 确认105份SV和`rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem`可读。ROM以进程工作目录解析，独立运行目录需要相同相对布局。
- 检查画像选择时，先比较维度、通道、类别、空间尺寸与极性，再看归一化距离。距离低而结构不兼容的输入仍应拒绝。
- 固定配置对应FC `0x01`、SRNN `0x10`、SCNN `0x20`。画像选择不意味着任意深度、时间步、精度或神经元规模已可运行时搜索。

## 切换与学习

查看软件`runtime_bridge.py`的停止接收、排空、加载/配置、对齐、派发与恢复动作，以及RTL相应握手；dry-run动作列表只证明软件决策，不证明板卡执行。

学习路径按实际写集表述：FC W2、SRNN M2 Wout、SCNN classifier。更新发布和后续query可见性须由同一版本、同一协议的结果支撑。版本屏障不等于已实现任意中途回滚。

## 证据分层

每项数字标清对象、输入范围、版本和测量层级：软件预测、RTL数值一致性、周期、综合、OOC布线、IP消费、板级。不能把独立SRNN和SCNN候选的最佳指标拼成一个版本。不要把null改成0，也不要把日志没有错误当作自然完成。

周期×时钟周期只在明确计数区间内推导时间；需要端到端结论时，查主机准备、数据传输和板级调度是否实际计入。15 ns OOC约束不能证明板级I/O、DDR、CDC与复位均已收敛。

## 真实失败情形

1. 不兼容输入被最近邻误选：保留兼容过滤，使用现有SMNIST/CIFAR10-DVS拒绝样例检验。
2. XPM响应在背压期间被覆盖：项目曾保留此类局部失败并加入held-response处理。检查消费者未握手时响应能否保持，不能只看无背压用例。
3. 混用同名候选或独立结果：按源清单和实现版本定位，再解释结果，不能用一次局部PASS覆盖系统集成缺口。
4. 物理实现超时仍写PASS：缺最终DCP、终态报告或自然退出记录时保留未完成状态，不根据已完成布局推定布线通过。

## 输出与执行范围

输出问题、证据位置、影响和最小改动建议。只做用户要求的整理时，不自动启动EDA、远端任务或新算法实验；需要额外数据才能判断时指出缺口。复现检查应覆盖变更的实际失败模式，不新增无由的hash、冻结合同或重复门禁。
