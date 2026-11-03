# 复现与评价说明

## 软件入口

在工程根目录使用Python 3.11运行README中的JSON指纹命令。标准库即可完成画像匹配并生成dry-run动作记录。交互页面`demo/decision_replay.html`回放已有软件记录，便于观察任务变化与配置过程。

## 按层准备环境

| 目的 | 环境与输入 |
| --- | --- |
| SHD H5指纹 | h5py、SHD H5 |
| 软件整数模型评价 | 对应脚本的numpy/torch依赖、checkpoint、数据及数值格式 |
| RTL数值仿真 | Verilator 5.048、C++工具链、所选测试平台、权重与输入 |
| FPGA OOC综合/实现 | Vivado 2022.2、Virtex-7器件支持与许可；XPM随工具提供 |
| 平台集成 | 实际板卡的时钟、复位、DDR、主机接口与约束 |

可选Python库列于`requirements_optional.txt`。执行模型评价前按选定脚本配置checkpoint、样本索引和support/query顺序。

公开版按[资源说明](public_release_scope.md)准备模型与输入。JSON指纹入口及内部算法ROM已随仓库提供。

## 生产闭包与原生IP

`rtl/final_sources.f`列出105份生产SV。独立候选位于`verification/experiments/`，分别使用对应源清单。ROM为`rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem`，按工具进程工作目录解析，因此从工程根目录启动。

```console
vivado -mode batch -source vivado/ooc/run_unified_h40_final_composition_v1_synth.tcl -tclargs <工程绝对目录> 1
```

脚本读取`rtl/`生产闭包，使用`xc7vx690tffg1761-3`和15 ns OOC约束。

R2项目IP保存在`ip/unified_snn_native/`，包含component.xml、核心源、连线wrapper和ROM。已有Vivado工程将本包`ip/`设为repository根目录，通过`ip/unified_snn_native_ip/register_repository.tcl`注册并准备XPM与ROM；平台层再接入主机总线和外部存储。

## 实验条件与指标解释

| 结果 | 评价范围 |
| --- | --- |
| 任务选择 | 三个固定硬件画像；先检查结构兼容性，再比较指纹距离。距离表示任务匹配程度 |
| 自动学习策略 | `--learning auto`采用工程阈值，后续按实际分布变化标定 |
| 软件准确率 | 对应任务、checkpoint和样本范围；SCNN为固定264样本子集 |
| 原生接口短桥 | learning-off合成小几何，验证切换、排空与状态握手 |
| RUN137资源与时序 | 15 ns内部核心OOC实现；板级I/O、DDR和端到端指标由平台集成测量 |
| 周期投影 | 仿真计数区间乘以时钟周期；主机准备与传输另计 |
| 优化候选 | 各候选按自己的功能、输入和实现版本评价 |

精选报告入口为`results/README.md`；完整历史DCP、波形和实验输入保存在研究原件。复算历史脚本时按其引用补齐对应文件。材料整理沿用既有实验结果，本轮检查覆盖源码一致性、语法、配置路径、文档链接和归档可读性。
