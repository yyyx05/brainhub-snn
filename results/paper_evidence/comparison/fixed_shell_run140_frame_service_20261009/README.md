# 当前frame-service源码：固定容量三模式复用对照

更新：2026-10-09。RUN140三臂综合均PASS并完整回传；统一mask7复用RUN136。
比较脚本和层次核查均PASS，无新RTL修改或额外统一臂重跑。

## 核心结果

| 指标 | 统一设计 | FC单模式 | SRNN单模式 | SCNN单模式 | 三套之和 | 相对三套之和减少 | 相对最大单套增加 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUT | 180969 | 139076 | 145194 | 112337 | 396607 | **54.37%** | **24.64%** |
| FF | 80160 | 57399 | 60725 | 62373 | 180497 | **55.59%** | **28.52%** |
| BRAM36等效 | 514.5 | 427 | 474.5 | 447 | 1348.5 | **61.85%** | **8.43%** |
| DSP | 1196 | 1126 | 1162 | 1134 | 3422 | **65.05%** | **2.93%** |

减少=(三套之和−统一)/三套之和；增加=(统一−各指标最大单套)/最大单套。
最大单套按各资源分别选择，不是假设一个同时取各最大资源的真实设计。
BRAM36等效=RAMB36+RAMB18/2。LUT、FF的统一复用系数分别为2.192、2.252，
它们是静态资源比，不是加速比或能效。

这批最新源码结果替代旧RUN128对照作为当前架构证据：统一设计避免三套独立shell的静态重复，
但多拓扑支持不是零成本，相对最大单套资源仍有增加。不能只展示前一个分母。

## 匹配条件与层次

同Vivado2022.2、xc7vx690tffg1761-3、15 ns、OOC综合、rebuilt层次、Default指令、
`UNIFIED_USE_XPM_MEMORY`、learning开启。RUN140与RUN136的rtl实际diff为空；
FC/SRNN/SCNN只改变`TOPOLOGY_MASK=1/2/4`，不是重新裁剪成最小专用加速器。

四臂层次均实际出现唯一`unified_binary_p4_h40_array12_path_v1`和12个H40 tile。
mask7必需8/8 client；mask1必需3/3、禁用5/5；mask2必需4/4、禁用4/4；mask4必需3/3、禁用5/5。
Vivado的括号self行排除后计数，避免重复累计父层与子层资源。

WNS/WHS依次为统一+0.006/+0.133、FC+4.123/+0.134、SRNN+3.980/+0.145、SCNN+0.738/+0.133 ns。
三新臂实际退出0，错误/严重警告/黑盒0，普通日志Warning367/379/534保留；未发现OOM。
这是综合结果，不是三单模式布线通过，也不能混用RUN137布线资源计算收益。

## 证据与复现

- [本包结果与完整归档说明](../../physical/run140_frame_service_fixed_shell/README.md)；原始脚本/DCP/full归档留主工程
- [统一臂RUN136](../../physical/run136_scnn_frame_service_synth/README.md)
- [资源CSV](comparison.csv)、[逐项比较原始输出](comparison_raw.txt)
- [层次JSON](hierarchy_summary.json)、[逐client原始输出](hierarchy_summary_raw.txt)

汇总命令见`scripts/summarize_fixed_shell_comparison.py`与`summarize_fixed_shell_hierarchy.py`，
只解析归档，不执行EDA。精确输入为RUN136目录及RUN140结果内`fc_only_mask1/srnn_only_mask2/scnn_only_mask4`，
层次期望使用已有`vivado/ooc/fixed_shell_expected_clients_v1.json`。

## 主张边界

不代表三模型同时执行的吞吐等价，不是与最小专用实现或异工艺同行芯片的面积直接比较。
不含板级I/O、DDR带宽、实测功耗、Fmax或能效主张；普通告警仍保留。
RUN139学习关闭版setup失败未被此独立综合改写；可引用的学习支持成本仍为RUN136−RUN138的匹配综合增量。
RUN141私有输出学习器对照是另一项实验，不能把本表当作所有学习算术共享的证明。
