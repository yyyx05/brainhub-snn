---
name: snn-service-optimization
description: Optimize SNN RTL service latency under existing arithmetic and memory budgets, using elastic handshakes, row-retirement overlap and held-response RAM semantics. Use for FC/SRNN/SCNN gradient, refill or backpressure bottlenecks.
---

# SNN服务与存储供给优化

围绕一个已观察到的等待源提出最小机制改动，用同事务的配对结果判断收益。适用于梯度归约器、cache-refill publisher与响应背压下的RAM端口复用。

## 输入与定位

读取目标版本的源清单、模块RTL、调用方握手、现有测试平台和结果计数。先给出供给、消费、等待、完成四类事件的时序关系，确认瓶颈属于数据依赖、调度空隙还是外部服务限速。

本项目生产入口为`rtl/final_sources.f`（105份SV及学习ROM）；隔离实验可能使用113项wrapper清单，按各自sources.f分析。画像切换任务先检查`hardware_profiles.json`兼容维度，再看指纹距离；固定画像为FC 0x01、SRNN 0x10、SCNN 0x20。学习写集分别为W2、M2 Wout、classifier。选择候选时沿用对应源闭包、数值位宽、RNE/饱和、single-credit、提交和排空语义。

## 按瓶颈选择机制

- **结果交接阻塞下一贡献：** 画出累加槽与结果槽的占用状态。若已有两个寄存器，可以让它们独立valid/ready推进；检查同拍结果消费、累加结果转移及新贡献接收的优先级。先覆盖连续流、慢供给、周期背压、双槽满、尾标记、溢出、abort/reset。
- **行间补给空隙：** 以真实行退休作为下一行首读触发点；核对末行、无效下一行、abort、响应tag和单未决credit。比较服务轨迹时令两臂每例从相同相位开始。若综合多出DSP，沿地址表达式追查重复乘加，优先复用已知索引条件下的已有运算。
- **RAM响应被阻塞：** 由已接受且未消费的请求记录mode/pair/port ownership；所有可能更新该输出寄存器的写操作都参与冲突判断。核对所选原语的read latency、write mode、enable语义。对本项目XPM READ_FIRST/latency1，A口写也更新输出；行为模型应体现这一点。定向覆盖无新请求、切换pair、同址冲突、非占用端口、消费同拍替换与复位。

## 最小验证与判断

1. 复用现有oracle，保留负值、极值、尾部与更新可见性；在相同输入和服务条件下验证地址、tag、数值、更新次序与完成事务。
2. 先用能揭示改动失败模式的局部测试。局部成立后，按任务授权执行一次对应整核协议，保留初始化、学习提交和后续query等原计时区间。
3. 分别报告完整区间周期、非重叠分账和嵌套计数。忙碌/背压计数上升可以伴随总周期下降，嵌套区间不求和。供给受限时的零收益也是有效结果。
4. 综合与物理按同器件、时钟和策略比较LUT/FF/BRAM/DSP及WNS。将周期收益与硬件代价并列；不同候选分别报告，合并后的性能以合并版本测量为准。

周期乘以约束时钟得到所计区间的时间投影；板级端到端结果由包含主机准备、传输与板级调度的实测支持。只读复盘沿用已有记录，新EDA按本次任务范围执行。

## 失效情形与调整

- 外部供给已限速：流水重叠可能零收益，转向实际限速段。
- 两臂相位不同：先修正测试条件，再判断候选，不按结果好坏删样本。
- 行为RAM与原厂语义不同：引入原厂模型定向反例，修正模型和持有响应的端口保护。
- 工具语法/参数导致未运行：记录工具失败，改正支持的调用；DUT判断来自真正运行的测试。
- 综合资源下降而布线恶化：根据路径和拥塞回看结构；超时记录保持未完成状态。

## 产物

提交一份瓶颈说明、机制差异、最小测试与反例、同事务周期表、资源时序代价，以及可直接定位的源码/日志路径。现有实验复盘见[三个案例](references/engineering_cases.md)，可改写调用示例见[提示语](references/prompts.md)。这些历史结果是技能设计依据；新一次使用的收益由新目标的实际验证记录说明。
