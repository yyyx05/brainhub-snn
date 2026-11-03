# 三个SNN工程优化案例

本复盘连接研发难题、机制修改和既有测量。路径相对于原始完整工程；随包精简结果位于`results/optimization_cases/`。三个候选独立评价，所列时序为内部OOC实现。

## SRNN：复用两级寄存器，重叠梯度归约与结果交接

**瓶颈：**结果等待下游期间，已有累加寄存器闲置；串行交接扩大了学习服务时间。

**结构变化：**把原128 bit累加寄存器与64 bit舍入结果寄存器组织为可独立停顿的两级弹性流水，允许结果退休、后继结果转移和新贡献接收同拍发生。端口、算术位宽、RNE、M2=512及单未决RMW保持。

**实验范围：**8条原代表序列、588步闭环；独立优化候选。物理为xc7vx690tffg1761-3、15 ns内部OOC同配方布线。

**局部观察：**8种batch×6种供给/背压，共48组配对；连续B1的20个梯度41→22拍，每三拍接受一个结果时61→61拍。

| 指标 | 数值（数组按原版→候选，三臂按原版→行退休→地址复用） |
|---|---|
| `cold_cycles` | [18924537, 17958877] |
| `cycles_saved` | 965660 |
| `reduction_percent` | 5.102688 |
| `learning_busy_cycles` | [9700398, 8734738] |
| `gradient_stalled_cycles` | [3050709, 3698172] |
| `gradient_inputs_outputs` | [3010560, 3010560] |
| `code_writes` | 481275 |
| `residual_writes` | 1017020 |
| `commits` | 588 |
| `steps_improved` | 572 |
| `steps_same` | 16 |
| `steps_worse` | 0 |
| `post_route_lut` | [194080, 194861] |
| `post_route_ff` | [80443, 80439] |
| `bram36_equivalent` | [594.5, 594.5] |
| `dsp` | [1196, 1196] |
| `post_route_wns_ns` | [0.151, 0.075] |
| `post_route_whs_ns` | [0.077, 0.078] |

**纠错与经验：**梯度背压计数增加647,463拍，但冷总周期下降965,660拍；判断收益应看完成相同事务的完整区间。布线LUT增加781、setup裕量减少76 ps，需要同时报告代价。相对更早正常RAM缓冲的28.923687%是组合端点收益，归约器独立增量是5.102688%。

**原始依据：**

- `results/paper_evidence/supplement_20261015/performance_optimization/srnn_gradient_elastic_core_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/srnn_gradient_elastic_core_r1/analysis/analysis.json`
- `results/paper_evidence/supplement_20261015/performance_optimization/gradient_elastic_local_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/gradient_elastic_impl_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/gradient_elastic_impl_r1/derived_metrics.json`
- `verification/experiments/srnn_gradient_elastic_core_r1/rtl/learning/unified_srnn_tp_wout_gradient_reducer.sv`

## SCNN：行退休首读重叠，再复用地址运算

**瓶颈：**classifier cache-refill publisher串行完成行退休和下一行首读，行间产生控制空隙。

**结构变化：**在真实行退休当拍启动下一合法行首读，沿用既有行缓冲与单未决credit。随后在退休子字索引固定为11的条件下复用已有地址加2通路，替代独立下一行地址运算。

**实验范围：**CPU显式Conv7特征、52个Conv8尾部描述符、3个冷B4T12学习窗口及query4；raw_source_active=0。独立publisher候选。

**局部观察：**512输入/12输出连续服务64,563→59,979拍；延迟3拍加周期背压202,754→202,754拍。

| 指标 | 数值（数组按原版→候选，三臂按原版→行退休→地址复用） |
|---|---|
| `protocol_cycles` | [220839641, 210661793] |
| `cycles_saved` | 10177848 |
| `reduction_percent` | 4.608705 |
| `setup_cycles` | [28431, 26561] |
| `cold_window_1_cycles` | [67977761, 64846547] |
| `cold_window_2_cycles` | [67996339, 64865125] |
| `cold_window_3_cycles` | [68004248, 64873034] |
| `query_cycles` | [16832862, 16050526] |
| `frames_improved` | 52 |
| `rmw` | 60273 |
| `commits` | 3 |
| `publications` | 7568 |
| `code_changes` | 662 |
| `fresh_reads` | 9437184 |
| `synth_arms` | ["original", "row_turnover", "address_reuse"] |
| `synth_lut` | [208751, 208002, 208081] |
| `synth_ff` | [80448, 80389, 80391] |
| `synth_dsp` | [1196, 1202, 1196] |
| `synth_wns_ns` | [-0.55, -0.708, -0.741] |
| `final_physical_pass` | false |
| `final_physical_exit` | null |
| `physical_timeout_seconds` | 7200 |
| `phase_aligned_cases` | 99 |
| `phase_aligned_improved` | 49 |
| `phase_aligned_same` | 50 |
| `phase_aligned_worse` | 0 |

**纠错与经验：**初次连续全局tick使两臂后续用例起始相位不同，出现53改善/37持平/9变慢；只修测试起点、保持RTL后，同相位99例49改善/50持平/0变慢。行退休候选新增6 DSP，地址复用恢复至1196但综合WNS为-0.741 ns。前一物理尝试level-6拥塞，复用版本后续补跑7200秒超时；现有结果支持周期收益和综合资源回收，最终物理收敛继续优化。

**原始依据：**

- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_turnover_core_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_turnover_core_r1/analysis/analysis.json`
- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_reuse_core_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_reuse_synth_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_turnover_impl_r1/RESULT.md`
- `results/paper_evidence/supplement_20261015/performance_optimization/scnn_row_reuse_impl_r2/RESULT.md`
- `verification/experiments/scnn_refill_phase_aligned_r1/RESULT.md`

## 存储供给：用原厂RAM反例修复背压响应保持

**瓶颈：**SCNN同一行读取复用释放端口后，填充冲突检查只参考当前请求。旧响应被背压时，即使没有新请求或新请求切换bank pair，旧请求仍占用A口输出。行为RAM写时保持输出，掩盖了真实READ_FIRST语义。

**结构变化：**由已接受且未消费请求的mode/pair/coalesced状态形成held_port_a_banks，并纳入fill冲突；FC/SRNN/M2保留四个A口、非合并SCNN保留旧pair两个A口、仅B口合并读不占A口。行为RAM写入时输出目标行旧值，与READ_LATENCY_A=1、READ_FIRST一致。

**实验范围：**40 lanes，4 banks，每bank 200 bit×578 rows；Vivado/XSim 2022.2未修改原厂XPM与行为RAM，同一77场景测试平台。

**局部观察：**四个定向场景区分两处覆盖、同pair既有保护和B口合并安全例；修复后的行为RAM与原厂XPM各77场景、4170检查通过。

| 指标 | 数值（数组按原版→候选，三臂按原版→行退休→地址复用） |
|---|---|
| `counterexample_cases` | 4 |
| `counterexample_vulnerable` | 2 |
| `counterexample_checks` | 182 |
| `term0_no_new_request` | [101, 303] |
| `term0_switch_pair` | [101, -404] |
| `repaired_cases_per_arm` | 77 |
| `repaired_checks_per_arm` | 4170 |
| `reads_per_arm` | 124 |
| `writes_per_arm` | 80 |
| `blocked_per_arm` | 132 |
| `held_cycles_per_arm` | 239 |
| `simultaneous_per_arm` | 28 |
| `resets_per_arm` | 53 |
| `hardware_cycles_speedup` | null |
| `hardware_resource_delta` | null |
| `hardware_timing_delta` | null |

**纠错与经验：**第一次Verilator在原厂deassign语法处停止；第二次XSim的--mt 1参数错误；第三次改--mt off后真实复现101→303及101→-404。保留原厂模型，修正执行选项和候选握手，明确区分工具失败与DUT反例。先用真实存储语义验证端口复用，再扩大到预取与整核。

**原始依据：**

- `verification/experiments/scnn_xpm_hold_r1/RESULT.md`
- `verification/experiments/scnn_xpm_hold_r1/runs/r3/run.log`
- `verification/experiments/scnn_held_response_r1/RESULT.md`
- `verification/experiments/scnn_held_response_r1/runs/r1/behavioral/run.log`
- `verification/experiments/scnn_held_response_r1/runs/r1/xpm/run.log`
- `verification/experiments/scnn_held_response_r1/rtl/scnn_held_response_weight_store_v1.sv`
