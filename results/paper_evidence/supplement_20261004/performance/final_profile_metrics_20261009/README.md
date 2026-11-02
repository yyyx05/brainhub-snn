# 最终固定画像指标：本地有界汇总（2026-10-09）

仅解析已有本地日志/JSON/时序报告，不调用 SSH、EDA、GPU 或仿真，不重跑实验，不修改 RTL/TB/共享文档。
输出不是新实验 PASS，也不是当前 live RTL 的重新资格认证；来源关联沿用归档说明，不新建 hash/基线/门禁。

## 复现

在项目根执行（仅 Python 标准库）：

```powershell
python -B scripts/summarize_final_profile_metrics.py
python -B -m unittest discover -s tests -p test_final_profile_metrics.py -v
```

默认写本目录；可用 --output-dir 指定本目录内的子目录，不能写其他共享路径。
Windows文件读写自动处理长路径、UNC及已有扩展前缀；JSON来源字段和README链接仍保持原相对路径，不写入扩展前缀。
metrics.json 是详细版本，完整保留 input/execution/counters/commits/ddr_bank_reads、原始样本ID及全部来源，不因CSV精简而删除。
profiles.csv 是论文表格/Excel用的精简版本：固定核心标量字段，另从timing提取period_ns，不把嵌套JSON塞进单元格。
activity.csv 只保留标量元数据及total_前缀总计数列；逐tile和mode明细留在metrics.json。stages.csv、software_accuracy.csv、missing_counters.csv同样只含标量字段，可独立读取。
JSON null / CSV 空字段表示未测量，不等于零；stages 中父区间与子区间不可重复相加。

## 主工作负载

| 记录 | B / T | 实测 cycles | 区间时间投影 ms | 仿真主机墙钟 s |
|---|---|---:|---:|---:|
| FC_support160_cache_r3_off | 16 / 99 | 10147245 | 152.208675 | 300.510 |
| FC_support160_cache_r3_on | 16 / 99 | 87013885 | 1305.208275 | 2239.830 |
| SRNN_raw700_M2_cache_r3_off | 1 / 8 | 12104533 | 181.567995 | 352.843 |
| SRNN_raw700_M2_cache_r3_on | 1 / 8 | 12250769 | 183.761535 | 355.891 |
| SCNN_real_final_B4T12_query4 | 4 / 12 | 2038360664 | 30575.409960 | 70133.664 |
| FC_current_source_B16T99_off | 16 / 99 | 1013344 | 15.200160 | 67.913 |
| FC_current_source_B16T99_on | 16 / 99 | 8714352 | 130.715280 | 265.412 |
| SRNN_current_source_B1T73_native_off | 1 / 73 | 110452102 | 1656.781530 | 3012.210 |
| SRNN_current_source_B1T73_native_on | 1 / 73 | 112087773 | 1681.316595 | 3051.002 |
| SCNN_current_source_B1T20_off | 1 / 20 | 781657379 | 11724.860685 | 27403.342 |
| SCNN_current_source_B1T20_on | 1 / 20 | 784525456 | 11767.881840 | 27487.712 |

### 输入、装载与学习边界

- 历史 FC cache_r3：2312→160→10，cache_r3_support160_r2 两臂，同一 exe，原 seed38472/+2 像素 support160 顺序，10×B16T99。保留事件重数和 padding，不截断为49槽；未覆盖为当前源码新记录。
- 历史 FC 区间排除首次装载/补载1542534 cycles 与状态初始化641 cycles，包含后续补载和批间初始化。含初始化时间另列 JSON/CSV；只提供该固定 support 区间的等效速率，不称板级或稳态吞吐。
- 历史 FC W2-only：开启30次 code 写、397621次 residual 写、990 commit，最终26地址变化；两臂各自匹配整数参考，t49首次可见。learning_cycles 是末样本后等待；refill_cycles 含首次补载，不能整体再加进区间。
- 历史 SRNN cache_r3：700→450循环骨干→450→256→20 M2，同一 production final top，官方 sample0 前8步，B1；实际 DUT spike 驱动 M2/循环反馈，不拼接两个独立 DUT 的周期。Wout-only，梯度仍固定 /64，不是B64准确率。
- 历史 SRNN 开启1680 code/4940 residual 写、8 commit/detach；cycles 从 initialized_cycle 后起算，包含工作集补载和 TB 读回检查。store=962560 是存储检查项数，不是访存握手。初次装载和各阶段时间未打印，不能恢复或外推完整 SHD samples/s。
- SCNN：真实最终组合顶层，seed38472、subject29→28、jitter1、support IDs 0..3，三个B4T4连续窗口构成B4T12，再对 query22 执行B1T4，即48训练帧+4查询帧，不是52个完整样本。模型为八卷积VGG9+2048→11，仅classifier可写。
- SCNN 三窗 residual 变化20041/20011/20221，code变化0/0/662，3 commit；地址18433由870变869，query current/logit各4组变化。fresh_reads=36868608 仅查询重读；ddr_bank_reads 是各bank整个执行的实际TB握手，不能用fresh_reads替代全程流量或称板级DDR带宽。

### 当前源码 FC 完整给定样本

独立 server_pair_r1 两臂均PASS，同一新编译exe、105生产SV加测试store和monitor。真实 first-saccade 输入为16×99×2312，有效长度94–99；全部99步含padding执行，不强补T100，不代表full Test或板级。旧cache_r3行及旧活动记录原样保留，数值偶然相同也不替代新来源。

| 当前 FC 臂 | runtime cycles | 15 ns投影 samples/s |
|---|---:|---:|
| off | 1013344 | 1052.6204987 |
| on | 8714352 | 122.4034405 |

上述full_sample_throughput_samples_s字段仅表示16个给定完整T99样本的15 ns区间投影，不是测得的板级吞吐、独立单样本时延或最优调度。runtime排除参数装载743040、首次补载799494、state/trace初始化641，以及初始学习配置（off 0/on 2）cycles；开启臂仍含逐commit保守W2补载。

| 被动计数 | off | on |
|---|---:|---:|
| 学习工作忙并集时钟 | 0 | 7900295 |
| 其中与前向重叠 | 0 | 612216 |
| 其中非前向/补载的忙时钟 | 0 | 7288079 |
| learning读请求握手 | 0 | 32243 |
| 16-bit residual写使能握手 | 0 | 32243 |
| 16-bit code写使能握手 | 0 | 0 |
| FC窗口commit握手 | 0 | 99 |

学习busy是frontend/reducer/update忙信号的并集，不是各项之和；非重叠忙时钟也不是反事实纯学习因果时延。末样本后等待7288871与非重叠忙7288079不是同一指标，612216重叠忙不可再加到前向周期。两臂差7701008 cycles还含调度及W2补载，不称纯学习成本。
commit的observation/update/done/valid/accept边沿及其差值保留在JSON与stages.csv，起止差不加1；只是FC学习窗握手，不是平台发布。缺失的last_write等时间戳转为null，off没有学习窗，不补成0时延。本批on有99 commit但code写为0，不继承support160的30次code写或新权重可见性结论。
各stage总线握手、mask有效payload位数保留；store/维护总线不是板级DDR字数，off logit是只读tap而非learner握手。当前12tile活动仅解析同一on/off.log，monitor包含初始化，不能套runtime分母。仿真沿用行为存储，不定义XPM宏，不是XPM原语仿真或新STA。

### 当前源码 SRNN 完整 native sample0

独立server_pair_r1的launcher/build/off/on退出均0；同一exe、105生产SV加验证store/monitor，原fixture从完整回传归档读取，不按服务器cwd去匹配本地工作区。两臂完整日志以现有run_srnn_profile_measurement.parse_log重解析，并核对原measurements/activity JSON和73步数值完结。
输入是未偏移的官方sample0，B1 native T73，raw700→循环骨干→M2在同一DUT执行；不是旧T8前缀，也不是官方firstbatch B64 padded T89。Wout-only和固定梯度/64保持，即使执行B1也不改为/1。单个给定完整样本通过，不代表full Test准确率、B64吞吐或板测。

| 当前 SRNN 臂 | sample service cycles | 15 ns投影 samples/s |
|---|---:|---:|
| off | 110452102 | 0.603579881772 |
| on | 112087773 | 0.594771979872 |

服务区间=终态end−initialized−TB检查仿真周期；含全部骨干/M2工作集refill和TB调度，不含初始120426 cycles（reset6+config2+参数装载120321+state初始化97）。TB存储检查8783360项使用层次peek，不推进仿真时钟，检查周期为0；主机检查耗时未知null，仿真墙钟不能当硬件时间。

| 真实阶段 / 信号计数 | off | on |
|---|---:|---:|
| 骨干前馈补载阶段 | 53042019 | 53042019 |
| 骨干循环补载阶段 | 34098519 | 34098519 |
| 骨干阶段 | 124076 | 124076 |
| M2含准备/前向/学习阶段 | 23187488 | 24823159 |
| learner busy高时钟 | 0 | 1906488 |
| learner busy且M2 engine空闲且处于M2阶段 | 0 | 1635671 |
| 骨干参数读握手 | 37777500 | 37777500 |
| M2补载读握手 | 8783360 | 8783360 |
| 学习读握手 | 0 | 145500 |
| code写使能握手 | 0 | 68040 |
| residual写使能握手 | 0 | 145500 |
| learner commit握手 | 0 | 73 |

stage区间连续分账，hardware的m2_prepare/m2_forward/pure_learning_busy是重叠子区间，不可再加到stage总数；on臂每步m2_forward与pure_learning_busy边沿重叠1周期，73步合计73周期，不能相加为互斥耗时。保留原始边沿、区间值和总周期，不修正为互斥。1906488 learner忙时钟中1635671满足上述predicate，余270817只表示不满足predicate，不能直接命名为前向重叠或额外学习耗时。两臂差1635671恰与predicate计数相同，仍不冒充反事实纯学习时延。
73个commit脉冲时间戳、每步准备/前向/学习忙区间、全程真实accepted/stalled通道握手均保留在JSON/stages.csv；没有独立commit或平台发布起止时延，留null。bus握手以通道分列，同一传输可经过多个边界，不跨通道加成DDR总流量；未独立测payload字节。fixture中的zero_skips/saturations是软件期望，不提升为新RTL计数。
本轮12tile只用同臂run.log，monitor覆盖reset释放后的初始化/补载/学习/调度时钟：off110572522、on112208193；与服务区间分母不同。行为存储仿真不定义XPM宏，不是XPM原语仿真、新STA或板测。

### SCNN 真实阶段

stages.csv 由 SCNN_TEMPORAL_LAYER 和 FRAME_STATUS 汇总416个descriptor与52帧。逐层周期是真实日志值，不借用旧T2实验。FRAME_PASS的八层累计requests/contexts/scalars按相邻帧差分，还原每descriptor的计数，不能当DDR字节或模型MAC数。
Conv8区间含pool+classifier；帧末还可能等待学习、发布及TB检查，frame_minus_layers 不标纯学习耗时。window 行用实际开始时间戳差，包含中间控制/设置。初始化71595 cycles含初始classifier publication 1892字，不能叫整模型独立装载时延。

| 互斥记账项 | cycles |
|---|---:|
| Conv1 | 12059944 |
| Conv2 | 43232176 |
| Conv3 | 93481960 |
| Conv4 | 177454056 |
| Conv5 | 268651760 |
| Conv6 | 529700080 |
| Conv7 | 453758864 |
| Conv8+pool+classifier | 454586032 |
| 全部frame内未分段余量 | 5364197 |
| 初始设置 | 71595 |
| frame外未归因余量 | 0 |

### 当前源码 SCNN 原始 sample0 完整T20

server_pair_r2的launcher/elab/build/off/on退出均0，同一新exe、105生产SV加验证store/TB/12tile被动monitor。现有run_scnn_profile_measurement.parse_result只读重解析两臂原始日志，与原summary及固定reference核对；不调用producer summarize、不重写主证据。elab/build各18条普通Verilator告警保留，结构阻断项为0；准备文件中的估时/not_measured是执行前元数据，不当作终态。
这是原始clean sample0，B1连续五个既有T4窗口构成完整T20，仍用固定640缩放；窗口间不reset/reload，没有fresh query，不是旧shifted 52帧B4T12+query4。每臂20帧/160 descriptor、40960 feature、220 current/logit和480 probe检查通过。on另有40960 classifier trace值检查、85600 RMW、0 code写、85600 residual写和5 commit，不等于全Test准确率或已证明适应收益。

| 当前 SCNN 臂 | runtime cycles | 15 ns投影 samples/s |
|---|---:|---:|
| off | 781657379 | 0.085288859874 |
| on | 784525456 | 0.084977059899 |

两臂均排除71594 cycles setup（reset释放至首个request，含初始classifier publication）。包含setup总周期off781728973/on784597050另列JSON/CSV。模型/oracle readmemh和165次golden检查在仿真中不推进时钟，golden_cycles=0；host_file_load_seconds/host_golden_seconds仍null，不能称0主机成本、完整模型DMA装载或板级吞吐。原报告的10ns仿真比率只保留在source_throughput，统一表只填RUN137关联的15ns投影，不是新STA。

| 被动计数（按各自范围） | off | on |
|---|---:|---:|
| runtime梯度/update busy时钟 | 0 | 611436 |
| 其中排除forward/publication重叠 | 0 | 611436 |
| runtime publisher busy时钟 | 0 | 357940 |
| forward/learning重叠时钟 | 0 | 0 |
| learning/publication重叠时钟 | 0 | 0 |
| 全监测窗口native参数读握手 | 184365568 | 184478208 |
| 全监测窗口learning读握手 | 0 | 85600 |
| code写使能握手 | 0 | 0 |
| residual写使能握手 | 0 | 85600 |
| 320-bit state restore行 | 1065400 | 1065400 |
| 320-bit trace restore行 | 1065400 | 1065400 |
| 320-bit state spill行 | 1065400 | 1065400 |
| 320-bit trace spill行 | 1065400 | 1065400 |

初始publication两臂均1892个原生fill握手字；on后续五次共9460字，总11352字，off总1892字。真实状态/trace搬运每行320bit=20个16bit标量=40byte；restore mask计数及各DDR-model bank read/response、code/residual有效写payload分列保留，不加address/tag/control，也不称板级DDR带宽。traffic为reset释放后的全窗口计数，active为首request后的sample计数，不能混用区间。store_loads=0是monitor打印常量，不升级为被动端口测量。
descriptor/Conv8/pool/classifier控制区间、frame/window父区间互相包含，不能相加。pool_control含Conv8 writeback收集，classifier_execute不含tail capture；不能把它们拆成互斥卷积/池化/分类器耗时。sample周期可以用descriptor+frame余量+frame外余量分账，但busy/控制/提交发布计数不能再加进去。
on有91393段碎片化学习忙区间，按五窗汇总段数、busy时钟、first_start/last_end及最小/最大段长；首尾范围含空隙，不伪装为连续时延。原始逐段learning_start/interval仍在本轮run_on.log/summary，已全部重解析核对，不在统一表重复展开九万行。learning_exclusive611436只是指定predicate忙时钟，不是所有学习支持或反事实纯学习成本；两臂差2868077 cycles也不是纯学习耗时。
五个commit_pending_to_pulse区间和backend publish request-to-done边沿均保留，前者包含发布与drain，不能与publication busy再加成独立时延。端到端平台发布/板级DDR时间仍null。两臂自己12tile monitor覆盖初始化至finish，off781728973/on784597050，不套runtime分母；不继承旧T2或52帧活动记录。

## 时序投影范围

投影公式为 cycles×period_ns/10^6 ms，独立于 simulator walltime；不是仿真器输出的虚拟时间或板测硬件时间。
历史FC support160/SRNN前8步仅关联cache_r3快照的RUN131 15 ns OOC核心报告；不能称为后来frame-service live源码PPA。旧SCNN B4T12 R2、新FC B16T99、SRNN native B1T73及新SCNN B1T20关联RUN136 exact-DCP的RUN137 15 ns OOC核心报告，仍只是区间投影。
没有移植旧V2、RUN125、旧单模式或私有学习器资源百分比，没有基于裕量重新估算Fmax。学习关闭是同一exe的运行时开关，不等于ENABLE_ONLINE_LEARNING=0物理裁剪。RUN139 R2/R3历史负setup保留；2026-10-10 R5学习关和RUN142私有臂已各自15 ns OOC实际布线闭合，性能计数不改。R5与RUN137策略不同，不称严格同recipe学习支持布线成本；RUN137/RUN142另有匹配布线资源表。
所有行保留普通告警、未约束板级I/O/DDR、无板测时间/功耗边界；历史短SRNN/SCNN窗口的full_sample_throughput始终留空，新FC/SRNN/SCNN完整给定样本仅填投影。

## lane / P4

七项历史记录、两项当前FC、两项当前SRNN及两项当前SCNN均重新解析各自被动bind日志，并与对应JSON核对。当前记录仅关联各自本轮主记录；历史监测不继承为当前源码，旧SCNN 52帧活动缺口也不借新T20补填。
全窗口lane分母=480×monitor cycles；P4分母=1920×monitor cycles；发射内lane分母=40×contributions；lane内source分母=4×lane_fires。使能项含零权重，不是非零MAC/DSP/FLOPs。
互斥桶分母=12×cycles，优先级abort→compute→issue_wait→region_wait→observation_wait→fill→internal→idle。internal/idle不是完整瓶颈因果解释。

| 独立记录 | monitor cycles | 全窗口lane % | 全窗口P4 % | 发射内lane % |
|---|---:|---:|---:|---:|
| fc_b16t99_on | 10257529 | 0.06472 | 0.01903 | 95.2830 |
| fc_b16t99_off | 2556519 | 0.25966 | 0.07635 | 95.2830 |
| srnn_raw700_b2t8 | 1222051 | 0.24349 | 0.08139 | 93.7500 |
| srnn_m2_observed_on | 2218337 | 0.18080 | 0.00530 | 88.3147 |
| srnn_m2_observed_off | 481697 | 0.83261 | 0.02442 | 88.3147 |
| scnn_graph_b1t2_r2 | 23911655 | 2.71698 | 0.12288 | 24.2253 |
| scnn_classifier_cold_b1t2 | 472477 | 0.02117 | 0.00689 | 2.5000 |
| fc_current_source_b16t99_off | 2556519 | 0.25966 | 0.07635 | 95.2830 |
| fc_current_source_b16t99_on | 10257529 | 0.06472 | 0.01903 | 95.2830 |
| srnn_current_source_b1t73_native_off | 110572522 | 0.02793 | 0.00361 | 90.4519 |
| srnn_current_source_b1t73_native_on | 112208193 | 0.02753 | 0.00355 | 90.4519 |
| scnn_current_source_b1t20_off | 781728973 | 0.83108 | 0.04800 | 24.2250 |
| scnn_current_source_b1t20_on | 784597050 | 0.82804 | 0.04783 | 24.2250 |

## 软件准确率与RTL一致性分列

主表是有界RTL数值/事务一致性，不填写数据集RTL准确率。下表只读取已完成软件报告，不运行CPU/GPU推理。

| 拓扑 | 软件臂 | 正确/样本 | % |
|---|---|---:|---:|
| FC | integer_raw24 | 9537/10000 | 95.370000 |
| FC | integer_q6 | 9537/10000 | 95.370000 |
| FC | same_weight_fp32_leak098 | 9539/10000 | 95.390000 |
| FC | same_weight_fp32_fixed_leak | 9539/10000 | 95.390000 |
| SRNN | full_integer | 1822/2264 | 80.477032 |
| SRNN | formal_fp32 | 1855/2264 | 81.934629 |
| SCNN | full_integer | 250/264 | 94.696970 |
| SCNN | same_banks_fp32 | 251/264 | 95.075758 |

FC同量化权重float不是原始FP32 checkpoint；SRNN完整整数不是仅M2权重量化；SCNN是固定264×20子集的严格同bank对照，不是完整DVS Test。软件输入范围也不同于主表在线窗口。

## 明确缺失的计数

以下是现有日志的精确缺口，不是假设为0，也不授权补跑实验。

- `FC_support160_cache_r3_off, FC_support160_cache_r3_on`：`pure_learning_active_cycles/commit_cycles`。只测末样本后的等待，学习与观察回放可重叠，未独立打印学习活跃/提交周期。
- `FC_support160_cache_r3_off, FC_support160_cache_r3_on`：`lane_fires/enabled_terms/exclusive_tile_buckets`。该support160执行未挂活动监测；B16T99是另一条独立执行。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`initial_parameter_load_cycles/initial_state_cycles`。initialized_cycle没有打印，无法恢复已排除的初始设置耗时。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`backbone_forward_cycles/M2_forward_cycles/refill_cycles/learning_cycles/commit_cycles`。只打印合并的每步累计周期，没有分段起止计数。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`forward_read_handshakes/learning_read_handshakes/refill_payload_bytes`。store=962560是TB参数检查项，不是总线流量。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`zero_skip_count/saturation_count`。参考JSON有软件期望，但当前RTL终态未打印这两个计数，不冒充实测。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`lane_fires/enabled_terms/exclusive_tile_buckets`。B2骨干与观察后M2监测来自独立DUT，不是此B1同DUT执行。
- `SRNN_raw700_M2_cache_r3_off, SRNN_raw700_M2_cache_r3_on`：`full_sequence_cycles`。仅官方sample0的前8步，不能算完整SHD样本吞吐。
- `SCNN_real_final_B4T12_query4`：`standalone_full_model_load_cycles/state_restore_cycles/state_spill_cycles`。71595为合并的初始设置/发布；状态搬运在层周期内，没有独立时延计数。
- `SCNN_real_final_B4T12_query4`：`state_restore_words/state_spill_words/trace_transfer_words`。有完整搬运一致性断言，但最新长测未独立打印这些端口的握手字数。
- `SCNN_real_final_B4T12_query4`：`Conv8_cycles/pool_cycles/classifier_cycles`。layer8 descriptor区间合并Conv8、pool和classifier，不能伪拆分。
- `SCNN_real_final_B4T12_query4`：`pure_learning_cycles/commit_latency_cycles/publication_latency_cycles`。RMW/commit/publication次数已知，独立阶段起止时间戳缺失。
- `SCNN_real_final_B4T12_query4`：`lane_fires/enabled_terms/exclusive_tile_buckets`。最新B4T12执行无被动活动计数，不能继承旧B1T2图监测。
- `SCNN_real_final_B4T12_query4`：`learning_off_B4T12_cycles`。没有同输入独立learning-off B4T12臂；B1T2为不同工作负载。
- `SCNN_real_final_B4T12_query4`：`full_T20_sample_cycles`。B4T12+queryT4是有界窗口，不能算完整DVS样本吞吐。
- `FC_current_source_B16T99_off, FC_current_source_B16T99_on`：`isolated_pure_learning_latency_cycles/platform_publication_latency_cycles`。已有学习忙并集、前向重叠和FC窗口commit边沿；非重叠忙时钟不是反事实纯学习时延，未测平台发布。
- `SRNN_current_source_B1T73_native_off, SRNN_current_source_B1T73_native_on`：`isolated_pure_learning_latency_cycles/commit_latency_cycles/platform_publication_latency_cycles`。已有predicate学习忙区间和commit脉冲边沿；不能将重叠区间或两臂差当独立学习/提交/平台发布时延。
- `SRNN_current_source_B1T73_native_off, SRNN_current_source_B1T73_native_on`：`bus_payload_bytes/tb_store_check_host_seconds`。保留真实accepted/stalled握手字数；未独立计payload字节或分离TB检查主机耗时，0仿真检查周期不等于0主机耗时。
- `SCNN_current_source_B1T20_off, SCNN_current_source_B1T20_on`：`standalone_full_model_load_cycles/initial_state_cycles/host_file_load_seconds/host_golden_seconds`。setup为初始化/首次classifier发布合并区间，readmemh和golden仿真周期为0，独立主机成本及完整模型DMA装载仍未知。
- `SCNN_current_source_B1T20_off, SCNN_current_source_B1T20_on`：`isolated_pure_learning_latency_cycles/platform_publication_latency_cycles/board_DDR_time`。忙信号、commit-pending和发布request-to-done已计数但互相包含，非反事实独立学习或端到端平台时延；原生DDR模型不能替代板测。
- `all`：`board_time/DDR_bandwidth/measured_power/full_dataset_RTL_accuracy`。均未测量；仿真墙钟只表示主机成本，软件准确率独立列出。

## 原始依据

主表evidence列给出精确执行目录；JSON sources列出所有读取文件。以下入口保留各源码范围：

- [rtl/fc_final_counts/README.md](../../rtl/fc_final_counts/README.md)
- [rtl/srnn_same_dut/README.md](../../rtl/srnn_same_dut/README.md)
- [rtl/scnn_real_final_graph/README.md](../../rtl/scnn_real_final_graph/README.md)
- [performance/h40_activity/README.md](../h40_activity/README.md)
- [当前FC B16T99独立配对](../current_source_measurements/fc/server_pair_r1/comparison.json)
- [当前SRNN native B1T73独立配对](../current_source_measurements/srnn/server_pair_r1/commands.json)

### SRNN归档的最小读取文件

轻包复算只需保留metrics.json sources列出的实际读取文件，路径按原项目相对目录放置；不是承诺重新仿真全部算法/EDA。SRNN build raw内仅读取以下两个原样文件，不需要镜像整个raw、obj、生产RTL、fixture mem或记录的source_files/dependencies树。解析端scripts需保留run_srnn_profile_measurement.py、build_srnn_profile_measurement_reference.py及已复用的run_fc_final_counts.py；其余既有FC/activity解析依赖不变。归档runner只作provenance，不从归档执行。

- [build/current_profile_measurements_20261009/srnn_complete_pair_r1_raw/results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/fixture_sample0_t73/reference.json](../../../../../build/current_profile_measurements_20261009/srnn_complete_pair_r1_raw/results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/fixture_sample0_t73/reference.json)
- [build/current_profile_measurements_20261009/srnn_complete_pair_r1_raw/scripts/run_srnn_profile_measurement.py](../../../../../build/current_profile_measurements_20261009/srnn_complete_pair_r1_raw/scripts/run_srnn_profile_measurement.py)

### SCNN归档的最小读取文件

SCNN reference使用原样复制的performance/current_source_measurements/scnn/fixed_sample0_b1t20_r1/manifest.json。run_off/on.log、原summary、preparation、四个阶段status、launcher.status、elab/build.log及sources/dependencies清单来自server_pair_r2，实际读取路径均列在metrics.json sources。清单中服务器绝对路径和依赖名仅记录provenance，不要求它们等于本地cwd，也不据此复制整个依赖树。
SCNN_RAW内仅读取以下三个原样源码文件，不执行归档脚本，不镜像raw/obj/105 RTL或全部fixture。解析端scripts需保留run_scnn_profile_measurement.py及其标准库/本地导入：prepare_scnn_profile_measurement.py、run_scnn_real_final_graph.py、generate_scnn_real_final_graph.py、generate_scnn_real_final_b4t12.py、supplement_scnn_same_dut_reference.py、supplement_scnn_graph_reference.py、analyze_scnn_b4t4_rtl_update_contract.py及既有run_fc_final_counts.py；只调用parse_result，不运行这些模块的prepare/build/reference入口。

- [build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/scripts/run_scnn_profile_measurement.py](../../../../../build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/scripts/run_scnn_profile_measurement.py)
- [build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/tb/include/scnn_profile_measurement_monitor.svh](../../../../../build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/tb/include/scnn_profile_measurement_monitor.svh)
- [build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/tb/include/scnn_profile_measurement_body.svh](../../../../../build/current_profile_measurements_20261009/scnn_complete_pair_r2_raw/tb/include/scnn_profile_measurement_body.svh)
- [results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn/server_pair_r2/summary.json](../current_source_measurements/scnn/server_pair_r2/summary.json)
- [results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn/fixed_sample0_b1t20_r1/manifest.json](../current_source_measurements/scnn/fixed_sample0_b1t20_r1/manifest.json)
