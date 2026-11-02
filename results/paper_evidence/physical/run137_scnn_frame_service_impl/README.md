# RUN137 最新 frame-service 源码 exact-DCP 实现

## 结论

RUN137于2026-10-09 14:21:22 +08:00终态通过。它直接读取RUN136原始`post_synth.dcp`，使用既有
15 ns OOC实现Tcl与约束，未修改RTL或时序例外。RUN136源码已与SCNN连续B4T12+query4功能长测源码
逐文件、逐字节核对一致，因此RUN137是当前frame-service候选的物理实现证据。

| 项目 | 结果 |
|---|---:|
| 工具 / 器件 | Vivado 2022.2 / xc7vx690tffg1761-3 |
| 目标周期 | 15.000 ns |
| Slice LUT | 178,070 / 433,200（41.11%） |
| Slice Registers | 80,138 / 866,400（9.25%） |
| BRAM36等效 | 514.5 / 1,470（35.00%） |
| DSP | 1,196 / 3,600（33.22%） |
| WNS / TNS | +0.102 / 0.000 ns |
| WHS / THS | +0.078 / 0.000 ns |
| 可布网络 / 完全布通 | 321,920 / 321,920 |
| 黑盒 / 未布局原语 / 布线错误网络 | 0 / 0 / 0 |
| Vivado退出 / 错误 / 严重警告 | 0 / 0 / 0 |

实现后全部用户时序约束满足。后布局和后布线拥塞报告均保留一个level-5全局窗口：North方向122%，
主要位于region1 tile1的P4权重存储、神经元区域和tile逻辑；初始router估计另有South方向level-5窗口，
覆盖2.696% tiles。尽管最终全部网络布通且时序通过，这仍不是“无拥塞”设计。

DRC报告保留3727条普通Warning，主要为DSP输入/输出流水、异步控制、RAMB输出寄存和板级配置；
methodology报告保留17477条普通Warning，主要为异步驱动检查、44个宽乘法器和1000个未约束I/O延迟。
Vivado日志含108条`WARNING:`行。错误和Critical Warning均为0，但不能称warning-clean。

## 证据文件

- `wrapper_status.txt`、`vivado_exit_code.txt`、`vivado_status.txt`及计数文件
- `utilization_post_route.rpt`、`utilization_hierarchical_post_route.rpt`
- `timing_summary_post_route.rpt`、setup/hold路径报告
- `route_status_post_route.rpt`、`congestion_post_place.rpt`、`congestion_post_route.rpt`
- `drc_post_route.rpt`、`methodology_post_route.rpt`、`qor_suggestions_post_route.rpt`
- `post_place_physopt.dcp`、`post_route_physopt.dcp`
- `power_post_route_estimated.rpt`仅为vectorless工具估计，不作实测功耗依据
- `run137_scnn_frame_service_impl_complete.tar.gz`为服务器完整回传归档

## 边界

结论仅为XC7VX690T、15 ns OOC核心布局布线闭合；板级I/O、HD.CLK_SRC、MIG/板卡时钟、bitstream、
上板、后仿真和实测功耗均未验证。当前setup裕量102 ps，仍应按明确约束引用，不能写成稳健板级Fmax。
learning-off配对需使用同一源码和同阶段结果，不能减旧RUN133。
