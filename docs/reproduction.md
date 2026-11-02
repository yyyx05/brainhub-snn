# 复现条件

## 不需要外部资产的入口

从仓库根目录运行README中的JSON指纹命令。Python 3.11标准库即可读取画像、进行兼容性过滤和生成dry-run动作记录。`demo/decision_replay.html`可直接本地打开，展示已有软件记录；两种入口均不执行板级操作。

生产核心由`rtl/final_sources.f`指定，共105份SV；算法ROM位于`rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem`。这些源文件与既有研究版本相同，当前公开整理没有改变设计。

## 需要额外环境的复现

| 目的 | 需要的环境/输入 |
| --- | --- |
| SHD H5指纹读取 | h5py及合法取得的SHD H5 |
| 量化模型评价 | 相应脚本的numpy/torch等依赖、checkpoint、数据与数值规则 |
| RTL数值仿真 | Verilator 5.048、C++工具链、选定测试平台及对应权重/输入 |
| FPGA OOC综合/实现 | Vivado 2022.2、Virtex-7器件支持与许可；XPM由Vivado提供 |
| 板级运行 | 实际板卡及其时钟、复位、DDR、主机接口和约束 |

`requirements_optional.txt`列出可选Python库。没有声明一个未经重新验证的完整锁定环境；JSON指纹入口不要求安装这些可选库。

## 模型和测试资产

公开版不包含`models/`、`data/`、`fixtures/`、`imports/`、`memory/`或`tb/data/`。画像JSON中保留部署资产的逻辑路径，供持有合法资产的使用者按相同目录放置。路径存在于配置中不代表文件已经提供，dry-run输出也不证明资产已载入硬件。

在拥有全部对应资产与工具后，可根据选定脚本参数运行数值测试。没有资产时可审阅源码与已有结果，不能仅凭本仓库重建全部研究数字。

## OOC与原生IP

历史评价使用`xc7vx690tffg1761-3`和15 ns约束。由仓库根目录调用：

```console
vivado -mode batch -source vivado/ooc/run_unified_h40_final_composition_v1_synth.tcl -tclargs <仓库绝对目录> 1
```

原脚本只读取`rtl/`生产闭包。ROM按进程工作目录解析；不能在其他目录运行后忽略ROM加载告警。此命令是复现方法，本次材料整理没有执行它。

`ip/unified_snn_native/`保留R2 component.xml、核心、纯连线wrapper和ROM。已有Vivado工程可将本仓库`ip/`作为repository根目录，并通过`ip/unified_snn_native_ip/register_repository.tcl`注册、配置XPM并准备ROM。原生IP不提供完整AXI/PCIe/DDR平台。

## 结果与候选

精选既有证据见`results/README.md`。原始结果说明中的其他历史文件不一定随仓库提供。完整DCP、波形、工具日志和实验输入未收入。候选在`verification/experiments/`独立保存，不能与生产源递归混编，也不能拼接独立候选的最佳指标。

软件准确率、RTL一致性、周期、OOC综合/布线和板级实测分开解释。周期乘以15 ns仅是所计区间的核心时间投影；缺少主机准备、传输和板级调度测量时，不称端到端延迟。

## 本轮检查范围

公开整理复用了已完成的生产源一致性与语法检查，并核对归档和新增文档链接；没有新增RTL仿真、综合、算法实验或板级测试。比赛包的模型资产齐全检查没有复制到公开版，因为它不适用于本目录。
