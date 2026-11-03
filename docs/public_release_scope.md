# 公开版资源说明

[公开仓库](https://github.com/yyyx05/brainhub-snn)提供生产105份SV与算法ROM、画像软件与JSON示例、演示页面、测试脚本、独立候选、Vivado配置、项目IP封装、精选结果和工程技能。

## 按任务准备资源

| 使用场景 | 所需资源 |
| --- | --- |
| HTML交互演示、JSON画像选择 | 使用仓库内文件即可 |
| 模型评价与RTL数值仿真 | 另行取得相应模型、数据与测试输入 |
| FPGA构建 | 本机安装Vivado与器件支持，配置对应许可 |

模型和输入按`models/`、`data/`、`fixtures/`、`imports/`、`memory/`、`tb/data/`目录约定放置；公开仓库保留这些资源的配置路径，资产本身由使用者按相应授权取得。内部算法ROM保存在`rtl/learning/data/`。

本地参赛RAR另外包含已有模型和派生测试输入。两种版本共用生产源码；公开版便于源码交流，参赛包用于作品审阅与本地复现。

原创代码采用[Apache-2.0](../license.txt)，项目归属见[NOTICE](../notice.txt)，外部资源沿用各自授权。
