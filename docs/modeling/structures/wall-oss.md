# Wall-OSS-0.5 适配记录

本记录覆盖 ModelZoo 接入、三图量化导出、部署推理、Golden 与开环数值评估。

## 来源和代码布局

- 官方模型：[x-square-robot/wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5)。
- 官方参考：[Wall-X](https://github.com/X-Square-Robot/wall-x)，固定 revision
  `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`。
- ModelZoo 原始基线：`5d9c5d5e`；独立仓库 `xh2modelzoo-wall-oss-05`。
- 模型实现：`xhmodel_merak/xh_other_model/models/wall_oss/`。
- 统一接入：`xhmodel_merak/model_integrations/wall_oss/`，注册模型 ID `wall-oss-0.5`。
- [构建、推理和开环执行命令](../../../examples_merak/vla/wall_oss/README.md)。
- [结构、精度、张量和缓存 ABI](../../../xhmodel_merak/xh_other_model/models/wall_oss/EXPLANATION.md)。
- Apache-2.0 来源声明：`THIRD_PARTY_NOTICES` 和 `licenses/wall-x/LICENSE`。

模型包负责源权重校验、processor、packed vision、双专家图、构建、runtime、Golden
及 workflow。统一 adapter 负责 recipe、发布和推理接入，公共机制没有新增模型名分支。

## 已验证范围

| 内容 | 范围 |
| --- | --- |
| 图拆分 | `visual`、`prefill`、`denoise`；两路相机复用视觉图 |
| 视觉 | 保持宽高比、长边 448、28 对齐、非方图、窗口与全局注意力 |
| 动作 | batch 1、H32、26 维状态/动作、512 档状态字符串、默认 10 步 Euler |
| 缓存 | 分块 prefill 256、总容量 2048、72 个 KV 状态、动作区域覆盖和请求隔离 |
| 部署精度 | W16A16、W8A16、W8A8 SEFP，group size 64，图位宽有实际审计 |
| 统一接口 | build / inference、CLI、Manifest、包搬移及脱离源 checkpoint 的推理 |
| Golden | 三精度、每精度三图，覆盖合成与真实相机输入，诊断与运行产物分离 |
| LIBERO 开环 | 40 任务、320 episode、960 观测、1,920 组官方 BF16/W8A8 配对 |
| FAST / 语言自回归 | 本次未支持，只接连续 Flow |
| NPU 性能 | 未验证，不以 GPU 吞吐代替芯片性能 |

checkpoint 配置 vocab_size=151936，实际 embedding 有 155765 行。按官方顺序添加
`<|propri|>`、`<|action|>`，保留完整 embedding 和动作模型，没有用通用 VLM 替代动作专家。
统计量来自源 checkpoint，随发布包保存；位置编码和视觉窗口采用实际几何和有效长度。

## 导出 YAML 与精度边界

| 精度 | 配置 |
| --- | --- |
| W16A16 | [wall_oss_05.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05.yaml) |
| W8A16 | [wall_oss_05_w8a16.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05_w8a16.yaml) |
| W8A8 | [wall_oss_05_w8a8.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05_w8a8.yaml) |

三个配置只在 `export.model.quant_scheme.quant_type` 上不同，沿用同一三图 ABI。
`quant: null` 跳过独立权重量化阶段，PTQ 在导出阶段执行。图接口/缓存为 FP16，
Host 动作状态、时间调度和 Euler 更新为 FP32，源 BF16 权重与部署 W16A16 不能混称。
官方 BF16 数值参考保留主干/视觉 BF16 及官方指定 FP32 模块，不将整个模型升到 FP32。

## 单样例与结构数值证据

受控横竖图中，官方联合前向与权重解释的最大速度误差为 `2.3842e-6`；前缀/动作拆分
首步误差 `2.0266e-6`，完整 10 步轨迹最大误差 `1.2696e-5`。
真实相机连通样例来自 `lerobot/aloha_static_coffee` 固定 v2.1 版本、episode 0，
使用受控 26 维状态与 `x2_normal`，只验证数值和接口，不作为该机器人任务成绩。
两路 640×480 相机预处理到 448×336。

该样例适配 FP32 对官方 FP32：视觉 RMSE `2.9793e-5`，最大误差 `0.0014734`；
完整动作轨迹 RMSE `3.2246e-7`，最大误差 `5.4240e-6`。

| 精度 | 全轨迹 RMSE | 最终动作 RMSE | 最终动作 cosine |
| --- | ---: | ---: | ---: |
| W16A16 | 0.000408406 | 0.000827204 | 0.999998589 |
| W8A16 | 0.003204902 | 0.004395923 | 0.999959634 |
| W8A8 | 0.011226881 | 0.021423831 | 0.999059213 |

以上是同一样例的归一化动作空间结果，不能代替多样本评估。三精度均做过改变
图像/状态/mask 后重跑原请求的逐位一致检查，以及脱离原 checkpoint 的包搬移推理。

## LIBERO 开环主报告

[完整报告与可提交证据](../../../examples_merak/vla/wall_oss/reports/libero_open_loop_20261026/README.md)
已归档在模型示例目录，包括原始统计 JSON、1,920 条结果 CSV、样本身份清单、
并发吞吐、复用审计、完整性审计和 SHA-256，不再只依赖忽略的工作目录查看结果。

数据固定为 `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。
40 任务 × 每任务 8 episode × 每 episode 3 时刻 × 2 seed，共 1,920 组、960 个不同观测。
原有 120 组 W8A8 预测校验身份后复用，新增 W8A8 1,800 组；BF16 全部重新运行。
两端采用完全相同的实际 FP32 初始噪声、图像、指令、状态、mask 和时间调度。
协议为双图长边 448、H32、26 维空间、`x2_normal`、右臂 10 维有效动作及 10 步 Euler。

| 最终归一化动作 | 全部 1,920 组 | 原 120 组 | 新增 1,800 组 |
| --- | ---: | ---: | ---: |
| MAE | 0.023253 | 0.022821 | 0.023282 |
| RMSE | 0.053900 | 0.056208 | 0.053743 |
| NRMSE | 10.211% | 10.591% | 10.185% |
| cosine | 0.99478178 | 0.99438600 | 0.99480847 |
| 样本 RMSE P95 | 0.102695 | 0.097683 | 0.102695 |
| 最差样本 cosine | 0.77693389 | 0.77693389 | 0.87932474 |

位置、6D 旋转、夹爪的 RMSE 分别为 0.032438、0.027306、0.146362。
总体接近不意味着每个样本偏差都小，夹爪和最差样本值得单独关注。
这些指标是描述性的数值差异，没有官方或用户确认的验收门限；不将 cosine 换算成
任务准确率。旧 120 组 FP32 对照及当时的自定筛查条件保留为历史诊断，不并入 BF16
主成绩，也不继续使用“未通过自定门限”直接断言量化不合格。

物理 GPU 6/7 上，每卡单/双引擎实测两卡合计吞吐：BF16 为 3.025/3.951 组每秒，
W8A8 为 1.039/1.515 组每秒，最终均采用每卡 2 引擎。短程吞吐测试至合并约 27.96 分钟，
不含实现和样本准备。重复预测逐位一致，源模型及部署包 21 个登记文件的哈希不变。

完整预测张量、实际噪声、图片、资源监控和日志保留在
`work_dirs/wall_libero_bf16_expanded_20261026/`，不随本次代码提交。
归档的 `provenance.json` 保留原始执行清单与模型/数据/产物身份；精确重放保留的旧
GPU 采样噪声还需要原始噪声张量，seed 本身不能代替跨设备噪声一致性。
