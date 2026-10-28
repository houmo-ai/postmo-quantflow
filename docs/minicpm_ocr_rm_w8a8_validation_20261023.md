# MiniCPM-OCR / OCR RM W8A8 验证（2026-10-23）

本次提供两个 **W8A8 SEFP 融合 GDR** 发布包。视觉、文本的 Linear / MatMul /
GroupMatMul / Conv 及 cumsum 操作数实际为 8/8，normalize 保留 FP32，接口与 cache
载体仍可为 float16。它们属于 XH2a HMONNX GPU 仿真产物，未执行芯片实机。

W8A8 不应沿用初次 W16 基线的精度结论：OCR 严格序列一致为 2/3，表格样本只存在
空格/换行差异；RM 的 12 个分类判定均与原生一致，但最大概率误差 0.04045719，
超过原先设定的概率门限。验证脚本保留非零退出状态，没有放宽门限。

## 发布包与运行入口

本仓库 `work_dirs/minicpm_ocr_20261023/` 下：

| 模型 | 发布目录 | ZIP |
| --- | --- | --- |
| MiniCPM-OCR | [minicpm-ocr-1.0_w8a8_gdr_release](../work_dirs/minicpm_ocr_20261023/minicpm-ocr-1.0_w8a8_gdr_release/) | [下载包](../work_dirs/minicpm_ocr_20261023/minicpm-ocr-1.0_w8a8_gdr_release.zip) |
| OCR RM | [ocr-rm-2b-value_w8a8_gdr_release](../work_dirs/minicpm_ocr_20261023/ocr-rm-2b-value_w8a8_gdr_release/) | [下载包](../work_dirs/minicpm_ocr_20261023/ocr-rm-2b-value_w8a8_gdr_release.zip) |

ZIP 只包含 Manifest 登记的运行资源和 Manifest 自身，排除运行缓存。
归档摘要见 [w8a8_SHA256SUMS](../work_dirs/minicpm_ocr_20261023/w8a8_SHA256SUMS)。
已解压读取两个 ZIP 中的全部运行文件，逐项校验 Manifest SHA-256 和 ZIP CRC，
结果见 [归档验证](../work_dirs/minicpm_ocr_20261023/reports/w8a8_archive_validation.json)。
根目录 `export_meta_info.json` 提供资源索引，组件继续按角色/档位平铺。

统一模型 ID 保持 `minicpm-ocr-1.0` 和 `ocr-rm-2b-value`，默认 workflow 改为本次
W8A8 配置，使用原始浮点 checkpoint 进行 SEFP 图量化，不生成独立 GPTQ checkpoint。
Manifest 的 `parameters.precision.checkpoint` 记录原始浮点权重档位，
`parameters.precision.deployment` 记录本次实际的 `w8a8h1_sefp`。标准图文件名仍沿用
源 checkpoint 的命名契约，核对位宽应读取 deployment 字段与节点属性。

```python
from xhmodel_merak.model_build import build_model

result = build_model(
    model_id="ocr-rm-2b-value",  # OCR 使用 minicpm-ocr-1.0
    model_dir="原始 checkpoint 路径",
    output_dir="新的发布目录",
    deployment_precision={
        "visual": "w8a8h1_sefp",
        "prefill": "w8a8h1_sefp",
        "decode": "w8a8h1_sefp",
    },
    device="cuda:0",
)
```

CLI 命令见 [OCR 示例](../examples_merak/llm/minicpm_ocr/README.md) 和
[RM 示例](../examples_merak/llm/ocr_rm/README.md)。组件精度覆盖只修改基础
QuantScheme，保留 workflow 中的融合策略和节点级例外。

## 为什么初次使用 W16、关闭 RM 的 GDR 融合

初次使用 W16 是建立原生实现与部署实现的一致性基线，当时没有 W8A8 的实测数据。
SDK 融合 GDR 的内部乘法固定为 8/8；关闭融合后，展开乘法可以跟随外层 W16 精度。
在当时相同 FP32 normalize、cumsum 实际 16/8 的设置下，RM 相对原生的最大概率误差
从融合版 0.00961179 降到展开版 0.00159222，因而选择展开图满足 0.005 的基线门限。

这一结果比较了不同的实际算子精度，不能直接推断融合算法本身损失精度。
本次 W8A8 默认启用 GDR 融合，另以相同 W8A8 配置关闭融合做受控对照。
历史基线与首次产物精度纠正见 [初次验证记录](minicpm_ocr_rm_validation_20261023.md)。

## 确保 cumsum 真正采用 8/8

初次检查发现两个配置链路问题，本次均已处理：

1. cumsum 的第二输入为固定三角矩阵，SDK 将其转换成 Linear。
   YAML 同时设置 `act_schema_2.man_bit=8` 与 `w_schema.man_bit=8`，覆盖转换后的权重。
2. 计算图中的 cumsum pragma 原先固定第一输入为 16-bit。现在从
   `cumsum_matmul_quant_config.act_schema.man_bit` 传到 wrapper 和图函数，使请求的 8-bit
   真正生效。未显式指定该字段的既有模型仍保留原先 16-bit 默认行为。

没有修改 SDK。对两个完整包的 prefill、decode 和全部视觉档位逐图检查：

- 显式 Linear / MatMul / GroupMatMul / Conv 操作数均为 8/8。
- 每个模型的 prefill 有 18 个 cumsum Linear，均为 8/8。
- prefill 保留 18 个 `GDRBlockTriInverse` 和 18 个 `GDRChunkScan`。
- decode 保留 18 个 `GDRRecurrentScan`；融合 Block/Chunk 内部乘法的 SDK 配置也为 8/8。

详细节点、图 SHA-256、SDK 配置来源及接口载体类型见
[OCR 位宽审计](../work_dirs/minicpm_ocr_20261023/reports/ocr_w8a8_precision.json) 和
[RM 位宽审计](../work_dirs/minicpm_ocr_20261023/reports/rm_w8a8_precision.json)。

## OCR 结果

保持相同三张原图、原生 processor、4x、9 个局部切片上限、关闭 thinking、greedy
decoding 和 256-token 生成上限。原生对照为 BF16。

| 样本 | 类型 | 原生 tokens | W8 tokens | 严格 token 一致 | 去空白文本一致 |
| --- | --- | ---: | ---: | --- | --- |
| s01 | 文字 | 13 | 13 | 是 | 是 |
| s05 | 公式 | 121 | 121 | 是 | 是 |
| s09 | 表格 | 89 | 92 | 否 | 是 |

表格样本的文字、标点、数字顺序一致，空格与换行不同，原始字符编辑距离为 7。
严格一致门禁为 **2/3，未通过**；去除空白后的文本一致为 **3/3**，是单独的辅助指标。
没有使用空白归一化替换严格比较，也没有据此宣称 OCRBench 全量准确率。

完整输出见 [ocr_w8a8_accuracy.json](../work_dirs/minicpm_ocr_20261023/ocr_w8a8_accuracy.json)，
差异统计见 [ocr_w8a8_text_comparison.json](../work_dirs/minicpm_ocr_20261023/reports/ocr_w8a8_text_comparison.json)。

## RM 结果

复测交付的文字、公式、表格全部 12 组样本，保持原图/渲染图顺序、完整 value prompt、
原生图像预算和完整 MRoPE，原生对照为 FP32，另记录生产 BF16 / batch=4 参考。

| 指标 | W8A8 融合版 |
| --- | ---: |
| 对原生 FP32 最大概率绝对误差 | 0.040457189083099365 |
| 对原生 FP32 平均概率绝对误差 | 0.004362892283950259 |
| 对生产 BF16 参考最大概率绝对误差 | 0.04245501756668091 |
| 与原生 0.5 阈值判定一致 | 12/12 |
| 与生产参考 0.5 阈值判定一致 | 12/12 |
| 原生误差门禁 0.005 | 未通过 |
| 跨 dtype 生产参考误差门禁 0.03 | 未通过 |

最大误差发生在 s08：原生 `P(1)=0.87207937`，W8A8 为 `0.83162218`，两者均判为 1。
这批样本包含原模型误判，判定一致率不代表任务准确率；本次结果支持“提供的 12 组样本
未新增分类判定变化”，不能据此概括整体 W8 任务精度。

完整逐样本概率、logits、标签及门禁布尔值见
[rm_w8a8_accuracy.json](../work_dirs/minicpm_ocr_20261023/rm_w8a8_accuracy.json)。
与历史 W16 版本的并列表格及原图、渲染图、OCR 文本见
[图片报告](../work_dirs/minicpm_ocr_20261023/reports/sample_report.html)，
表格可下载为 [CSV](../work_dirs/minicpm_ocr_20261023/reports/sample_scores.csv)。

## 同位宽 GDR 融合对照

以解析后的 W8A8 BuildPlan 为基础，仅把 `fuse_gdr_ops` 和
`fuse_gdr_block_recurrent_ops` 改成 false，经相同统一 executor 生成
[W8A8 展开对照包](../work_dirs/minicpm_ocr_20261023/ocr-rm-2b-value_w8a8_unfused_release/)。
这是诊断用的配置对照，默认公共构建仍启用融合。复现脚本为
[build_rm_w8a8_unfused.py](../work_dirs/minicpm_ocr_20261023/build_rm_w8a8_unfused.py)。

直接比较两个包的 effective workflow，确认只有两个融合标志不同。
图审计确认展开版同样采用实际 8/8（包括 cumsum），prefill/decode 均不包含 GDR 融合算子。
证据为 [配置差异](../work_dirs/minicpm_ocr_20261023/reports/rm_w8a8_fusion_config_diff.json)
和 [展开版位宽审计](../work_dirs/minicpm_ocr_20261023/reports/rm_w8a8_unfused_precision.json)。

| 12 组样本上的指标 | 结果 |
| --- | ---: |
| 融合 / 展开最大概率差 | 0 |
| 融合 / 展开最大最终 logit 差 | 0 |
| 最终概率完全相同 | 12/12 |
| 展开版相对原生 FP32 最大误差 | 0.040457189083099365 |

关闭融合没有改善这批 W8A8 样本的评分；本次保留融合。该对照说明本轮未观察到融合
带来额外的最终输出差异，不是对所有输入或中间张量逐位一致的证明。
也没有以 GPU 仿真结果宣称芯片实机吞吐提升。
详细结果见 [融合对照统计](../work_dirs/minicpm_ocr_20261023/reports/rm_w8a8_fusion_comparison.json)
和 [展开版精度原始报告](../work_dirs/minicpm_ocr_20261023/rm_w8a8_unfused_accuracy.json)。

## 公共接口与环境

两个完整融合包均由公共 `build_model` 构建，并从 `/tmp` 使用公共 `inference_model`
离线加载，未传入源 checkpoint 路径。输出与配对精度验证的 s01 完全一致：
[OCR 统一推理](../work_dirs/minicpm_ocr_20261023/ocr_w8a8_unified.json)、
[RM 统一推理](../work_dirs/minicpm_ocr_20261023/rm_w8a8_unified.json)。
历史 W16 包也已通过更新后 recipe 的兼容性校验。

两个最小真实模型已通过公共构建、产物搬迁、删除源 checkpoint、离线推理，
并新增实际图节点位宽及 GDR 算子存在性断言。日志为
[w8a8_tiny_3.log](../work_dirs/minicpm_ocr_20261023/w8a8_tiny_3.log)（2 passed）。

运行环境为 Python 3.12.14、Torch 2.8.0+cu128、Transformers 5.16.1、本地 SDK
distribution 0.9.2 / 模块版本 0.4.0、A100 80GB。沿用初次记录的隔离环境及声明依赖；
全新环境安装和芯片实机没有验证，环境清单中的 `blocked_distribution` 状态保留。
当前环境没有 `ocr` 命令，未执行 `ocr review`，已逐项复核改动与实际验证证据。

## CI 范围与依赖缺口

由于修改了共享 Qwen3.5 的 cumsum pragma 位宽传递，按当前 impact 选择运行了 50 个
测试目标、1102 项检查，初次结果为 **1069 passed / 33 failed**，耗时 1896.44 秒。
日志见 [w8a8_ci.log](../work_dirs/minicpm_ocr_20261023/w8a8_ci.log)，失败用例列表见
[w8a8_ci_initial_failures.json](../work_dirs/minicpm_ocr_20261023/reports/w8a8_ci_initial_failures.json)。
不能将此结果视为全仓库或本轮广泛回归全部通过。

两个 OCR 模型的真实小模型构建、搬迁、离线推理和位宽检查在该轮均通过。
新增文档另跑快速 policy gate：**133 passed**，日志为
[w8a8_docs_ci.log](../work_dirs/minicpm_ocr_20261023/w8a8_docs_ci.log)。

初次广泛回归的失败集中在以下依赖问题：

- 8 项 MoE LoRA 用例：本机 SDK `MoeBlock` 缺少 `set_lora`。
  已在不导入 ModelZoo 模型的独立进程中检查该 SDK 类，源文件相对 SDK HEAD 无修改。
  证据为 [SDK 能力检查](../work_dirs/minicpm_ocr_20261023/reports/ci_sdk_lora_failure.json)
  和 [调用栈](../work_dirs/minicpm_ocr_20261023/reports/ci_sdk_lora_failure_trace.txt)。
- 23 项其他模型的量化集成用例：隔离环境安装的 PyPI AutoRound 0.12 缺少仓库定制的
  `int_sym_xh` 和本地 JSONL 数据支持，分别表现为量化函数返回 None 和 dataset KeyError。
  这些失败发生在量化生产阶段，早于本次 cumsum 导出代码。
- 2 项 Qwen3-Next 集成用例：该模型声明 `Defuser==0.0.25`，初始环境为 0.0.28。

定向复查显式使用 `PYTHONPATH` 指向 `gptqmodel/third_party/auto-round` 源码，保留
初次广泛运行和复查运行的独立日志。该定制源码的模块版本为 0.12.0，构建 wheel 的
SCM 版本为 `0.12.0.dev79+g304f5612`，未把它伪装成 release pin 或替换原环境分发包。
源码位置、commit 和 wheel 摘要见
[AutoRound 来源记录](../work_dirs/minicpm_ocr_20261023/reports/autoround_fork_provenance.json)。
两个 OCR 模型使用 SEFP 图量化，其构建没有执行 AutoRound/GPTQ 权重量化生产流程。

复查结果分别记录，未将不同环境下的通过数伪装成一次完整 CI：

| 复查 | 结果 | 日志 |
| --- | --- | --- |
| 使用本地 AutoRound 定制源码，第一组 6 项 | 6 passed | [日志](../work_dirs/minicpm_ocr_20261023/w8a8_ci_autoround_fork.log) |
| 使用相同源码，其余 19 项 | 15 passed / 4 failed | [日志](../work_dirs/minicpm_ocr_20261023/w8a8_ci_remaining_fork_2.log) |
| 在任务隔离环境对齐 Defuser 0.0.25，复查 Qwen3-Next | 2 passed | [日志](../work_dirs/minicpm_ocr_20261023/w8a8_ci_next_defuser.log) |

19 项复查中的另外两项失败为 Qwen3.6 MoE LoRA 路径缺少 SDK `MoeBlock.has_lora`，
与初次的 8 项 `set_lora` 缺口同属 SDK MoE LoRA 能力不完整。
合并各用例最新状态为 **1092 passed / 10 项 SDK MoE-LoRA 失败**；这不是一次全绿 CI。
未修改 SDK、降低断言或跳过这些用例。明细见
[CI 汇总](../work_dirs/minicpm_ocr_20261023/reports/w8a8_ci_summary.json)。

构建诊断 wheel 曾在 AutoRound 源码树生成带 dev 版本的 egg-info，导致一次 19 项复查
被版本门禁提前拒绝。已把本次生成的 metadata 转存到任务目录，恢复此前源码路径运行
环境后，才执行表中的 19 项复查。该过程未改源码版本或安装该 dev wheel，记录见
[构建元数据清理](../work_dirs/minicpm_ocr_20261023/reports/autoround_build_metadata_cleanup.json)。
