# MiniCPM-OCR 与 OCR RM 适配验证（2026-10-23）

**OCR RM 旧 W16 包已废弃（2026-10-26）**：其基础权重采用禁止的 SEFP h0。
本页保留历史证据，RM 的旧精度数据不代表 h1 版本。请使用
[重新导出的 SEFP h1 包及验证结果](ocr_rm_sefp_h1_validation_20261026.md)。
此说明仅针对 OCR RM，未对本页 MiniCPM-OCR 产物作新的验证结论。

本页记录初次 W16 基线。后续 W8A8 融合版本、真实位宽修复和对照结果见
[W8A8 验证记录](minicpm_ocr_rm_w8a8_validation_20261023.md)。统一构建默认已切换为 W8A8，
本页列出的历史包与原始报告仍保留。

两个交付包已解压，并通过统一 `build_model` 导出 XH2a HMONNX。
最终包使用仓库共用的 Transformers 5.16.1；OCR 三个样本共 223 个生成 token
与原生 BF16 完全一致，RM 全部 12 个样本相对原生 FP32 最大概率误差为
0.00159222。本次改动选中的 17 个 CI 文件共 **804 passed**。

## 输入与产物

原始压缩包保留在 `/data01/llm_models/minicpm_ocr/`，分别解压到同名目录：

| 统一模型 ID | 压缩包 | 解压后的权重位置 |
| --- | --- | --- |
| `minicpm-ocr-1.0` | `hw_vllm_minicpm_20261016.tar.gz` | `hw_vllm_minicpm_20261016/MiniCPM-OCR-1.0` |
| `ocr-rm-2b-value` | `rm_hf_infer_demo_v1_20261015_with_ckpt_20261016.tar.gz` | `rm_hf_infer_demo_v1_20261015_with_ckpt_20261016/rm_hf_infer_demo_v1_20261015_with_ckpt/qwen35_2b_l2_triple_value_checkpoint-3000` |

最终产物位于本仓库 `work_dirs/minicpm_ocr_20261023/`：

| 模型 | 最终包 | 布局 |
| --- | --- | --- |
| MiniCPM-OCR | [minicpm-ocr-1.0_shared_release](../work_dirs/minicpm_ocr_20261023/minicpm-ocr-1.0_shared_release/) | `visual_m384/`、`prefill/`、`decode/` |
| OCR RM Value | [ocr-rm-2b-value_shared_release](../work_dirs/minicpm_ocr_20261023/ocr-rm-2b-value_shared_release/) | `visual_m96/`、`visual_m384/`、`visual_m1536/`、`prefill/`、`decode/` |

包根包含 `export_meta_info.json`、`model_build_manifest.json`、
`environment.json`、`environments/`、有效 workflow、运行 metadata、本地 HF
processor/tokenizer、embedding。资源索引使用包内相对路径；Manifest 记录源权重
身份及发布文件哈希。训练记录、导出日志和中间模型不进入最终包。
`work_dirs` 下的早期探查目录不是本次交付包。

交付包内 vocab-head demo 指向的另一份 RM checkpoint 未提供，故只登记实际存在的
value-head 模型。两份模型均未登记独立 GPTQ/AutoRound 或统一 Golden 能力。

## 实现与协议

接入包含底层 master 注册、workflow/YAML、可发现的 `ModelRecipe`、统一构建与推理
适配器、示例、精度脚本和当前 CI/impact 声明。代码分别在
`xhmodel_merak/xh_llm/models/{minicpm_ocr,ocr_rm}` 与
`xhmodel_merak/model_integrations/{minicpm_ocr,ocr_rm}`。

OCR 文本为 24 层、hidden size 1024 的 Qwen3.5 混合注意力，保留 tied embedding
装载语义。视觉复用现有 MiniCPM token-capacity 表达，宿主保持原生动态切片。
本次只登记单图 4x 路径：每片最多 1536 patch tokens、384 image tokens，
`max_slice_nums=9`，8K context、256-token prefill。16x、adaptive resize 和视频未登记。
图片或上下文超出部署容量时拒绝，不降低图片分辨率来绕过容量限制。

原包包含 Transformers 5.19.0.dev0 fork。为保持仓库 5.16.1 环境，接入复用已安装的
MiniCPM-V-4.6 4x 主干，并移植 OCR 的两段图像几何方法：2x patch 对齐及切片网格
选择。独立加载原包 5.19 实现后，对文字、公式、表格三个真实输入比较：
token IDs、pixel values、target sizes 完全一致，FP32 视觉特征最大绝对误差均为 0。
最终配对验证的原生 token 序列也与原包 5.19 基线一致。
证据见 [ocr_compatibility.json](../work_dirs/minicpm_ocr_20261023/ocr_compatibility.json)、
[原包生成基线](../work_dirs/minicpm_ocr_20261023/ocr_native.json)。
移植代码保留上游版权和 Apache-2.0 标识，归属记录在
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES)，许可证为
[licenses/transformers/LICENSE](../licenses/transformers/LICENSE)。

RM 文本为 24 层、hidden size 2048 的 Qwen3.5；真实权重为二分类
`score.weight [2, 2048]`，没有语言模型输出头。原配置误写为生成架构，加载器显式
使用 `Qwen3_5ForSequenceClassification` 并检查训练好的 score 权重。
输出为最后一个非 padding token 的 `softmax(score(hidden))[1]`。
复用共享导出图时只替换输出投影，不改 embedding vocabulary。

RM 严格保留原图、渲染图的顺序和原包 value prompt：无 system、无 assistant 头、
末尾保留换行。保留原生最少 64 image tokens、processor 最大预算及 199:1 长宽比补白。
每图选择最小可容纳视觉档位，超出 1536 image tokens 时拒绝。
完整 prompt 上计算 MRoPE 并散入视觉特征，然后按 chunk 切分，支持图片跨 chunk。
评分使用 prefill，包内 decode 供共享图加载契约使用。

当前 SDK fused GDR 的内部 MatMul 固定使用 A8/W8，即使外层声明 W16A16。
RM 最终配置关闭 `fuse_gdr_ops` 和 `fuse_gdr_block_recurrent_ops`，启用
`normalize_force_fp32`，使用已有展开图达到评分精度要求。本次没有修改 SDK。

**产物精度审计纠正：** YAML 的 `cumsum_matmul_quant_config` 请求两个输入均为
16-bit，但当前 SDK 会把第二输入为固定三角矩阵的 cumsum MatMul 转成 Linear。
Linear 使用 `w_schema`，该覆盖中的 `act_schema_2` 没有控制其权重精度。
直接读取最终 HMONNX，prefill 中 18 个 cumsum Linear 实际均为 **16/8**，
普通显式 MatMul 共 666 个，均为 16/16；无融合 GDR 算子。
旧版的 cumsum 同样为 16/8，融合 GDR 内部乘法为 SDK 固定的 8/8。
此前“最终 cumsum 为 16/16”的表述是依据配置做出的错误判断，现已纠正。
上述精度分数属于当前实际的 16/8 cumsum 产物；真实 16/16 或 8/8 cumsum 对照版
尚未重新导出。逐节点证据见
[precision_audit.json](../work_dirs/minicpm_ocr_20261023/reports/precision_audit.json)。

## 运行环境与验证边界

- Linux x86_64、实际运行 Python 3.12.14、PyTorch 2.8.0+cu128、Transformers 5.16.1。
- 本地 `xhquanttool` distribution 版本 0.9.2，运行模块版本字符串为 0.4.0。
- A100 80GB GPU，目标为 XH2a，执行方式为 **HMONNX GPU simulation**，未执行芯片实机。
- 使用 `work_dirs/minicpm_ocr_20261023/env_ci` 隔离环境；原包 fork 仅在
  `env_ocr` 中提供独立基线，没有替换系统环境。
- 包内环境清单沿用仓库声明依赖契约。尚未做全新环境完整安装验证；其
  `blocked_distribution` 状态保留了 SDK/ModelZoo 不可变发行包、CUDA wheel 来源和
  现有依赖 metadata 冲突等待解决项，不能据此宣称跨机器安装已验证。
  声明的 Python pin 为 3.12.13，与本机实际解释器的 patch 版本不同。

## OCR 配对结果

输入取交付 RM demo 的原图，使用相同 prompt `请识别图片中的文字。`、原生切片、
关闭 thinking、greedy decoding，生成上限 256 tokens。原生为 BF16，HMONNX 为
W16A16 SEFP。

| 样本 | 类型 | prompt tokens | 生成 tokens（含结束符） | 序列一致 |
| --- | --- | ---: | ---: | --- |
| s01 | 文字 | 274 | 13 | 是 |
| s05 | 公式 | 297 | 121 | 是 |
| s09 | 表格 | 1088 | 89 | 是 |

总计 **3/3 序列、223/223 token 一致**。完整原生/HMONNX token 与文本见
[ocr_accuracy_final.json](../work_dirs/minicpm_ocr_20261023/ocr_accuracy_final.json)。
这是导出后的配对一致性验证，不是 OCRBench/OmniDocBench 全量任务准确率。

## RM 概率结果

覆盖交付 `samples/manifest.json` 的全部 12 个样本。逐样本比较原生 FP32 与最终
HMONNX 包，同时记录生产链 BF16 / batch=4 参考值。原生概率误差门限为 0.005；
生产参考采用原包跨 dtype 验证门限 0.03，分别判定。

| 样本 | 类型 | 原生 FP32 P(1) | HMONNX P(1) | 生产参考 P(1) | 对原生绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| s01 | 文字 | 0.03208300 | 0.03099683 | 0.02675928 | 0.00108617 |
| s02 | 文字 | 0.99522996 | 0.99529076 | 0.99509466 | 0.00006080 |
| s03 | 文字 | 0.00002658 | 0.00002679 | 0.00002931 | 0.00000021 |
| s04 | 文字 | 0.99674606 | 0.99673969 | 0.99677747 | 0.00000638 |
| s05 | 公式 | 0.00109367 | 0.00106289 | 0.00101624 | 0.00003078 |
| s06 | 公式 | 0.98761660 | 0.98749614 | 0.98717880 | 0.00012046 |
| s07 | 公式 | 0.00003067 | 0.00003060 | 0.00003024 | 0.00000008 |
| s08 | 公式 | 0.87207937 | 0.87048715 | 0.87407720 | 0.00159222 |
| s09 | 表格 | 0.00002360 | 0.00002265 | 0.00002754 | 0.00000094 |
| s10 | 表格 | 0.98839462 | 0.98850292 | 0.98858041 | 0.00010830 |
| s11 | 表格 | 0.00000090 | 0.00000084 | 0.00000103 | 0.00000006 |
| s12 | 表格 | 0.91393137 | 0.91313583 | 0.91243619 | 0.00079554 |

相对原生 FP32 最大误差 **0.0015922188758850098**，平均误差
**0.00031682797388536227**；相对生产参考最大误差
**0.004237549379467964**，两个概率门禁均通过。以 0.5 为阈值，与两份参考的判定
均为 **12/12 一致**。交付样本包含原模型误判，一致率不等于任务准确率。
完整 logits、概率、误差见
[rm_accuracy_final.json](../work_dirs/minicpm_ocr_20261023/rm_accuracy_final.json)。

## 统一接口与 CI

两个真实大模型均由公共 `build_model` 构建最终发布包，并通过公共
`inference_model` 从 `/tmp`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 加载。
推理只使用最终包和输入图片/文本，没有传入源权重目录：

- [OCR 统一推理结果](../work_dirs/minicpm_ocr_20261023/ocr_unified_final.json)：s01 输出 13 tokens。
- [RM 统一推理结果](../work_dirs/minicpm_ocr_20261023/rm_unified_final.json)：s01 `prob_1=0.030996832996606827`。

当前 CI 新增两个最小真实模型，执行公共 build、产物搬迁、删除源 checkpoint、离线
公共 inference，未 mock 图导出。小模型覆盖线性/全注意力、真实视觉处理器、RM 二分类头。
契约测试另外覆盖 source identity、配置档位漂移、跨 chunk 图片位置、最后非 pad
pooling、输入边界、资源闭包和路径越界。真实大模型源目录未删除。

已有测试中的 registry/impact 期望同步新增实际消费者，没有修改全局 impact policy。
仅为本任务生成 [changed_files.txt](../work_dirs/minicpm_ocr_20261023/changed_files.txt)，
避免将工作区原有 Xing4/hm_eval 改动混入此次验证范围。

```bash
PATH="$PWD/work_dirs/minicpm_ocr_20261023/env_ci/bin:$PATH" \
CI_TEST_CHANGED_FILES_FILE="$PWD/work_dirs/minicpm_ocr_20261023/changed_files.txt" \
CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=4 \
./benchmark_test/ci_test_transformer5.16.0/run.sh -q
```

选择 17 个测试文件，结果 **804 passed, 2 warnings in 126.61s**，警告为第三方 SWIG
弃用提示。日志为 [ci_final_2.log](../work_dirs/minicpm_ocr_20261023/ci_final_2.log)。
这不是全仓库全量 CI 结论。
补充验证文档和 RM README 后，相关选择再跑 **147 passed**，日志为
[ci_docs_final.log](../work_dirs/minicpm_ocr_20261023/ci_docs_final.log)。

按 `AGENTS.md` 检查过 `ocr` 命令，当前环境未安装，故未执行 `ocr review`；已人工
复核新增集成、模型、运行时、精度脚本及已有测试的改动。CodeGraph 工具不可用。
用户原有改动保留，未提交或推送代码。

## 复现入口

构建与统一推理命令、模型输入要求见
[MiniCPM-OCR README](../examples_merak/llm/minicpm_ocr/README.md) 和
[OCR RM README](../examples_merak/llm/ocr_rm/README.md)。在仓库根目录可复测最终包：

```bash
TASK_DIR="$PWD/work_dirs/minicpm_ocr_20261023"
MODEL_ROOT=/data01/llm_models/minicpm_ocr
RM_ROOT="$MODEL_ROOT/rm_hf_infer_demo_v1_20261015_with_ckpt_20261016/rm_hf_infer_demo_v1_20261015_with_ckpt"
SAMPLES_DIR="$RM_ROOT/rm_hf_infer_demo_v1_20261015"

CUDA_VISIBLE_DEVICES=0 "$TASK_DIR/env_ci/bin/python" \
  examples_merak/llm/minicpm_ocr/verify_accuracy.py \
  --model-dir "$MODEL_ROOT/hw_vllm_minicpm_20261016/MiniCPM-OCR-1.0" \
  --artifact-dir "$TASK_DIR/minicpm-ocr-1.0_shared_release" \
  --image "$SAMPLES_DIR/samples/images/s01_orig.png" \
  --image "$SAMPLES_DIR/samples/images/s05_orig.png" \
  --image "$SAMPLES_DIR/samples/images/s09_orig.png" \
  --output /tmp/minicpm_ocr_accuracy.json

CUDA_VISIBLE_DEVICES=0 "$TASK_DIR/env_ci/bin/python" \
  examples_merak/llm/ocr_rm/verify_accuracy.py \
  --model-dir "$RM_ROOT/qwen35_2b_l2_triple_value_checkpoint-3000" \
  --artifact-dir "$TASK_DIR/ocr-rm-2b-value_shared_release" \
  --samples-dir "$SAMPLES_DIR" --native-dtype float32 \
  --max-prob-error 0.005 --reference-max-prob-error 0.03 \
  --output /tmp/ocr_rm_accuracy.json
```

精度验证需要源权重用于原生对照；部署推理只需要最终包及仓库运行时代码。
权重、HMONNX、原始 JSON 报告与日志位于本机忽略目录，没有加入版本库。
