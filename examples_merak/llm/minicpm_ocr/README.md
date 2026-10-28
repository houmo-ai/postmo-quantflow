# MiniCPM-OCR 1.0

实现位于 `xhmodel_merak/xh_llm/models/minicpm_ocr`，统一模型 ID 为
`minicpm-ocr-1.0`。视觉复用 MiniCPM token-capacity 图表达，文本复用 Qwen3.5
混合注意力、split-conv cache 和 prefill/decode。原生 tied embedding 保持正确装载，
导出时独立保存 embedding 和输出投影。

## 环境与模型

`hw_vllm_minicpm_20261016.tar.gz` 解压后，权重在 `MiniCPM-OCR-1.0/`，
配套 Transformers 源码在 `hw_codehub/new-model-addition-mincpm-ocr/`。
生产接入沿用仓库 Transformers 5.16.1，复用已安装的 MiniCPM-V-4.6 4x 主干。
`native.py` 保留 OCR 的 2x patch 对齐及切片网格选择，原包新增的 16x merger 和
adaptive resize 未登记。三个真实样本的 token/像素张量与 FP32 视觉特征，均已与
原包 5.19.0.dev0 实现逐位比较一致。移植的两段图像几何代码及其 Apache-2.0
归属见根目录 `THIRD_PARTY_NOTICES` 和 `licenses/transformers/LICENSE`。

使用 recipe 声明的 SDK、Transformers 5.16.1 和直接依赖；发布包内 `environment.json`
和 `environments/` 同步记录版本。无需安装整份原包 Transformers。

## 构建和推理

从仓库根目录执行，`MODEL_DIR` 为原始权重目录，`ARTIFACT_DIR` 必须不存在：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/minicpm_ocr/unified_demo.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --deployment-precision w8a8h1_sefp --device cuda:0

CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/minicpm_ocr/unified_demo.py \
  --artifact-dir "$ARTIFACT_DIR" --image "$IMAGE" \
  --prompt '请识别图片中的文字。' --max-new-tokens 512
```

默认 XH2a / W8A8 SEFP（`w8a8h1_sefp`）、8K context、256-token prefill。
保留 GDR 融合；cumsum 乘法实际为 8/8，normalize 保留 FP32。
`--deployment-precision` 只覆盖基础图精度，保留融合策略与节点级例外。
只登记已执行的 4x 图像路径：
每个切片最多 1536 patch tokens，输出最多 384 image tokens；宿主保持原生动态切片，
最多 9 个局部切片，另含一张全图。超出图档位或上下文时明确拒绝，不额外降低图片分辨率。该入口接收单图，
未登记视频、16x、独立 GPTQ/AutoRound 量化或统一 Golden 能力。

发布包平铺 `visual_m384/`、`prefill/`、`decode/`，包含原生 processor/tokenizer、
embedding `.pt/.bin`、运行 metadata 和构建 Manifest。集成入口为根目录
`export_meta_info.json`，专用运行契约为 `runtime_meta_info.json`。推理不读取原始权重。

## 精度复现

同一 processor、4x、9 切片上限、关闭 thinking、greedy decoding 配对比较：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/minicpm_ocr/verify_accuracy.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --image "$TEXT_IMAGE" --image "$FORMULA_IMAGE" --image "$TABLE_IMAGE" \
  --output /tmp/minicpm_ocr_accuracy.json
```

2026-10-23 的 W8A8 融合包在三个样本中，文字、公式逐 token 一致，表格的空格、换行
不同（89 → 92 tokens）；严格序列一致为 2/3，去除空白后的文本一致为 3/3。
严格比较脚本因此退出 1，归一化文本比较单独记录，不替代严格门禁。
此前 W16 基线共 223 个生成 token，三个序列均与原生 BF16 一致。
这属于配对一致性验证，不是 OCRBench/OmniDocBench 全量准确率。
详见 [W8A8 验证记录](../../../docs/minicpm_ocr_rm_w8a8_validation_20261023.md)
和 [初次 W16 基线](../../../docs/minicpm_ocr_rm_validation_20261023.md)。
