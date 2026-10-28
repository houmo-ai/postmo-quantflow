# OCR RM Qwen3.5 2B Value Head

统一模型 ID 为 `ocr-rm-2b-value`，实现位于
`xhmodel_merak/xh_llm/models/ocr_rm`。接入的是交付包中实际存在的
`qwen35_2b_l2_triple_value_checkpoint-3000`；包内 vocab-head demo 指向的另一份权重
未包含在压缩包中，本次没有为它登记模型。

此模型执行 OCR 一致性二分类：输入原图、OCR 文本和其渲染图，输出
`softmax(score(last_nonpad_hidden))[1]`。`score.weight` 为 `[2, 2048]`，无 bias。
交付配置误写为生成架构，加载器显式选择 `Qwen3_5ForSequenceClassification`，
缺少训练好的 `score.weight` 时终止，避免随机语言头替代评分头。

## 输入和部署约定

- 图像顺序为原图在前、渲染图在后；prompt 无 system、无 assistant 头，末尾保留换行。
- 原生图像下限为 64 tokens，保留 processor 的最大预算及 199:1 的补白保护。
  本次部署视觉档位为 96 / 384 / 1536 image tokens，每图选择最小可容纳档；超限拒绝。
- 位置编码在完整 prompt 上计算，再按 256-token chunk 切分。保留 interleaved MRoPE，
  不在一个图片被 chunk 截断后重新计算其位置。
- 默认 XH2a / W8A8 SEFP（`w8a8h1_sefp`）、8K context。复用现有 Qwen3.5 图及共享权重加载契约，
  包含 prefill/decode；评分只运行 prefill，不进行文本生成。
- W8A8 默认启用两项 GDR 融合，文本、视觉和 cumsum 的乘法输入采用 8/8，
  normalize 保留 FP32。cumsum 同时声明 `act_schema_2` 和转换为 Linear 后的
  `w_schema`；图中 pragma 读取请求位宽，避免把第一输入固定为 16。
- `--deployment-precision` 只修改基础图精度，保留 workflow 的融合策略与节点级例外。
  公开选项仅接受 `w8a8h1_sefp` / `w16a16h1_sefp`。OCR RM 禁止 SEFP h0，
  导出前检查有效配置，发布前逐图检查实际序列化的激活、权重 `hidden_bit=1`。
  不能把默认 W8 workflow 的基础精度覆盖为 W16 就称为全 W16：其 GDR 与 cumsum 仍为 8/8。
  W16A16h1 展开基线使用独立的 `ocr_rm_2b_value_xh2a_w16a16.yaml`，关闭 GDR 融合。
  当前 SDK 导出的 cumsum Linear 实际仍为 16/8h1，其余受审计的矩阵乘为 16/16h1；
  不能把这份基线称为所有矩阵乘均为 16/16。
- 统一推理返回 `output_type=classification`，`stats` 包含 `prob_0`、`prob_1`、
  `logit_0`、`logit_1` 和 `n_tokens`。概率表示“一致”的方向性分数。

## 构建和推理

使用 recipe 声明的 Transformers 5.16.1 环境和本地 SDK。`MODEL_DIR` 为上述 value
checkpoint 目录，`ARTIFACT_DIR` 必须不存在。仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ocr_rm/unified_demo.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --deployment-precision w8a8h1_sefp --device cuda:0

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ocr_rm/unified_demo.py \
  --artifact-dir "$ARTIFACT_DIR" --orig-image "$ORIG_IMAGE" \
  --render-image "$RENDER_IMAGE" --ocr-text-file "$OCR_TEXT_FILE"
```

发布目录使用 `visual_m96/`、`visual_m384/`、`visual_m1536/`、`prefill/`、`decode/`，
保留本地 processor/tokenizer、embedding、运行 metadata、档位清单和 Manifest。
原始目录中的训练记录和导出日志不进入发布包。共享图权重通过 HMONNX V2 加载，
runtime 只在创建会话时局部启用该选项，并恢复调用进程的环境。

导出时自动执行 h1 审计；也可单独复查现有发布包的全部五张图：

```bash
python examples_merak/llm/ocr_rm/audit_precision.py \
  --artifact-dir "$ARTIFACT_DIR" --output /tmp/ocr_rm_h1_audit.json
```

检查范围包括 prefill、decode、三个 visual 档位及子图中的 SEFP hidden-bit 属性；
若使用 GDR 融合，还检查当前 SDK 内部 MatMul 的 h1 配置。该检查不替代精度验证。

## 精度验证

`SAMPLES_DIR` 指向解压后的 `rm_hf_infer_demo_v1_20261015/`，含
`samples/manifest.json`。验证文本、公式、表格的全部 12 个样本，分别记录原生与
生产参考分数，不把分类判定一致等同于概率误差达标：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ocr_rm/verify_accuracy.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --samples-dir "$SAMPLES_DIR" --native-dtype float32 \
  --max-prob-error 0.005 --reference-max-prob-error 0.03 \
  --output /tmp/ocr_rm_accuracy.json
```

任一概率误差超过对应阈值时保存完整报告并以非零状态退出。生产参考为 BF16 /
batch=4，使用原包跨 dtype 的 0.03 阈值，与原生 FP32 的 0.005 阈值分别判断。
W8A8 融合包相对原生 FP32 最大误差为 0.04045719，相对生产参考最大误差为
0.04245502，超过上述概率门限，验证脚本按预期退出 1。两个参考的分类判定均为
12/12 一致。样本含原模型误判，判定一致率不代表任务准确率。
历史 W16 展开基线使用了禁止的 h0，其 0.00159222 误差不能用作当前 h1 版本的验证结论；
该旧包已废弃，保留记录仅供追溯。
相同 W8A8 配置仅关闭 GDR 融合的对照，在这 12 组样本上的最终 logits 和概率与融合版
完全相同，因此当前 W8A8 默认保留融合。该结论只覆盖本批样本。
当前重新导出的 h1 包、节点审计及精度结果见
[SEFP h1 验证记录](../../../docs/ocr_rm_sefp_h1_validation_20261026.md)。
历史 W8A8 对照见 [W8A8 验证记录](../../../docs/minicpm_ocr_rm_w8a8_validation_20261023.md)，
已废弃的初次 RM 基线见 [历史 W16 记录](../../../docs/minicpm_ocr_rm_validation_20261023.md)。
