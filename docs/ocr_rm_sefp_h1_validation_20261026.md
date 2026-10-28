# OCR RM：SEFP h1 重新导出与验证（2026-10-26）

本次只处理 `ocr-rm-2b-value`。最初 W16 展开包的基础 QuantScheme 为
`w16a16h0_sefp`，实际图中基础 Linear 权重的 `hmfp_weight_hidden_bit=0`，
不符合禁止 SEFP h0 的要求，旧包废弃，其精度数字仅保留用于历史追溯。
随后交付的 W8 融合包实际已为 h1；本次仍从原始 checkpoint 重新导出新包。

## 配置与防回退

- 默认文本和视觉精度保持 `w8a8h1_sefp`，两项 GDR 融合保持启用，normalize 保留 FP32。
- W16 展开 YAML 的文本和视觉基础精度改为 `w16a16h1_sefp`，保持关闭 GDR 融合。
- 两份 YAML 的 cumsum `act_schema`、`act_schema_2`、`w_schema` 均显式设置
  `hidden_bit: true`；W8 请求 8/8，W16 请求 16/16。常量第二输入会下沉为 Linear，
  因此必须检查 `w_schema` 及序列化图，不能只看 `act_schema_2`。
- recipe 和示例公开精度选项只接受 W8h1 / W16h1，移除 W16h0。
- OCR RM workflow 在导出前递归检查有效配置，包括节点级例外；发布前检查 prefill、decode、
  三个视觉档位及 ONNX 子图的实际 hidden-bit 属性。发现 h0 即终止。
- 融合 GDR 的内部乘法由当前 SDK 的 `GDR_DEFAULT_HMFP_CONFIG` 指定，
  BlockTriInverse / ChunkScan 使用 8/8、两侧 hidden-bit 为 true；RecurrentScan
  当前使用逐元素更新，不调用该 MatMul。没有更改 SDK 实现。

这里的 W8/W16 指 SEFP 运算精度，不是整数 INT8，也不表示 normalize 等所有操作都采用该精度。
`--deployment-precision w16a16h1_sefp` 只覆盖默认 workflow 的基础图配置；不会关闭 GDR
或修改 cumsum 的节点级例外。本次额外的 W16 基础精度展开基线使用独立 W16 YAML。
实际审计发现当前 SDK 下沉的 18 个 cumsum Linear 仍为 16/8，即使 YAML 的 `w_schema`
请求为 16；此项不是全 16/16，报告明确保留这个位宽例外。两侧 hidden-bit 都为 h1。

## 新产物与节点审计

新工作目录为 [ocr_rm_sefp_h1_20261026](../work_dirs/ocr_rm_sefp_h1_20261026/)。
两份发布目录均采用统一布局：根目录 `export_meta_info.json` 资源清单、
`model_build_manifest.json`、运行 metadata、`prefill/`、`decode/`、
`visual_m96/`、`visual_m384/`、`visual_m1536/` 与包内 tokenizer/processor/embedding。
ONNX 文件名可能包含源 checkpoint 的 `w16a16`；实际部署精度以 Manifest、有效配置和节点审计为准。

W8 使用公开 `unified_demo.py` → `build_model`。W16 展开基线通过
[可复现脚本](../work_dirs/ocr_rm_sefp_h1_20261026/build_w16a16h1.py) 解析同一 recipe，
替换为修正后的 W16 YAML 及其 SHA-256，再执行统一 `execute_build`；没有增加另一个模型 ID。

| 新包 | SEFP h1 属性数 / h0 | 最大 FP32 误差 | 最大 BF16 参考误差 | 两参考判定一致 | 概率门限 |
| --- | ---: | ---: | ---: | --- | --- |
| [W8A8h1，GDR 融合 ZIP](../work_dirs/ocr_rm_sefp_h1_20261026/ocr-rm-2b-value_w8a8h1_gdr_release.zip) | 2470 / 0 | 0.04045719 | 0.04245502 | 12/12、12/12 | 未通过 |
| [W16 基础精度 h1，展开 ZIP](../work_dirs/ocr_rm_sefp_h1_20261026/ocr-rm-2b-value_w16a16h1_unfused_release.zip) | 3694 / 0 | 0.00186841 | 0.00345531 | 12/12、12/12 | 通过 |

两份均独立检查 prefill、decode、三个 visual 档位。W8 中受审计的 Linear / MatMul /
GroupMatMul 均为 8/8h1；W16 的 18 个 cumsum Linear 为 16/8h1，其余为 16/16h1，且没有 GDR 融合节点。

原始证据：[W8 节点审计](../work_dirs/ocr_rm_sefp_h1_20261026/w8a8h1_audit.json)、[W16 节点审计](../work_dirs/ocr_rm_sefp_h1_20261026/w16a16h1_audit.json)、
[W8 精度](../work_dirs/ocr_rm_sefp_h1_20261026/w8a8h1_accuracy.json)、[W16 精度](../work_dirs/ocr_rm_sefp_h1_20261026/w16a16h1_accuracy.json)。
含原图、渲染图、OCR 文本和全部分数的 [新样本报告](../work_dirs/ocr_rm_sefp_h1_20261026/reports/sample_report.html)，
另有 [CSV](../work_dirs/ocr_rm_sefp_h1_20261026/reports/sample_scores.csv)。

旧 W16h0 包的审计明确失败，见 [拒绝日志](../work_dirs/ocr_rm_sefp_h1_20261026/reject_previous_h0.log)。
W8 精度脚本退出 1，W16 退出 0；两份图的 h1 审计均退出 0。

ZIP 仅包含 Manifest 登记的运行文件及 Manifest 本身，不包含推理缓存或日志；
逐文件复核 ZIP CRC 和 payload SHA-256，校验值见 [SHA256SUMS](../work_dirs/ocr_rm_sefp_h1_20261026/SHA256SUMS)。

两份均在 `/tmp`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 条件下仅通过公开
`inference_model` 加载包内模型资源，s01 的两类 logits、概率及 token 数与各自完整报告完全一致。
结果见 [W8 离线推理](../work_dirs/ocr_rm_sefp_h1_20261026/w8a8h1_offline_inference.json)、
[W16 离线推理](../work_dirs/ocr_rm_sefp_h1_20261026/w16a16h1_offline_inference.json)。

## 精度验证协议

使用交付包全部 12 个样本，包含文本、公式、表格。保持原图 → OCR 文本 → 渲染图的 prompt，
保留图像预算、MRoPE、最后有效 token pooling、8K context 和 256-token prefill chunk。
原生参考使用相同 checkpoint 的 `Qwen3_5ForSequenceClassification` / FP32，
另与包内生产 BF16 / batch=4 参考分别比较。

保持既定门限：原生 FP32 最大概率绝对误差 ≤0.005，生产 BF16 参考 ≤0.03。
概率未达标时脚本保存完整报告并退出 1，未调整门限。分类判定以 P(1)≥0.5 判断，
判定一致不代表概率达标，也不等于相对人工标签的任务准确率。

W8 新包在全部样本上的两类 logits 和概率与上次 W8h1 包完全一致，
最大差值为 0，见 [逐项对照](../work_dirs/ocr_rm_sefp_h1_20261026/previous_w8_comparison.json)。
这支持上次 W8 本身已是 h1 的结论；不能据此推断 h0 与 h1 数值等价。

## 复现

在仓库根目录，使用任务虚拟环境
`work_dirs/minicpm_ocr_20261023/env_ci/bin/python`。环境为 Python 3.12.14、
PyTorch 2.8.0+cu128、Transformers 5.16.1、xhquanttool 发行包 0.9.2、ONNX 1.18.0。
部署模型不运行 GPTQ / AutoRound 权重量化。`MODEL_DIR` 指向交付的
`qwen35_2b_l2_triple_value_checkpoint-3000`，`SAMPLES_DIR` 指向含 `samples/manifest.json`
的 `rm_hf_infer_demo_v1_20261015`。

```bash
python examples_merak/llm/ocr_rm/unified_demo.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --deployment-precision w8a8h1_sefp
python examples_merak/llm/ocr_rm/audit_precision.py \
  --artifact-dir "$ARTIFACT_DIR" --output /tmp/ocr_rm_h1_audit.json
python examples_merak/llm/ocr_rm/verify_accuracy.py \
  --model-dir "$MODEL_DIR" --artifact-dir "$ARTIFACT_DIR" \
  --samples-dir "$SAMPLES_DIR" --native-dtype float32 \
  --max-prob-error 0.005 --reference-max-prob-error 0.03 \
  --output /tmp/ocr_rm_h1_accuracy.json
```

## CI、审查与范围

本次 12 个变更路径由当前 impact selector 选中 17 个测试文件，**810 passed**，
耗时 174.34 秒。记录见 [选择及原因](../work_dirs/ocr_rm_sefp_h1_20261026/ci_selection.txt)、
[CI 日志](../work_dirs/ocr_rm_sefp_h1_20261026/ci.log)。前置小模型及契约检查为 20 passed，
已包含在 810 项中，不重复累计。

回归包括公开精度拒绝 h0、基础/视觉/节点/cumsum 隐藏覆盖拒绝、真实小模型统一构建、
移动并删除源目录后的推理；在实际 SDK 导出图中将一个权重 hidden-bit 注入为 0，
确认审计拒绝，即使声明的 QuantScheme 仍为 h1。
7 个本次修改的 Python 源文件通过 Black 检查；人工复核模型包、recipe、CLI、workflow、
实际图审计和精度报告。当前环境未提供 CodeGraph 工具，也没有 `ocr` 可执行文件，
因此未执行仓库要求的 `ocr review --audience agent`，没有把人工复核写成工具审查通过。

以上为 A100 上的 HMONNX GPU 仿真及本批样本验证；未做 XH2a 真机验证或全量任务评测。
本次 CI 范围为本次 h1 变更，不代表整仓所有未提交改动通过全量 CI。
