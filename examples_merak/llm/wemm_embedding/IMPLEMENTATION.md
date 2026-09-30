# WeMM-Embedding 实现说明

## 设计目标

WeMM-Embedding 的适配目标不是复制一套 Qwen3.5，而是在复用已维护的 Qwen3.5 多模态导出、量化和 HMONNX 基础设施的同时，隔离 embedding 模型独有的输出、pooling、接口和验证规则。

当前实现覆盖 WeMM-Embedding 2B、4B 和 9B，并保持以下不变量：

1. 导出图输出 hidden states，不输出词表 logits；
2. 一个输入 item 只产生一行 embedding；
3. 每条输入的最后一个有效 token 必须是 `<embedding>`，pooling 使用该 token；
4. 最终 embedding 使用 FP32 L2 归一化；
5. embedding 维度来自 checkpoint，而不是写死为某个规格；
6. 当前运行时顺序重复执行 prefill 图并复用缓存，不对外提供生成式 decode；
7. 4bit 权重量化的 `group_size` 固定为 64。

## 模型适配

专用注册位于 `xhmodel_merak/xh_llm/models/wemm_embedding/`。其核心行为如下：

1. `XHWeMMEmbeddingModel` 继承 Qwen3.5 导出实现；
2. `init_wrap_model()` 在 Qwen3.5 完成 Hugging Face 模型包装后，将 `lm_head` 替换为 `torch.nn.Identity`；
3. 第一个语言图输出按 embedding 语义命名为 `hidden_states`；
4. `_extra_export_metadata()` 从 checkpoint 的 `text_config.hidden_size` 读取 embedding 维度；
5. metadata 记录 `embedding_output`、`embedding_pooling`、`embedding_normalize` 和 `embedding_dimension`；
6. HMONNX runtime 验证输出最后一维与 metadata 一致，防止旧图的 `[B,T,vocab_size]` logits 被误当成 embedding。

三种规格的 hidden size 分别为 2048、2560 和 4096。

## 推理与 pooling

`embed()` 接收包含 text、image、video 或交错多模态内容的 mapping。图片 patch 通过 Qwen3.5 的 `[96, 196, 384, 704, 1536]` 视觉档位图编码；视频处理器输出的扁平 patch 按原始顺序划分为双帧时间单元，逐单元复用相应档位图。Qwen3.5 原生视觉注意力也按时间单元独立计算。语言预处理器分别注入图像和视频特征，并按时间戳展开视频 RoPE grid。`embed_items()`、`embed_texts()` 和 `embed_images()` 提供批处理便利接口，但内部仍保持“一条 item 对应一行向量”的契约。

运行时步骤：

1. processor 将 item 转换为模型输入；
2. 检查 processed-token 长度不超过 `context_max_length`；
3. 对完整输入生成文本或多模态位置编码；
4. 按 `prefill_chunk_length` 切分，尾段 padding 到固定图长度；
5. 顺序执行 HMONNX prefill，并在各段之间复用 KV、卷积和 recurrent cache；
6. 校验最后一个有效 token 是 `<embedding>`，并从最后一段中选择该 token 的 hidden state；
7. 转为 FP32 并执行 L2 normalize；
8. 校验输出 shape 为 `[batch, embedding_dimension]`。

当前 W4 HMONNX 图的 `prefill_chunk_length` 为 256，metadata 中 `context_max_length` 为 2048。运行时已实现 256-token 图的重复执行，因此 257～2048 processed tokens 会分段处理；超过 2048 会在执行前明确报错。

## AutoRound 量化

六份 YAML 都把 `quant.algorithm` 设为 `gptqmodel`（GPTQModel 库后端），`quant.method` 才是具体算法：`autoround` 或 `gptq`。二者不是重复字段。

AutoRound 配方的 `mode: llm-only` 由 AutoRound API 接收。GPTQ API 没有
`mode` 参数；其 Qwen3.5 VL 模型定义只遍历
`model.language_model.layers` 进行权重量化，视觉权重保留并在导出时按 W8A8
方案处理。GPTQ 的 `preset: full_vlm` 保留视觉模型及 processor，
不表示视觉权重参与 GPTQ 权重量化。

AutoRound recipe 的统一参数：

- GPTQModel `qwen35_autoround` recipe；
- 仅量化语言分支；
- W4、对称量化、group size 64；
- 400 次迭代；
- 256 个校准样本；
- 校准 `seqlen=2048`；
- seed 42；
- 保存为 GPTQModel-compatible `auto_gptq` checkpoint；
- HMONNX 语言分支 `w4a8h0_ssfp`；
- HMONNX 视觉分支 `w8a8h1_sefp`；
- 视觉输入使用与 Qwen3.5 相同的 `[96, 196, 384, 704, 1536]` patch 档位，实际 patch 按图像网格补齐到档位容量。

共享调用为：

```text
gptqmodel.recipes.qwen35_autoround.quantize_qwen35_autoround
```

共享后端生成 `qwen35_autoround_recipe_provenance.json`，WeMM workflow 在不删除原文件的前提下复制为 `wemm_embedding_autoround_recipe_provenance.json`。

`qwen35_autoround` API 不接收 `check_quant_vision_demo` 或 `skip_inference`，也不运行 Qwen3.5 的生成式校验；AutoRound YAML 因此不写无效的 `quant.validation` 字段。导出后的 embedding 质量通过独立的检索与 Native/HMONNX cosine 评测检查。

## GPTQ 量化

GPTQ 接入不在 xh2modelzoo 中重新实现 GPTQ 算法。GPTQModel 已提供 `WeMMEmbeddingQModel(Qwen3_5_VLQModel)` 和自动模型映射，xh2modelzoo 继续复用 Qwen3.5 的 adapter：

```text
gptqmodel.recipes.qwen35.quantize_qwen35
```

GPTQ recipe 的统一参数：

- `method: gptq`；
- W4、group size 64；
- `hessian_mse: true`；
- 256 个校准样本；
- 校准 `seqlen=1024`；
- `batch_size=1`；
- `wait_for_submodule_finalizers=true`；
- `quant_nontext_module=false`；
- 跳过共享 Qwen3.5 脚本的 generate/PPL 后验；
- seed 42；
- 输出 GPTQModel HF checkpoint；
- 后续按语言 `w4a8h0_ssfp`、视觉 `w8a8h1_sefp` 导出 HMONNX。

GPTQModel 后端生成 `qwen35_quant_recipe_provenance.json`，WeMM workflow 复制一份 `wemm_embedding_gptq_recipe_provenance.json` 作为交付别名。共享脚本的 generate、PPL 和视觉生成校验面向 Qwen3.5 生成模型，因此 GPTQ recipe 使用 `check_quant_vision_demo=false` 和 `skip_inference=true`。导出步骤不自动执行 embedding 精度后验；完整检索和 Native/HMONNX cosine 由独立评测流程承担。

AutoRound 与 GPTQ 使用不同校准参数：AutoRound 的迭代优化使用 256×2048，GPTQ 的 Hessian/误差估计使用 256×1024。二者均不改变 HMONNX 单次 prefill 图的 256-token 长度或单条输入累计 2048-token 上限。

## Workflow 组织

CLI `preflight_recipe()` 在量化前检查模型目录、`config.json` 和校准数据，并在预检结果中返回解析后的 `calibration_jsonl`。AutoRound 的 `xh2modelzoo://` 校准文件与 GPTQ 的 `gptqmodel://` 校准文件分别解析到仓库和 GPTQModel 包中的实际文件；测试确认两种解析结果与 Qwen3.5 量化适配器最终使用的路径一致。YAML 中的 `export.model.hf_model: null` 是占位。WeMM workflow 先确认该字段存在且原值为 YAML null，再由 `BaseLLMWorkflow._build_export_config()` 覆盖成量化产物目录；覆盖后仍为 null、空字符串，或目标不是目录时直接失败。语言与视觉 `quant_scheme.ops` 均为空，MatMul 使用当前 Qwen3.5 9B 配方的默认量化规则；移除旧的第二操作数 16bit 覆盖后须重新验证精度。

统一入口 `wemm_embedding_workflow.py` 使用两个正交参数：

- `--size {2b,4b,9b}` 选择模型规格；
- `--quant-method {autoround,gptq}` 选择量化方法。

AutoRound 保持原有目录：

```text
work_dirs/wemm_embedding/<size>/wemm_embedding_quant
work_dirs/wemm_embedding/<size>/wemm_embedding_export
```

GPTQ 使用独立目录：

```text
work_dirs/wemm_embedding/<size>/wemm_embedding_gptq_quant
work_dirs/wemm_embedding/<size>/wemm_embedding_gptq_export
```

这种隔离避免 GPTQ 覆盖已有 AutoRound checkpoint、导出图和精度证据。`--skip-quant` 与 `--quant-model-dir` 对两种方法都适用，但调用者必须选择与 checkpoint 匹配的 `--quant-method`。

Golden 数据通过 WeMM 的 `embed()` 生成，而不是调用 Qwen3.5 的生成接口。支持 text、image 和 mixed 三种代表性输入。

## 评测设计

Flickr30K 通过通用 `RetrievalDataset` 契约接入。数据集 loader 负责返回图像路径和每张图像的一条或多条正样本 caption，共享评测代码负责：

- 样本顺序；
- Native 和 HMONNX embedding 生成；
- Recall@1/5/10；
- MRR@10 和 nDCG@10；
- embedding 持久化；
- 同序 Native/HMONNX 全量逐行 cosine；
- 并行 worker shard 的有序合并，缺失、重叠或空洞区间直接失败。

新增 image-caption 数据集时应注册新 loader，而不是复制一套 evaluator 和指标实现。若数据集文本显著变长，还应记录 tokenizer 后的长度分布，并分别评估校准覆盖、截断策略和运行时多 chunk 能力。

## CI 设计

便携测试覆盖：

- WeMM 专用模型和 workflow 注册；
- 2B/4B/9B 动态 embedding 维度；
- `lm_head=Identity`；
- hidden-state pooling 和 FP32 L2 normalize；
- logits 维度防回归；
- 256-token 多段 prefill、真实尾段长度和 cache/state 复用；
- 视觉 token 跨越 256 边界时的 feature 切片和完整位置编码；
- 2048/2049 context 边界；
- 统一 embedding 便利接口；
- AutoRound recipe 字段；
- GPTQ recipe 字段及 Qwen3.5 adapter 参数翻译；
- GPTQ workflow 分支路由；
- AutoRound/GPTQ provenance 双文件；
- CLI 的规格和量化方法路径解析；
- 检索指标、数据集注册和结果汇总工具；
- `hf_model: null` 占位由 `XHWeMMEmbeddingWorkflow._build_export_config()` 处理：基类覆盖前检查字段存在且原值是真正的 YAML null，覆盖后拒绝 leftover null、文件路径和缺失目录；
- `resolve_xh2modelzoo_uri()` 解析 `xh2modelzoo://` / `repo://` / `gptqmodel://` / `file://`、绝对路径和相对路径；
- Flickr30K 并行 shard 用 `validate_shard_ranges()` 检查半开区间合并、相邻合法、重叠/空洞，以及覆盖长度必须等于数据集大小；`wait_for_worker_processes()` 在 worker 失败、超时或 join 异常时 fail closed，不会合并残缺结果；
- 当前 CI 目录中的主契约、support 和 eval 测试，以及与它们同名的 impact 声明；support
  impact 显式包含 WeMM 直接复用的 Qwen3.5 源文件。

本次提交没有交付依赖真实 checkpoint/HMONNX 产物的 pytest 集成测试。2B/4B/9B 的 AutoRound
导出、Flickr30K embedding 和 MMEB-v3 结果由评测脚本在仓库外部运行产物上验证；原始产物位于
忽略提交的 `work_dirs`，提交中只保留可复现脚本、固定协议和汇总报告。

## 验证边界

旧产物及其精度结果已清理。当前 AutoRound 配方使用 `int_sym_xh`，2B/4B/9B 的真实量化、HMONNX 导出和图像精度证据见 [精度复测报告](ACCURACY_REPORT.md)。MMEB-v3 本轮仅覆盖 Visual7W 和 ChartQA 两项图像任务。

GPTQ 当前证明范围是：

- GPTQModel 已注册 WeMM 模型定义；
- xh2modelzoo 三规格 GPTQ recipe 可解析；
- recipe 参数可正确翻译到共享 Qwen3.5 GPTQ 后端；
- WeMM workflow 能进入 GPTQ 分支；
- GPTQ provenance 可生成 WeMM 别名；
- 旧 GPTQ 量化和 HMONNX 导出产物已清理；

GPTQ 尚未执行 Flickr30K-1000 全量检索，因此不能引用 AutoRound 的 Flickr 指标作为 GPTQ 精度结论。
