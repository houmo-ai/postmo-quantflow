# WeMM-Embedding 与 Qwen3.5 差异说明

## 总览

WeMM-Embedding 使用 Qwen3.5 多模态主干和相近的 remote-code 模块树，因此可以复用 Qwen3.5 的模型加载、视觉编码、线性注意力、量化 recipe 和 HMONNX 导出基础设施。但两者的任务目标、输出语义、运行时接口和精度评价完全不同。

| 对比项 | WeMM-Embedding | Qwen3.5 生成式模型 |
| --- | --- | --- |
| 主要任务 | 多模态 embedding、检索、相似度 | 自回归文本生成、多模态理解与回答 |
| 最终输出 | 固定维度向量 | 词表 logits 和后续 token |
| `lm_head` | 导出时替换为 `Identity` | 保留词表投影层 |
| `num_logits_to_keep` | 0 | 通常为 1 或生成配置要求的值 |
| 输出图语义 | 全序列 hidden states | prefill/decode logits、cache/state |
| 后处理 | 最后有效 token pooling + FP32 L2 normalize | sampling/argmax、token decode |
| 公开接口 | `embed()`、`embed_items()`、`embed_texts()`、`embed_images()` | `generate()` 或逐 token prefill/decode |
| decode | 不支持，也不需要 | 生成流程的核心阶段 |
| page attention | 不属于当前 embedding runtime | 可用于生成式 cache 管理 |
| 评价指标 | Recall、MRR、nDCG、embedding cosine | 生成准确率、PPL、任务得分、文本质量 |

## 模型结构复用与专用修改

### 复用部分

WeMM 复用 Qwen3.5 的：

- 多模态模型包装和模块命名；
- 视觉编码器导出；
- 语言主干和线性注意力实现；
- HMONNX 基础导出逻辑；
- GPTQModel 的 Qwen3.5 AutoRound 和 GPTQ recipe；
- W4A8/W8A8 的 HMONNX 量化配置能力；
- processor 的主要多模态输入组织能力。

### WeMM 专用部分

WeMM 单独注册 `WeMMEmbedding`，并实现以下行为：

1. 导出包装完成后把 `lm_head` 替换为 `Identity`；
2. 将语言图输出定义为 `hidden_states`；
3. 从 checkpoint 动态读取 embedding dimension；
4. 运行时选择最后一个有效 token；
5. 使用 FP32 执行 L2 normalize；
6. 拒绝词表维度 logits，避免生成式旧图被误用；
7. processor 不追加用于生成回答的 generation prompt；
8. Golden 数据调用 `embed()`，不调用生成接口。

三种 WeMM 规格输出维度：

| 规格 | 输出维度 |
| --- | ---: |
| 2B | 2048 |
| 4B | 2560 |
| 9B | 4096 |

## 输入与输出语义

Qwen3.5 的输入通常被组织为对话 messages，模型输出下一个 token 的 logits，并在 decode 阶段不断更新 cache，直到生成结束。

WeMM 的输入被组织为一个待编码 item，例如：

```python
{"text": "A dog playing in a park."}
{"image": "/path/to/image.jpg"}
{"image": "/path/to/image.jpg", "text": "Represent this image."}
```

WeMM 不生成回答。它只执行编码并返回：

```text
[batch, embedding_dimension]
```

输出为 FP32，且每一行的 L2 norm 约等于 1，可直接用于 cosine/点积相似度。

## Prefill、chunk 和长输入差异

### Qwen3.5 生成式模型

当 Qwen3.5 接收约 2000 token prompt 时，运行时可以按静态 prefill 图长度多次执行图，并在各段之间维护线性注意力状态、卷积状态或其他 cache。完成整个 prompt 的 prefill 后，模型基于累积状态产生第一个新 token，再进入 decode。

因此，生成式模型的“多段 prefill”不仅是把每段结果相加，而是严格按模型定义更新和传递状态；最终目标是得到正确的下一 token logits 和后续 decode 初始状态。

### WeMM embedding

当前 WeMM HMONNX 图同样使用 `prefill_chunk_length: 256`，底层 Qwen3.5 配置保留 `context_max_length: 2048` 和线性注意力 `linear_chunk_size: 64`。公开的 WeMM runtime 会为完整输入生成位置编码，再按 256 token 重复执行 prefill。

一条 2000-token 输入会分 8 段执行。实现中会：

- 复用 full-attention KV cache；
- 复用线性注意力 recurrent state 和卷积 cache；
- 从完整序列切片三维位置编码，避免每段重新从零编号；
- 为末段构造有效长度和 linear-attention mask；
- 只从末段最后一个有效位置取得 hidden state。

因此当前公开接口的累计上限由 `context_max_length=2048` 决定，而单次图长度仍是 256。多段调度已有单元测试覆盖，但长输入 embedding 与 Native 全长前向的真实精度对齐仍需使用实际 HMONNX 产物验证。

## `linear_chunk_size=64` 的含义

`linear_chunk_size: 64` 是线性注意力模块内部进行 chunkwise 计算时的块大小，来源于当前 Qwen3.5/WeMM 导出实现与算子配置。它与外层 HMONNX prefill 图的 256-token 静态长度不同：

- 外层一次 prefill：最多 256 token；
- 线性注意力内部：可按 64-token 子块执行 chunkwise 状态更新；
- 一次 256-token prefill 内部通常可包含 4 个 64-token 子块；
- 2000-token 输入需要多次 256-token prefill，并在每次图执行之间继续传递状态。

把内部 chunk 从 64 改成其他值会影响算子实现、状态语义、性能和数值行为，不能只作为普通评测参数随意调整。

## 量化复用关系

### 为什么 WeMM provenance 中出现 `qwen35`

WeMM 的语言和多模态模块树与 Qwen3.5 对齐，GPTQModel 已将 WeMM 定义为：

```text
WeMMEmbeddingQModel(Qwen3_5_VLQModel)
```

因此真实调用的共享量化函数是：

```text
AutoRound: gptqmodel.recipes.qwen35_autoround.quantize_qwen35_autoround
GPTQ:      gptqmodel.recipes.qwen35.quantize_qwen35
```

后端原始 provenance 使用 `qwen35_*` 文件名是为了准确记录 recipe 来源，不表示量化后的权重变成了 Qwen3.5 模型，也不表示加载了 Qwen3.5 checkpoint。实际 `model_dir` 仍是 WeMM-Embedding checkpoint，GPTQModel 通过 WeMM auto-map 选择 `WeMMEmbeddingQModel`。

为兼顾真实来源和交付可读性，WeMM workflow 同时保留：

```text
qwen35_autoround_recipe_provenance.json
wemm_embedding_autoround_recipe_provenance.json

qwen35_quant_recipe_provenance.json
wemm_embedding_gptq_recipe_provenance.json
```

同一方法的两个文件内容一致。

### 量化与导出边界

GPTQModel 负责把 FP 权重量化并保存为 GPTQModel HF checkpoint；xh2modelzoo 负责选择 recipe、解析校准资源、组织目录，并将量化 checkpoint 导出为 XH2a HMONNX。

WeMM 的语言分支使用 4bit 权重，视觉分支按 W8A8 HMONNX 导出。Qwen3.5 与 WeMM 可以共享底层量化算法，但不能共享任务精度结论：生成模型需要评估生成任务，embedding 模型需要评估检索和向量对齐。

共享 Qwen3.5 GPTQ 脚本默认包含生成、PPL 和可选视觉生成校验。WeMM GPTQ recipe 明确跳过这些生成式校验，因为它们无法证明 embedding 的 shape、归一化、cosine 或检索精度。导出后通过独立数据集评测检查检索质量和 Native/HMONNX cosine。

## 校准序列与运行时长度

AutoRound WeMM recipe 使用 `seqlen=2048`，GPTQ recipe 使用 `seqlen=1024`。这些参数描述量化校准样本长度，用于覆盖权重在不同 token 位置和上下文下的激活/误差分布。

校准序列较长可以帮助量化过程看到更接近长文本的激活，但不会自动生成多 chunk HMONNX runtime。当前运行时已经实现多段调度，累计输入上限为 `context_max_length=2048`；校准 `seqlen` 仍不等于这个运行时上限。

如果未来改用 caption 更长的数据集，应分别处理两个问题：

1. **量化覆盖**：统计 tokenizer 后长度分布，根据目标流量选择校准样本和 `seqlen`；
2. **推理能力**：验证超过 256 token 的多 chunk embedding 精度，并在数据超过 2048 token 时明确采用扩容、截断或文档分段策略。

这两个问题相关，但不能互相替代。

## 精度评价差异

Qwen3.5 的常见精度验证关注：

- PPL；
- 任务准确率；
- 多模态问答结果；
- 生成文本质量；
- Native 与 HMONNX logits/token 一致性。

WeMM 的当前精度验证关注：

- text-to-image Recall@1/5/10；
- image-to-text Recall@1/5/10；
- MRR@10 和 nDCG@10；
- Native BF16 与 HMONNX embedding 的逐行 cosine；
- embedding shape、dtype 和 L2 norm；
- 输入顺序及图文正样本映射。

因此，即使 WeMM 和 Qwen3.5 共享 backbone 和量化 recipe，也必须保留独立的 embedding 评测流程和模型卡证据。

## 结论

WeMM-Embedding 是“基于 Qwen3.5 多模态主干的 embedding 模型适配”，不是“去掉生成循环的普通 Qwen3.5”。正确交付必须同时处理 hidden-state 输出、pooling、归一化、长度契约、统一 item 接口、检索精度和 provenance；共享 Qwen3.5 代码只解决底层模型和量化基础设施复用问题。
