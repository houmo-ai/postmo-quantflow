# 推测解码的权重所有权

## Qwen3.5 dense/MoE 与 Qwen3-Next

主模型拥有唯一的 head 量化载荷。MTP/DFlash/DFlash2 在支持的模型架构上继承
位宽、整数权重、scale 和格式。普通版/推测版必须从同一个 checkpoint 导出；
相同 seed 的两次独立搜索不能作为权重一致性保证。LoRA 只作用于主干增量分支，
MTP 输入使用带 LoRA 的主模型 hidden state，head 不单独校准。

unified 负责来源校验、冻结产物复用和发布；recipe 管理默认策略；GPTQModel
负责实际校准和保存，ModelZoo 负责消费与运行时存储复用。客户命令见
[unified 示例](../../examples_merak/unified_api/README.md)。

## Gemma4：结构调查结论及后续接入边界

Gemma4 当前 assistant 不是 Qwen 那种直接复用 target head 的 MTP 层。
[Gemma4AssistantDraftModule](../../xhmodel_merak/xh_llm/models/gemma4_series/gemma4_series_mtp_model.py)
从 assistant checkpoint 加载权重，`lm_head.weight` 绑定 assistant 的
`model.embed_tokens.weight`。`pre_projection` 把拼接的 target 输入映射到
assistant hidden size；`post_projection` 把 assistant hidden 映射回 target
hidden size。因此即使两个 head 的 vocab size 相等，也不能据此覆盖 assistant
head，更不能把它标记成 Qwen 的 target-owned shared_head。

- target prefill/verify：同一 target 权重应只量化一次并共享存储。
- assistant embedding/head：同一浮点参数的别名需保留；导出后只有量化格式、
  scale、轴/分组和整数载荷全部相同才可共享物理存储。
- E2B/E4B 的 `use_ordered_embeddings=true` 还包含 centroids、候选词选择和
  被选 embedding 的打分。centroids 不是 target head，不能拿 head 替换。
- assistant 与 target 的共享 KV 是已有的独立 ABI，不等价于共享权重；必须
  继续验证 full/sliding cache 的位置、长度与只读语义。

本轮没有注册未经验证的 Gemma4 unified MTP，也没有修改其 assistant 权重。
后续应分别用 E2B/E4B ordered 路径和 12B/31B/26B 全词表路径验证：权重来源、
量化次数、加载后的存储身份、candidate logits，以及关闭/开启 MTP 的确定性
解码结果。确认这些契约后再注册 unified 的 MTP 变体，不强制套用 Qwen Head W5。
