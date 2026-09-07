# Ling MLA：先融合再量化，K/V 分开缓存

适用范围：Ling-3.0-Flash、XH2a Merak、G64 W8 attention。核对日期：2026-09-07。
这里的融合是权重吸收，不是把 GPU 的单 latent cache 搬到芯片上。

## 不能改变的部署约束

令 C 为 KV latent 维度，R 为 RoPE 维度。当前模型 C=512、R=64、H=32。

| 路径 | 图中的形状 | SEFP 指数对齐方向 |
| --- | --- | --- |
| Q | `[B,H,S,C+R]` | QK 的规约维，即 head_dim |
| K cache | `[B,1,T,C+R]`，内容为 latent＋RoPE | head_dim；当前 64 个元素一组 |
| V cache | `[B,1,T,C]`，内容为另一份 latent | PV 的规约维，即 token 轴；当前 64 个元素一组 |

两个 cache 包含相同的浮点 latent，不等于包含相同的量化值。一个 outlier 在
K 侧影响同一 token 的一组通道，在 V 侧影响同一通道的一组 token。
因此不能共享 mantissa、exponent、cache 存储，也不能先按 K 量化再转置当 V 使用。
可以共用浮点 latent 的计算，但必须从浮点值分别进入两条量化路径。

`_LingMLAAttention` 继续使用两个 `LLMCacheV2`。cache shape、更新轴、runtime
输入顺序没有改变；打开 FlashAttention 也不改变这套 ABI。
这是当前 XH2a 数据流的约束，不推广为所有芯片或所有 MLA kernel 的限制。

## 修复的是权重量化顺序

旧路径：原始 Q/K/V/output 分别 GPTQ 或 AutoRound → 解包 → 相乘融合 → 新矩阵 W8。
最后一次 W8 未参与前面的 Hessian/输出误差优化，存在二次量化。
原矩阵的整数和 scale 不能直接贴到矩阵乘积上。

新路径：**原始浮点权重 → FP32 相乘融合 → GPTQ/AutoRound → 打包 → 直接导出**。
GPTQ 在新 Linear 上采集 Hessian；AutoRound 优化融合后的 decoder block。
这对齐权重边界，不声称训练模拟了全部 SEFP 激活、累加和 cache 舍入。

按每个 head、PyTorch `[out,in]` 约定：

```
q_absorbed_proj = W_UK.T @ W_Q_nope
q_rope_proj     = W_Q_rope
absorbed_dense = concat_heads(W_O_head @ W_UV)
```

Q 侧保留 `q_lora_rank=None` 和 `q_a_proj → norm → q_b_proj` 两种入口；
KV down-proj、norm、RoPE、gate 不删除。softmax scale 保留原模型值（含 YaRN 修正），
不能按融合后的 C+R 重新计算。矩阵乘法用 FP32，结果落回源权重 dtype 后再优化。

当前 `xh2_absorbed_v1` 支持无 gate、head-wise gate，以及原有 bias。
head-wise gate 可移到 latent context；element-wise gate 不能穿过 W_UV，
必须 value-up → gate → dense，因此新布局明确拒绝它，不能强行融合。
旧 `hf_unabsorbed` 导出仍保留 element-wise 分支。

## 跨仓库分工与格式

- GPTQModel：`models/definitions/ling3_flash_mla.py` 拥有变换、浮点 forward、
  布局校验和 checkpoint hook；公共 Ling recipe 的两种算法共用它。
- GPTQModel 的 `Ling3FlashQModel.module_tree` 包含三个新 Linear。
  KDA decay/beta、MoE 位宽和 MTP 排除规则不变；只变换 decoder 的 MLA，不变换 MTP。
- ModelZoo 的 [adapter](../../../xhmodel_merak/xh_llm/models/ling_3_flash/quant_adapter.py)
  和两份 YAML 默认传 `mla_weight_layout: xh2_absorbed_v1`。
  后端通用 recipe 保留 `hf_unabsorbed` 默认；直接调用后端须显式选择 XH2 布局。
- `config.json` 保存 `ling_mla_layout` 和 `ling_mla_transform`。原有 config 已保存
  head/latent/RoPE 维度、gate；recipe provenance 记录来源、布局和训练参数。
- 保存时只给**产物副本**的 `modeling_bailing_moe_v3.py` 加配置驱动的构造 hook，
  不修改原始权重目录。HF/GPTQModel 重载时先建融合后的空结构，再装 packed tensor；
  resave 保留 hook 和布局。需要配套 GPTQModel；缺少实现会明确失败，不随机初始化旧投影。
- [ModelZoo wrapper](../../../xhmodel_merak/xh_llm/models/ling_3_flash/_model.py)
  识别新布局后复用同一 Linear 和 `quant_weight`，只接算子/cache，不再乘权重。
  普通加载和分层加载使用同一契约。未知版本、结构与 metadata 不符会报错。

必须从原始未量化 HF 权重开始。新布局不接受旧 W8 模型作为融合输入，
也不支持 disk offload 下未物化的原权重。旧量化产物仍可导出，但仍有原来的二次量化；
改目录名或改 config 标签不能将其升级成新布局。

## RoPE 是派生状态，也必须校验

Ling 的 `inv_freq` 不写入 checkpoint，远端 `_init_weights` 又只处理 Linear/Embedding。
HF meta 加载后可能留下未初始化的 `inv_freq`，即使 missing/unexpected keys 都为空。
小模型实测出现过校准前 cos/sin 已含 NaN，不能把它归因于 AutoRound 优化。

后端加载后按 rotary 模块自己的 config/初始化函数重建 `inv_freq` 和
`original_inv_freq`，恢复 attention scaling；不能用普通 attention head_dim 替代 R，
也不能只做 `nan_to_num`。量化 loss 的非有限保护仍保留。

## 验证与使用边界

[当前 CI](../../../benchmark_test/ci_test_transformer5.16.0/test_ling_mla_quantization_boundary.py)
覆盖原分支浮点等价性、新分支整数载荷保留、SSFP unfold、不同 SEFP 对齐轴的反例、
独立 cache、错误布局，以及普通/分层 packed 加载。loader fixture 使用
[仓库内固定版本的建模代码与小配置](../../../benchmark_test/ci_test_transformer5.16.0/fixtures/ling3_flash/README.md)，
构造一层 MLA、小 MoE 和确定性权重，不构造与本用例无关的 KDA。仅 CI 建模副本将
FLA 导入延迟到 KDA 构造/执行；MLA 和 MoE 实现不变。不依赖 FLA、本地模型目录或
联网下载，也不运行量化训练。无 FLA 子进程回归覆盖打包和两种实际加载路径，
不替代后端完整混合结构的量化验证。

后端 `tests/test_ling3_flash_mla.py` 覆盖浮点 prefill/连续 decode、Q LoRA、bias/gate、
梯度、真实小型 KDA＋MLA＋MoE＋MTP 的 GPTQ/AutoRound、save/load/resave 与整数逐项一致。
小模型保留实际的 32 个 head，避免 gate 等输出宽度破坏 GPTQ 打包对齐。

这些用例不代表整模 PPL、业务精度或真机认证。发布前仍需固定数据/seed/n256/i400
重跑整模，比较旧新布局 PPL 与任务集，并验证目标 runtime 的 prefill/decode。
不能仅凭训练 loss 更低或短句能生成宣布精度改善。

2026-09-07 的小模型验证：GPTQ 普通加载和 AutoRound 分层加载均完成 prefill/decode
HMONNX 导出；三个融合 Linear 的整数码与各自 checkpoint 逐项一致。HMONNX 生成
也已跑通。现有 SDK 会从 FP16 反量化权重重建 SSFP scale，本次最大相对差约 0.095%，
整数码未变；这不是再次吸收权重，但也不能宣称导出 scale 位级无损。本次未修改 SDK。
验证记录在工作区 `work_dirs/ling_mla_absorbed_20260907.7GiOrZ/validation.md`，不随代码发布。

## 上游参考

此前核对的 GPTQModel main `f37cd5b9ef5586206ee99b82464c74453166eabe` 中，
[Bailing](https://github.com/ModelCloud/GPTQModel/blob/f37cd5b9ef5586206ee99b82464c74453166eabe/gptqmodel/models/definitions/bailing_moe.py)
是 `query_key_value` 树；
[DeepSeek V3](https://github.com/ModelCloud/GPTQModel/blob/f37cd5b9ef5586206ee99b82464c74453166eabe/gptqmodel/models/definitions/deepseek_v3.py)
仍量化原始 Q/KV/output 投影，没有可直接复用的 Ling XH2 预吸收方案。
GPU 的 absorb 路径可参考数学，但不能替代本页的量化边界和双 cache 约束。

未来若 backend/SDK 统一提供新布局，可删除本地适配；前提是浮点、整数、导出和
runtime 状态回归均通过，而不只是出现同名参数。共享 cache 则需新的目标 ABI 和指数域设计。
