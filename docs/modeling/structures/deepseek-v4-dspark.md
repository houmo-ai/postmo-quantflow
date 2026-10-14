# DeepSeek-V4 Flash 0731 DSpark

Flash 0731 的三个草稿 block 是完整的 MoE/mHC block。BF16 解包后的主体为
39,463,490,972 字节；它是导出中间态，部署采用非共享专家 W4、共享专家及其他线性层
W8、G64。目标与草稿复用同一个量化 LM head，默认 W5，也支持 W4/W8。
归一化参数、常量和 embedding 查表参数保持各自的浮点表示。

## 与当前 DFlash / DFlash2 的差异

| 项目 | DFlash / DFlash2 | Flash 0731 DSpark |
| --- | --- | --- |
| 主模型特征 | 各 checkpoint 的目标层选择 | 第 40、41、42 层 mHC 通道均值，依次拼为 12288 维 |
| context | 目标特征生成草稿 KV | main_proj/main_norm 后，每个草稿层独立生成一份共享 latent K/V |
| KV | DFlash2 使用 sliding window，但 K/V 分开 | 每层一份 512 维 latent，window=128 |
| 草稿网络 | DFlash transformer；DFlash2 增加卷积和 selector | 三层训练过的 MoE/mHC、独立草稿 HC head/norm、共享目标 LM head |
| 草稿位置 | 现有 DFlash 路径移除 anchor 输出 | anchor 本身也预测一个候选，block=5 时产生 5 个候选 |
| 条件修正 | DFlash2 selector 还使用当前位置 hidden state | 两张 `[129280,256]` 表计算前一 token 对下一 token 的 Markov bias |
| 置信度 | DFlash2 selector 的自身接口 | confidence head 读取草稿 hidden state 与 Markov embedding；首版使用固定验证长度 |

四个组件平铺为 `dspark_context/`、`dspark_context_decode/`、
`dspark_draft_decode/`、`dspark_markov/`。context 输入档位为 256；context_decode
为 K+1；draft_decode 为 K；Markov 为单 token。context 的三份 latent 权重
在 context、context_decode、draft_decode 间按实际 tensor 内容去重复。

主模型在图内对三层 mHC 输出求均值并 concat，context 直接接收 `[1,256,12288]`。
context、context_decode 和 draft_decode 均内置 checkpoint RoPE 表，通过 `past_length`
执行 DynamicSlice，分别取得 256、K+1、K 个位置；Host 不传 sin/cos。位置使用绝对
token 索引，不随滑窗回绕。表尾覆盖固定档位的 padding，支持接近最大上下文的短前缀。
draft_decode 接收 Host 生成的 embedding，三层均为 learned routing，不保留 `input_ids` 输入。
RoPE 频率从 checkpoint 配置以 FP32 重建，再生成 FP16 查表常量，避免 HF 模型先执行
`half()` 后损失频率精度。2026-10-13 的正式 A8/A16 新旧接口对照中，量化权重一致，
跨窗口的 context cache、draft logits 和 head hidden 均逐值一致；两套发布包已更新。

## 验证与缓存提交

1. 主模型 prefill 输出三个目标层特征，context 提交有效前缀。
2. 主模型采样 anchor。draft 输入 anchor 加 K-1 个 noise token，允许整块非因果注意力。
3. Markov 按前一个实际采样 token 逐位置修正 base logits，保留真实 proposal 概率 q。
4. 主模型 decode 档位为 K+1，输入 anchor 和 K 个候选，输出所有候选 logits 与目标特征。
5. 按 `min(1,p/q)` 接收；首次拒绝时从归一化的 `max(p-q,0)` 采样。greedy 使用相等判断。
6. 草稿 proposal 的临时 KV 一律恢复。全接收时提交主模型状态；拒绝时恢复主模型状态，
   重跑 anchor 加已接收前缀，再由 context_decode 提交这些目标特征。

主模型回滚保存完整的小 SWA/carry 状态，以及本次验证可能写入的压缩缓存行，
不会复制整个 256K CSA 缓存。首版保持单请求并发，每次生成从新的请求状态开始。

## 权重保存与当前证据

GPTQModel 保存 DeepSeek-V4 时，原始 `mtp.*` 单独保存到 `dspark/`，保留官方 FP4/FP8
张量、原生配置、索引和摘要；它不改变主模型的 index/config/shard。当前量化 checkpoint
缺失的 4705 个 MTP 张量已补齐，原生组件为 10,862,838,300 字节。BF16 转换由
producer 的 `prepare_deepseek_v4_dspark_bf16` 完成，ModelZoo 负责图和 runtime。

2026-10-12，真实三层 W4/W8 A8 与 A16 的四个组件均完成导出及 GPU 重载。
共享加载器权重占用为 11,380,569,964 字节，GPU allocation 为 11,388,980,224 字节；
其中包含尚未与目标包合并去重复的共享 LM head。W4 runtime 打包节省
9,663,676,416 字节。SDK HMONNX 文件仍以展开 int8 保存，磁盘约 21 GB，
加载时才进行 W4 半字节打包；不能把磁盘文件大小写成 11 GB。

组件数值对照、A8/A16 主模型 K+1 图、共享 W4/W5/W8 head、缓存回滚与最小真实 SWA 模型
greedy 等价性均有当前 CI 用例。混合 SWA/CSA/HCA 模型还验证了跨 128-token 压缩边界时，
回滚后的验证 logits 与仅重建已提交前缀的同形状目标前向逐值一致；A8 目标的
1-token/6-token 数值本身存在差异，因此不把跨有效查询长度的 bit-exact 作为该用例契约。
两个公开变体都通过四层混合 SWA/CSA/HCA
目标模型的真实构建、打包、目录迁移、删除源权重和加载推理；DSpark 用例还包含
三层真实草稿和完整五候选验证块。该 fixture 使用真实 G64 RTN 打包权重，验证格式
消费与运行闭环，不作为 Hessian GPTQ 量化算法或训练模型接收率的证据。

2026-10-12 21:17，完整 43 层 A8/A16 目标模型及四个草稿图完成导出。
完整包包含六个图，通过统一打包、外部文件引用、权重指纹和目录迁移检查。
共享加载器的唯一权重为 165,165,665,576 字节，W4 打包节省 148,176,371,712 字节。
GPU 执行设备在创建 HMONNX V2 session 时指定；decode 和草稿组件支持 CUDA Graph，
prefill 保持普通执行。双 GPU 的图重放与上述缓存契约均有真实 CI 回归。

四方验证使用六个手写问题，单并发、K=5、thinking max、temperature=1、top_p=1、
输出上限 32768；数学、翻译、代码、逻辑和信息保持均正常回答。

| 目标模型 | 接收 / 提议 token | 接收率 |
| --- | ---: | ---: |
| HMONNX A8 | 1765 / 3575 | 49.37% |
| HMONNX A16 | 1642 / 3475 | 47.25% |
| 原始官方模型，vLLM 0.31 | 1689 / 4215 | 40.07% |
| AutoRound weight-only，vLLM 0.31 | 977 / 2430 | 40.21% |

这是回答与接收率诊断，不是 MMLU-Pro 精度结果。各实现的随机采样轨迹不同，不能由
这六题推断任意数据集精度或把接收率差异直接归因于某一量化层。之后恢复的
SPEED-Bench、MMLU-Pro 和 HLE 小样本评测分别记录协议与结果，不沿用该六题结论。

## vLLM Merak 服务接入

`DeepSeekV4DSparkProposer` 使用完整包的四个草稿图，复用 ModelZoo 的 Markov
采样和 SWA/CSA/HCA 事务。vLLM 负责目标采样及 standard rejection；插件向其传入
Markov 修正后实际用于采样的 q，不使用并行 base logits 代替条件概率。
缓存事务在目标验证前开始；接受结果返回后，拒绝分支恢复目标状态并重放有效输入，
context_decode 仅提交这些重放特征，下一轮才处理恢复/bonus token。

当前服务契约为单请求、禁用 prefix caching 和 async scheduling，K 最多等于导出的
block_size。DeepSeek 专用 proposer 不把异构状态转换成普通 PageAttention K/V。
prefill 可以只返回最后一行 logits；不支持 prompt logprobs。decode 及草稿图沿用
HMONNX CUDA Graph，vLLM 本身使用 eager 调度。

在独立的 Merak 服务环境中启动：

```bash
CUDA_VISIBLE_DEVICES=4,5 PORT=18185 \
  bash vllm_merak/examples/deepseek_v4_dspark_service.sh /path/to/released_a8
CUDA_VISIBLE_DEVICES=6,7 PORT=18184 \
  bash vllm_merak/examples/deepseek_v4_dspark_service.sh /path/to/released_a16
```

客户端显式使用 temperature=1、top_p=1、max_tokens=32768，并使用官方 Flash0731
编码器生成 thinking max 提示。直接 HMONNX 与 vLLM 服务的速度和结果分别保留，
不把直接调用的 token/s 作为服务性能。

## A8/A16 有效查询长度的数值一致性限制

在 A8 四层最小真实模型中，目标单独执行也会出现有效查询长度为 1 或 6 的差异。
CUDA Graph 与普通执行给出相同的该差异，排除了草稿网络和回滚作为来源。
首个实际上下文差异位于第 0 层 SWA 的 value MatMul；控制实验中，第一个 query
的 softmax 系数和所有非零系数对应的 V 都完全相同，浮点上下文差为 0，但只替换
被屏蔽位置的 V 就使 SEFP 上下文变化 0.0048828125。该算子的 V 使用 W8/G64 动态
共享指数，组内被屏蔽的值仍可改变有效 V 的舍入。128-token 压缩边界处的目标
logits 最大差为 0.161376953125，接近的候选可因此改变 greedy 排名。

按 query 在 V 量化前屏蔽不可见位置，并以 query 为 MatMul 的 batch 轴，在同一
算子控制实验中消除了该差异。这个候选数值修正尚未应用到发布图；需要先验证完整
图的成本与接口，再重新导出，不能将它描述为已解决或保证 A8 跨档位 bit-exact。

A16 也不保证跨有效查询长度逐值或逐 token 一致。当前 A16 发布图的 QK 和 value
MatMul 两侧均使用 16 位 SEFP、G64 共享指数，累加为 FP24、输出为 FP16；动态共享
指数和舍入仍然存在。较高位宽通常减小量化误差，但是否改变 token 排名还取决于
候选 logits 的间隔。上述控制实验测量的是 A8；目前没有 A16 在相同前缀下的误差
幅度及排名变化频率统计，不能把“更少出现差异”写成已验证结论。

官方参考：[Flash 0731 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)、
[vLLM 0.31 DSpark 实现](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/deepseek_v4/nvidia/dspark.py)。

## 使用

统一接口示例：

```bash
python examples_merak/unified_api/deepseek_v4.py \
  --checkpoint /path/to/flash0731-W8E4G64 \
  --output-dir /path/to/flash0731-dspark-a8 --activation-bits 8
```

checkpoint 必须含 producer 保留的 `dspark/`。workflow 默认按需准备临时 BF16 草稿，
导出后清理这个临时目录。需要 W4/W8 LM head 时，在底层 workflow YAML 的
`export.model.dspark.head_weight_bits` 设置 4/8；已有低位 head 必须保持其实际位宽。
运行时所有组件通过同一个 initializer pool 加载。公开 recipe 只声明已实现的
low_memory 导出，使用现有 GPTQModel/AutoRound 主模型 checkpoint，不重新量化主模型。
