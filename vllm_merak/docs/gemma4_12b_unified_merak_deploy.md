# Gemma4 12B Unified vLLM-Merak 部署

## 支持范围

`Gemma4UnifiedForConditionalGeneration` 使用独立
`Gemma4UnifiedMerakForCausalLM` loader，支持：

- 文本；
- 图像：固定 280-token HMONNX 输入容量，实际 token 按 position ID 在 Host 过滤；
- 视频：固定每帧 70-token HMONNX 输入容量，逐帧过滤后按视频拼接；
- 音频：固定 750 x 640 HMONNX 输入，按 `ceil(samples / 640)` 展开 placeholder，
  图执行后按 processor mask 过滤；
- 图像、视频只进入 bidirectional vision range；音频始终保持 causal。

`max_soft_tokens` 以 checkpoint 的
`vision_config.num_soft_tokens=280` 为准。服务请求不能覆盖为 560/1120。

Transformers 5.13 的可见性语义必须逐层保持：

- full/global attention 始终 causal；
- sliding/local attention 使用
  `sliding_window AND (causal OR visual_blockwise)`。

因此同一张混合 attention 图中，只有 sliding FlashAttention/PageAttention
节点可以连接 `mm_prefix_ranges`；full 节点连接该输入会改变 checkpoint
语义。另一方面，vLLM 请求进入 HMONNX preprocessor 时必须根据 placeholder
token 恢复 `mm_token_type_ids`（image=1、video=2、其余包括 audio=0），否则
导出 preprocessor 会把真实视觉范围静默生成为 `[0,0]`。

## Artifact 合同

服务目录必须由包含无-mask frontend 改动的 xh2modelzoo 版本重新导出，并至少包含：

```text
MODEL_DIR/
├── merak_config.json
├── golden_meta_info.json
├── hf_config/                  # config/tokenizer/processor/chat template
├── visual/                     # pixel_values + pixel_position_ids
├── video_visual/               # pixel_values + pixel_position_ids
├── audio/                      # input_features
├── prefill/
└── decode/
```

不要复用旧的三输入 vision 或两输入 audio side graph。新的 ABI 是：

```text
visual.forward(pixel_values, normalized_position_ids)
video_visual.forward(pixel_values, normalized_position_ids)
audio.forward(input_features)
```

## 重新导出和测试前预检

12B Unified 使用统一的 `xhquant_55` 导出环境（PyTorch 2.8、
Transformers 5.13）。已有 W4G64 量化权重可以复用，不需要再次量化：

```bash
conda activate xhquant_55
cd /data01/home/yujy/work/xh2modelzoo

python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir /path/to/HF_MODEL_DIR \
  --existing-hf-model-dir /path/to/QUANT_DIR \
  --artifact-format gptqmodel_hf \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/gemma4_12b_unified_full.yaml \
  --export-output-dir /path/to/NEW_EXPORT_DIR \
  --export-device cuda:0 \
  --force
```

Gemma4 Unified 必须通过 Transformers 5.13 的模型类加载 GPTQModel-HF
checkpoint；不能使用 GPTQModel 5.8 的 generic Torch loader。后者会把未量化的
视觉 frontend 权重静默加载成全零或非有限值。导出器会对
`pos_embedding`、`patch_dense.weight` 和 `embedding_projection.weight`
执行 fail-closed 检查。检查失败时应修复加载路径，禁止绕过后继续导出。

先在新目录完成导出，不要直接删除旧产物。找到新生成的 `hmquant_*` 目录，
补齐 Merak 入口配置后运行静态 ABI 检查：

```bash
MODEL_DIR=/path/to/NEW_EXPORT_DIR/hmquant_xxx
cp /data01/home/yujy/work/xh2modelzoo/vllm_merak/data/merak_config.json \
  "$MODEL_DIR/merak_config.json"

conda activate xhquant_55
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak
python tools/check_hmonnx_attention_abi.py "$MODEL_DIR"
```

检查必须确认 prefill/decode 的 FlashAttention 输出为 BHSD，并且紧接
`Transpose([0,2,1,3])` 转回模型侧 BSHD；同时验证 contract-v2、无
`sliding_attention_mask`、full attention 无 `mm_prefix_ranges` 以及三类
side graph 输入 ABI。只有检查通过后才切换部署目录或覆盖旧产物。

## 启动服务

Gemma4 contract-v2 的 image span 最长为 280，调度预算必须至少为 320；当前固定
sliding cache 只支持单请求 cache owner，因此 `max-num-seqs` 必须为 1。

```bash
conda activate vllm
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
ENABLE_HMINFERENCE_V2=1 \
vllm serve /path/to/MODEL_DIR \
  --config-format merak_llm \
  --load-format merak_llm \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 262144 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":1,"audio":1}'
```

`--max-model-len` 必须与 artifact 的 `spec_decode.context_length` 一致。当前正式
artifact 按 262144 导出，服务也必须使用 262144；不能为了降低显存占用擅自改成
2048。cache 容量和 valid length 均服从导出合同，长上下文持续 decode 时不需要
额外搬运或重新分配。`2560` 的调度预算用于容纳 2496-token 的多模态单项预算和
320-token prefill chunk；仅设为 320 会在 vLLM 启动阶段被多模态预算检查拒绝。

服务启动日志中必须出现：

```text
Resolved architecture: Gemma4UnifiedMerakForCausalLM
```

## 前处理验收

部署前至少核对一次 processor 结果：

| 模态 | HMONNX 输入 | 有效 embedding 数 |
|---|---|---|
| image | `[280, 6912]` + `[280, 2]` | 非 `[-1,-1]` position 数 |
| video | `[frames, 70, 6912]` + `[frames, 70, 2]` | 各帧有效 position 数之和 |
| audio | `[750, 640]` | `input_features_mask.sum()` |

641 个 16 kHz sample 必须得到 2 个 audio soft tokens，而不是旧 audio tower
estimator 的结果。audio token ID 不得出现在 PageAttention vision ranges 中。

图像端到端验收不能只检查 HTTP 200。固定回归图
`xh2modelzoo/data/images/houmo_logo.jpg` 应能识别出“后摩智能 /
HOUMO.AI”；若输出类似“HDM.A”，先检查 HMONNX preprocessor 实际收到的
`mm_token_type_ids` 和 `mm_prefix_ranges`，不要误判为视觉 side graph 权重问题。

## MTP 只读 PageAttention

推荐使用：

```text
configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/
gemma4_12b_unified_full_mtp_page_attention.yaml
```

该配置导出 `gemma4_mtp_readonly_page_attention_v2`。assistant 的四个 attention
层只计算 Q，并按 attention 类型绑定 target 中最后一个非共享 KV owner；draft
不分配、不复制、不更新 KV。每次 MTP draft 的输入长度固定为 1：

```text
target committed KV length = N
attention mask position     = N - 1
assistant RoPE position     = N
past_seq_length             = N - 1
current_input_length        = 1
full visible KV             = [0, N)
sliding visible KV          = [max(0, N - W), N)
```

因此 causal PageAttention 与旧 v1 的显式 `[start,end)` Padding Full
Cross-Attention 数值等价。Full 层可直接走 causal paged decode/split-K 路径；
Sliding 层额外接收两个紧凑标量：

```text
kv_window_start_abs = max(0, N - W)
kv_valid_length     = N - kv_window_start_abs
```

这两个值在同一轮的 K 次 proposal 中原地复用，不构造逐 query 的 `[start,end)`
range tensor，也不在图内执行 `Clip`。旧
`gemma4_mtp_readonly_page_attention_v1` 保留为兼容回退，但新导出应使用 v2。

verify 会暂时向 target page cache 写入整块候选。accept/reject 只提交新的逻辑
长度；被拒绝的尾部不搬运也不清零，下次 draft 由逻辑长度和上述紧凑窗口元数据
共同隐藏。Sliding 物理 ring 按导出合同保留
`window + prefill_chunk_length` 容量，因此回退后仍在新窗口内的历史页不会被
候选尾部覆盖。

MTP 服务必须显式启用 speculative scheduler：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
ENABLE_HMINFERENCE_V2=1 \
VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1 \
vllm serve /path/to/MTP_MODEL_DIR \
  --config-format merak_llm \
  --load-format merak_llm \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 262144 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":1,"audio":1}' \
  --speculative-config '{
    "method":"custom_class",
    "model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer",
    "num_speculative_tokens":4
  }'
```

仅 artifact 内包含 MTP graph 不会自动启用投机解码；启动日志中的
`speculative_config` 不能为 `None`。
