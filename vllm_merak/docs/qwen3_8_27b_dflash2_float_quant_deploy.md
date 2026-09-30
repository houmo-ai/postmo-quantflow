# Qwen3.8-27B DFlash2 浮点与量化部署全流程

本文给出 `Qwen3.8-27B` + `Qwen3.8-27B-DFlash2` 的两条可复现链路：

- 浮点：vLLM 原生 DFlash2；
- 量化：AutoRound W4G64 target、HMONNX W4A8、vLLM Merak DFlash2。

文中的接收率口径固定为 DFlash2 官方推荐采样配置。不要省略
`draft_sample_method`：vLLM 合入版默认是 `greedy`，而 DFlash2 PR 和官方
非零温度验收使用的是 probabilistic selector walk。省略该字段会让浮点和
量化结果不可比较。

## 1. 已验证范围和固定口径

### 1.1 模型

```bash
export FLOAT_TARGET=/data01/datasets/Qwen3.8-27B
export FLOAT_DRAFT=/data01/datasets/Qwen3.8-27B-DFlash2
```

本地 `xh2modelzoo/weights` 是 `/data01/datasets` 的软链接，也可以使用：

```bash
export FLOAT_TARGET=${MODELZOO_ROOT}/weights/Qwen3.8-27B
export FLOAT_DRAFT=${MODELZOO_ROOT}/weights/Qwen3.8-27B-DFlash2
```

### 1.2 接收率参数

| 参数 | 固定值 |
| --- | ---: |
| 数据集 | DFlash 官方五套：GSM8K 1319、MATH-500 500、HumanEval 164、MBPP sanitized test 257、MT-Bench first-turn 80，共 2320 请求 |
| prompt 顺序 | Python `random.Random(42)` shuffle，完整集无遗漏 |
| concurrency | 1 |
| reasoning | GSM8K/MATH-500=`xhigh`；HumanEval/MBPP/MT-Bench=`medium` |
| temperature | 1.0 |
| top-p | 0.95 |
| top-k | 20 |
| max new tokens | 按数据集设置，见 7.1 |
| draft K | 7 |
| draft sample | `probabilistic` |
| rejection sample | `standard` |
| request seed | `seed + benchmark_index`，基础 seed 为 0 |

这两个开关控制不同阶段：

```json
{
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "standard"
}
```

- `draft_sample_method` 决定 Path Selector 是逐步 argmax，还是按 selector
  分布 q 抽样并保存 q；
- `rejection_sample_method=standard` 使用 target 分布 p 与 q 做无损拒绝采样；
- draft 使用 `greedy` 不代表 target 请求是 greedy。target 仍由请求里的
  temperature/top-p/top-k 决定；
- target 本身 `temperature=0` 时建议 draft 也使用 `greedy`，避免缓存 q 的
  额外显存；本文的非零温度官方口径必须使用 `probabilistic`。

## 2. 仓库和环境

```bash
export WORK_ROOT=/data01/home/yujy/work
export VLLM_SOURCE=${WORK_ROOT}/v2/vllm
export MODELZOO_ROOT=${WORK_ROOT}/v2/xh2modelzoo
export VLLM_MERAK_ROOT=${MODELZOO_ROOT}/vllm_merak
export XHQUANT_ROOT=${WORK_ROOT}/xhquanttool
export GPTQMODEL_ROOT=${WORK_ROOT}/gptqmodel
```

本次验证环境：

| 环境 | 用途 | Python | Torch | Transformers |
| --- | --- | --- | --- | --- |
| `xhquant_55` | AutoRound、HMONNX 导出、Golden | 3.12.13 | 2.8.0+cu128 | 5.13.0 |
| `vllm` | 浮点 vLLM、量化 Merak 服务 | 3.12.13 | 2.11.0+cu129 | 5.13.0 |

2026-08-26 全量验收起始 revision：

| 仓库 | commit |
| --- | --- |
| vLLM DFlash2 浮点源码 | `9c8e90eb2637a863ca14e47fd436b10ed7ba6536` |
| vllm-merak | `0d8a5f729399293e5d68ee12044fbfaecb08cf6d` + 本文对应工作区变更 |
| xh2modelzoo | `726047c533c8ed2ae8677da485e9ede5b6319162` + DFlash2 工作区变更 |
| xhquanttool | `ddb6ee1edec259de5c32819bce5485375caef6a2` |
| GPTQModel | `d083cf0f2374921285aea2cb55a6014adaf4d7e3` |

量化 Merak 当前使用 Conda 中的 vLLM 0.24.0 加本仓库 bridge；浮点 DFlash2
使用上表 vLLM 源码树。两条链路的 engine 实现不同，但数据、prompt、采样、
K、随机 seed、最大输出和统计方法完全相同。

记录源码 revision：

```bash
for repo in \
  "${VLLM_SOURCE}" \
  "${MODELZOO_ROOT}" \
  "${XHQUANT_ROOT}" \
  "${GPTQMODEL_ROOT}"; do
  printf '%-48s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
done
```

不要在 `vllm` 环境重新导出 HMONNX。Torch/CUDA 扩展具有 ABI 绑定：导出只在
`xhquant_55` 中进行，vLLM/Merak 推理才切到 `vllm`。

## 3. 浮点 vLLM DFlash2

### 3.1 版本要求

vLLM 必须包含 DFlash2 PR `#52816`。验证：

```bash
git -C "${VLLM_SOURCE}" log --oneline --all --grep='DFlash2' -n 3
```

本次浮点验收使用源码树，而不是 Conda 环境中可能较旧的 site-packages：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH=${VLLM_SOURCE}:${PYTHONPATH:-}
```

`vllm-merak` 注册了 platform plugin。跑纯浮点基线时显式禁用插件，避免把
Merak 环境差异混进上游 vLLM 对照：

```bash
export VLLM_PLUGINS=none
```

### 3.2 启动浮点服务

单卡 A100 80GB 已验证的 8K 功能/接收率配置：

```bash
export SERVED_NAME=qwen3_8_27b_dflash2_float
export PORT=18501

CUDA_VISIBLE_DEVICES=0 \
VLLM_PLUGINS=none \
PYTHONPATH=${VLLM_SOURCE}:${PYTHONPATH:-} \
vllm serve "${FLOAT_TARGET}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --speculative-config \
  '{"method":"dflash","model":"/data01/datasets/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

启动日志必须出现：

```text
Using FLASH_ATTN attention backend
Capturing model for DFlash2 speculator
```

浮点 27B target + draft + 256K KV 不适合直接照搬到单张 A100 80GB。部署长
上下文前应先按实际 HBM、KV dtype、并发和 TP 重新核算容量。上面的
8192 是单卡 smoke/GSM8K 配置；官方五数据集完整复验按 7.1 节使用
8K/32K/64K 的数据集分级上限。

### 3.3 API smoke

```bash
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"${SERVED_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Compute 17 * 23 and explain briefly.\"}],
    \"temperature\":1.0,
    \"top_p\":0.95,
    \"top_k\":20,
    \"max_tokens\":256,
    \"seed\":0
  }" | python -m json.tool
```

## 4. Target W4G64 量化

### 4.1 安装导出环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55

python -m pip install -v -e "${XHQUANT_ROOT}" \
  --no-build-isolation \
  --config-settings editable_mode=compat
python -m pip install -v -e "${GPTQMODEL_ROOT}" \
  --no-build-isolation
python -m pip install -v -e "${MODELZOO_ROOT}" \
  --use-pep517 --no-build-isolation \
  --config-settings editable_mode=compat
```

### 4.2 AutoRound W4G64

已验证 recipe 为 LLM-only、W4G64、symmetric GPTQ format；视觉权重不在这一步
量化，HMONNX export 使用自己的视觉 quant scheme：

```bash
cd "${MODELZOO_ROOT}"
export QUANT_MODEL=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/autoround_quant
export CALIB=${MODELZOO_ROOT}/data/calib_data/NeelNanda-pile-10k.jsonl

CUDA_VISIBLE_DEVICES=0 \
python "${GPTQMODEL_ROOT}/third_party/auto-round/scripts/quantize.py" \
  --model "${FLOAT_TARGET}" \
  --mode llm-only \
  --output_dir "${QUANT_MODEL}" \
  --llm_bits 4 \
  --llm_group_size 64 \
  --data_type int_sym_gptq \
  --seed 42 \
  --iters 200 \
  --nsamples 128 \
  --seqlen 2048 \
  --batch_size 8 \
  --dataset "${CALIB}" \
  --format auto_gptq \
  --device cuda:0 \
  --sym \
  --low_gpu_mem_usage
```

量化完成后保留独立 checkpoint；不要为 Base/MTP/DFlash 重复量化 target。

最终运行口径中，target checkpoint 是 W4G64，HMONNX 激活为 A8，loader 在
启动时把 target W4 在线 pack 给 W4A8 kernel；DFlash2 assistant 主体按
`w8a8h1_sefp` 导出，`core_lm_head` 由 `spec_draft_head_weight_bits=4` 单独使用
4-bit 权重。不要仅根据目录名推断每个子图的位宽，必须查看 metadata。

## 5. 导出 256K FA + GDR DFlash2 HMONNX

### 5.1 正式 workflow

```bash
export DFLASH2_CONFIG=${MODELZOO_ROOT}/configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_8_27b_full_dflash2_256k_fa.yaml
export EXPORT_ROOT=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/dflash2_256k_flash_gdr
```

该 YAML 必须保留：

```yaml
context_max_length: 262144
prefill_chunk_length: 256
max_pe_length: 262144
flash_attention:
  enable: true
fuse_gdr_ops: true
fuse_gdr_block_recurrent_ops: true
spec_decode_mode: dflash
num_draft_tokens: 7
spec_draft_head_weight_bits: 4
dflash_config:
  hf_model: weights/Qwen3.8-27B-DFlash2
  dtype: bfloat16
  activation_residual_scale: 128
```

这里 Qwen3.8 是已完成完整 ABI 验证的专用配置，两个 GDR fusion 都必须开。
通用 Qwen3.5 文档中“默认不打开 `fuse_gdr_ops`”的保守规则不适用于该专用
workflow。

DFlash2 assistant 用 BF16 checkpoint 加载。直接把整条 assistant 路径改成
FP16 会在 two-tap dynamic convolution 的残差中溢出；HMONNX 外部张量仍可为
FP16，但进入图前后的残差必须使用精确的 `activation_residual_scale=128` 合约。

### 5.2 导出命令

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir "${QUANT_MODEL}" \
  --config-path "${DFLASH2_CONFIG}" \
  --export-from-quanted-model \
  --export-output-dir "${EXPORT_ROOT}" \
  --device cuda:0 \
  --context-max-length 262144 \
  --num-draft-tokens 7 \
  --enable-flash-attention \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --overwrite
```

`--overwrite` 会清理目标导出目录，只能用于明确的新导出目录；已有正式产物
不要原地覆盖。

### 5.3 DFlash2 图和计算职责

最终 artifact 包含 target `prefill/decode`，以及三张 draft 图：

| 图 | 调用时机 | 职责 |
| --- | --- | --- |
| `dflash_draft_context` | target prompt prefill 后 | 用 target hidden 初始化 draft cross-attention K/V |
| `dflash_draft_context_decode` | target 提交接受 token 后 | 只把已提交 hidden 增量写入同一 draft cache |
| `dflash_draft_decode` | 每轮 proposal | two-tap dynamic convolution、候选 top-k、Path Selector 分数 |

最终 ABI 没有第四张独立 `draft_prefill` 图；DFlash 的 draft prefill 语义就是
`dflash_draft_context`。三张 draft 图绑定同一份 persistent private draft cache。

结构复核结论是：相对 legacy DFlash，新增的学习模块主要就是 two-tap dynamic
convolution 和 Scheme-B Path Selector，但部署 ABI 还多了几项不能漏掉的变化：

- decode 输入从纯 mask block 变成 `[anchor, mask, ..., mask]`；anchor 前一行按零
  padding，第一条真实 draft 路径消费 anchor，因此图内保留完整的 `K+1` 行，最后
  只输出 K 个 proposal；
- 每个 draft layer 的 attention、MLP 两条 branch 都是
  `dynamic-conv prepare -> branch -> dynamic-conv finish`，不只是加一次卷积；
- 第一跳 selector 分数依赖 anchor，后续跳使用前一候选到下一候选的
  `selector_top_k x selector_top_k` transition score，所以必须把
  `anchor_token_id`、候选表和两类 score 一起纳入图 ABI；
- checkpoint 的 embedding scale、output multiplier、可选 logit softcap 和
  `activation_residual_scale=128` 都属于数值合约。它们不是新的主干模块，但漏掉任意
  一项都会破坏浮点/HMONNX 对齐；
- context/context-decode 只提交 target 已确认的 hidden/KV；decode 产生的
  anchor/mask query state 是临时状态，拒绝后不能泄漏到下一轮。

因此现有四个逻辑入口的兼容关系是：`draft_context`、`draft_context_decode`、
`draft_decode` 可直接保留职责；`draft_prefill` 只作为上层语义别名映射到
`draft_context`，不能再寻找或导出第四张图。`draft_decode` 的返回值则必须从 legacy
DFlash 的逐位置 logits 切到下面三项 selector 输出。

`dflash_draft_decode` 输出：

```text
candidate_ids
selector_first_scores
selector_transition_scores
```

Merak runtime 按这些输出完成 Scheme-B path walk。`probabilistic` 模式按 q 抽样
并把 q 交给 standard rejection sampler；`greedy` 模式仍运行 Path Selector，
只是逐步选择当前最高分。

### 5.4 产物审计

`MODEL_DIR` 必须直接包含 `golden_meta_info.json` 和 `merak_config.json`：

```bash
mapfile -t MODEL_DIRS < <(
  find "${EXPORT_ROOT}" -name golden_meta_info.json -printf '%h\n' | sort
)
test "${#MODEL_DIRS[@]}" -eq 1
export MODEL_DIR=${MODEL_DIRS[0]}
```

若检查不是恰好一个目录，显式选择本次新导出的目录；禁止用 `head -1` 静默
拿到同一 work directory 里的旧产物。

执行静态合约检查：

```bash
python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
meta = json.loads((root / "golden_meta_info.json").read_text())
merak = json.loads((root / "merak_config.json").read_text())
cfg = meta["model_config"]
spec = meta["spec_decode"]
draft = spec["draft"]

assert cfg["context_max_length"] == 262144
assert merak["enable_page_attention"] is True
assert cfg["flash_attention"]["enable"] is True
assert cfg["fuse_gdr_ops"] is True
assert cfg["fuse_gdr_block_recurrent_ops"] is True
assert cfg["num_draft_tokens"] == 7
assert cfg["dflash_config"]["dtype"] == "bfloat16"
assert cfg["dflash_config"]["activation_residual_scale"] == 128.0
assert spec["mode"] == "dflash"
assert spec["verify_length"] == 8
assert draft["abi"] == "qwen_dflash2_paged_shared_v1"
assert draft["architecture"] == "DFlash2DraftModel"
assert draft["cache_mutation"] == "page_attention"
assert draft["selector_scheme"] == "precomputed_kxk_v1"
assert draft["decode_outputs"] == [
    "candidate_ids",
    "selector_first_scores",
    "selector_transition_scores",
]
for key in ("context_hmonnx", "context_decode_hmonnx", "decode_hmonnx"):
    path = root / draft[key]
    assert path.is_file(), path
print("DFlash2 artifact contract: PASS", root)
PY
```

8K 非 FA、`enable_page_attention=false` 的临时图只能用于调试，不能替代这里的
256K `qwen_dflash2_paged_shared_v1` 正式产物。

## 6. vLLM Merak 量化服务

### 6.1 安装和环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

python -m pip install -v -e "${XHQUANT_ROOT}" \
  --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
python -m pip install -v -e "${MODELZOO_ROOT}" \
  --no-build-isolation --no-deps
PYTHON_BIN=python "${VLLM_MERAK_ROOT}/tools/install_from_modelzoo.sh"
```

量化 DFlash2 当前是串行 runtime：

```bash
export ENABLE_HMINFERENCE_V2=1
export VLLM_MERAK_USE_V2=1
export VLLM_MERAK_AUTO_OFFLOAD=1
export VLLM_MERAK_CUDA_GRAPH=1
export VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=0
export VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1
```

- `max_num_seqs` 必须为 1；HTTP 请求可以排队，但不能并行 forward；
- prefix caching 必须关闭；复用 target prefix 时没有对应 draft private state；
- async scheduling 必须关闭；
- context/context-decode 保持 eager，draft-decode 使用已验证的 HMONNX CUDA
  Graph；
- `--enforce-eager` 只关闭 vLLM 外层 graph，不会关闭 HMONNX V2 graph。

### 6.2 启动量化服务

```bash
cd "${VLLM_MERAK_ROOT}"
export SERVED_NAME=qwen3_8_27b_dflash2_w4a8
export PORT=18502

CUDA_VISIBLE_DEVICES=0 \
ENABLE_HMINFERENCE_V2=1 \
python examples/vllm_chat_server.py \
  --model "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --config-format merak_llm \
  --load-format merak_llm \
  --enforce-eager \
  --use-v2 \
  --auto-offload \
  --cuda-graph \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --max-model-len 262144 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":4}' \
  --speculative-config \
  '{"method":"custom_class","model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

纯文本部署可去掉 `--limit-mm-per-prompt`。正式 256K 服务保留 262144；
接收率 benchmark 为了与浮点单卡严格对齐，两边使用相同的数据集分级
`max_model_len`：GSM8K=8192、MBPP/MT-Bench=32768、MATH-500/HumanEval=65536。

## 7. 官方五数据集完整接收率复验

### 7.1 Benchmark 脚本

```bash
export ACCEPT_SCRIPT=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/run_dflash2_acceptance.py
export MERGE_SCRIPT=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/merge_dflash2_acceptance.py
export COMPARE_SCRIPT=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/compare_dflash2_acceptance.py
export ACCEPT_DIR=${MODELZOO_ROOT}/work_dirs/qwen3_8_27b_release_20260815/acceptance
```

本地使用的官方 DFlash 源码是 `z-lab/dflash@07ebd93db9f4`。runner 中的数据
定义和 prompt formatter 已对官方 `dflash/benchmark.py` 做全量逐字符串校验：

| `--dataset` | Hugging Face dataset/config/split | 完整请求数 | prompt 规则 |
| --- | --- | ---: | --- |
| `gsm8k` | `openai/gsm8k`, `main`, `test` | 1319 | 题目 + step-by-step/boxed 后缀 |
| `math500` | `HuggingFaceH4/MATH-500`, `test` | 500 | problem + 相同 boxed 后缀 |
| `humaneval` | `openai/openai_humaneval`, `test` | 164 | 官方 Python fenced-code wrapper |
| `mbpp` | `google-research-datasets/mbpp`, `sanitized`, `test` | 257 | 原始 prompt |
| `mt-bench` | `HuggingFaceH4/mt_bench_prompts`, `train` | 80 | 只取第一轮 |

MT-Bench 的 80 请求不是漏跑第二轮：官方 OpenAI-compatible benchmark 构造请求时
只使用 `item["turns"][0]`；只有官方 Transformers/MLX in-process 路径才把模型第一轮
回答追加到 history 后继续第二轮。本文对比的是 vLLM/Merak OpenAI serving 口径，因此
固定 first-turn，并把该策略写入 summary。

输出预算不能在不同任务之间统一写成 4096。runner 的
`qwen3.8-dflash2-dataset-specific/v3` token budget 与
`qwen3.8-dflash2-dataset-specific/v1` reasoning 默认策略为：

| `--dataset` | max new tokens | runtime `max_model_len` | reasoning effort |
| --- | ---: | ---: | --- |
| `gsm8k` | 4096 | 8192 | `xhigh` |
| `math500` | 32768 | 65536 | `xhigh` |
| `humaneval` | 32768 | 65536 | `medium` |
| `mbpp` | 16384 | 32768 | `medium` |
| `mt-bench` | 16384 | 32768 | `medium` |

这不是把 context 一律放大：MATH 和 HumanEval 长尾使用 64K，MBPP 和 MT-Bench
使用 32K，只有 GSM8K 保持 8K。
早期统一 4096 的诊断轮里 MATH-500 有明显 `finish_reason=length`。HumanEval 不能
通过一味增加上限来解决：量化 `xhigh` 预跑 97 条时，中位数 1987、P90 16458，17 条
超过 8K，且 4 条在 32768 仍未结束、没有最终答案。对这 4 个同题同 seed 样本，浮点
`xhigh` 分别生成 29733、16747、32768 和 24969 tokens；说明主触发器是简单代码题
使用了 `xhigh`，而量化会进一步放大长尾。改用 `medium` 后，浮点分别为 760、1588、
4276、3110，量化分别为 559、1196、3938、1814，8 次都自然 stop 并产出最终答案。
因此旧 `xhigh` HumanEval 文件只作为诊断，不进入正式合并；32K 仍保留为安全上限，
但默认 reasoning 改为 `medium`。64K 双后端单卡冒烟已验证：浮点峰值约 74.4 GiB；
Merak virtual token capacity 为 68850，大于 65536。若正式结果仍有
`finish_reason=length`，必须记录截断数并复核输出，不能直接把截断结果标为完整集。

脚本会：

1. 按 `--dataset` 下载或读取缓存中的官方完整 split；
2. 用 shuffle seed 42 排序；
3. 给每题使用稳定 request seed；
4. concurrency=1，并关闭 prefix cache；
5. 为每题记录 acceptance histogram、token IDs、耗时和结束原因；
6. 分别报告 per-request mean 与 pooled mean。

### 7.2 浮点 8 卡完整集

每张卡运行一个 shard；`SHARD` 分别取 0 到 7：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

SHARD=0
CUDA_VISIBLE_DEVICES=${SHARD} \
VLLM_PLUGINS=none \
PYTHONPATH=${VLLM_SOURCE}:${PYTHONPATH:-} \
python "${ACCEPT_SCRIPT}" \
  --model "${FLOAT_TARGET}" \
  --draft-model "${FLOAT_DRAFT}" \
  --backend float \
  --dataset gsm8k \
  --output "${ACCEPT_DIR}/vllm_float_gsm8k_full1319_aligned_v1.shard${SHARD}.jsonl" \
  --num-prompts 1319 \
  --num-shards 8 \
  --shard-index "${SHARD}" \
  --num-spec-tokens 7 \
  --max-model-len 8192 \
  --max-tokens 4096 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --draft-sample-method probabilistic
```

### 7.3 量化 Merak 8 卡完整集

量化 shard 仍是单卡、单请求，只是 8 个独立进程并行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

SHARD=0
CUDA_VISIBLE_DEVICES=${SHARD} \
ENABLE_HMINFERENCE_V2=1 \
VLLM_MERAK_USE_V2=1 \
VLLM_MERAK_AUTO_OFFLOAD=1 \
VLLM_MERAK_CUDA_GRAPH=1 \
VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=0 \
VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1 \
python "${ACCEPT_SCRIPT}" \
  --model "${MODEL_DIR}" \
  --backend merak \
  --dataset gsm8k \
  --output "${ACCEPT_DIR}/vllm_merak_gsm8k_full1319_aligned_cg_v2.shard${SHARD}.jsonl" \
  --num-prompts 1319 \
  --num-shards 8 \
  --shard-index "${SHARD}" \
  --num-spec-tokens 7 \
  --max-model-len 8192 \
  --max-tokens 4096 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --draft-sample-method probabilistic \
  --enforce-eager
```

### 7.4 合并和完整性检查

```bash
python "${MERGE_SCRIPT}" \
  "${ACCEPT_DIR}"/vllm_float_gsm8k_full1319_aligned_v1.shard{0..7}.jsonl \
  --output "${ACCEPT_DIR}/vllm_float_gsm8k_full1319_aligned_v1.jsonl"

python "${MERGE_SCRIPT}" \
  "${ACCEPT_DIR}"/vllm_merak_gsm8k_full1319_aligned_cg_v2.shard{0..7}.jsonl \
  --output "${ACCEPT_DIR}/vllm_merak_gsm8k_full1319_aligned_cg_v2.jsonl"

python "${COMPARE_SCRIPT}" \
  "${ACCEPT_DIR}/vllm_float_gsm8k_full1319_aligned_v1.jsonl" \
  "${ACCEPT_DIR}/vllm_merak_gsm8k_full1319_aligned_cg_v2.jsonl" \
  --expected-count 1319 \
  --output "${ACCEPT_DIR}/vllm_float_vs_merak_gsm8k_full1319_paired.json"
```

merge 会要求 benchmark index 精确覆盖 `[0, 1319)`；缺题、重复冲突或不同 shard
参数不一致都会失败。compare 还会逐题核对 dataset index、question、prompt、
source ID、reference answer 和所有采样设置，再输出配对差值及其 95% 区间。

其余四套保持 sampling、selector 和 rejection-sampling 参数不变；
`--dataset`、完整条数、reasoning effort、token/context 上限和文件 stem
按下表切换：

```text
math500   500  max_new_tokens=32768  max_model_len=65536
humaneval 164  max_new_tokens=32768  max_model_len=65536
mbpp      257  max_new_tokens=16384  max_model_len=32768
mt-bench   80  max_new_tokens=16384  max_model_len=32768  # 文件名 stem 使用 mtbench
```

例如 MATH-500 的 shard 参数是：

```bash
--dataset math500 \
--num-prompts 500 \
--reasoning-effort xhigh \
--output "${ACCEPT_DIR}/vllm_float_math500_full500_mnt32768_re-xhigh_aligned_v1.shard${SHARD}.jsonl"
```

Merak 对应文件名为
`vllm_merak_math500_full500_mnt32768_re-xhigh_aligned_cg_v1.shard${SHARD}.jsonl`。
建议始终在文件名中保留 `re-<effort>`；否则 HumanEval 的旧 `xhigh`
诊断文件很容易被误当成正式 `medium` 结果。未显式传
`--max-tokens/--max-model-len` 时 runner 会从上表选择默认值。四套数据都必须分别
merge；随后以相同数据集的 float JSONL 和 Merak JSONL 调用
compare。不要跨数据集合并，也不要用 `--num-prompts 1024` 让短数据集重复采样后称为
“全集”。

若 `xhigh` 的少数长题造成 shard 尾拖，可以把**尚未开始**的 benchmark index 放到
独立 helper 输出；逐题 seed 仍是 `seed + benchmark_index`，因此换卡、换 shard 不改变
该题采样 seed：

```bash
--benchmark-indices 403,456 \
--output "${ACCEPT_DIR}/vllm_merak_math500_tail_helper_gpu7.jsonl"
```

两个进程绝不能同时追加同一个 JSONL。原进程完成当前题并 flush 后再交接；helper 必须使用
完全相同的 model、sampling、reasoning effort、token budget 和 runtime flags。中断后的
文件用同一组参数加 `--summary-only` 做 index 严格校验并重建 sidecar，该模式不加载 vLLM
和 27B 权重。最后仍由 merge 要求 index 精确覆盖 `[0, N)`，缺失、重复冲突或设置不一致
都会失败。这个选项只用于调度慢尾，不能拿来筛掉难题。

### 7.5 2026-08-26 至 2026-08-27 五套全量结果

正式结论使用五套完整 split，共 2320 个请求，不再用早期 128 题或重复采样结果替代。
五份 paired report 的 benchmark index 和 prompt fields 都是 exact：

| dataset | N | effort | max new / model | BF16 stop / length | W4A8 stop / length |
| --- | ---: | --- | ---: | ---: | ---: |
| GSM8K | 1319 | xhigh | 4096 / 8192 | 1281 / 38 | 1270 / 49 |
| MATH-500 | 500 | xhigh | 32768 / 65536 | 497 / 3 | 495 / 5 |
| HumanEval | 164 | medium | 32768 / 65536 | 164 / 0 | 164 / 0 |
| MBPP | 257 | medium | 16384 / 32768 | 257 / 0 | 257 / 0 |
| MT-bench first-turn | 80 | medium | 16384 / 32768 | 80 / 0 | 80 / 0 |

“完整 split”表示每个请求都有结果，不表示每个请求都自然结束。GSM8K 和 MATH-500
的 `length` 行仍是不完整 completion；不能因为它们已计入接收率就把输出质量写成
“无截断”。

下表每格依次为 per-request mean acceptance length、pooled mean acceptance length、
pooled draft acceptance rate。最后一列是逐题 `W4A8 - BF16` 及 normal-approximation
95% CI：

| dataset | vLLM BF16 | Merak W4A8 | paired delta (95% CI) |
| --- | --- | --- | --- |
| GSM8K | 5.468605 / 4.742928 / 53.4704% | 5.386262 / 4.645932 / 52.0847% | -0.082344 [-0.112267, -0.052420] |
| MATH-500 | 5.254855 / 4.289215 / 46.9888% | 5.159649 / 4.159207 / 45.1315% | -0.095207 [-0.135718, -0.054695] |
| HumanEval | 6.290638 / 6.153722 / 73.6246% | 6.212432 / 6.032013 / 71.8859% | -0.078207 [-0.145881, -0.010532] |
| MBPP | 5.400811 / 5.346009 / 62.0858% | 5.387490 / 5.357667 / 62.2524% | -0.013320 [-0.058213, 0.031572] |
| MT-bench | 4.342918 / 3.562391 / 36.6056% | 4.285682 / 3.499718 / 35.7103% | -0.057236 [-0.152680, 0.038207] |

完全对齐后，五套的 W4A8 per-request mean 都没有高于 BF16。GSM8K、MATH-500
和 HumanEval 的区间在 0 以下；MBPP 和 MT-bench 的区间跨 0。MBPP 的 W4A8
pooled mean/rate 略高，但逐题 mean 略低且区间跨 0：这是两边 target 随机轨迹不同、
completion 长度和 speculative step 权重不同造成的口径差异，不是量化准确率更高。
本节只测 acceptance，不等价于数据集 pass rate 或答案准确率。

[vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) 公布的 GSM8K 两项是
per-request `5.34` 和 pooled `4.65`，不是固定的 `5.37`。本地同口径 128 题曾得到
`5.389 / 4.639`；短样本出现约 5.37 属于抽样差异。正式判断使用上面的 1319 题
配对结果，不能拿短跑数、默认 greedy 数或不同 rejection sampler 混比。

completion token 分布如下，P90 使用线性插值：

| dataset | BF16 median / P90 / max | W4A8 median / P90 / max |
| --- | ---: | ---: |
| GSM8K | 294 / 957.2 / 4096 | 287 / 961.8 / 4096 |
| MATH-500 | 605.5 / 6836.1 / 32768 | 633 / 10462.2 / 32768 |
| HumanEval | 967 / 1705.0 / 10092 | 991 / 1787.1 / 7083 |
| MBPP | 880 / 1881.4 / 6048 | 892 / 1936.4 / 7407 |
| MT-bench | 958 / 2366.6 / 6031 | 967.5 / 2444.8 / 9749 |

GSM8K 的 4096 是官方对齐 profile，不是所有数据集的统一上限；若目标是答案完整性，
应另跑更大 budget 的 production profile，不能覆盖或冒充官方接收率结果。MATH-500
即使给到 32768，`xhigh` 仍有 BF16 3 条、W4A8 5 条触顶。触顶样本的局部推理大多
仍有语义关联，有些已经多次写出 boxed 候选答案后继续复核，但没有闭合 reasoning 的
请求仍是不完整结果。

对共享异常样本 benchmark index 42 另做 `medium` 诊断：BF16 在 25440 tokens、W4A8
在 22080 tokens 自然 `stop`，两边最终都是 `2\sqrt{113}`；同一题的正式 `xhigh`
在两边都触及 32768 且没有 `</think>`。这一个对照不能替代全集质量评测，但足以证明
超长轨迹主要受 reasoning effort 影响，不是 W4A8 数值溢出或 rejection sampling
计算错误。HumanEval 的正式 `medium` 结果则两边 164/164 都自然结束，旧 `xhigh`
诊断中的超长自我修正不应视为健康输出。

速度口径也必须分开。GSM8K 的 BF16/W4A8 请求内汇总分别为 107.15/17.81 token/s；
MATH-500 为 91.61/13.39 token/s。它们是 A100、concurrency=1、排除加载和 warmup 的
`sum(completion_tokens) / sum(request_seconds)`，不是 8 卡聚合吞吐；而且 MATH-500
W4A8/BF16 分别生成 1601933/1198339 tokens，不能逐 token 配对。正式日志确认 target
和 draft CUDA Graph replay，artifact 的 FlashAttention 及两类 GDR flag 都为 true，
所以当前吞吐差距不是回退 eager 或漏开 FA/GDR；仍需继续优化 HMONNX CUDA/A100 kernel
与调度路径。接收率已对齐不等于吞吐已追平。

五套 canonical JSONL、同名 `.summary.json` 和 paired report 位于
`work_dirs/qwen3_8_27b_release_20260815/acceptance/`：

```text
vllm_float_gsm8k_full1319_aligned_v1.jsonl
vllm_merak_gsm8k_full1319_aligned_cg_v2.jsonl
vllm_float_vs_merak_gsm8k_full1319_paired.json
vllm_float_math500_full500_mnt32768_re-xhigh_aligned_v1.jsonl
vllm_merak_math500_full500_mnt32768_re-xhigh_aligned_cg_v1.jsonl
vllm_float_vs_merak_math500_full500_mnt32768_re-xhigh_paired.json
vllm_float_humaneval_full164_mnt32768_re-medium_aligned_v1.jsonl
vllm_merak_humaneval_full164_mnt32768_re-medium_aligned_cg_v1.jsonl
vllm_float_vs_merak_humaneval_full164_mnt32768_re-medium_paired.json
vllm_float_mbpp_full257_mnt16384_re-medium_aligned_v1.jsonl
vllm_merak_mbpp_full257_mnt16384_re-medium_aligned_cg_v1.jsonl
vllm_float_vs_merak_mbpp_full257_mnt16384_re-medium_paired.json
vllm_float_mtbench_full80_mnt16384_re-medium_aligned_v1.jsonl
vllm_merak_mtbench_full80_mnt16384_re-medium_aligned_cg_v1.jsonl
vllm_float_vs_merak_mtbench_full80_mnt16384_re-medium_paired.json
```

归档 SHA256：

| dataset | BF16 JSONL | W4A8 JSONL | paired report |
| --- | --- | --- | --- |
| GSM8K | `928f93b56ac378d16bcaa6b3bccebaed2b655e55388d69dbe27174ee93df948f` | `56c0185538777a331da26c826dccf5b09dc96ecf1b660179815764c8299d0673` | `6c3dea49c51185eb62cbdb2bcbc84aa57334d46fac504b753014ec612beba9bd` |
| MATH-500 | `9ca67c0e4717094242af440e8da8e3b607b4eb93c61093ee5354cc77d3cacd66` | `0c621057c2fc07df9e759423c1a93e7ade9666de20928e4f6942555a2479407d` | `3b7452f5d228aca2e3e518d1748c87b8014b226db435acc06565c044b22fb8aa` |
| HumanEval | `64f91f38560641b361e07def05d534405cc11ade8f0e7c688ae993e7017e9b3b` | `ffde7beafecd49be41f08fbdafbb9beccd4725961d5a4d1a35d2ca9d334a7137` | `f2c1186889c1dd5a5eb5bf789d719117573e24fbd9eae68098ab74601b391067` |
| MBPP | `3a1cee0491994c0418507dcc6a5c11e31a22d152ac05e06f231e5aef72aa5835` | `4ed3f9246013dab50561bd799c267790420514ec38da867c2c597bbdfbed3d16` | `6e9bc04a1e1796d609597dd5ab8534002dfd906272ef1d6650b3220c778da60c` |
| MT-bench | `91087c1a791239f4258138e32f027006cc1517c163bdfa48d35493c4cc3cc09e` | `b633b5a995062f98adf352ee30197d8a28b0e90bedeeb72a6602b1acf3e4c5e4` | `8dd952a6393b37dbe5e8af2332f4480657a0af906a5a4e36db55581794f3673f` |

## 8. Rejection sampling 正确性

设 target 经 temperature/top-p/top-k 处理后的分布为 p，DFlash2 selector 的提案
分布为 q，提议 token 为 y：

```text
accept(y) = min(1, p(y) / q(y))
reject recovery distribution ∝ max(p - q, 0)
```

全部 K 个 draft 接受后，再追加一个只由 target 产生的 bonus token。该算法保证
最终输出仍服从 target 的 p；投机模型只能改变速度和生成轨迹所消耗的随机数，
不能改变目标分布。

`greedy` draft 把 q 当作提议 token 上的 one-hot，非零温度 target 下单步接受
概率退化为 `p(y*)`。`probabilistic` 的期望接受概率为
`sum(min(p, q)) = 1 - TV(p, q)`，因此只要 selector q 与 target p 接近，通常
显著高于 greedy。

## 9. 常见故障

### 9.1 浮点只有约 4.8，量化却约 5.3

先检查浮点 speculative config 是否遗漏：

```json
"draft_sample_method": "probabilistic"
```

vLLM 合入版默认 `greedy`。早期 128 题“量化高于浮点”的主要原因就是 Merak
走 probabilistic，而浮点走默认 greedy，并非量化提高了 assistant 精度。

### 9.2 DFlash2 FP16 出现 inf/NaN

不要把 assistant checkpoint 强制转成 FP16。保持：

```yaml
dflash_config:
  dtype: bfloat16
  activation_residual_scale: 128
```

受控的 83 个 stage probe 中，FP16 + scale 1 有 69 个 stage 出现非有限值，首个
故障点是 `layer0.mlp`（8 个 Inf）；BF16 + scale 1 为 0 个，FP16 + scale 128
也为 0 个。正式链路仍按 checkpoint 用 BF16 加载，并把 scale 128 固化到导出和
runtime 合约，不能因为缩放后的单次 FP16 probe 有限就把 checkpoint dtype 改掉。

同时检查导出 metadata 中两个字段仍然存在。scale 必须在 PyTorch reference、
HMONNX export 和 runtime 反缩放三处一致。

### 9.3 看不到 FlashAttention

- 浮点日志必须看到 vLLM 选择 `FLASH_ATTN`；
- 量化 artifact 的 `model_config.flash_attention.enable` 必须为 true；
- 正式量化 artifact 的 draft ABI 必须为 `qwen_dflash2_paged_shared_v1`，且
  `merak_config.json` 中 `enable_page_attention=true`；
- 不要用非 FA 8K debug artifact 做正式性能测试。

### 9.4 Merak 启动或首题很慢

首次加载包括 W4 pack、HMONNX initializer 去重和 CUDA Graph capture。接收率
计时前先跑一个独立 warmup；不要把启动/capture 时间算入单题 decode。正式
benchmark 脚本已经做了 64-token warmup。

`VLLM_MERAK_USE_V2=1` 还会首次物化并缓存 target 的 parallel-Linear fused
prefill/decode 图。该步骤可能持续数分钟并增加约 32GB 派生文件，但有文件锁且只需
完成一次；多 worker 冷启动前先单进程预热。正式日志应同时看到 target 的
`CudaGraphAutoOffLoadGraphModuleInterpreter`、`borrowed_outputs>0`，以及 draft
decode 的 `CUDA Graph replay active`。只看到最后一项说明仅 draft 开了 graph，
实测 decode 吞吐会低约一半。本次用同两个固定 seed 请求做 A/B：仅 draft graph
分别为 6.85、5.31 token/s；target + draft graph 为 15.21、12.81 token/s。
两组的 token IDs 和 acceptance 明细逐项一致，因此该开关只修复性能，不改变
采样结果。

### 9.5 vLLM 退出时报 `libnvrtc.so.13`，但 summary 已写出

当前混合 CUDA 环境可能在 EngineCore teardown 导入可选 allocator/deep-gemm
扩展时报告 CUDA 13 动态库缺失。若所有请求、JSONL 和 summary 已完整写出，
这是退出清理告警，不是推理失败；仍应单独修正运行环境的 CUDA extension ABI，
不要在生产服务中长期忽略。

### 9.6 多进程冷启动报派生图或 recurrent-state 合约错误

target PageAttention 派生图使用稳定的 hash 文件名。旧实现会让多个首次启动的进程
直接写同一个最终 `.onnx`，另一个进程可能在写完前执行 `onnx.load`。当前工作区已改为
在目标目录生成进程私有临时文件，再通过原子 replace 发布最终图。

本次并发诊断还遇到过一次
`decode recurrent-state input layers mismatch: expected=0..47, actual=[]`。当时最终 ONNX
可正常读取，其他已预热 worker 也能加载同一 artifact，因此不能把它解释为 DFlash2
没有导出 48 层状态，更不能通过关闭状态校验绕过。它属于多进程冷启动期间的派生图可见性/
初始化时序故障；原子发布解决了半写 protobuf，但不等价于允许多个全冷 worker 同时完成
所有派生图和 runtime 合约初始化。

无论是否已经包含原子发布修复，生产部署都先用一个进程完成 smoke/warmup，确认 target
和 draft 的派生图、48 层 recurrent state 合约及 CUDA Graph replay 全部就绪，再启动其余
worker。不要让 8 个全冷进程同时物化同一 artifact。失败后先停止该 worker，确认最终图
大小和 SHA256 已稳定，再单独重试；不要删除原始 prefill/decode 图或 external data。
若顺序重试仍复现，再按 runtime input name 和派生图 hash 定位，不能笼统归因于模型结构。
可在 `vllm` 环境做只读 protobuf 检查：

```bash
python - "${MODEL_DIR}/decode/<name>_decode_with_act_page_attention_<hash>.onnx" <<'PY'
import onnx
import sys

path = sys.argv[1]
model = onnx.load(path, load_external_data=False)
print(len(model.graph.node), len(model.graph.initializer))
PY
```

### 9.7 HumanEval 出现异常长 reasoning

这不是 DFlash2 cache 或 rejection sampling 的正常副作用。`xhigh` 会让 Qwen3.8 在一些
很短的代码题上不断补边界条件、推翻并重写方案；抽查时多数相邻推理步骤有语义关联，不能
简单归类为随机乱码，但它是**输出控制异常**：一旦触及 token 上限、没有 `</think>` 或
没有给出最终代码，就算局部推理合理，该请求也仍是不完整结果，不能用于质量结论。

正式 HumanEval 因此使用 `reasoning_effort=medium`、
`max_new_tokens=32768/max_model_len=65536`。164 题中 BF16 和 W4A8 都是 164/164
自然 `stop`；completion token 中位数分别为 967 和 991，最大值分别为 10092 和 7083。
这说明旧 `xhigh` 长输出不是必须的模型计算流程，也不是把统一上限从 4096 提高后就应该
接受的现象。审计时至少同时检查 reasoning 是否闭合、最终代码是否出现、finish reason
是否为 `stop`，以及代码执行/单测结果；只看文字“似乎有逻辑”不够。

## 10. 发布检查单

- [ ] target 和 DFlash2 checkpoint revision/hash 已记录；
- [ ] 浮点 vLLM 包含 DFlash2 PR；
- [ ] target 使用 W4G64 symmetric AutoRound checkpoint；
- [ ] 256K FA workflow 同时开启两类 GDR fusion；
- [ ] assistant 以 BF16 加载，activation residual scale 为 128；
- [ ] K=7、verify length=8、selector top-k=16；
- [ ] 三张 draft 图存在并共享 private PageAttention cache；
- [ ] artifact ABI 为 `qwen_dflash2_paged_shared_v1`；
- [ ] Merak `max_num_seqs=1`，prefix cache/async scheduling 关闭；
- [ ] 非零温度部署显式写 `probabilistic + standard`；
- [ ] 浮点日志确认 `FLASH_ATTN`；
- [ ] 官方五套共 2320 请求无缺失；初始 shard 与 slow-tail helper 的 sidecar 参数一致，canonical merge index exact；
- [ ] 五份 paired report 的请求数分别为 1319/500/164/257/80，且 `benchmark_indices_exact`、`prompt_fields_exact` 均为 true；
- [ ] 同时报告 per-request 与 pooled 指标，不能混用；
- [ ] full-dataset 结果和原始 JSONL/summary 一并归档。
