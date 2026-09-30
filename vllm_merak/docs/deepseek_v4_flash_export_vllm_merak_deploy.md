# DeepSeek V4 Flash 量化、HMONNX 导出与 vLLM Merak 部署

本文给出 DeepSeek-V4-Flash-0731 从原始模型、W8E4G64 checkpoint、
W4A8/W4A16 HMONNX 导出，到 vLLM Merak OpenAI 服务部署和验收的完整流程。
默认目标为 XH2a、全部 43 层、256-token prefill chunk 和 262144-token
上下文。

本文同时记录 2026-08-20 实际部署中已经遇到的问题，包括：W4 命名歧义、
首次请求 CUDA Graph 捕获时间、`flashinfer` 版本冲突、Torch 扩展 ABI、
`reasoning_effort` 档位，以及 Cherry Studio 中长时间重复“让我写代码”的原因。

## 1. 先理解精度名称

DeepSeek V4 Flash 的正式输入 checkpoint 不是“全模型 W4”：

- 原始官方 checkpoint 混合使用 FP8 E4M3 和 routed-expert E2M1 FP4；
- GPTQModel/AutoRound 部署 checkpoint 中，非共享 routed experts 为 W4G64；
- shared expert、attention 等已登记的普通量化 Linear 为 W8G64；
- compressor、indexer、router、norm、mHC 等敏感控制路径保持浮点；
- HMONNX 的 `w8a8h1_sefp` / `w8a16h1_sefp` 表示图的基础权重和激活档；
- routed experts 仍从 E4 checkpoint 直接进入 W4 runtime。

因此本文采用以下名称：

| 名称 | 权重 | 激活 | 推荐用途 |
| --- | --- | --- | --- |
| W4A8 | W8E4G64；routed experts W4 | A8 | 默认部署，速度优先 |
| W4A16 | W8E4G64；routed experts W4 | A16 | 较高激活精度，对照和精度优先 |

产物目录里出现 `w8a8` 或 `w8a16` 是正常的，不能据此断言 routed experts
没有使用 W4。最终应以服务加载日志中的 `W4 runtime contract audit` 为准。

## 2. 仓库、环境与版本

### 2.1 仓库职责

```bash
export WORK_ROOT=/data01/home/yujy/work
export XHQUANT_ROOT=${WORK_ROOT}/xhquanttool
export MODELZOO_ROOT=${WORK_ROOT}/xh2modelzoo
export VLLM_MERAK_ROOT=${MODELZOO_ROOT}/vllm_merak
export GPTQMODEL_ROOT=${WORK_ROOT}/gptqmodel
```

| 仓库 | 作用 |
| --- | --- |
| `gptqmodel` | 原生 FP8/FP4 到 BF16 staging，以及 GPTQ/AutoRound W8E4G64 量化 |
| `xh2modelzoo` | DeepSeek V4 模型适配、workflow YAML、HMONNX 导出和 Golden |
| `xhquanttool` | HMONNX loader/runtime、W4 在线 Pack、多卡放置和 CUDA Graph |
| `xh2modelzoo/vllm_merak` | vLLM 插件、OpenAI 服务、tokenizer/parser 和请求调度 |

正式量化或发布前记录准确 revision，不要只记分支名：

```bash
for repo in \
  "${GPTQMODEL_ROOT}" \
  "${MODELZOO_ROOT}" \
  "${XHQUANT_ROOT}"; do
  printf '%-48s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
done
```

### 2.2 已验证版本

2026-08-20 的完整 43 层 A8/A16 服务使用了以下组合：

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| `xhquant_55` | BF16 staging、量化、导出、Golden | Python 3.12、Torch 2.8.0+cu128、Transformers 5.13.0、GPTQModel 5.8.0 |
| `vllm0271_test` | vLLM Merak 服务 | Python 3.12.13、Torch 2.13.0+cu129、Transformers 5.13.0、vLLM 0.27.1+cu129 |
| `vllm0271_test` | runtime 依赖 | vLLM-Merak 0.20.0、FlashInfer 0.6.16.post3、Triton 3.7.1 |

量化/导出和服务环境应分开。Torch C++/CUDA 扩展与 ABI 绑定，不要把在
Torch 2.8 环境生成的 `hsum_gpu.so` 直接交给 Torch 2.13 使用。

### 2.3 安装量化/导出环境

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

验证实际 import 来源和版本：

```bash
python - <<'PY'
import torch
import transformers
import gptqmodel
import xhmodel_merak
import xhquant

print("torch       =", torch.__version__)
print("transformers=", transformers.__version__)
print("gptqmodel   =", gptqmodel.__file__)
print("modelzoo    =", xhmodel_merak.__file__)
print("xhquant     =", xhquant.__file__)
PY
```

### 2.4 用 clone 环境升级 vLLM

不要直接覆盖当前生产环境。先 clone，再在 clone 中升级和回归：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n vllm_dsv4_0271 --clone vllm
conda activate vllm_dsv4_0271

# 使用团队发布源或已验收 wheel；下面是使用 Python 包索引的形式。
python -m pip install --upgrade "vllm==0.27.1"

# 插件和模型代码使用当前源码，避免安装器再次改写 Torch 依赖。
python -m pip install -v -e "${MODELZOO_ROOT}" \
  --no-build-isolation --no-deps
python -m pip install -v -e "${XHQUANT_ROOT}" \
  --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
RUNTIME_PROFILE=deepseek_v4_vllm_0271 \
PYTHON_BIN=python "${VLLM_MERAK_ROOT}/tools/install_from_modelzoo.sh"
```

如果 `xhquanttool` 源码树会在不同环境中原地生成扩展，推荐为 runtime 创建
独立 worktree，防止两个 Torch ABI 互相覆盖：

```bash
export XHQUANT_RUNTIME_ROOT=${WORK_ROOT}/xhquanttool-vllm
git -C "${XHQUANT_ROOT}" worktree add --detach \
  "${XHQUANT_RUNTIME_ROOT}" \
  "$(git -C "${XHQUANT_ROOT}" rev-parse HEAD)"

python -m pip install -v -e "${XHQUANT_RUNTIME_ROOT}" \
  --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
```

若 worktree 已存在，不要重复创建；先确认它与量化侧 revision 相同。

升级后至少执行：

```bash
python -m pip check
python - <<'PY'
import torch
import transformers
import vllm
import vllm_merak
import xhquant

print("torch       =", torch.__version__)
print("transformers=", transformers.__version__)
print("vllm        =", vllm.__version__)
print("vllm_merak  =", vllm_merak.__file__)
print("xhquant     =", xhquant.__file__)
PY
```

vLLM-Merak 使用 vLLM 的部分内部 API。升级 vLLM 不能只验证 DeepSeek V4，
还要回归 Qwen3.5 和 Gemma4 的加载、首 token、连续请求、工具调用和长上下文。

## 3. 从原始 checkpoint 生成 W8E4G64

如果已经有验收通过的 GPTQModel/AutoRound W8E4G64 checkpoint，可直接跳到
第 4 节。量化阶段只需要执行一次，A8 和 A16 导出复用同一份 checkpoint。

### 3.1 设置路径

```bash
export SOURCE_MODEL=/data01/datasets/DeepSeek-V4-Flash-0731
export BF16_MODEL=/path/to/deepseek-v4-flash-bf16-full
export QUANT_MODEL=/path/to/deepseek-v4-flash-AutoRoundV2-W8E4G64
export CALIB_JSONL=/path/to/NeelNanda-pile-10k.jsonl
```

原始模型必须包含：

```text
config.json
model.safetensors.index.json
model-*.safetensors
tokenizer.json
tokenizer_config.json
```

`config.json` 应为 `model_type=deepseek_v4`，并带原生 FP8 量化描述。

### 3.2 为什么需要 BF16 staging

原始 checkpoint 的普通权重为 FP8 E4M3，routed experts 为打包 E2M1 FP4，
GPTQ/AutoRound 校准不能把它们当普通 BF16 Linear 直接处理。GPTQModel recipe
会按 shard 流式反量化为 BF16 staging，同时保留：

- absorbed latent-attention 的 `q_a/q_b`、shared KV；
- grouped `o_a/o_b` 分解；
- 真实层数和 tokenizer/config 资产；
- 默认排除 MTP 权重。

它不会展开传统 K/V 投影，也不会合并 `o_a/o_b`。完整模型省略
`--num-layers`；只有诊断时才用 `--num-layers 6` 等前缀模型。

可以显式先准备 BF16：

```bash
cd "${GPTQMODEL_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
python examples/quantization/examples/example_deepseek_v4_flash.py \
  --model "${SOURCE_MODEL}" \
  --prepared-model "${BF16_MODEL}" \
  --out "${QUANT_MODEL}" \
  --prepare-only \
  --preparation-device cuda:0
```

成功后应有 `deepseek_v4_flash_bf16_preparation.json` 和完整的 safetensors
index。recipe 拒绝覆盖已有输出目录；不要在半成品目录上直接重跑。

### 3.3 先打印量化计划

```bash
cd "${MODELZOO_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/deepseekv4/quantize.py \
  --method autoround \
  --model "${SOURCE_MODEL}" \
  --prepared-model "${BF16_MODEL}" \
  --output "${QUANT_MODEL}" \
  --calibration-jsonl "${CALIB_JSONL}" \
  --dry-run
```

计划必须显示：

- base W8；
- routed non-shared experts W4；
- group size 64；
- symmetric quantization；
- `offload_to_disk=false`；
- `auto_forward_data_parallel=false`。

### 3.4 AutoRound V2 正式量化

```bash
cd "${MODELZOO_ROOT}"
CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/deepseekv4/quantize.py \
  --method autoround \
  --model "${SOURCE_MODEL}" \
  --prepared-model "${BF16_MODEL}" \
  --output "${QUANT_MODEL}" \
  --calibration-jsonl "${CALIB_JSONL}" \
  --text-key text \
  --nsamples 128 \
  --seqlen 2048 \
  --batch-size 1 \
  --base-bits 8 \
  --expert-bits 4 \
  --group-size 64 \
  --device cuda:0 \
  --preparation-device cuda:0 \
  --moe-batch-size 128 \
  --auto-round-version v2 \
  --auto-round-iters 200 \
  --seed 42
```

GPTQ 使用相同的 mixed-precision 规则，只需改为：

```bash
--method gptq
```

默认按 decoder layer、subset 和 MoE module chunk 串行量化，单张 80-GiB GPU
即可。不要因为看到多张卡就默认开启 `auto_forward_data_parallel`；它会复制当前
层的 forward replay，不是完成 43 层的必要条件。磁盘 offload 也只在明确需要时
通过 `--offload-to-disk --offload-dir ...` 开启。

> 2026-08-20 用于打通端到端流程的现有 artifact provenance 为
> `nsamples=2`、`seqlen=64`、AutoRound `iters=2`。它通过了当时的 5 个历史
> bad case，但这是工程 smoke 配置，不能替代 `128/2048/200` 的正式精度量化。

### 3.5 检查量化产物

```bash
test -f "${QUANT_MODEL}/config.json"
test -f "${QUANT_MODEL}/quantize_config.json"
test -f "${QUANT_MODEL}/model.safetensors.index.json"
python -m json.tool "${QUANT_MODEL}/quantize_config.json" >/dev/null
```

重点检查：

- `bits` 为 8；
- `dynamic` 中 experts 的 `bits` 为 4、`group_size` 为 64；
- `sym=true`、`desc_act=false`；
- 所有 43 层 routed expert 的 gate/up/down 权重均被覆盖；
- tokenizer、generation config 和量化 provenance 已随 checkpoint 保存。

## 4. 导出 W4A8 和 W4A16 HMONNX

### 4.1 正式 workflow 配置

配置位于：

```text
configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731/
├── deepseek_v4_flash_0731_w4a8.yaml
└── deepseek_v4_flash_0731_w4a16.yaml
```

两者只有 `quant_type` 不同：

| 字段 | A8 | A16 |
| --- | --- | --- |
| `quant_scheme.quant_type` | `w8a8h1_sefp` | `w8a16h1_sefp` |
| `context_max_length` | 262144 | 262144 |
| `prefill_chunk_length` | 256 | 256 |
| `max_pe_length` | 262144 | 262144 |
| `max_layers` | `null`，全部 43 层 | `null`，全部 43 层 |
| `packed_weight_only` | `true` | `true` |
| `runtime.low_memory` | `true` | `true` |

`packed_weight_only=true` 表示直接消费 GPTQ/AutoRound 整数和 FP16 scale，不在
导出阶段物化 dense FP 权重。`enable_auto_offload=false` 是导出阶段设置；服务
加载后的四卡 runtime 放置是另一层配置，两者不要混淆。

### 4.2 导出 A8

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

export A8_EXPORT_ROOT=/path/to/exports/deepseek_v4_flash_w4a8_full43

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/deepseekv4/deepseek_v4_workflow.py \
  --model-dir "${QUANT_MODEL}" \
  --config-path configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731/deepseek_v4_flash_0731_w4a8.yaml \
  --export-output-dir "${A8_EXPORT_ROOT}" \
  --model-name deepseek_v4_flash_0731_a8_full43 \
  --context-length 262144 \
  --quant-type w8a8h1_sefp \
  --low-memory
```

### 4.3 导出 A16

```bash
export A16_EXPORT_ROOT=/path/to/exports/deepseek_v4_flash_w4a16_full43

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/deepseekv4/deepseek_v4_workflow.py \
  --model-dir "${QUANT_MODEL}" \
  --config-path configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731/deepseek_v4_flash_0731_w4a16.yaml \
  --export-output-dir "${A16_EXPORT_ROOT}" \
  --model-name deepseek_v4_flash_0731_a16_full43 \
  --context-length 262144 \
  --quant-type w8a16h1_sefp \
  --low-memory
```

不要传 `--max-layers`，否则得到的是诊断用前缀模型，不是完整 43 层。A8 和
A16 可以复用同一 `QUANT_MODEL`；在资源不确定时顺序导出，避免两个 550-GiB
级工作目录同时写盘造成容量或 I/O 问题。

workflow 的 `quant: null` 表示这里不会重新量化；`workflow.quant()` 只把已有
checkpoint 作为 export 输入继续传递。

### 4.4 低层诊断导出

需要绕过 workflow 定位问题时，使用低层 CLI：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/deepseekv4/export_hmonnx.py \
  --model "${QUANT_MODEL}" \
  --output-dir /path/to/deepseek_v4_diag_a8 \
  --model-name deepseek_v4_flash_0731_diag \
  --chip-arch XH2a \
  --context-length 262144 \
  --quant-type w8a8h1_sefp \
  --max-layers 6 \
  --packed-weight-only \
  --low-memory
```

该接口适合 1/6 层快速定位。正式产物仍应使用 workflow YAML，保证配置、命名和
Golden 流程一致。

### 4.5 Golden

需要 Golden 时，在同一次 workflow 导出调用中追加 `--dump-golden`，让脚本在
导出完成、释放 PyTorch export model 后继续生成正式 prefill/decode `step_0`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples_merak/llm/deepseekv4/deepseek_v4_workflow.py \
  --model-dir "${QUANT_MODEL}" \
  --config-path configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731/deepseek_v4_flash_0731_w4a8.yaml \
  --export-output-dir "${A8_EXPORT_ROOT}" \
  --model-name deepseek_v4_flash_0731_a8_full43 \
  --dump-golden \
  --golden-device-map 0 1 2 3 \
  --golden-prompt '17乘以3等于多少？只回答结果。'
```

注意：每个 export 都会有名为 `golden_meta_info.json` 的运行元数据；只有显式
`--dump-golden` 才会生成用于数值对比的 Golden tensor/step 目录。

如果 HMONNX 已经导出，不要对同一输出目录重跑 workflow；直接指定实际
`hmquant_*` 目录运行生成脚本：

```bash
export MODEL_DIR=/path/to/a8/hmquant_xh2_deepseek_v4_flash_0731_...

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples_merak/llm/deepseekv4/deepseek_v4_xh_hmonnx_generate.py \
  --config "${MODEL_DIR}" \
  --device 0,1,2,3 \
  --prompt '17乘以3等于多少？只回答结果。' \
  --max-new-tokens 2 \
  --pack-w4 \
  --dump-golden
```

## 5. 检查 HMONNX 产物

workflow 会在 export root 下创建一个 `hmquant_*` 实际模型目录。服务的
`--model` 必须指向这一层，而不是它的父目录：

```text
hmquant_xh2_deepseek_v4_flash_0731_*_w8a*_256_256k_mpe256k_YYYYMMDD/
├── merak_config.json
├── golden_meta_info.json
├── quant_embedding.pt
├── hf_config/
│   ├── config.json
│   ├── generation_config.json
│   ├── quantize_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── prefill/
│   ├── *_prefill_with_act.onnx
│   └── *_prefill_external_data
└── decode/
    ├── *_decode_with_act.onnx
    └── *_decode_external_data
```

服务首次加载后还可能在 prefill/decode 目录生成带 hash 的
`*_linear_fused_*` 缓存，这是 loader 优化产物，不应被当成另一版模型。

### 5.1 静态检查

```bash
export MODEL_DIR=/path/to/hmquant_xh2_deepseek_v4_flash_0731_...

test -f "${MODEL_DIR}/merak_config.json"
test -f "${MODEL_DIR}/golden_meta_info.json"
test -f "${MODEL_DIR}/quant_embedding.pt"
test -d "${MODEL_DIR}/hf_config"
test -d "${MODEL_DIR}/prefill"
test -d "${MODEL_DIR}/decode"
python -m json.tool "${MODEL_DIR}/merak_config.json" >/dev/null
python -m json.tool "${MODEL_DIR}/golden_meta_info.json" >/dev/null
```

`golden_meta_info.json` 至少应满足：

- `model_type=DeepseekV4ForCausalLM`；
- `num_layers=43`；
- `context_max_length=262144`；
- `prefill_chunk_length=256`；
- `max_pe_length=262144`；
- `packed_weight_only=true`；
- `kv_cache.num_layers=43`；
- `prefill_hmonnx` 和 `decode_hmonnx` 指向存在的图；
- `hf_config`、tokenizer 和 embedding 引用均存在。

这里的两个 `256` 含义不同：

- `262144` 是最大上下文；
- `256` 是固定 prefill chunk，也是服务端单次 batch token budget；
- 把 `--max-num-batched-tokens 256` 误认为只支持 256 上下文是错误的。

### 5.2 加载审计

加载但不生成文本：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm_dsv4_0271
cd "${MODELZOO_ROOT}"

export PYTHONPATH="${VLLM_MERAK_ROOT}:${MODELZOO_ROOT}:${XHQUANT_RUNTIME_ROOT:-${XHQUANT_ROOT}}:${PYTHONPATH:-}"
export ENABLE_HMINFERENCE_V2=1
export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples_merak/llm/deepseekv4/deepseek_v4_xh_hmonnx_generate.py \
  --config "${MODEL_DIR}" \
  --device 0,1,2,3 \
  --pack-w4 \
  --load-only
```

2026-08-20 的 A8/A16 完整产物对 prefill 和 decode 都得到：

```text
nodes=43
initializers=129
source_nbytes=277025390592
packed_nbytes=138512695296
saved_nbytes=138512695296
op_types={'MoeBlock': 43}
```

这才是 43 个 routed MoE 层按 W4 在线打包并复用的直接证据。

## 6. 启动 vLLM Merak 服务

### 6.1 通用环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm_dsv4_0271
cd "${VLLM_MERAK_ROOT}"

export XHQUANT_RUNTIME_ROOT=${XHQUANT_RUNTIME_ROOT:-${XHQUANT_ROOT}}
export PYTHONPATH="${VLLM_MERAK_ROOT}:${MODELZOO_ROOT}:${XHQUANT_RUNTIME_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_PLUGINS=merak
export ENABLE_HMINFERENCE_V2=1
export VLLM_MERAK_USE_V2=1
export VLLM_MERAK_AUTO_OFFLOAD=1
export VLLM_MERAK_CUDA_GRAPH=1
export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1
export XHQUANT_HSUM_JIT=1
```

设置模型和日志路径：

```bash
export A16_MODEL=/path/to/a16/hmquant_xh2_deepseek_v4_flash_0731_...
export A8_MODEL=/path/to/a8/hmquant_xh2_deepseek_v4_flash_0731_...
export SERVICE_LOG_ROOT=/path/to/service_logs
mkdir -p "${SERVICE_LOG_ROOT}"
```

### 6.2 参数说明

两套服务共同使用：

```bash
COMMON_ARGS=(
  --served-model-name deepseek-v4-flash-hmonnx
  --host 0.0.0.0
  --tokenizer-mode deepseek_v4
  --trust-remote-code
  --dtype float16
  --max-model-len 262144
  --enforce-eager
  --config-format merak_llm
  --load-format merak_llm
  --generation-config vllm
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v4
  --max-num-batched-tokens 256
  --max-num-seqs 1
  --no-async-scheduling
)
```

关键点：

- `--tokenizer-mode deepseek_v4` 不能省略，否则 chat/tool/reasoning prompt 不一致；
- `VLLM_MERAK_USE_V2=1`、`VLLM_MERAK_AUTO_OFFLOAD=1` 和
  `VLLM_MERAK_CUDA_GRAPH=1` 必须同时设置；三者共同让 Merak 把全部可见 GPU
  写入 HMONNX `device_map`。仅设置 `ENABLE_HMINFERENCE_V2=1` 会让整网落到
  单卡并在加载阶段 OOM；
- vLLM 可能将上述 `VLLM_MERAK_*` 打印为 unknown environment variable，变量
  实际由 Merak 插件读取，这个 warning 不影响生效；
- reasoning parser 和 tool parser 决定 `reasoning_content` 与 tool calls 的拆分；
- `--enforce-eager` 关闭的是 vLLM 自身编译图，HMONNX runtime 仍会懒捕获自己的 CUDA Graph；
- `--max-num-seqs 1` 是当前 cache owner 和请求隔离约束，不表示 HTTP 不能排队；
- prefix cache 保持关闭；当前已验证启动日志中 `enable_prefix_caching=False`；
- `--max-model-len 262144` 必须与 artifact 一致；
- `--max-num-batched-tokens 256` 必须与固定 prefill chunk 协同，不能改成 256K。

### 6.3 GPU 0-3 启动 A16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${A16_MODEL}" \
  --port 18184 \
  "${COMMON_ARGS[@]}" \
  >"${SERVICE_LOG_ROOT}/deepseek_v4_w4a16_18184.log" 2>&1 &
echo $! >"${SERVICE_LOG_ROOT}/deepseek_v4_w4a16_18184.pid"
```

### 6.4 GPU 4-7 启动 A8

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${A8_MODEL}" \
  --port 18185 \
  "${COMMON_ARGS[@]}" \
  >"${SERVICE_LOG_ROOT}/deepseek_v4_w4a8_18185.log" 2>&1 &
echo $! >"${SERVICE_LOG_ROOT}/deepseek_v4_w4a8_18185.pid"
```

两个端口复现已验证基线时使用同一个 served name；生产注册中心若需要同时展示
两档，建议分别命名为 `deepseek-v4-flash-w4a16` 和
`deepseek-v4-flash-w4a8`，并同步修改客户端 `model` 字段。

停止服务时只终止 pidfile 中确认过的 PID：

```bash
ps -fp "$(cat "${SERVICE_LOG_ROOT}/deepseek_v4_w4a16_18184.pid")"
kill "$(cat "${SERVICE_LOG_ROOT}/deepseek_v4_w4a16_18184.pid")"
```

### 6.5 健康检查

服务从启动到 `/health` 可用需要加载和多卡放置完整权重。首次启用 V2 时还会生成
并行 Linear 融合图、计算 initializer 哈希并建立共享权重缓存，可能需要 15 分钟以上；
缓存命中后的启动会明显缩短。不要用 30 秒探针反复杀进程。

```bash
curl -fsS --max-time 10 http://127.0.0.1:18184/health
curl -fsS --max-time 10 http://127.0.0.1:18184/version \
  | python -m json.tool
curl -fsS --max-time 10 http://127.0.0.1:18184/v1/models \
  | python -m json.tool
```

`/version` 应返回 `0.27.1`，`/v1/models` 中的 `max_model_len` 应为
`262144`。A8 端口同理。

### 6.6 首次请求预热

服务健康不代表 CUDA Graph 已捕获。首次真实请求会依次触发：

1. prefill 四个 stage 的 HMONNX CUDA Graph 捕获，实测约 4.5 分钟；
2. 首次多 token decode 的四个 stage 捕获，实测约 70 秒；
3. 后续请求进入稳定时延。

预热请求应给足客户端超时，且不要在超时后自动重试十几次：

```bash
curl -fsS --max-time 900 http://127.0.0.1:18184/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash-hmonnx",
    "messages":[{"role":"user","content":"只回复：OK"}],
    "reasoning_effort":"none",
    "temperature":0,
    "max_tokens":8
  }' | python -m json.tool
```

再发一次 32-token 以上的请求完成 decode 捕获。不要把首次捕获时间计入稳定性能。

## 7. OpenAI API、思考档位与 Cherry Studio

### 7.1 无思考模式

代码生成、工具调用和普通聊天优先从 `none` 开始：

```bash
curl -fsS --max-time 900 http://127.0.0.1:18185/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash-hmonnx",
    "messages":[{"role":"user","content":"写一个单文件 HTML 飞机大战游戏，只输出代码。"}],
    "reasoning_effort":"none",
    "temperature":1.0,
    "top_p":0.95,
    "max_tokens":8192
  }' | python -m json.tool
```

### 7.2 `high` 和 `max`

`reasoning_effort` 是请求 JSON 的顶层字段，不是 system prompt，也不是
`messages` 的一部分：

```json
{
  "model": "deepseek-v4-flash-hmonnx",
  "messages": [{"role": "user", "content": "请证明这个结论。"}],
  "reasoning_effort": "high",
  "max_tokens": 8192
}
```

最大档：

```json
{
  "model": "deepseek-v4-flash-hmonnx",
  "messages": [{"role": "user", "content": "请完整分析并给出最终结论。"}],
  "reasoning_effort": "max",
  "max_tokens": 16384
}
```

vLLM 0.27.1 的 DeepSeek V4 tokenizer 实际映射为：

| 请求值 | DeepSeek V4 模式 |
| --- | --- |
| `none` | chat，不思考 |
| 未传 | 由 `enable_thinking` 决定；本文服务默认开启 |
| `minimal` / `low` / `medium` / `high` | 统一归一化为 `high` |
| `xhigh` / `max` | 统一归一化为 `max`，额外加入最大推理强度前缀 |

因此这版 DeepSeek V4 并没有四个真正不同的 low/medium/high 档，核心是
`none`、普通 thinking/high、`max` 三种行为。

`high/max` 会先输出较长 `reasoning_content`，最终答案在 `content`。如果
`max_tokens` 太小，token 可能全部消耗在 reasoning，客户端看到的 `content`
为空。这不是服务未返回，先查看完整 JSON，再增加输出预算。

### 7.3 Cherry Studio 设置

在 OpenAI Compatible provider 的模型自定义参数或“额外请求 Body”中加入：

```json
{"reasoning_effort":"none"}
```

需要思考时改为：

```json
{"reasoning_effort":"high"}
```

或：

```json
{"reasoning_effort":"max"}
```

同时把最大输出 token 调到与档位匹配的值。若客户端界面没有该字段，先用
`curl` 验证服务，再检查客户端是否支持透传自定义 OpenAI Body；不要把
`reasoning_effort high` 写进用户正文代替参数。

### 7.4 为什么会重复“让我写代码”

实际“飞机大战”请求出现过模型长时间重复内部计划：

```text
让我写完整的代码。
让我把最终的代码写出来。
...
```

这通常是 thinking 输出在短预算或不合适 prompt 下进入自循环，不是 HMONNX
权重没有加载，也不一定是 HTTP 请求发了多次。排查顺序：

1. 查 access log/request id，确认客户端是否因首轮超时自动重试；
2. 查看原始 JSON，区分 `reasoning_content` 和最终 `content`；
3. 代码生成先设 `reasoning_effort=none`；
4. 必须思考时提高 `max_tokens`，避免在 reasoning 中被截断；
5. 多轮对话不要把上轮完整 reasoning 再作为 assistant `content` 回灌；
6. 同一 prompt 用 `temperature=0` 做 greedy 复现，检查请求隔离；
7. 不要用随意的 stop word 掩盖循环，它可能截断合法代码。

把 vLLM 升级到 0.27.1 有助于统一当前 tokenizer/parser 接口，但不会自动消除
这种模型级 reasoning 自循环。已验证的飞机大战请求在
`reasoning_effort=none` 后直接进入 HTML 输出。

### 7.5 工具调用 smoke

```bash
curl -fsS --max-time 900 http://127.0.0.1:18185/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash-hmonnx",
    "messages":[{"role":"user","content":"请读取 report.txt 的前 1000 个字符。"}],
    "tools":[{
      "type":"function",
      "function":{
        "name":"read_file",
        "description":"读取文件内容",
        "parameters":{
          "type":"object",
          "properties":{
            "filename":{"type":"string"},
            "offset":{"type":"integer"},
            "limit":{"type":"integer"}
          },
          "required":["filename"]
        }
      }
    }],
    "tool_choice":"auto",
    "reasoning_effort":"none",
    "temperature":0,
    "max_tokens":512
  }' | python -m json.tool
```

必须确认响应进入结构化 `tool_calls`，而不是只在普通文本里打印伪 JSON。

## 8. 精度、长上下文和请求隔离验收

### 8.1 最小验收集

A8 和 A16 分别执行：

1. `只回复：OK`，验证基本 prefill/decode；
2. `17*3` greedy，期望 `51`；
3. 普通知识问答，检查最终答案；
4. 同一 greedy 请求重复多次，token 必须确定；
5. arithmetic → 长文检索 → arithmetic，检查请求隔离；
6. 带 tools 的 attachment/read_file 请求；
7. 32+ token 输出，确保 decode CUDA Graph 已捕获；
8. 接近 262144 的长 prompt，检查 chunk prefill 和边界错误。

2026-08-20 的历史 bad-case 回归结果为 A8 5/5、A16 5/5。两档均命中
`read_file` 工具调用，郭淮长文检索没有出现历史上的无关朝代内容污染，greedy
算术和诸葛亮 sanity case 也通过。

### 8.2 长上下文规则

请求必须满足：

```text
prompt_tokens + max_tokens <= 262144
```

不要依赖 tokenizer 的静默截断。曾用于回归的完整《三国演义》文本约
562626 tokens，超过 256K；正确做法是提取相关段落、分块检索或显式设计
滑窗，而不是只截书的前 256K 后声称完成全文检索。

### 8.3 稳定性能记录

记录以下阶段的独立时间：

- 进程启动到 `/health`；
- HMONNX loader 和 W4 pack；
- 首次 prefill CUDA Graph 捕获；
- 首次 decode CUDA Graph 捕获；
- 完成预热后的 TTFT、decode TPS 和端到端时延。

已验证环境完成预热后，`OK` 两 token 请求约为 A8 3.77 秒、A16 5.25 秒；
A8 的 32-token 飞机大战请求约为 12.3 秒。该数据只用于确认量级，不替代独占
GPU、固定输入输出长度的正式性能测试。

## 9. 常见故障

### 9.1 `flashinfer` 版本冲突

若 `flashinfer-python` 已升级到 0.6.16，但环境残留
`flashinfer-cubin==0.6.12`，import 会因版本不一致失败。先检查：

```bash
python -m pip show flashinfer-python flashinfer-cubin
```

只在确认是旧残留且当前 wheel 不需要它时移除不匹配的 cubin 包：

```bash
python -m pip uninstall flashinfer-cubin
```

更一般的处理是安装与 `flashinfer-python` 完全匹配的 cubin。不要把关闭版本检查
作为正式修复。

### 9.2 `hsum_gpu.so: undefined symbol`

这是 Torch ABI 不匹配。使用服务环境专属的 `xhquanttool` worktree 重新安装或
触发 JIT 构建。首次 JIT 可能需要约 2 分钟；这段时间不要认为进程卡死。

如果 cache lock 长时间不动：

1. 先确认没有编译器、Python 或另一服务正在拥有该 lock；
2. 只删除已经确认 stale 的那个具体 lock；
3. 不要清空整个共享扩展缓存。

### 9.3 `/health` 已好，但第一个请求超时

这是 HMONNX CUDA Graph 懒捕获。将第一次请求超时设为 10-15 分钟，观察日志中
每个 stage 的 `capture starting/completed`。捕获完成后再测稳定性能。

### 9.4 `content` 为空

先检查 `reasoning_content`。`high/max` 下输出预算可能在最终答案前耗尽。提高
`max_tokens`，或将简单任务改成 `reasoning_effort=none`。

### 9.5 输出持续重复内部计划

先禁用 thinking、关闭客户端自动重试、检查多轮消息是否回灌 reasoning，并用
greedy 请求复现。单纯升级 vLLM 或增大 context 不能保证解决该问题。

### 9.6 产物叫 `w8a8`，为什么日志说 W4

`w8a8` 是图的基础量化档；routed experts 来自 E4G64 checkpoint，loader 会在
服务加载时将 43 个 MoeBlock 的 129 个 initializer 打包为 W4 runtime layout。
以 `W4 runtime contract audit` 的 node/count/bytes 为最终依据。

### 9.7 导出日志出现 `Failed to set buffer path meta info`

2026-08-20 的已验收导出日志中出现过这条中间转换告警，但导出最终打印
`DeepSeek-V4 export completed`，metadata、图 MD5、loader 审计和推理均通过。
不能只按单个 `ERROR` 字样判断失败；也不能直接忽略。必须继续核对最终完成标记、
产物完整性、MD5、load-only 和 Golden。

### 9.8 升级后 DeepSeek 正常，Qwen/Gemma 异常

这说明升级越过了插件的兼容边界。不要在原环境回滚覆盖；切回旧环境保持生产，
在 clone 环境中按以下顺序定位：

1. `pip check` 和 Torch/CUDA/FlashInfer 版本；
2. vLLM SchedulerOutput、loader、tokenizer/parser 私有 API 变化；
3. DeepSeek/Qwen/Gemma 的最小服务启动；
4. cache、连续请求、MTP/DFlash、视觉和工具调用专项回归。

## 10. 发布检查单

- [ ] 原始模型、BF16 staging、W8E4G64、A8、A16 是独立目录；
- [ ] 四个仓库 revision 和两个 Conda 环境版本已记录；
- [ ] 正式量化不是 `2/64/2` smoke 配置；
- [ ] quant config 为 base W8、routed experts W4、G64、symmetric；
- [ ] A8/A16 复用同一个验收通过的 W8E4G64 checkpoint；
- [ ] 正式导出没有传 `--max-layers`，metadata 显示 43 层；
- [ ] context/max PE 为 262144，prefill chunk 为 256；
- [ ] `MODEL_DIR` 指向含 `merak_config.json` 的实际 `hmquant_*` 层；
- [ ] prefill/decode 图、external data、embedding、hf_config 均存在；
- [ ] W4 audit 为 43 nodes、129 initializers，packed bytes 约为 source 一半；
- [ ] A8/A16 分别在预期的四张卡和端口启动；
- [ ] `/health`、`/version`、`/v1/models` 通过；
- [ ] prefill 和 decode CUDA Graph 都已预热；
- [ ] `none/high/max` 的 API 行为和输出预算已验证；
- [ ] 工具调用、greedy 确定性、请求隔离、历史 bad case 和长上下文通过；
- [ ] vLLM 升级 clone 已完成 DeepSeek、Qwen 和 Gemma 回归后再替换生产。

相关文档：

- [Qwen3.5/Qwen3.6/Qwen3-Next W4 导出部署](qwen3_5_series_w4_export_vllm_merak_deploy.md)
- [Gemma4 全系列 W4 导出部署](gemma4_series_w4_export_vllm_merak_deploy.md)
