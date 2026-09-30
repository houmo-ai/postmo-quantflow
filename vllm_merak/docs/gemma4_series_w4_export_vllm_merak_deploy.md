# Gemma4 全系列 W4 量化、HMONNX 导出与 vLLM Merak 部署

本文给出 Gemma4 系列从源码环境、W4G64 量化、HMONNX 导出，到
vLLM Merak Base/MTP 服务部署和验收的完整流程。文中的正式产物统一按
`context_max_length=262144` 导出，并在加载时启用在线 Pack W4。

适用模型：

- Gemma4 E2B；
- Gemma4 E4B；
- Gemma4 12B Unified；
- Gemma4 26B-A4B；
- Gemma4 31B。

适用模式：

- Base，旧版 Mask + Softmax Attention；
- Base，FlashAttention/运行时 PageAttention；
- MTP，旧版 Mask + Softmax Attention；
- MTP，FlashAttention/只读 PageAttention，推荐用于正式部署。

> Gemma4 当前没有 DFlash 模式。本文所说的“所有模式”指 Base 和 MTP
> 各自的非 FlashAttention、FlashAttention/PageAttention 版本。

## 1. 三个仓库及职责

假设三个主仓库位于：

```bash
export WORK_ROOT=/data01/home/yujy/work
export XHQUANT_ROOT=${WORK_ROOT}/xhquanttool
export MODELZOO_ROOT=${WORK_ROOT}/xh2modelzoo
export VLLM_MERAK_ROOT=${MODELZOO_ROOT}/vllm_merak
```

`xhquanttool` 的 `hsum_gpu.so` 会生成在源码树内。导出环境和服务环境的
Torch ABI 不同，正式性能环境推荐为同一 revision 建一个独立 runtime
worktree：

```bash
export XHQUANT_RUNTIME_ROOT=${WORK_ROOT}/xhquanttool-vllm
git -C "${XHQUANT_ROOT}" worktree add --detach \
  "${XHQUANT_RUNTIME_ROOT}" \
  "$(git -C "${XHQUANT_ROOT}" rev-parse HEAD)"
```

它仍然是同一个 `xhquanttool` 仓库的工作树，不是第四个集成仓库。若目录已
存在，先验证两边 `git rev-parse HEAD` 相同，不要重复执行 `worktree add`。

| 仓库 | 作用 |
| --- | --- |
| `xhquanttool` | HMONNX 加载、在线 Pack W4、W4A8/W4A16 CUDA/Triton kernel、HMONNX V2 runtime |
| `xh2modelzoo` | Transformers 模型适配、量化配置、Golden 生成、HMONNX 导出 |
| `xh2modelzoo/vllm_merak` | vLLM 插件、Paged KV cache、Base/MTP 调度、接受/拒绝与 cache 回退 |

GPTQModel 是量化和量化 checkpoint 加载依赖，不算上述三个集成仓库，但导出
环境需要安装当前项目验证过的魔改版本：

```bash
export GPTQMODEL_ROOT=${WORK_ROOT}/gptqmodel
```

每次生成正式产物前记录四个仓库的准确 revision，不要只记录分支名：

```bash
for repo in \
  "${XHQUANT_ROOT}" \
  "${MODELZOO_ROOT}" \
  "${GPTQMODEL_ROOT}"; do
  printf '%-48s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
done
```

## 2. 环境准备与安装

量化/导出和在线服务使用两个 Conda 环境。不要在一个环境中反复切换 Torch，
也不要把 Torch 2.8 编译出的 `.so` 直接拿到 Torch 2.11 环境使用。

### 2.1 已验证版本

| 环境 | 用途 | Python | Torch | Transformers |
| --- | --- | --- | --- | --- |
| `xhquant_55` | 量化、Golden、HMONNX 导出 | 3.12 | 2.8.0+cu128 | 5.13.0 |
| `vllm` | vLLM Merak 在线服务 | 3.12 | 2.11.0+cu129 | 5.13.0 |

Gemma4 12B Unified 必须使用 Transformers 5.13 的模型结构和 conversion
体系。GPTQModel 也必须包含与其匹配的 key conversion；不能用
`xh2modelzoo` 中绕开通用加载流程、重写整段加载逻辑的临时补丁代替。

### 2.2 安装量化/导出环境

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

验证导出环境没有被依赖解析意外降级：

```bash
python - <<'PY'
import torch
import transformers
import xhquant
import xhmodel_merak

print("torch       =", torch.__version__)
print("transformers=", transformers.__version__)
print("xhquant     =", xhquant.__file__)
print("modelzoo    =", xhmodel_merak.__file__)
PY
```

### 2.3 安装 vLLM Merak 环境

HMONNX 执行 CUDA kernel，vLLM 本体采用当前项目验证的 CPU target 构建：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

# VLLM_SOURCE 指向与本项目版本匹配的 vLLM 源码。
export VLLM_SOURCE=${WORK_ROOT}/vllm
CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
VLLM_TARGET_DEVICE=cpu \
VLLM_USE_PRECOMPILED=0 \
python -m pip install -v -e "${VLLM_SOURCE}" \
  --no-build-isolation --no-deps

# --no-deps 很重要：不要让 editable install 改写运行环境的 Torch。
# 独立 worktree 会为 Torch 2.11 编译自己的 hsum_gpu.so。
python -m pip install -v -e "${XHQUANT_RUNTIME_ROOT}" \
  --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
python -m pip install -v -e "${MODELZOO_ROOT}" \
  --no-build-isolation --no-deps
PYTHON_BIN=python "${VLLM_MERAK_ROOT}/tools/install_from_modelzoo.sh"
```

验证：

```bash
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

如果日志出现 `xhquant/lib/hsum_gpu*.so: undefined symbol`，说明扩展是由另一个
Torch ABI 编译的。Python fallback 可以用于功能排查，但正式性能测试必须在
`vllm` 环境重新编译对应扩展并消除该告警。

## 3. 模型、量化方法和能力矩阵

### 3.1 模型来源

默认目录命名如下；实际目录可以通过命令行传入，不要把机器绝对路径写回 YAML：

| 模型 | 主模型示例 | MTP assistant 示例 | 多模态 |
| --- | --- | --- | --- |
| E2B | `weights/gemma-4-E2B-it` | `weights/gemma-4-E2B-it-assistant` | 文本/图像/视频；音频以 checkpoint 和导出产物为准 |
| E4B | `weights/gemma-4-E4B-it` | `weights/gemma-4-E4B-it-assistant` | 文本/图像/视频；音频以 checkpoint 和导出产物为准 |
| 12B Unified | `weights/gemma-4-12B-it` | `weights/gemma-4-12B-it-assistant` | 文本/图像/视频/音频 |
| 26B-A4B | `weights/gemma-4-26B-A4B-it` | `weights/gemma-4-26B-A4B-it-assistant` | 文本/图像/视频 |
| 31B | `weights/gemma-4-31B-it` | `weights/gemma-4-31B-it-assistant` | 文本/图像/视频 |

量化均为 W4、group size 64。当前推荐：

- E2B：AutoRound；
- E4B、12B Unified、26B-A4B、31B：GPTQ；
- 已有合法 W4G64 checkpoint 时，HMONNX 导出不重新量化。

### 3.2 配置矩阵

配置根目录：

```text
configs_merak/workflows/xh2a/llm_models/gemma4_series
```

下表给出每个模型的 GPTQ 导出配置；这些配置完整覆盖正式 Base/MTP
Attention 模式。

| 模型 | Base 非 FA | Base FA | MTP 非 FA | MTP PageAttention |
| --- | --- | --- | --- | --- |
| 12B Unified | `12b_unified/gemma4_12b_unified_full.yaml` | `12b_unified/gemma4_12b_unified_full_flash_attention.yaml` | `12b_unified/gemma4_12b_unified_full_mtp.yaml` | `12b_unified/gemma4_12b_unified_full_mtp_page_attention.yaml` |
| E2B | `e2b/gemma4_e2b_full.yaml` | `e2b/gemma4_e2b_full_flash_attention.yaml` | `e2b/gemma4_e2b_full_mtp.yaml` | `e2b/gemma4_e2b_full_mtp_page_attention.yaml` |
| E4B | `e4b/gemma4_e4b_full.yaml` | `e4b/gemma4_e4b_full_flash_attention.yaml` | `e4b/gemma4_e4b_full_mtp.yaml` | `e4b/gemma4_e4b_full_mtp_page_attention.yaml` |
| 26B-A4B | `26b_a4b/gemma4_26b_a4b_full.yaml` | `26b_a4b/gemma4_26b_a4b_full_flash_attention.yaml` | `26b_a4b/gemma4_26b_a4b_full_mtp.yaml` | `26b_a4b/gemma4_26b_a4b_full_mtp_page_attention.yaml` |
| 31B | `31b/gemma4_31b_full.yaml` | `31b/gemma4_31b_full_flash_attention.yaml` | `31b/gemma4_31b_full_mtp.yaml` | `31b/gemma4_31b_full_mtp_page_attention.yaml` |

E2B/E4B/26B-A4B/31B 还提供：

```text
*_autoround.yaml
*_autoround_flash_attention.yaml
*_autoround_mtp.yaml
```

它们用于 AutoRound 量化或旧版导出。若要把已有 AutoRound W4 checkpoint
导出为推荐的 MTP PageAttention 产物，仍使用上表
`*_full_mtp_page_attention.yaml`，并通过 `--existing-hf-model-dir` 跳过
配置中的 GPTQ 量化阶段。

### 3.3 Attention 配置规则

1. `*_full.yaml`/`*_full_mtp.yaml` 是非 FlashAttention ABI，保留
   Mask + Softmax 图。
2. `*_full_flash_attention.yaml` 和
   `*_full_mtp_page_attention.yaml` 是 contract-v2 图，运行时降低为
   PageAttention。
3. YAML 中出现 `flash_attention` 块表示启用；新配置可显式写
   `enable: true`。`enable: false` 才表示禁用。
4. `attention_contract_version` 由导出模式推导：FA 为 2，非 FA 为 1。
   不应为了 256K 导出在所有 YAML 中手工增加或批量修改该字段。
5. 正式上下文长度由导出命令的单个 override 控制。MTP draft 的 cache
   容量跟随主模型，不再单独维护第二个 context 参数。

## 4. W4 量化

### 4.1 从浮点模型量化并导出

下面以 12B Unified Base FlashAttention 为例：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

export FLOAT_MODEL=weights/gemma-4-12B-it
export CONFIG=configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/gemma4_12b_unified_full_flash_attention.yaml
export QUANT_OUT=work_dirs/gemma4_12b_w4g64_gptq/quant
export EXPORT_OUT=work_dirs/gemma4_12b_w4g64_gptq/base_fa

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir "${FLOAT_MODEL}" \
  --config "${CONFIG}" \
  --quant-output-dir "${QUANT_OUT}" \
  --artifact-format gptqmodel_hf \
  --export-output-dir "${EXPORT_OUT}" \
  --quant-device cuda:0 \
  --export-device cuda:0 \
  --override export.model.context_max_length=262144 \
  --dump-golden \
  --force
```

量化很耗时。生产流程建议先独立保留 `QUANT_OUT`，后续 Base/MTP/Attention
组合全部复用该 checkpoint；不要为每种 HMONNX 模式重复量化。

### 4.2 复用已有 W4 checkpoint

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

export FLOAT_MODEL=weights/gemma-4-26B-A4B-it
export QUANT_MODEL=/path/to/gemma4_26b_a4b_w4g64_quant
export CONFIG=configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full_flash_attention.yaml
export EXPORT_OUT=work_dirs/gemma4_26b_a4b_w4g64/base_fa

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir "${FLOAT_MODEL}" \
  --config "${CONFIG}" \
  --existing-hf-model-dir "${QUANT_MODEL}" \
  --artifact-format gptqmodel_hf \
  --export-output-dir "${EXPORT_OUT}" \
  --export-device cuda:0 \
  --override export.model.context_max_length=262144 \
  --dump-golden \
  --force
```

已验证过的历史量化目录示例：

```text
work_dirs/qtl384_gemma4_requant_export_demo_20260621_111057/
  26b_a4b_gptq/gemma4_26b_a4b_unified/quant-export/quant
  31b_gptq/gemma4_31b_unified/quant-export/quant
```

不要清理这些 GPTQModel/AutoRound 量化权重；只清理临时 HMONNX 测试产物。

### 4.3 Gemma4 12B 量化 checkpoint 验收

12B Unified 是 Transformers 5.13 新支持结构。量化后至少检查：

- language layers 中存在 W4 packed/quantized 权重；
- `pos_embedding` 不是全零；
- vision projection 不是全零；
- LayerNorm 参数全部有限；
- processor/config 能由 Transformers 5.13 正常加载。

若 frontend 被清零或出现非有限值，应修正 GPTQModel 的
`HF_CONVERSION_MAP_REVERSED + Transformers mapping + manual override`
加载体系后重新量化，而不是在 `xh2modelzoo` 中复制一条私有加载路径。

## 5. 导出所有模式

### 5.1 Base 非 FlashAttention

使用配置矩阵中的 `*_full.yaml`，按 4.2 节命令导出。该产物主要用于精度
对照和兼容旧 runtime。

### 5.2 Base FlashAttention

使用 `*_full_flash_attention.yaml`。这是 Base 正式部署推荐产物：

```bash
export CONFIG=.../gemma4_31b_full_flash_attention.yaml
export EXPORT_OUT=work_dirs/gemma4_31b_w4g64/base_fa_256k
# 其余命令同 4.2，必须保留：
# --override export.model.context_max_length=262144
```

### 5.3 MTP 非 FlashAttention

使用 `*_full_mtp.yaml`，并增加 assistant：

```bash
export CONFIG=.../gemma4_e4b_full_mtp.yaml
export ASSISTANT_MODEL=weights/gemma-4-E4B-it-assistant
export EXPORT_OUT=work_dirs/gemma4_e4b_w4g64/mtp_legacy_256k

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir "${FLOAT_MODEL}" \
  --config "${CONFIG}" \
  --existing-hf-model-dir "${QUANT_MODEL}" \
  --artifact-format gptqmodel_hf \
  --mtp-assistant-model-dir "${ASSISTANT_MODEL}" \
  --export-output-dir "${EXPORT_OUT}" \
  --export-device cuda:0 \
  --override export.model.context_max_length=262144 \
  --dump-golden \
  --force
```

### 5.4 MTP FlashAttention/PageAttention

将上面的配置替换为 `*_full_mtp_page_attention.yaml`。这是 MTP 正式部署
推荐产物；MTP draft token 数为 4。

Gemma4 MTP 有以下特殊契约：

- draft 读取主模型最后一层 KV cache，不写 draft 自己的 KV cache；
- full attention cache 和 sliding-window hybrid cache 均由主模型拥有；
- draft RoPE position 使用主模型已提交长度；
- PageAttention 以 `[kv_start, kv_end)` 表达当前有效区域，不为每轮 draft
  移动整段 sliding cache；
- verify 拒绝时只回退逻辑有效长度，未接受尾部不可见；跨 sliding window
  边界时由 hybrid cache 参数恢复正确窗口。

非 FlashAttention 版本仍使用 sliding attention mask；不要把游离的
`sliding_attention_mask` 删除或误改为 full attention mask。

### 5.5 产物目录识别

`--export-output-dir` 下通常还会生成一层带日期和量化信息的目录。部署时
`MODEL_DIR` 必须指向**直接包含**以下文件/目录的那一层：

```text
golden_meta_info.json
merak_config.json
hf_config/
prefill/
decode/
visual/                 # 多模态产物
mtp_*                   # MTP 产物才有
```

查找：

```bash
find "${EXPORT_OUT}" -name golden_meta_info.json -printf '%h\n'
```

## 6. Pack W4：导出格式不变，加载时在线打包

Pack W4 不改变 HMONNX 存储格式，也不要求重新导出模型。加载器发现 W4
initializer 后在 CPU 上流式打包一次，再把 packed tensor 送入 CUDA：

1. 整图审计所有 W4 initializer；
2. 对 Base/MTP 多图共享的 initializer 去重，只打包一次；
3. 释放 external-data mmap/源 tensor 后再计入更小的 packed tensor；
4. 禁止先把展开的 W4 权重搬上 GPU，因此 pack 阶段 GPU 显存不增长；
5. 请求执行期直接调用 W4A8/W4A16 kernel，不再逐请求 unpack。

packed 权重约为旧 int8 展开存储的一半。当前 kernel 路径：

| 场景 | 典型 M | 默认实现 |
| --- | ---: | --- |
| 普通 decode | 1 | native CUDA split W4 kernel |
| MTP verify | 5 | native CUDA small-M W4 kernel |
| Gemma prefill | 320 | native CUDA W4 MMA；Triton 为 fallback |

生产环境固定：

```bash
export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1
export USE_TRITON_MATMUL=1
export USE_FAST_TRITON_MATMUL=0

# 默认已经是 cuda；只有做受控 A/B 时才改成 triton。
export XHQUANT_W4_SMALL_M_BACKEND=cuda
export XHQUANT_W4_PREFILL_PACK_BACKEND=cuda
```

`XHQUANT_PACKED_W4` 当前默认开启，但部署脚本仍应显式设置。严格开关
`XHQUANT_FORBID_CUDA_W4_UNPACK=1` 可在任何算子试图把 W4 CUDA 权重展开时
立即失败，避免服务悄悄退回慢路径。

## 7. vLLM Merak 部署

### 7.1 通用环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
cd "${VLLM_MERAK_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${VLLM_MERAK_ROOT}:${XHQUANT_RUNTIME_ROOT}:${MODELZOO_ROOT}:${PYTHONPATH:-}"
export VLLM_PLUGINS=merak
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ENABLE_HMINFERENCE_V2=1

export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1
export USE_TRITON_MATMUL=1
export USE_FAST_TRITON_MATMUL=0
export XHQUANT_W4_SMALL_M_BACKEND=cuda
export XHQUANT_W4_PREFILL_PACK_BACKEND=cuda
```

当前 Gemma4 sliding cache 只支持单请求 cache owner，因此
`--max-num-seqs` 必须为 1，prefix cache 必须关闭。HTTP 请求可以排队，
但不能把该限制理解为已支持多请求并行 forward。

### 7.2 Base 服务

```bash
export MODEL_DIR=/path/to/artifact-containing-golden-meta
export SERVED_NAME=gemma4_12b_w4_base
export PORT=18500

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
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1,"video":1,"audio":1}'
```

26B-A4B/31B 不支持音频时改成：

```bash
--limit-mm-per-prompt '{"image":1,"video":1}'
```

E2B/E4B 必须以该 artifact 的 `golden_meta_info.json` 和 `hf_config` 实际
导出的模态为准；不要仅凭模型名称开放未导出的 modality。

### 7.3 MTP 服务

MTP 在 Base 命令上增加：

```bash
export VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=0
export VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1

--speculative-config \
'{"method":"custom_class","model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer","num_speculative_tokens":4}'
```

完整示例：

```bash
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
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1,"video":1,"audio":1}' \
  --speculative-config \
  '{"method":"custom_class","model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer","num_speculative_tokens":4}'
```

服务端 `--max-model-len` 必须和导出长度一致。这里按导出 HMONNX 的
262144 分配 cache，不再引入一套 `max_model_len + lookahead` 的私有容量
策略。请求的 prompt token、视觉/音频 token 和输出 token 总和不得超过
262144。

## 8. 验收

### 8.1 静态产物

```bash
test -f "${MODEL_DIR}/golden_meta_info.json"
test -f "${MODEL_DIR}/merak_config.json"
test -d "${MODEL_DIR}/hf_config"
python -m json.tool "${MODEL_DIR}/golden_meta_info.json" >/dev/null
python -m json.tool "${MODEL_DIR}/merak_config.json" >/dev/null

cd "${VLLM_MERAK_ROOT}"
python tools/check_hmonnx_attention_abi.py "${MODEL_DIR}"
```

检查内容：

- context/max sequence length 均为 262144；
- Base 产物没有 speculative graph；
- MTP K=4，且 draft context length 跟随 target；
- FA 产物是 attention contract v2，非 FA 产物是 legacy contract；
- full/sliding attention 层顺序与 HF config 一致；
- `visual`/`audio` 文件和 metadata 引用一致。

### 8.2 启动日志

必须看到：

- Gemma4 对应 Merak architecture 被正确解析；
- `GPU KV cache size` 至少覆盖 262144；
- HMONNX V2、auto-offload、CUDA Graph 正常初始化；
- `W4 runtime contract audit`；
- `packed_nbytes` 明显小于 `source_nbytes`，W4 主权重通常约减半；
- 没有 CUDA W4 unpack、ABI mismatch、NaN 或 cache contract 告警。

### 8.3 OpenAI API smoke

```bash
curl -s "http://127.0.0.1:${PORT}/v1/models" | python -m json.tool

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"${SERVED_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"请用三句话介绍南京。\"}],
    \"temperature\":0,
    \"max_tokens\":128
  }" | python -m json.tool
```

多模态 smoke 必须分别覆盖该模型实际支持的 image/video/audio，并确认 MTP
draft prefill 复用目标模型产生的多模态特征，而不是对视觉/音频占位 token
重复做 token embedding。

### 8.4 精度、拒绝回退和长上下文

正式验收至少包括：

1. 固定 seed、greedy decoding，对比 HF/Golden、Base legacy、Base FA；
2. 对比 Base FA 和 MTP PageAttention 的最终 token；
3. 构造全接受、部分接受、全拒绝三类 MTP 请求；
4. 跨 sliding-window 边界连续 decode，验证拒绝后窗口和 RoPE position；
5. 验证 full-attention KV 仅通过 valid length 隐藏未接受尾部；
6. 验证 hybrid/sliding cache 在回退后有效区域正确；
7. 验证接近 262144 上下文以及最后一个可分配 block 的边界；
8. 串行压测多轮请求，确认前一个请求的 cache 不污染下一个请求。

MTP 接受率异常低不是“正常波动”。优先检查 assistant/target tokenizer 和
权重是否匹配、RoPE position、只读 KV 的 `[start,end)`、sliding mask、
verify 后已提交长度，以及部分接受时 HMFP shared-exponent V page 的恢复。

## 9. 常见问题

### 9.1 为什么不直接修改所有 YAML 为 256K？

YAML 默认值用于资源较小的本地调试，不应为一次部署批量改写。正式导出统一
使用：

```bash
--override export.model.context_max_length=262144
```

这一个值同时控制 target 和 MTP draft cache。

### 9.2 为什么导出后还要 Pack W4？

HMONNX 保持已有存储 ABI，便于兼容和复用。Pack W4 是 runtime
layout 优化：加载时只执行一次，并让 prefill、decode、MTP verify 共享同一份
packed 权重；不是每次请求重新 pack。

### 9.3 为什么 MTP cache 没有自己的更新输出？

这是 Gemma4 的模型定义，不是导出丢失。MTP 读取 target 最后一层 KV，
draft 本身不更新 KV；接受/拒绝由 target cache 的已提交长度和
sliding-window 有效区间管理。

### 9.4 FlashAttention 图为什么还出现 sliding mask 输入？

非 FA 版本需要显式 `sliding_attention_mask`。FA/PageAttention 版本应通过
`kv_start`/`kv_valid_length` 表达有效区域。两种 ABI 必须区分，不能为了
统一输入而混用。

### 9.5 同卡可以启动多个服务吗？

只要加载后的 packed 权重、262144 cache 和 CUDA Graph 总显存可容纳即可。
多个服务不会跨进程共享 packed 权重，性能测试也会互相抢占 GPU；功能联调
可同卡，正式吞吐/时延数据必须逐服务独占 GPU 测量。

## 10. 发布检查单

- [ ] 两个 Conda 环境版本和四个源码 revision 已记录；
- [ ] 12B GPTQModel conversion 和未量化 frontend 已验收；
- [ ] W4G64 量化权重独立保存，未被清理；
- [ ] 所有正式 HMONNX 均通过命令行 override 导出为 262144；
- [ ] Base legacy/Base FA/MTP legacy/MTP PageAttention 模式齐全；
- [ ] MTP assistant 与 target 型号严格匹配，K=4；
- [ ] `MODEL_DIR` 指向包含 `golden_meta_info.json` 的实际产物层；
- [ ] Pack W4 和禁止 CUDA unpack 的严格开关已开启；
- [ ] Pack 阶段 GPU 显存无增长，日志中 packed bytes 约为源 W4 展开存储一半；
- [ ] Attention ABI、Golden、Base/MTP、拒绝回退和 256K 边界均通过；
- [ ] 服务监听 `0.0.0.0`，`/v1/models` 和文本/多模态请求通过。

进一步的模型细节可参阅：

- [Gemma4 12B Unified 部署说明](gemma4_12b_unified_merak_deploy.md)
- [Gemma4 历史部署说明](gemma4_merak_deploy.md)
- `xh2modelzoo/docs/gemma4_series_merak_workflow_guide_20260621.md`
- `xh2modelzoo/docs/merak_transformers_5_13_compatibility_20260727.md`
