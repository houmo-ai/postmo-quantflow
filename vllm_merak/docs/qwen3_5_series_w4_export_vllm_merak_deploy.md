# Qwen3.5/Qwen3.6/Qwen3-Next W4 量化、HMONNX 导出与 vLLM Merak 部署

本文覆盖 Qwen3.5、Qwen3.6 和 Qwen3-Next 当前接入型号，从三个仓库的
环境准备、W4G64 量化、256K HMONNX 导出，到 vLLM Merak 的 Base、MTP、
DFlash 和视觉服务部署。HMONNX 存储格式保持不变，服务加载阶段统一在线
Pack W4。

覆盖型号：

- Qwen3.5 0.8B、2B、4B、9B；
- Qwen3.6 27B、35B-A3B；
- Qwen3.5 122B-A10B；
- Qwen3-Next 80B-A3B。

覆盖模式：

- Base；
- MTP（模型有对应配置时）；
- DFlash（模型有对应 assistant 时）；
- full visual 和独立 visual-only；
- 非 FlashAttention 兼容导出和 FlashAttention/PageAttention 正式导出。

## 1. 三个仓库及职责

```bash
export WORK_ROOT=/data01/home/yujy/work
export XHQUANT_ROOT=${WORK_ROOT}/xhquanttool
export MODELZOO_ROOT=${WORK_ROOT}/xh2modelzoo
export VLLM_MERAK_ROOT=${MODELZOO_ROOT}/vllm_merak
export GPTQMODEL_ROOT=${WORK_ROOT}/gptqmodel
```

`xhquanttool` 的 `hsum_gpu.so` 生成在源码树内，而两个 Conda 环境使用不同
Torch ABI。正式性能环境推荐为同一 revision 创建独立 runtime worktree：

```bash
export XHQUANT_RUNTIME_ROOT=${WORK_ROOT}/xhquanttool-vllm
git -C "${XHQUANT_ROOT}" worktree add --detach \
  "${XHQUANT_RUNTIME_ROOT}" \
  "$(git -C "${XHQUANT_ROOT}" rev-parse HEAD)"
```

若该目录已经存在，只需确认它和 `XHQUANT_ROOT` 的 commit 相同。它仍是
`xhquanttool` 的第二个工作树，不是额外的集成仓库。

| 仓库 | 作用 |
| --- | --- |
| `xhquanttool` | HMONNX loader/runtime、在线 Pack W4、W4A8/W4A16 CUDA/Triton kernel |
| `xh2modelzoo` | Transformers 模型、量化配置、Base/MTP/DFlash 图导出、Golden |
| `xh2modelzoo/vllm_merak` | vLLM 插件、Paged KV/linear state 管理、投机 draft/verify/接受/拒绝 |

GPTQModel 是量化依赖。正式产物应记录所有代码 revision：

```bash
for repo in \
  "${XHQUANT_ROOT}" \
  "${MODELZOO_ROOT}" \
  "${GPTQMODEL_ROOT}"; do
  printf '%-48s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
done
```

## 2. 环境准备与安装

### 2.1 已验证环境

| Conda 环境 | 用途 | Python | Torch | Transformers |
| --- | --- | --- | --- | --- |
| `xhquant_55` | 量化、Golden、HMONNX 导出 | 3.12 | 2.8.0+cu128 | 5.13.0 |
| `vllm` | vLLM Merak 服务 | 3.12 | 2.11.0+cu129 | 5.13.0 |

不要在导出环境运行 vLLM 性能测试，也不要在服务环境重新导出模型。
Torch 编译扩展具有 ABI 绑定，两个环境需要各自构建。

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

验证：

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

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

export VLLM_SOURCE=${WORK_ROOT}/vllm
CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
VLLM_TARGET_DEVICE=cpu \
VLLM_USE_PRECOMPILED=0 \
python -m pip install -v -e "${VLLM_SOURCE}" \
  --no-build-isolation --no-deps

python -m pip install -v -e "${XHQUANT_RUNTIME_ROOT}" \
  --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
python -m pip install -v -e "${MODELZOO_ROOT}" \
  --no-build-isolation --no-deps
PYTHON_BIN=python "${VLLM_MERAK_ROOT}/tools/install_from_modelzoo.sh"
```

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

如果启动时 `xhquant/lib/hsum_gpu*.so` 报 `undefined symbol`，说明它由
Torch 2.8 而不是当前 Torch 2.11 编译。Python fallback 只能做功能排查，
最终性能环境必须重新编译并消除 ABI 告警。

## 3. 模型和模式矩阵

### 3.1 能力矩阵

| 模型 | Base | MTP | DFlash | 视觉 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 是 | 否 | 否 | 是 | checked-in quant 默认 W8，做 Pack W4 必须显式量化为 W4 |
| Qwen3.5-2B | 是 | 否 | 否 | 是 | checked-in quant 默认 W8，做 Pack W4 必须显式量化为 W4 |
| Qwen3.5-4B | 是 | 否 | 否 | 是 | 当前提供 AutoRound W4 |
| Qwen3.5-9B | 是 | K=4 | K=9 | 是 | Base/MTP/DFlash 完整支持 |
| Qwen3.6-27B | 是 | K=4 | K=9 | 是 | Base/MTP/DFlash 完整支持 |
| Qwen3.6-35B-A3B | 是 | K=4 | K=9 | 是 | MoE |
| Qwen3.5-122B-A10B | 是 | K=4 | K=9 | 是 | MoE；超大模型分块导出 |
| Qwen3-Next-80B-A3B | 是 | K=4 | 否 | 否 | 纯文本；旋转权重只可跑 Base |

这里 K 是 `num_speculative_tokens`：

- 普通 decode 的 target GEMM 行数 `M=1`；
- MTP K=4，target verify 行数 `M=5`；
- DFlash 默认 K=9，target verify 行数 `M=10`；
- runtime/kernel 必须守护普通 decode 和所有 verify 宽度 `M=1..16`；
- Qwen prefill chunk 的典型 `M=256`。

不要为了接收率临时把所有 DFlash YAML 改成 K=15。默认保持 K=9；只有受控
实验才用 `--num-draft-tokens 15`，且服务端必须使用相同 K。

### 3.2 Base 配置

配置根目录：

```text
configs_merak/workflows/xh2a/llm_models
```

| 模型 | AutoRound | GPTQ |
| --- | --- | --- |
| 0.8B | `qwen3_5/0.8b/qwen3_5_0_8b_full.yaml` | `qwen3_5/0.8b/qwen3_5_0_8b_full_gptq.yaml` |
| 2B | `qwen3_5/2b/qwen3_5_2b_full.yaml` | `qwen3_5/2b/qwen3_5_2b_full_gptq.yaml` |
| 4B | `qwen3_5/4b/qwen3_5_4b_full.yaml` | — |
| 9B | `qwen3_5/9b/qwen3_5_9b_full.yaml` | `qwen3_5/9b/qwen3_5_9b_full_gptq.yaml` |
| 27B | `qwen3_5/27b/qwen3_6_27b_full.yaml` | `qwen3_5/27b/qwen3_6_27b_full_gptq.yaml` |
| 35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml` | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_gptq.yaml` |
| 122B-A10B | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml` | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_gptq.yaml` |
| Next 80B-A3B | `qwen3_next/80b_a3b/qwen3_next_80b_a3b_full.yaml` | `qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_gptq.yaml` |

0.8B/2B 两组 YAML 当前 `quant.bits: 8`。完整 W4 流程必须在从浮点权重量化时
显式传 `--bits 4`，或复用已验收的 W4G64 checkpoint；否则 Pack W4 审计
不会命中主权重，不能把 W8 结果写成“Pack W4”。

35B 还存在 `*_w5a8.yaml`、`*_w8a8.yaml`、`*_fa_all16.yaml`、LoRA 和
no-quant 配置；它们是专项实验，不属于本文 W4 正式矩阵。

### 3.3 MTP 配置

| 模型 | AutoRound | GPTQ |
| --- | --- | --- |
| 9B | `qwen3_5/9b/qwen3_5_9b_full_mtp.yaml` | `qwen3_5/9b/qwen3_5_9b_full_mtp_gptq.yaml` |
| 27B | `qwen3_5/27b/qwen3_6_27b_full_mtp.yaml` | `qwen3_5/27b/qwen3_6_27b_full_mtp_gptq.yaml` |
| 35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp.yaml` | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp_gptq.yaml` |
| 122B-A10B | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_mtp.yaml` | — |
| Next 80B-A3B | — | `qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_mtp_gptq.yaml` |

Qwen MTP 使用主模型 checkpoint 中的 MTP config/head，不需要再在 YAML
增加一个独立 `mtp.hf_model`。主模型 context 参数同时控制 target 和 MTP
cache。

### 3.4 DFlash 配置

| 模型 | AutoRound | GPTQ | 默认 assistant |
| --- | --- | --- | --- |
| 9B | `qwen3_5/9b/qwen3_5_9b_full_dflash.yaml` | `qwen3_5/9b/qwen3_5_9b_full_dflash_gptq.yaml` | `weights/Qwen3.5-9B-DFlash` |
| 27B | `qwen3_5/27b/qwen3_6_27b_full_dflash.yaml` | `qwen3_5/27b/qwen3_6_27b_full_dflash_gptq.yaml` | `weights/Qwen3.6-27B-DFlash` |
| 35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml` | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash_gptq.yaml` | `weights/Qwen3.6-35B-A3B-DFlash` |
| 122B-A10B | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_dflash.yaml` | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_dflash_gptq.yaml` | `weights/Qwen3.5-122B-A10B-DFlash` |

DFlash target layer IDs、head 数和架构信息必须来自 assistant checkpoint 的
config。不要在 YAML 复制一份并手工覆盖，否则 checkpoint 与导出图很容易
错位。

### 3.5 Visual-only 配置

独立视觉导出用于复用或调试；full 配置本身已经包含默认 448×448 visual：

| 模型 | 448 | 896 |
| --- | --- | --- |
| 9B | `qwen3_5/9b/qwen3_5_9b_visual_only_448{,_gptq}.yaml` | `qwen3_5/9b/qwen3_5_9b_visual_only_896{,_gptq}.yaml` |
| 27B | `qwen3_5/27b/qwen3_6_27b_visual_only_448{,_gptq}.yaml` | `qwen3_5/27b/qwen3_6_27b_visual_only_896{,_gptq}.yaml` |
| 35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_visual_only_448{,_gptq}.yaml` | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_visual_only_896{,_gptq}.yaml` |

4B 当前有 `qwen3_5/4b/qwen3_5_4b_visual_only_896.yaml`。0.8B/2B/122B
使用 full 配置中的 visual。Qwen3-Next 的 `visual_config: null`，按纯文本部署。

## 4. W4 量化

### 4.1 从浮点模型量化并导出

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

export FLOAT_MODEL=weights/Qwen3.5-9B-mode1-llm-only
export CONFIG=configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml
export QUANT_OUT=work_dirs/qwen3_5_9b_w4g64_gptq/quant
export EXPORT_OUT=work_dirs/qwen3_5_9b_w4g64_gptq/base_fa_256k

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir "${FLOAT_MODEL}" \
  --config-path "${CONFIG}" \
  --quant-output-dir "${QUANT_OUT}" \
  --export-output-dir "${EXPORT_OUT}" \
  --device cuda:0 \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --overwrite
```

0.8B/2B 要在上面额外增加：

```bash
--bits 4
```

量化结果必须独立保留。Base、MTP、DFlash、FA/非 FA 都应复用同一份合法
W4 checkpoint，而不是重复量化。

### 4.2 从已有 W4 checkpoint 导出

这是日常最常用命令：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xhquant_55
cd "${MODELZOO_ROOT}"

export QUANT_MODEL=/path/to/w4g64-gptq-or-autoround-model
export CONFIG=configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml
export EXPORT_OUT=work_dirs/qwen3_5_9b_base_fa_256k

CUDA_VISIBLE_DEVICES=0 \
python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir "${QUANT_MODEL}" \
  --config-path "${CONFIG}" \
  --export-from-quanted-model \
  --export-output-dir "${EXPORT_OUT}" \
  --device cuda:0 \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --overwrite
```

`--export-from-quanted-model` 会跳过 quant 阶段。配置文件仍负责模型拓扑、
模式和 HMONNX export contract。

当前已知模型/量化源示例：

```text
weights/Qwen3.5-9B-mode1-llm-only
weights/Qwen3.6-27B-mode1-llm-only
weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400
work_dirs/qwen3_5_122B_quant
weights/Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse
```

最后一个 Qwen3-Next checkpoint 做过 rotation，只能验证 Base；MTP head
和旋转后的 target 权重不匹配。Next MTP 必须重新使用 `rotation: false` 的
AutoRound/GPTQ W4 checkpoint。

### 4.3 122B 超大模型

122B 导出前显式使用分块/串行模式，先以单 worker 保证峰值内存可控：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
export XH2MODELZOO_EXPORT_WORKERS=1
```

确认内存充足后才逐步增加 worker。不要删除 split-conv-cache、linear
attention mask view、`flat_split_outputs` 等超大模型分块导出接口；它们同时
服务 Dense/MoE 两条路径。

### 4.4 不要默认开启的选项

- 必须开启 `--enable-fuse-gdr-block-recurrent-ops`；
- 不要默认开启 `--enable-fuse-gdr-ops`。后者会改变 prefill recurrent-state
  I/O，只能在对应 ABI 完整验证后使用；
- `LAYER_TAG_ENABLE=1` 仅用于调试，不是服务产物要求，正式导出可保持未设置；
- DFlash 默认 K=9，不传 `--num-draft-tokens` 即可。

## 5. 导出 Base、MTP、DFlash 和 Attention 变体

### 5.1 Base FlashAttention（正式推荐）

使用 3.2 节对应 Base 配置，执行 4.2 节命令，并保留三个关键参数：

```bash
--context-max-length 262144
--enable-flash-attention
--enable-fuse-gdr-block-recurrent-ops
```

FlashAttention 图在 vLLM Merak 运行时转换为 PageAttention。

### 5.2 Base 非 FlashAttention

用于 legacy ABI/精度对照：

```bash
--context-max-length 262144
--disable-flash-attention
--enable-fuse-gdr-block-recurrent-ops
```

非 FA 图保留显式 mask + softmax。不要把这类图当作 PageAttention 图加载。

### 5.3 MTP

把 `CONFIG` 换成 3.3 节对应 MTP YAML，其余命令不变。MTP 的主模型、
visual encoder 和 target cache 与 Base 相同；视觉 MTP draft prefill 直接
使用主模型产生的视觉特征，不对视觉 placeholder 再做一次 token embedding。

Qwen3.5/Qwen3.6 hybrid linear attention 的 verify 必须按接受长度提交状态：

- full-attention KV 通过已提交 valid length 隐藏未接受尾部；
- linear-attention state 选择“最后一个已接受 token 对应的 state”作为下轮
  初始状态；
- 未接受 token 的 recurrent state 必须丢弃；
- conv cache 和 recurrent state 与同一 committed length 原子更新。

### 5.4 DFlash

把 `CONFIG` 换成 3.4 节对应 DFlash YAML，默认不传
`--num-draft-tokens`，即 K=9。DFlash 导出包含三个 draft 阶段：

| 图 | 职责 | cache |
| --- | --- | --- |
| `dflash_draft_context` | prompt 后构造 draft context cache | 更新共享 cache，不执行 cross-attention |
| `dflash_draft_context_decode` | target 接受 token 后增量更新 context | 更新同一 cache，不执行 cross-attention |
| `dflash_draft_decode` | 生成 draft token | 只读共享 context cache，执行 cross-attention |

FA 模式的 `dflash_draft_decode` 必须是 **non-causal Padding Full
Cross-Attention**，输入 current input length 和 KV valid length，并在
vLLM Merak 中降低为只读 PageAttention。它不是下三角 causal attention。
非 FA 模式才使用 mask + softmax。

三个图共享同一组 cache allocation；图 I/O/metadata 可以用 in-place alias
表达，但不能导出成三个互不相关的 cache，更不能把 cache 接口从 context
和 context-decode 图中优化掉。

DFlash context 阶段生成的 KV 直接由后续 draft 读取，不需要 CPU/GPU 搬运。
不要为了 CUDA capture 把含 `.item()` 的 cache update 强行整图 capture；
当前安全配置是 context eager、draft decode CUDA Graph。

### 5.5 Visual-only

使用 3.5 节 visual-only YAML。该流程只生成视觉产物，不需要给视觉图设置
262144 上下文。可通过：

```bash
--max-size-h 896 --max-size-w 896
```

覆盖视觉尺寸。full HMONNX 默认已经包含 448×448 visual，只有需要独立复用
或 896 输入时才额外导出。

### 5.6 产物目录

`--export-output-dir` 往往包含一层自动命名子目录。部署的 `MODEL_DIR`
必须是直接包含以下内容的目录：

```text
golden_meta_info.json
merak_config.json
hf_config/
prefill/
decode/
visual/                         # 视觉模型
mtp_*                           # MTP
dflash_draft_context/           # DFlash
dflash_draft_context_decode/    # DFlash
dflash_draft_decode/            # DFlash
```

查找：

```bash
find "${EXPORT_OUT}" -name golden_meta_info.json -printf '%h\n'
```

## 6. Pack W4

### 6.1 工作方式

现有 HMONNX 导出和 external-data 格式不变。`xhquanttool` loader 在加载时
发现 W4 initializer 后：

1. 在 CPU 上流式 pack；
2. 对 prefill/decode/MTP/DFlash 图的共享 initializer 去重；
3. pack 完一份就释放源 mmap/tensor；
4. 只把 packed tensor 搬入 GPU；
5. 运行时由 W4A8/W4A16 kernel 直接消费 packed 权重。

因此 pack 只发生一次，不增加 GPU 峰值；packed 权重约为旧 int8 展开存储
的一半。若 W4 源权重先出现在 CUDA 上，严格模式会直接报错而不是产生瞬时
双份显存。

### 6.2 kernel 路径

| 场景 | M | 默认路径 |
| --- | ---: | --- |
| 普通 decode | 1 | native CUDA split W4 |
| MTP verify | 5 | native CUDA small-M W4 |
| DFlash verify | 10 | native CUDA small-M W4 |
| 其他 verify | 2～16 | native CUDA；完整宽度守护 |
| prefill | 256 | native CUDA W4 MMA；Triton fallback |

MoE expert/group GEMM 同样使用 packed W4 路径，但总速度还会受路由、
expert dispatch、shared expert、linear attention 和 kernel launch 数影响。
不能仅以“激活参数 A3B 相同”推断 35B-A3B 与 80B-A3B 速度相同。

### 6.3 生产环境变量

```bash
export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1
export USE_TRITON_MATMUL=1
export USE_FAST_TRITON_MATMUL=0
export XHQUANT_W4_SMALL_M_BACKEND=cuda
export XHQUANT_W4_PREFILL_PACK_BACKEND=cuda
```

说明：

- Pack W4 当前默认开启，部署脚本仍显式设置；
- packed W4 CUDA 执行要求 `USE_TRITON_MATMUL=1`；
- `USE_FAST_TRITON_MATMUL=0` 才会进入当前 native W4 kernel 选择逻辑；
- `*_BACKEND=triton` 仅用于受控 A/B，不是当前默认生产选择；
- `XHQUANT_FORBID_CUDA_W4_UNPACK=1` 用于阻止任何静默展开慢路径。

## 7. vLLM Merak 部署

### 7.1 公共环境

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

当前 cache owner/投机事务以单请求为正式契约：

```text
--max-num-seqs 1
--no-enable-prefix-caching
--no-async-scheduling
```

多个 HTTP 请求可以排队，但不能并行 forward。

### 7.2 Base 服务

```bash
export MODEL_DIR=/path/to/artifact-containing-golden-meta
export SERVED_NAME=qwen3_5_9b_w4_base
export PORT=18443

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
  --limit-mm-per-prompt '{"image":4}'
```

Qwen3-Next 纯文本服务去掉 `--limit-mm-per-prompt`。若产物没有 `visual/`
也应去掉该参数。当前 Qwen visual 路径按图像验收，不要在没有实际导出和
测试的情况下开放 video。

`--max-num-batched-tokens 256` 是调度/prefill chunk budget，不是最大上下文。
最大上下文由 `--max-model-len 262144` 决定。大模型在显存验证后可把
batched-token budget 调为 256 的倍数；不得因此把 max model length 降为
2048。

### 7.3 MTP 服务

```bash
export VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=0
export VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1

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
  '{"method":"custom_class","model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer","num_speculative_tokens":4}'
```

Next 80B 同样去掉视觉参数。服务端 K 必须与 artifact 的 MTP K 一致。

### 7.4 DFlash 服务

环境：

```bash
export VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=0
export VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=1
```

启动参数与 MTP 相同，只把 speculative K 改成 9：

```bash
--speculative-config \
'{"method":"custom_class","model":"vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer","num_speculative_tokens":9}'
```

DFlash context、context-decode 和 draft-decode 使用同一份 GPU cache，运行
过程中不做 host copy。context 更新中若仍有 `.item()`，保持 eager；
只 capture 已验证安全的 draft decode。不要在 capture 失败后保留约十几 GiB
无效 CUDA reserve，启动失败必须退出进程并重新启动。

### 7.5 256K cache 规则

服务按导出 HMONNX 的 262144 容量建立 cache，不私自改成
`max_model_len + lookahead` 的另一套模型容量。投机 lookahead 必须在最后
物理 block 做边界守护；接近 256K 且无法容纳完整 verify block 时，runtime
退回普通 M=1 decode。

请求的文本 token、视觉 token 和生成 token 总和不得超过 262144。长上下文
持续 decode 不会触发整段 cache 搬运，正常路径只推进 valid length/page
metadata。

## 8. Cache、verify 与拒绝语义

### 8.1 Target full attention

- prefill/decode 写 target Paged KV；
- verify 临时产生 K+1 个候选位置；
- 接受后只提交实际接受长度；
- 未接受尾部通过 valid length 隐藏；
- 部分接受时需要恢复事务开始后的未提交 page 内容，尤其是 HMFP
  shared-exponent V page。

### 8.2 Linear attention

Qwen3.5/Qwen3.6/Qwen3-Next 的 recurrent state 不能只靠 KV valid length：

1. draft 前保存已提交 state；
2. target verify 同时输出每个候选位置对应的新 state；
3. 接受 `a` 个 draft token 后，提交正确的第 `a` 个状态；
4. 全拒绝时提交 target 自己产生的单 token 状态；
5. 其余 speculative state 丢弃；
6. split conv cache 与 recurrent state 使用同一个 token 边界。

### 8.3 DFlash 私有 context cache

- prefill 后执行一次 `draft_context`；
- target 每次提交接受 token 后，`draft_context_decode` 只增量更新已提交 token；
- `draft_decode` 把 context cache 当只读 cross-attention K/V；
- 拒绝的 token 不得进入下一轮 context cache；
- cache 本身留在 GPU，三个图通过 runtime cache owner 共享。

接受率特别低时优先检查这一提交顺序、assistant checkpoint、non-causal
cross-attention mask/valid length 和 linear state，而不是先修改 K。

## 9. 验收

### 9.1 静态产物

```bash
test -f "${MODEL_DIR}/golden_meta_info.json"
test -f "${MODEL_DIR}/merak_config.json"
test -d "${MODEL_DIR}/hf_config"
python -m json.tool "${MODEL_DIR}/golden_meta_info.json" >/dev/null
python -m json.tool "${MODEL_DIR}/merak_config.json" >/dev/null
```

核对：

- context/max sequence length 为 262144；
- quant checkpoint 和 artifact 都是 W4G64；
- FA 开关与实际 ONNX attention ABI 一致；
- Base/MTP/DFlash graph 集合和模式一致；
- MTP K=4，DFlash K=9；
- DFlash 三图共享 cache contract 完整；
- `fuse_gdr_block_recurrent_ops=true`；
- visual metadata 指向真实文件；
- Next MTP 使用未旋转 checkpoint。

### 9.2 Pack W4 日志

启动日志必须出现类似：

```text
W4 runtime contract audit: graph=..., nodes=..., initializers=...,
source_nbytes=..., packed_nbytes=..., saved_nbytes=..., op_types=...
```

验收标准：

- `packed_nbytes < source_nbytes`，主 W4 权重通常约减半；
- prefill/decode/draft 的共享权重没有重复计入；
- pack 前后 GPU allocated/reserved 没有因源 W4 展开权重增加；
- 没有 `No non-expanding kernel`；
- 没有 CUDA W4 unpack；
- 没有 Torch ABI mismatch。

### 9.3 API smoke

```bash
curl -s "http://127.0.0.1:${PORT}/v1/models" | python -m json.tool

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"${SERVED_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"在南京，怎样安排一天的城市漫步？\"}],
    \"temperature\":0,
    \"max_tokens\":256
  }" | python -m json.tool
```

视觉请求需另测一张、多张图片以及长文本 + 图片组合。首字延迟和输出 TPS
要从服务端同一计时口径统计，不能使用客户端把思考文本、网络等待或多个并发
服务争抢 GPU 混入的数字。

### 9.4 精度与投机解码

每个型号至少执行：

1. HF/Golden 与 Base 非 FA；
2. Base 非 FA 与 Base FA；
3. Base FA 与 MTP 最终 greedy token；
4. Base FA 与 DFlash 最终 greedy token；
5. MTP/DFlash 的全接受、部分接受、全拒绝；
6. normal decode `M=1` 和 verify `M=2..16` kernel 守护；
7. linear state、conv cache、full KV 的接受/拒绝边界；
8. 视觉 prompt 的 target/draft prefill；
9. 接近 262144 的边界和投机退化为普通 decode；
10. 多轮串行请求无 cache 污染。

除最终输出外，还要记录：

- draft 每步耗时；
- target verify 耗时；
- 平均接受 token 数和接受率；
- 普通 decode、MTP、DFlash 的 decode TPS；
- prefill 吞吐和 TTFT；
- packed 权重、KV cache、CUDA Graph 的显存分项。

MTP 接受率约 85%、K=4 时通常应接收多个 token；若端到端仍慢于 Base，
应拆分 draft、verify、scheduler、sampling、cache commit 和 kernel 时间，
而不是只看接受率推断一定加速。

## 10. 常见问题

### 10.1 为什么服务的 token budget 是 256，不是 256K？

`--max-num-batched-tokens 256` 是单次 scheduler/prefill chunk 的预算，
`--max-model-len 262144` 才是模型总上下文。两者不是同一参数。

### 10.2 为什么 0.8B/2B 启动后没有 Pack W4 日志？

它们 checked-in YAML 默认 W8。必须用 `--bits 4` 重新量化，或者提供合法
W4 checkpoint，再检查 artifact quant metadata；不能只把目录名写成 w4。

### 10.3 为什么 DFlash 三个模型不能各自维护 cache？

`context` 和 `context_decode` 构造的是同一份 draft context，`draft_decode`
只读它。分成三份会增加显存、引入搬运并破坏接受/拒绝后的有效长度。

### 10.4 为什么 DFlash draft decode 不是 causal？

它的 query 是当前 anchor/mask，K/V 是已经构造好的 context，语义是
non-causal Padding Full Cross-Attention。只有 target 自回归 attention 才按
causal 规则。

### 10.5 Qwen3-Next 旋转权重能不能跑 MTP？

不能。旋转后的 Base 可以运行，但未同步旋转的 MTP head 与 target 表示空间
不一致。重新用 `rotation: false` 量化，再导出 Base/MTP 做对照。

### 10.6 一张卡可以启动三个 9B 服务吗？

在当前 A100 验证中，三个 9B packed W4 Base/MTP/DFlash 服务可在一张卡
容纳，但每个进程仍独立持有权重、262144 cache 和 CUDA Graph。它适合功能
联调，不适合性能比较；正式 TPS 测试必须逐服务独占 GPU。

## 11. 发布检查单

- [ ] `xhquant_55` 和 `vllm` 环境版本正确，各自扩展 ABI 正确；
- [ ] 四个源码 revision 已记录；
- [ ] W4G64 量化 checkpoint 已保存，0.8B/2B 没有误用默认 W8；
- [ ] 所有正式产物都传入 262144、FlashAttention、block recurrent fusion；
- [ ] Base/MTP/DFlash/visual-only 按能力矩阵导出；
- [ ] DFlash K=9，assistant config、共享 cache 和 non-causal cross-attention 正确；
- [ ] MTP K=4，linear state/conv cache 接受与回退正确；
- [ ] Next MTP 使用未旋转 checkpoint；
- [ ] `MODEL_DIR` 指向含 `golden_meta_info.json` 的实际目录；
- [ ] Pack W4 严格模式已启用，pack 阶段 GPU 显存无增长；
- [ ] Base、MTP、DFlash、视觉、拒绝回退和 256K 边界均通过；
- [ ] 服务监听 `0.0.0.0`，`/v1/models` 和 chat completion 均正常。

补充资料：

- [Qwen3.8-27B DFlash2 浮点与量化部署全流程](qwen3_8_27b_dflash2_float_quant_deploy.md)
- [Qwen3.5 9B 部署说明](qwen3_5_9b_merak_deploy.md)
- [Qwen3.6 35B-A3B 部署说明](qwen3_6_35b_a3b_merak_deploy.md)
- [投机解码与 W4 验证记录](qwen3_5_spec_decode_w4_validation_20260727.md)
- `xh2modelzoo/examples_merak/llm/qwen3_5/README_workflow.md`
- `xh2modelzoo/examples_merak/llm/qwen3_next/README.md`
- `xh2modelzoo/docs/qwen3_5_hmonnx_io_spec.md`
