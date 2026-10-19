# Qwen3.5 / Qwen3.6 / Qwen3.8 workflow

Dense 和 MoE 共用 `qwen3_5_workflow.py`：量化 → 导出 → golden → HMONNX 验证。
模型结构和精度放在 YAML，命令行只覆盖本次运行需要改的项目。

在仓库根目录运行：

```bash
conda activate xhquant_55
```

需要配套的 GPTQModel 和 AutoRound 支持 `int_sym_xh`。当前 CI 使用
Transformers 5.16.1，见 [CI 环境与测试](../../../benchmark_test/ci_test_transformer5.16.0/README.md)。

XH2A 与 Yuehui 配置不要混用。Yuehui 的 Qwen3.5-4B YAML 在
`configs_merak/workflows/yuehui/llm_models/qwen3_5/4b/`，导出前确认环境已安装
对应的 xhquant/runtime 依赖。

## 选配置

以下路径都相对于 `configs_merak/workflows/xh2a/llm_models/`。

| 模型 | AutoRound 配置 |
| --- | --- |
| Qwen3.5-0.8B / 2B / 4B | `qwen3_5/{0.8b,2b,4b}/` 下的 `*_full.yaml` |
| Qwen3.5-9B | `qwen3_5/9b/qwen3_5_9b_full.yaml` |
| Qwen3.6-27B | `qwen3_5/27b/qwen3_6_27b_full.yaml` |
| Qwen3.8-27B | `qwen3_5/27b/qwen3_8_27b_full.yaml` |
| Qwen3.6-35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml` |
| Qwen3.5-122B-A10B | `qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml` |

`*_gptq.yaml` 使用 GPTQ；`*_mtp.yaml` 和 `*_dflash*.yaml` 选择投机解码结构。
Visual-only 配置以 `*_visual_only_token_gears_99p_candidate.yaml` 结尾。
不是每个模型都有全部组合，以目录中已有配置为准。

AutoRound 默认 `int_sym_xh`、G64、256 个校准样本、400 次迭代、bs8/ga1。
逐层位宽、seed 和校准数据入口仍由原 YAML 决定；Qwen3.5 系列的 pile-10k
别名会解析到本仓库的 `data/calib_data/NeelNanda-pile-10k.jsonl`。
`quant.data_type` 可显式覆盖。GPTQ 配置和已有 checkpoint 不会因此改变。

## 从原始权重量化并导出

以 9B 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --quant-output-dir work_dirs/Qwen3.5-9B_autoround_quant \
  --export-output-dir work_dirs/Qwen3.5-9B_fuse_gdr_export \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --quick-test
```

35B-A3B 只需换模型、配置和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.6-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml \
  --quant-output-dir work_dirs/Qwen3.6-35B-A3B_autoround_quant \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_fuse_gdr_export \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --quick-test
```

量化输出使用新目录。需要重做导出时加 `--overwrite`；它会删除已有导出目录，
不是量化断点续跑开关。导出名来自 YAML 的 `export.model.model_name`；
`--model-name` 只改发布名，不选择或替换原始权重。

## 复用量化模型导出 MTP

Base 和 MTP 共用同一份量化 HF 目录，不再量化一次：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/Qwen3.5-9B_autoround_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/Qwen3.5-9B_mtp_fuse_gdr_export \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --quick-test
```

35B-A3B 对应 `qwen3_6_35b_a3b_full_mtp.yaml`。量化目录需包含 MTP 权重；
DFlash 还需准备 YAML 中指定的 draft checkpoint。
`--export-from-quanted-model` 跳过主干量化；配置了低位 head 或 MTP 校准时，
仍会检查并按需补齐对应权重。

35B-A3B 的 MTP 非共享专家可单独做 W4 校准，算法默认按 head 补量化的
`inherit` 规则跟随主体：AutoRound 对应 AutoRound，GPTQ 对应 GPTQ。其他 MTP 主干保持
RTN SEFP W8，head 复用主模型已量化权重。现有 YAML 默认行为不变；需要开启时
在 `quant` 中增加：

```yaml
quant:
  mtp:
    algorithm: inherit
    expert_bits: 4
    iters: 400
```

校准数据沿用 `quant.calibration`，另留 4 条文本做 draft 输出比较。
完整量化在主干/head 完成后补 MTP；已量化模型加 `--export-from-quanted-model`
时只补 MTP，默认保存到输入同级 `<模型目录名>_mtp_w4`。已有匹配 W4 MTP
校验后直接跳过。导出使用新目录，根据 checkpoint 自动绑定 MTP 专家 W4
和共享 head 的量化配置。MTP 不参与主干自动混合精度搜索。

也可以仅调用 GPTQModel 的 `gptqmodel.recipes.qwen35_mtp.quantize_qwen35_mtp`
先生成新 checkpoint，再按普通 MTP YAML 导出。

主模型 export 的 `lm_head` 是唯一 head 位宽来源，并与 `quant.head.bits` 校验。
MTP/DFlash/DFlash2 自动继承，不需要再填 `spec_draft_head_weight_bits`；旧字段显式
不一致时会报错。规则适用于整个 Qwen3.5 dense/MoE 系列及 Qwen3-Next MTP。
W5 沿用本地 8 bit 槽位，导出逻辑位宽为 5；draft 复用主 head 的完整量化载荷，
包括 W8 路径也不单独 RTN。完整 W5＋MTP 和混合精度＋W5＋量化 MTP 的配置见
[标准入口](../qwen3_5_moe/README.md#共享-head-与-mtp-标准入口)。

GPU HMONNX 的 full causal MTP 验证按 token 顺序更新和读取 SEFP V cache，
避免后续草稿改变已有 token 的共享指数；拒绝草稿后的提交也保持相同顺序。
CUDA Graph 首次捕获会恢复预热改写的缓存，并区分验证块跨越 split-KV
阈值的情况。该运行路径需要配套更新 xhquant 与 ModelZoo；GPU 数值对齐
结果不代替 M50 编译器和芯片验证。

35B-A3B 的 `qwen3_6_35b_a3b_lora.yaml` 默认对 LoRA A/B 分支使用 W8A16：
`export.model.lora.w_schema.bits: 8`、`a_schema.bits: 16`。这两项仅控制
adapter 分支，主模型精度继续由 `quant_scheme` 决定。验证 adapter 量化时，
使用同一 adapter 分别对照 FP 底座、所选量化底座和 W8A16 adapter。
workflow 会为 `lora/<adapter>/` 子目录生成独立的 `merak_config.json`，
验证指定该子目录即可加载对应 adapter；根目录仍是无 adapter 的基座视图。
同时导出 MTP 时，子目录包含自己的 target 图与共享草稿图的本地副本，
草稿和 visual 的大权重文件通过相对链接复用根目录文件。发布和搬迁时保留
整个导出目录，不能只复制 `lora/<adapter>/`；图文件本身的解析路径保持在
adapter 子目录内，以满足 MTP 加载契约。

## Yuehui Qwen3.5-4B

Yuehui 命令统一使用 `configs_merak/workflows/yuehui/` 下的 YAML。原始 HF 模型
走 `full.yaml` 时会量化并导出；量化后的 HF/GPTQModel 目录加
`--export-from-quanted-model`；跳过量化但输入仍是原始 HF 权重时用
`no_quant.yaml`。

完整模型：量化并导出

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir data/models/Qwen3.5-4B \
  --config-path configs_merak/workflows/yuehui/llm_models/qwen3_5/4b/qwen3_5_4b_full.yaml \
  --quant-output-dir work_dirs/yuehui_qwen3_5_4b_quant \
  --export-output-dir work_dirs/yuehui_qwen3_5_4b_export \
  --device cuda:0 \
  --overwrite
```

完整模型：跳过量化，直接导出

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir data/models/Qwen3.5-4B \
  --config-path configs_merak/workflows/yuehui/llm_models/qwen3_5/4b/qwen3_5_4b_no_quant.yaml \
  --export-output-dir work_dirs/yuehui_qwen3_5_4b_no_quant_export \
  --device cuda:0 \
  --overwrite
```

仅导出视觉塔：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir data/models/Qwen3.5-4B \
  --config-path configs_merak/workflows/yuehui/llm_models/qwen3_5/4b/qwen3_5_4b_visual_only_896.yaml \
  --quant-output-dir work_dirs/yuehui_qwen3_5_4b_visual_quant \
  --export-output-dir work_dirs/yuehui_qwen3_5_4b_visual_export \
  --device cuda:0 \
  --overwrite
```

visual-only 命令会先执行 YAML 中的 AutoRound，再导出 visual tower。
如果 `--model-dir` 已经是量化后的 HF/GPTQModel 目录，需要补充
`--export-from-quanted-model`，否则 workflow 会再次量化：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir data/models/Qwen3.5-4B \
  --config-path configs_merak/workflows/yuehui/llm_models/qwen3_5/4b/qwen3_5_4b_visual_only_896.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/yuehui_qwen3_5_4b_visual_export \
  --device cuda:0 \
  --overwrite
```

## 常用导出选项

- 长上下文：`--context-max-length 262144 --enable-flash-attention`。
  先确认该 YAML 的 cache/模型配置支持；单卡长上下文 MTP/DFlash 验证可加
  `--resource-tight-mode`，让 prefill 在分配 decode cache 前释放。
- GDR：`--enable-fuse-gdr-ops` 控制 GDRChunkScan；
  `--enable-fuse-gdr-block-recurrent-ops` 控制块求逆和 recurrent scan。
  对照实验用各自的 `--disable-...`。不传则按 YAML。
- FA 的 Q/K/V/S/P bits 写在 `export.model.flash_attention`，不由量化 W4/W8 决定。
- 分层导出和多卡 golden 见 [MoE 补充说明](../qwen3_5_moe/README.md)。
- 层标记：运行前设置 `LAYER_TAG_ENABLE=1`，方便切分 HMONNX 和分配 GPU。

Qwen3.8 的 Base/MTP 同样使用上面的流程，换成 `qwen3_8_27b_full*.yaml`。
DFlash2 的 8K 与 256K 配置分别为 `qwen3_8_27b_full_dflash2_8k.yaml`、
`qwen3_8_27b_full_dflash2_256k_fa.yaml`，不要只改目录名来切换上下文。
共享 W5 head 时选对应的 `*_head_w5.yaml` 配置。已有 AutoRound 主干若只有浮点
`lm_head.weight`，`--export-from-quanted-model` 会先通过 GPTQModel 校准 head，
在输入目录同级写出 `<输入目录名>_head_w5`，然后用这份新 checkpoint 导出；
再次导出可直接传新目录。DFlash2 的 head 继承主模型 W5 载荷，
`max_pe_length` 仍是 262144，8K/256K 配置分别控制 KV cache 长度。
统一接口复用 checkpoint 时只接受已补齐的 W5 head，不会在构建中隐式补校准。

动态剪枝现在只作用于 Decode，Prefill 始终使用原始 Top-K。
普通 MoE workflow 中保留 ACEP 时可显式配置：

```yaml
export:
  model:
    dynamic_prune:
      mode: decode  # off 关闭；省略整个 dynamic_prune 也默认关闭
      algorithm: acep
      threshold: 0.85  # ACEP 累计代理概率覆盖率
      auto_s_scalar: true
      s_scalar_path: null
```

两份剪枝 workflow 现在默认使用 SAFE，覆盖率 `0.85`，仅作用于 Decode：

- `qwen3_6_35b_a3b_dynamic_prune.yaml`：独立 text-only 导出，默认上下文 2048。
- [`qwen3_6_35b_a3b_full_head_w4_decode_prune_256k_fa.yaml`](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4_decode_prune_256k_fa.yaml)：完整视觉档位、W4 head、256K FlashAttention。

两份配置默认 `auto_s_scalar: true`。没有显式 `.pt` 产物时，会在 HF 包装前
从真实专家权重计算 AMP/AIMER，再用 WikiText-2 raw 训练集全部行按 `"\n".join`
拼接后的前连续 `128 × 2048` token 校准 ExFold，默认每批 1 个窗口。
校准只累计同 token 共同路由专家的配对统计，系数回归使用 `ridge=1e-3`、
裁剪到 `[-4,4]`；未观察配对禁止折叠。无需提前手动生成 `s_scalar`。
算法、文件格式及固定策略见 [SAFE 接入说明](../../../docs/qwen3_5_safe_decode.md)。

本机 text-only 导出示例使用 `xh2` 环境与配套 `xhquanttool_3`；以下 offline 选项要求
本机已有 tokenizer 和 WikiText 数据缓存。`USE_TRITON_MATMUL=0`、`XHQUANT_HSUM_JIT=0`
是本机编译环境的运行选项，不是所有机器的通用要求；SAFE 目前使用常规导出：

```bash
HUGE_MODEL_EXPORT_ENABLED=0 CUDA_VISIBLE_DEVICES=0 USE_TRITON_MATMUL=0 \
XHQUANT_HSUM_JIT=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=.:/data01/home/zukang.xu/workspace/xh2/xhquanttool_3 \
/data01/home/zukang.xu/anaconda3/envs/xh2/bin/python \
  examples_merak/llm/qwen3_5/qwen3_5_dynamic_prune_workflow.py \
  --model-dir /data01/datasets/Qwen3.6-35B-A3B \
  --export-output-dir work_dirs/qwen3_6_35b_safe_export_20261019
```

脚本默认导出目录仍是 `work_dirs/qwen3_5_safe_export`；上述命令显式指定本次本机目标目录。
完整导出不要加 `--only-first-block`。

自动校准文件写入导出工作目录的 `safe_calibration/priors.pt`、
`safe_calibration/exfold.pt`，随后打包为 `moe_safe_s_scalar_v4.pt`。
缓存 identity 绑定 checkpoint、实际 token SHA256、校准配置和模型 dtype；
不匹配会报错，可设置 `overwrite_s_scalar: true` 重建。

可用 `--calibration-text-path` 指向本地原始训练文本；
`--calibration-nsamples`、`--calibration-sequence-length`、`--calibration-batch-size`
分别覆盖样本数、窗口长度和批大小。更改样本数、序列长度或数据来源时应记录实际配置，
不再称为标准 `128 × 2048` WikiText 协议。

显式文件优先：同时传 `--ace-priors-path`、`--exfold-artifact-path`，
或只传 `--s-scalar-path` 使用已生成的 `safe_exfold_v4` 缓存。
也可写入 YAML 的 `export.model.dynamic_prune` 同名字段。
`--no-auto-s-scalar` 要求提供这些手动文件。一维 ACEP scalar 不能替代 SAFE 二维缓存。
CLI 的 `--prune-algorithm` 默认遵从 YAML；保留 ACEP 的自定义 YAML 不应携带 SAFE 专用字段。

W4 head / 256K 配置使用普通 `qwen3_5_workflow.py`，可沿用自动 SAFE 校准，
或在 YAML 中填写已有 SAFE 文件路径，再传 `--export-from-quanted-model`
复用已经量化且包含 W4 head 的 checkpoint。该路径保留完整视觉档位和 head 精度检查。

当前 SAFE 走常规导出，不支持低内存 placeholder 或 MTP/DFlash verify。
Prefill 不带剪枝数据，Decode 带 `prune_threshold=0.85` 和 FP16 SAFE 二维 `s_scalar`。
导出后可使用 text-only runtime 验证连续生成：

```bash
CUDA_VISIBLE_DEVICES=0 USE_TRITON_MATMUL=0 python \
  examples_merak/llm/qwen3_5/qwen3_5_dynamic_prune_hmonnx_generate.py \
  --config work_dirs/qwen3_5_safe_export \
  --device 0 \
  --prompt "用中文简单介绍 Qwen3.5。" \
  --max-new-tokens 64 \
  --min-output-tokens 8
```

以上生成命令使用脚本默认导出目录；若使用本机命令的日期目录，应同步修改 `--config`。
`USE_TRITON_MATMUL=0` 用于绕开本机通用 Linear Triton kernel 的编译兼容问题。

以下是保留的 ACEP 算法说明（使用显式 ACEP YAML 时适用）：
`MoeBlock` 把 `gate_score * s_scalar` 归一化成代理概率，降序累计并保留使累计概率首次达到覆盖率的最短专家前缀，
同时始终保留 Router Top-1 专家。`--auto-s-scalar` 在模型包装后遍历真实主体层，使用当前消费的专家权重和
RMSNorm 的有效系数 `1 + HF weight` 计算 Method1 scalar，不匹配 checkpoint 中的
fused 名称，也不遍历 MTP。这允许直接使用 AutoRound/GPTQ checkpoint，无需文本校准
或重新量化 head。缓存写入工作目录 `moe_acep_s_scalar.pt`，记录算法版本、配置/index
摘要、shard 文件版本和 `s_scalar_eps`；缓存来源或计算参数不匹配时明确报错，可设置 `overwrite_s_scalar: true`
重建。显式 `.pt/.json` scalar 文件必须完整覆盖本次导出的主体层，且每个专家的分数
均为有限正数；缺层不会静默回退为 1。
Scalar 仅用于决定保留哪些专家；合并输出时仍使用保留专家的原始 router 权重并重新归一化。

## 验证产物

完整包的主入口是导出目录内的 `golden_meta_info.json`，其中记录
`prefill/`、`decode/`、`visual_m*` 和可选的 draft 子图相对路径。
五档视觉图位于包根目录的 `visual_m96`、`visual_m196`、`visual_m384`、
`visual_m704`、`visual_m1536`，与 `prefill`、`decode` 同级；路由清单为
根目录的 `visual_gears.json`。完整包名的视觉后缀使用下划线：
`visualm96_196_384_704_1536`。

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config work_dirs/Qwen3.5-9B_fuse_gdr_export/hmquant*/golden_meta_info.json \
  --prompt "用中文介绍一下北京。" \
  --no-sample \
  --max-new-tokens 128 \
  --auto-offload \
  --cuda-graph \
  --use-v2
```

需要从同一次请求生成 HMONNX golden 时，在上述入口增加 `--golden`。decode golden 至少要让
decode 图实际执行一次，因此 `--max-new-tokens` 不得小于 2：

```bash
ENABLE_HMINFERENCE_V2=1 \
python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config <export_dir>/golden_meta_info.json \
  --prompt "用中文介绍一下北京。" \
  --golden \
  --no-sample \
  --max-new-tokens 2
```

`--golden` 除了把 HMONNX 图输入、输出及导出图暴露的 activation 写入
`prefill/step_*`、`decode/step_*` 和本次命中的 `visual_m*/step_*`，还会把本次请求在
prefill 前的原始输入保存到同级 `golden_input/`：

```text
<export_dir>/
├── golden_input/
│   ├── prompt.txt
│   ├── input_image.<原扩展名>   # 本次实际提供图片时
│   ├── tokenizer_output.txt
│   ├── token_ids.npy
│   └── input_meta.json
├── prefill/
│   └── step_0/
└── decode/
    └── step_0/
```

其中 `tokenizer_output.txt` 保留 chat template 的 special tokens 和原始换行；
`token_ids.npy` 是 padding 到静态 prefill 长度之前的二维 token IDs。Host 侧 embedding 结果
仍属于 HMONNX prefill 图输入，按当前图 ABI 保存为 `prefill/step_*/input_1.npy`；embedding
权重由 metadata 中的 `quant_embedding`/`token_embedding_file` 指向，不复制到
`golden_input/`。`golden_input/` 只记录整次请求和 prefill 前处理，不创建 decode 子目录；
每次运行 `--golden` 会清理并覆盖上一轮内容。不传 `--golden` 时不会创建或修改该目录。

首版 TCIM 适配用于 Qwen3.5-9B 普通 prefill/decode 与 visual gears，不包含 MTP、DFlash 或
其他 speculative 路径。先用已编译 HMM 生成独立 metadata：

```bash
PYTHONPATH=$PWD python -m xhmodel_merak.exec_backends.link_hmm \
  --hmm_path <qwen3_5_9b_compiled_hmm_dir> \
  --hmonnx_path <directory_containing_golden_meta_info.json>
```

再使用同一个示例入口，显式传 HMM metadata：

```bash
XHMODEL_INFERENCE_BACKEND=tcim PYTHONPATH=$PWD \
python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config <directory_containing_golden_meta_info.json>/hmm_meta_info.json \
  --prompt "用中文介绍一下北京。" \
  --no-sample \
  --max-new-tokens 128
```

TCIM 不在运行时搜索或替换 ONNX/HMONNX 路径；`hmm_meta_info.json` 中 prefill、decode 和
每个 `visual_m*` 必须明确指向 `.hmm`/`.hmms`。CUDA Graph、auto-offload 和 fast mode
选项不会改变已经编译的 HMM；`--golden` 在 TCIM 后端明确不支持。

`--dump-golden` 保存对齐用的中间结果，`--quick-test` 检查导出后的运行链路；
两者都不能代替 PPL 和任务精度评测。视觉模型还需用真实图片跑端到端验证。
`--dump-golden` 先为五档视觉图分别生成 `visual_m*/step_0`，再用真实图片
验证 visual → prefill → decode；命中档位的视觉 golden 写入 `step_1`。
真实图片验证不能代替全部五档的覆盖。visual-only 的入口是根目录的
`visual_meta_info.json`，图和 golden 同样位于 `visual_m*`，路由清单仍为
根目录的 `visual_gears.json`。

每个 gear 的 step 链接必须指向该 gear 自己的图和 external data。
发布布局校验与链接修复只支持根目录 `visual_m*`，不支持旧 `visual/m*`
或独立导出的 `m*` 布局；旧包需要重新导出，并按元数据中的相对路径加载。

投机解码接收率用 [spec-decode 测试](debug_scripts/qwen3_5_xh_spec_decode_test.py)，
PPL 用 [PPL 脚本](debug_scripts/qwen3_5_xh_ppl_eval.py)。
张量与 cache 布局见 [HMONNX I/O 说明](../../../docs/qwen3_5_hmonnx_io_spec.md)。

## Python 调用

```python
from xhmodel_merak.workflows import AutoWorkflow

workflow = AutoWorkflow.from_config(
    model_dir="weights/Qwen3.5-9B",
    config_path="configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml",
)
quant = workflow.quant(output_dir="work_dirs/qwen35_quant", device="cuda:0")
export = workflow.export(
    quant_result=quant, output_dir="work_dirs/qwen35_export", device="cuda:0",
)
```

临时参数通过 `quant()` / `export()` 的 `config_overrides` 传入，如
`{"quant.calibration.nsamples": 128, "quant.iters": 200}` 可做快速实验。
默认配置仍是 n256/i400。

## DFlash2 滑窗 cache

`DFlash2DraftModel` 的 context prefill/context decode 使用 HybridCache，draft decode
使用逐 query 的 mask 和 xhquant `MaskedAdd`。窗口 2048、prefill 256、block 8 时，
draft cache 容量为 2320；容量计算、绝对位置与临时写入语义见
[DFlash2 cache 契约](../../../docs/modeling/structures/cache.md#dflash2滑窗上下文与临时查询-kv)。
主模型的 256K FlashAttention 不受影响；旧 `DFlashDraftModel` 保留原有 cache/FA 路径。

新 DFlash2 产物使用 `qwen_dflash2_hybrid_v1` ABI，需要配套的加载端；旧 DFlash2
产物需重新导出。接受率验证应记录数据集、抽样、采样参数、输出上限、结束原因，
同时报告逐请求平均接受长度与 pooled draft 接受率。
