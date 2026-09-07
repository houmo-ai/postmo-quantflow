# Ling-3.0-Flash

Ling 是文本模型，使用与 Qwen3.5 相同的 Merak workflow。
量化由 `gptqmodel.recipes.ling3_flash.quantize_ling3_flash` 执行；
ModelZoo 负责参数映射、产物加载和导出。

在仓库根目录使用 `conda activate xhquant_55`。需要配套 GPTQModel 的
Ling GPTQ/AutoRound recipe、`int_sym_xh` 和 `xh2_absorbed_v1` 支持。
多份工作区并存时用 `PYTHONPATH="$PWD"` 指向当前仓库，避免导入另一份 editable 安装。

## 配置

配置目录：`configs_merak/workflows/xh2a/llm_models/ling_3_flash/`。

| 配置 | 校准与训练 |
| --- | --- |
| `ling_3_flash_autoround.yaml` | `int_sym_xh`，n256/i400，bs1/ga1 |
| `ling_3_flash_gptq.yaml` | GPTQ，128 个样本，batch=1 |

两者都是 G64、routed experts W4、其余参与量化的 Linear W8；
KDA decay/beta 投影和 MTP 权重保留 BF16。校准 JSONL 与 seed 沿用原配置。
GPTQ 默认关闭磁盘 offload。GDR 两项融合默认打开，FlashAttention 默认关闭。

## 量化并导出

模型结构和参数统一放在 YAML：

```python
from xhmodel_merak.workflows import AutoWorkflow

workflow = AutoWorkflow.from_config(
    model_dir="weights/Ling-3.0-flash",
    config_path="configs_merak/workflows/xh2a/llm_models/ling_3_flash/ling_3_flash_autoround.yaml",
)
quant = workflow.quant(output_dir="work_dirs/Ling-3.0-flash_autoround_quant", device="cuda:0")
export = workflow.export(
    quant_result=quant,
    output_dir="work_dirs/Ling-3.0-flash_export",
    device="cuda:0",
)
```

只做量化也可用原脚本；`--dry-run` 打印 recipe 实际接收的配置：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ling_3_flash/quantize.py \
  --method autoround \
  --model weights/Ling-3.0-flash \
  --output work_dirs/Ling-3.0-flash_autoround_quant \
  --calibration-jsonl data/calib_data/NeelNanda-pile-10k.jsonl
```

输出目录不要复用。快速实验可显式传 `--nsamples 128 --iters 200`；
`--data-type` 只作用于 AutoRound。

## 导出已有量化模型

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ling_3_flash/export.py \
  --model-dir work_dirs/Ling-3.0-flash_autoround_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/ling_3_flash/ling_3_flash_autoround.yaml \
  --export-output-dir work_dirs/Ling-3.0-flash_export \
  --export-from-quanted-model \
  --low-memory \
  --dump-golden
```

`export.py` 只导出，不重新运行 GPTQ/AutoRound。
`--low-memory` 分层加载，`--no-low-memory` 强制整模加载；
都不传时沿用 `HUGE_MODEL_EXPORT_ENABLED`。
`--overwrite` 会删除指定导出目录，重跑前先确认路径。

两份配置默认使用 `mla_weight_layout: xh2_absorbed_v1`：先融合浮点 MLA 权重，
再做 GPTQ/AutoRound；导出直接复用量化载荷，不再二次融合。K/V cache 仍分开维护。
必须从原始 HF 权重重新量化，不能靠改名或改 config 升级旧产物。
旧布局可通过 `--mla-weight-layout hf_unabsorbed` 做对照，仍有原来的二次量化。
新布局支持当前模型的 head-wise gate；不支持 element-wise gate 和磁盘 offload。
格式与芯片约束见 [MLA 量化边界](../../../docs/modeling/structures/ling-mla-quantization.md)。

## 验证

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/ling_3_flash/infer.py \
  --meta work_dirs/Ling-3.0-flash_export/hmquant*/golden_meta_info.json \
  --device cuda:0 \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 32 \
  --print-token-ids
```

推理默认使用 HMONNX V2、packed W4 和 CUDA Graph。
`--no-use-v2`、`--no-pack-w4`、`--no-cuda-graph` 用于定位问题。
普通加载和分层加载都需要验证；短句能生成不代表精度评测通过。
