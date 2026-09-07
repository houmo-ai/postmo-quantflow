# Qwen3.5 / Qwen3.6 / Qwen3.8 workflow

Dense 和 MoE 共用 `qwen3_5_workflow.py`：量化 → 导出 → golden → HMONNX 验证。
模型结构和精度放在 YAML，命令行只覆盖本次运行需要改的项目。

在仓库根目录运行：

```bash
conda activate xhquant_55
```

需要配套的 GPTQModel 和 AutoRound 支持 `int_sym_xh`。当前 CI 使用
Transformers 5.16.1，见 [CI 环境与测试](../../../benchmark_test/ci_test_transformer5.16.0/README.md)。

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
`--export-from-quanted-model` 跳过量化，直接导出 `--model-dir` 中的模型。

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

动态剪枝使用 [独立 workflow](qwen3_5_dynamic_prune_workflow.py)，
`--auto-s-scalar` 可从原始 HF 权重计算并缓存 expert scalar。
完整导出不要加 `--only-first-block`。

## 验证产物

主入口是导出目录内的 `golden_meta_info.json`，其中记录
`prefill/`、`decode/`、`visual/` 和可选的 draft 子图相对路径。

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

`--dump-golden` 保存对齐用的中间结果，`--quick-test` 检查导出后的运行链路；
两者都不能代替 PPL 和任务精度评测。视觉模型还需用真实图片跑端到端验证。
五档 visual golden 覆盖 96/196/384/704/1536 token；visual-only 的入口是
`visual_meta_info.json`。

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
