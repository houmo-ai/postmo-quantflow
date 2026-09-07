# Qwen3.5 MoE

Dense/MoE 已共用 [Qwen3.5 系列 workflow](../qwen3_5/readme.md)。
本目录保留兼容脚本，已有调用不用改；新任务使用
`examples_merak/llm/qwen3_5/qwen3_5_workflow.py`。

## 配置与量化

配置在 `configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/`：

- Qwen3.6-35B-A3B：`35b_a3b/qwen3_6_35b_a3b_full.yaml`。
- Qwen3.5-122B-A10B：`122b_a10b/qwen3_5_122b_a10b_full.yaml`。
- MTP/DFlash 选对应后缀；GPTQ 选 `*_gptq.yaml`。
- `ornith-1.0-35b_full.yaml` 是独立命名配置，不要用它替代 Qwen3.6 的发布名。

AutoRound 默认 `int_sym_xh`、n256/i400、bs8/ga1、G64。
Attention、shared expert 和 routed expert 的位宽看 YAML 的 `quant.moe`；
W8、FA16 等对照配置不因默认 dtype 变化而改变。
MTP 直接复用 Base 的量化目录，导出时加 `--export-from-quanted-model`。

## 122B 分层导出

122B 需要启用 placeholder 分层导出，降低峰值内存：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/Qwen3.5-122B-A10B_autoround_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/Qwen3.5-122B-A10B_export \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --quick-test
```

多卡并行导出可设 `XH2MODELZOO_EXPORT_WORKERS`，默认 1；可见 GPU 和机器内存
都要够用。多卡 golden 另外传 `--golden-device-map cuda:0 cuda:1`，
设备编号相对于 `CUDA_VISIBLE_DEVICES`。

维护 placeholder wrapper 时，attention、GDR、MoE 子图返回值只能是 tensor
或扁平 tensor 容器，不能带 `None`，否则会破坏主图与子图的端口对应关系。

## 旧入口

本目录的生成、PPL、legacy meta 转换脚本继续保留。新产物直接使用
`golden_meta_info.json`，一般不需要转换 meta；调用方式见
[统一验证说明](../qwen3_5/readme.md#验证产物)。
