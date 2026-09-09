# Qwen3.5 MoE

Dense/MoE 已共用 [Qwen3.5 系列 workflow](../qwen3_5/readme.md)。
本目录保留兼容脚本，已有调用不用改；新任务使用
`examples_merak/llm/qwen3_5/qwen3_5_workflow.py`。

## 配置与量化

配置在 `configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/`：

- Qwen3.6-35B-A3B：`35b_a3b/qwen3_6_35b_a3b_full.yaml`；head W4/W5 分别选
  `qwen3_6_35b_a3b_full_head_w4.yaml` / `qwen3_6_35b_a3b_full_head_w5.yaml`，见下文。
- Qwen3.5-122B-A10B：`122b_a10b/qwen3_5_122b_a10b_full.yaml`。
- MTP/DFlash 选对应后缀；GPTQ 选 `*_gptq.yaml`。
- `ornith-1.0-35b_full.yaml` 是独立命名配置，不要用它替代 Qwen3.6 的发布名。

AutoRound 默认 `int_sym_xh`、n256/i400、bs8/ga1、G64。
Attention、shared expert 和 routed expert 的位宽看 YAML 的 `quant.moe`；
W8、FA16 等对照配置不因默认 dtype 变化而改变。
MTP 直接复用 Base 的量化目录，导出时加 `--export-from-quanted-model`。

## Qwen3.6-35B-A3B head W4 / W5

现有 Qwen3.5/Qwen3.6 YAML 和 W8 head 路径保持不变。只有导出 head 小于 8 bit
时要求校准；W8 不要求 GPTQModel 产物包含量化 head，继续使用导出侧的 RTN SEFP。

提供三份可直接使用的完整配置：

| 配置 | quant.head.bits | export head |
| --- | --- | --- |
| [qwen3_6_35b_a3b_full.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml) | 未配置 | `w8a8h1_sefp` |
| [qwen3_6_35b_a3b_full_head_w4.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml) | 4 | `w4a8h1_sefp` |
| [qwen3_6_35b_a3b_full_head_w5.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w5.yaml) | 5 | `w5a8h1_sefp` |

两份新配置基于 `full.yaml`，只增加 `quant.head.bits` 并调整 export 的 lm_head 位宽。
主体保持 AutoRound W4、attention/shared expert W8，使用 `int_sym_xh`、G64、
n256/s2048/i400；视觉和其他导出参数均沿用原配置。W5 head 使用本地 5 bit 数值、8 bit 槽位的 GPTQ 布局。
quant/export 的 head 位宽必须一致；低位宽导出缺少 `quant.head` 会提前报错。
同一份配置兼容原生 HF 整模量化和已有量化产物补 head，不需要 `scope`。

原生浮点 HF 输入照常运行完整 recipe。GPTQModel 内部先处理主体，再校准 head，
最终只保留指定输出目录。head 默认跟随主体算法，AutoRound 跟随 AutoRound、GPTQ
跟随 GPTQ；校准样本和迭代数沿用 quant 设置。遇到 tied weights，会先复制 head，
保持 embedding 数值不变，并在新 checkpoint 中将根配置/text_config 的
`tie_word_embeddings` 设为 false，避免重载时重新绑定。

已有 GPTQModel/AutoRound 产物沿用 `--export-from-quanted-model`。该参数跳过主体
量化，但保留 YAML 中的 head 要求和校准设置：

- 已量化且符合目标：直接复用传入目录，不读校准数据、不前向主体、不复制模型。
- head 仍为浮点：warning 提示补校准算法和输出路径，校准后保存同级完整 checkpoint，
  目录后缀为 `_head_w4` 或 `_head_w5`，本次导出自动消费新目录。
- 已打包但位宽/分组/格式不符：报错，避免默默将低位宽导出变成二次量化。

先执行 `conda activate xhquant_55`，从仓库根运行现有 CLI。`PYTHONPATH=.` 确保使用当前
工作树，避免 conda 环境的 editable 安装指向另一份仓库：

从原生 HF 模型开始量化并导出 W4 head：

```bash
PYTHONPATH=. python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.6-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml \
  --quant-output-dir work_dirs/Qwen3.6-35B-A3B_autoround_head_w4_n256_iter400 \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_head_w4_export
```

已有 GPTQModel/AutoRound 产物跳过主体量化并导出 W4 head：

```bash
PYTHONPATH=. python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/Qwen3.6-35B-A3B_autoround_int_sym_xh_n256_iter400 \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_head_w4_export
```

W5 使用相同命令，将配置改为 `qwen3_6_35b_a3b_full_head_w5.yaml`，输出目录的
`head_w4` 相应改为 `head_w5`。

若需要补校准，新模型路径为
`work_dirs/Qwen3.6-35B-A3B_autoround_int_sym_xh_n256_iter400_head_w4`。
后续将它直接作为 `--model-dir`，配合同一份 W4 head YAML 和跳过量化参数。
同名输出已存在时不会覆盖，请直接传入该新模型复用，或先整理已有目录。

补校准默认按产物生产信息选择算法；AutoRound 常写入 `quant_method=gptq`，应优先
看 provider/provenance，不能据此误判成 GPTQ 算法。补校准只替换 head 张量及相应
配置，保留主体、视觉/MTP、tokenizer 和其他辅助文件，并记录
`lm_head_quantization.json`。head 输入来自已量化主体的真实路由及 final norm。
当前消费契约为 G64、对称、无 activation ordering、int32 打包字、不启用 rotation。

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
