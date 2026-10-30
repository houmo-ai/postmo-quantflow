# Spark X2.5

以下命令均从仓库根目录运行。`CUDA_VISIBLE_DEVICES=6` 选择物理 GPU 6，
程序内使用逻辑设备 `cuda:0`。输出目录需使用新路径。

1.7B 和 4B 配置保留 FlashAttention，禁用 PageAttention；Spark 的滑动窗口输入
处理器尚不支持 PageAttention，显式开启会在量化或模型初始化前报错。

KV cache 按 `layer_types` 分配：滑窗层的物理长度为
`align16(sliding_window + prefill_chunk_length)`，全局层为 `context_max_length`。
例如 4B 的 512-token 滑窗和 256-token prefill 对应 768-token 滑窗缓存；
其 27 个滑窗层不再各自分配完整 context，9 个全局层仍保留完整历史。
滑窗层使用 `HybridCacheTensor` 记录有效长度，prefill/decode 复用同一容量；
attention 实际消费的宽度为 `align16(sliding_window + 当前图输入长度 - 1)`。
导出 metadata 的 `layer_types`、`layer_kv_shapes` 描述逐层布局，加载时校验并按其分配。
旧产物缺少该布局或仍使用全长滑窗缓存时需要重新导出，不能只修改 metadata。

## 1.7B：RTN W8A8

模型目录：`data/models/Spark-X2.5-1.7B`。
配置：`configs_merak/workflows/xh2a/llm_models/xh_token_spark/1_7b/xh_token_spark_1_7b_xh2a_w8a8.yaml`。

该配置为 `quant: null`，跳过独立权重量化，由 xhquant 在导出阶段执行 RTN W8A8。

### Workflow 导出

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_workflow.py \
  --model-dir data/models/Spark-X2.5-1.7B \
  --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/1_7b/xh_token_spark_1_7b_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/xh_token_spark_2_5_1_7b_export
```

### HMONNX 推理测试

推理脚本的 `--config` 接收导出产物的 `golden_meta_info.json`，不是 workflow YAML。
先从上面的导出目录查找 metadata；以下命令适用于目录内只有一份导出产物的情况：

```bash
MODEL_META_1_7B="$(rg --files work_dirs/xh_token_spark_2_5_1_7b_export | rg '/golden_meta_info\.json$')"
```

如果目录内有多次导出，将 `MODEL_META_1_7B` 设为要测试的那份 metadata 的完整路径，
也可以直接使用导出日志打印的路径。保留 metadata 周围的 HMONNX 图、权重和 tokenizer 文件。

普通文本生成测试：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_1_7B" \
  --prompt "用中文简单介绍自己。" \
  --max-new-tokens 64
```

思考模式测试（增加 token 预算以容纳思考内容）：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_1_7B" \
  --prompt "计算 17 乘以 23，并解释步骤。" \
  --think \
  --max-new-tokens 256
```

Golden 数据测试：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_1_7B" \
  --prompt "用中文简单介绍自己。" \
  --golden
```

`--golden` 开启 HMONNX Golden 数据输出，并将生成长度固定为 2 个 token。
普通推理会流式打印生成文本，结束后打印回答内容、耗时和内存统计；
带 `--think` 时还会分开打印思考内容与回答。

### 统一接口

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

build_model(
    model_id="spark-x2.5-1.7b",
    model_dir="data/models/Spark-X2.5-1.7B",
    output_dir="work_dirs/spark_x2_5_1_7b_release",
    device="cuda:0",
)
result = inference_model(
    artifact_dir="work_dirs/spark_x2_5_1_7b_release",
    inputs={"prompt": "用中文简单介绍自己。", "max_new_tokens": 64},
    device="cuda:0",
)
print(result.output_text)
```

## 4B：AutoRound 混合 W4/W8

模型目录：`data/models/Spark-X2.5-4B`。
配置：`configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml`。
标定数据：`examples_merak/llm/xh_token_spark_2_5/calib_data/spark_tokens_4b_generated.jsonl`。

AutoRound 使用 group size 64：attention QKV/输出投影，以及第 0、1、34、35 层
MLP 使用 W8，其余 MLP 使用 W4。embedding、tied head、norm、g_proj 保持浮点。
默认使用 128 条、每条 2048 token 的标定样本，迭代 200 次。

独立入口 `spark_tokens_autoround.py` 默认也使用上述 4B 标定数据，并启用短样本拼接；
`--no-concat-calibration` 可关闭拼接。当前 recipe 只支持对称量化，`--no-sym` 会直接报错。

Spark recipe 位于本工程的
`xhmodel_merak/xh_llm/models/xh_token_spark_2_5/autoround_recipe.py`，
从已安装 GPTQModel 的位置定位 `third_party/auto-round`。
标定数据准备复用 GPTQModel 的 `prepare_calibration_dataset`，无需修改 GPTQModel。
导出时 packed Linear 保留 checkpoint 的 W4/W8 位宽，浮点模块使用 RTN W8。

### 独立 mixed 脚本的精度规则

`spark_tokens_autoround_mixed.py` 从 checkpoint 的 `config.json` 读取层数 N，
取 `k = min(2, N // 2)`。所有 attention 的 `q_k_v_proj`、`out_proj` 使用 W8；
首 k 层和末 k 层的 MLP `gate_proj`、`up_proj`、`down_proj` 使用 W8，
其余 MLP 跟随 `--bits`（默认 W4）。层号从 0 开始：

| N | 使用 W8 的 MLP 层 |
| --- | --- |
| 1 | 无；仅 attention 使用 W8 |
| 2 | 0、1 |
| 3 | 0、2 |
| 4 | 0、1、2、3 |
| 5 | 0、1、3、4 |
| 36 | 0、1、34、35 |

mixed 配置对 `self_attn.g_proj` 和 `lm_head` 显式设置 `bits:16`，
要求 AutoRound 跳过其权重量化；该值不强制把原始 BF16 权重转换为 FP16。
embedding 不启用量化，RMSNorm 不属于 Linear 量化目标。
这里的“保持浮点”指 AutoRound checkpoint 阶段，后续 HMONNX 导出仍按部署配置处理。
默认输出目录的精度标记随全局位宽变化：W4 为 `w4w8`，W8 为 `w8`，W5 为 `w5w8`。

两个独立脚本 `spark_tokens_autoround.py` 和 `spark_tokens_autoround_mixed.py`
的 `--data-type` 默认值均已从 `int_sym_gptq` 改为 `int_sym_xh_hessian`，
并作为全局 `--data_type` 传给 AutoRound；需要原值时显式传入
`--data-type int_sym_gptq`。单层 override 的 `data_type:int` 保持自身配置。
这项默认值变更仅描述这两个 CLI，不改变 workflow YAML 或 recipe 自身的默认值。
普通脚本仍使用自身的 W4/浮点 gate 策略，不采用上述首尾层规则。

### Workflow 量化与导出

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_workflow.py \
  --model-dir data/models/Spark-X2.5-4B \
  --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml \
  --quant-output-dir work_dirs/xh_token_spark_2_5_4b_quant \
  --export-output-dir work_dirs/xh_token_spark_2_5_4b_export
```

复用量化产物时，将 `--model-dir` 设为量化日志中 `quanted_model_dir` 的实际路径。
以下命令从已有的 `work_dirs/xh_token_spark_2_5_quant`（4B 混合 W4/W8）重新导出，
跳过 AutoRound，保留 YAML 中的 FlashAttention 配置：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_workflow.py \
  --model-dir work_dirs/xh_token_spark_2_5_quant \
  --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/xh_token_spark_2_5_4b_reexport_20261009
```

### HMONNX 推理测试

推理脚本的 `--config` 接收导出产物的 `golden_meta_info.json`，不是 workflow YAML。
先从上面的导出目录查找 metadata；以下命令适用于目录内只有一份导出产物的情况：

```bash
MODEL_META_4B="$(rg --files work_dirs/xh_token_spark_2_5_4b_export | rg '/golden_meta_info\.json$')"
```

如果目录内有多次导出，将 `MODEL_META_4B` 设为要测试的那份 metadata 的完整路径，
也可以直接使用导出日志打印的路径。保留 metadata 周围的 HMONNX 图、权重和 tokenizer 文件。

2026-10-09 从 `work_dirs/xh_token_spark_2_5_quant` 重新导出的 metadata 为：

```bash
MODEL_META_4B=work_dirs/xh_token_spark_2_5_4b_reexport_20261009/hmquant_xh2_xh_token_spark_4b_w4a8_256_2k_mpe1024k_20261009/golden_meta_info.json
```

该产物的 prefill/decode 图各有 9 个 FlashAttention 节点、96 个 W4 Linear 和
121 个 W8 Linear；两张图的 180 个 SSFP 投影均无负 scale 或非有限 scale。
使用下方推理入口完成了最多 16-token 的生成测试，输出
“您好，我是一款人工智能助手，专注于为您提供各类知识解答、信息处理与”。
导出检查和推理日志保存在重新导出目录的 `export_validation.json`、
`workflow_export.log` 和 `hmonnx_generate_smoke.log`；此次为生成测试，未进行完整精度评测。

普通文本生成测试：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_4B" \
  --prompt "用中文简单介绍自己。" \
  --max-new-tokens 64
```

思考模式测试（增加 token 预算以容纳思考内容）：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_4B" \
  --prompt "计算 17 乘以 23，并解释步骤。" \
  --think \
  --max-new-tokens 256
```

Golden 数据测试：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python examples_merak/llm/xh_token_spark_2_5/xh_token_spark_2_5_hmonnx_generate.py \
  --config "$MODEL_META_4B" \
  --prompt "用中文简单介绍自己。" \
  --golden
```

`--golden` 开启 HMONNX Golden 数据输出，并将生成长度固定为 2 个 token。
普通推理会流式打印生成文本，结束后打印回答内容、耗时和内存统计；
带 `--think` 时还会分开打印思考内容与回答。

### 统一接口

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

build_model(
    model_id="spark-x2.5-4b",
    model_dir="data/models/Spark-X2.5-4B",
    output_dir="work_dirs/spark_x2_5_4b_release",
    quant_method="autoround",
    device="cuda:0",
)
result = inference_model(
    artifact_dir="work_dirs/spark_x2_5_4b_release",
    inputs={"prompt": "用中文简单介绍自己。", "max_new_tokens": 64},
    device="cuda:0",
)
print(result.output_text)
```

## 推理参数

| 参数 | 用途 |
| --- | --- |
| `--config` | 导出产物中的 `golden_meta_info.json` 路径 |
| `--prompt` | 输入文本，也可以传入保存 prompt 的文本文件路径 |
| `--max-new-tokens` | 最大生成 token 数，默认 10 |
| `--think` | 开启思考模式，默认关闭 |
| `--golden` | 开启 Golden 数据输出，生成长度固定为 2 |
| `--fast` | 调用 runtime 的 `to_fast()` 切换执行模式 |
| `--debug` | 开启 SDK 调试日志 |

## Workflow 推理调试

`debug_scripts/spark_tokens_xh2_generate.py` 通过 workflow 加载原生或已量化的 HF checkpoint，
使用 `--eval-type` 选择模型执行模式。此入口不执行 AutoRound 标定量化。

此脚本与 `debug_scripts/native_spark_tokens_generate.py` 均在贪心和采样模式应用
`--repetition-penalty`。原生入口的 `--dtype float16|bfloat16|float32` 会传给模型加载器。

以下命令使用已有的 4B 量化 checkpoint；1.7B 可替换模型目录和对应的 W8A8 workflow YAML。

```bash
CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/xh_token_spark_2_5/debug_scripts/spark_tokens_xh2_generate.py \
  --model-dir work_dirs/xh_token_spark_2_5_4b_autoround_quant \
  --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml \
  --eval-type wrap \
  --prompt '用中文简单介绍自己。' \
  --no-think --max-new-tokens 64
```

使用同一提示词逐一测试各模式，并分别保存日志：

```bash
for eval_type in wrap fronted quanted_disable quanted_fast quanted_aligned; do
  CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/xh_token_spark_2_5/debug_scripts/spark_tokens_xh2_generate.py \
    --model-dir work_dirs/xh_token_spark_2_5_4b_autoround_quant \
    --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml \
    --eval-type "$eval_type" \
    --work-dir "work_dirs/spark25_workflow_modes_fixed/$eval_type" \
    --prompt '用中文简单介绍自己。' \
    --no-think --max-new-tokens 16 || break
done
```

2026-10-09 在 SSFP 转换和浮点 FlashAttention decode 修复后，使用上述 4B checkpoint、
W4A8 YAML、float16 和相同提示词完成五种模式的测试（贪心生成，最多 16 tokens）：

| `eval_type` | 输出 |
| --- | --- |
| `wrap` | 您好，我是词元星火研发的认知智能大模型，专注于通过 |
| `fronted` | 您好，我是词元星火研发的认知智能大模型，专注于通过 |
| `quanted_disable` | 您好，我是词元星火研发的认知智能大模型，专注于通过 |
| `quanted_fast` | 您好，我是词元星火研发的认知智能大模型，专注于通过 |
| `quanted_aligned` | 我是词元星火研发的认知智能大模型，专注于通过自然语言 |

结果汇总保存在 `work_dirs/spark25_workflow_modes_fixed/results.json`，其中记录各模式的
实际日志路径。此次测试验证加载、prefill 和缓存 decode，未进行完整精度评测。
脚本会在日志中记录生成 token 数、token ID 和包含特殊 token 的原始输出，
避免普通解码隐藏重复特殊 token。

Spark checkpoint 的 `config.json` 必须声明 `model_type: spark2_5`。
加载器对 `spark3`、其他类型或缺少 `model_type` 的配置直接报错，
不会自动改写模型身份或创建转换后的临时 checkpoint。

可选模式：`wrap`、`fronted`、`quanted_disable`、`quanted_fast`、`quanted_aligned`。
`quanted_*` 会按 YAML 的部署配置构建
对应的量化执行模型；`wrap`、`fronted` 用于检查模型包装和图转换。
默认模式为 `quanted_aligned`；`none` 和 `exported` 不支持此入口的生成调用。
当前 Spark eager 路径存在 RMSNorm dtype 不兼容，入口暂不提供 `eager_fast` 和 `eager_aligned`。

`--device` 可选择推理设备，计算类型沿用 float16；`--work-dir` 可指定日志目录。
未指定日志目录时保存到 `work_dirs/<配置名>/<eval-type>/`。
`--input` 支持 JSON/JSONL 输入，`--limit` 控制记录数，`--do-sample` 开启采样。

2026-10-09 修复前使用上述 `work_dirs/xh_token_spark_2_5_4b_autoround_quant` checkpoint、
4B W4A8 YAML 和同一提示词进行了 16-token 贪心推理测试：
`wrap`、`fronted`、`quanted_disable`、`quanted_fast` 均输出
“您好，我是词元星火研发的认知智能大模型，专注于通过”。
`quanted_aligned` 能完成执行，但输出重复的“决”。修复前的日志和结果汇总位于
`work_dirs/spark25_4b_autoround_inference/<eval-type>/` 和该目录的 `results.json`。

### `quanted_aligned` 重复输出的原因

对 `work_dirs/xh_token_spark_2_5_4b_autoround_quant` 的逐节点对比定位到第 0 层
`model.layers.0.self_attn.q_k_v_proj`。两种模式的输入 RMSNorm 输出完全相同，
HM 整数权重与 scale 反量化后也完全相同，但投影输出的余弦相似度仅为 0.000530。
该投影的 245,760 个 scale 中有 123,499 个为负数（50.25%）。

SSFP 内核的输入约定是非负 scale，计算使用 `abs(scale)`。
问题出在 `xhquant.nn.GPTQPackedWeight.to_hm_ssfp()`：转换时保留了 GPTQ 的负 scale，
却没有将其符号折入整数权重。因此 ALIGNED 实际计算的是
`W = signed_integer * abs(scale)`，与 checkpoint 的
`W = signed_integer * scale` 不同。该投影的 ALIGNED 输出与 `abs(scale)` 参考输出的
余弦相似度为 1.0；FAST 直接用带符号的 scale 反量化，所以修复前能输出可读文本。

2026-10-09 已在 xhquanttool 的 `to_hm_ssfp()` 中修复转换：
对负 scale 对应的整数权重在 int16 中取反，再输出 `abs(scale)` 的 FP16 值。
W4 的 `-8`、W8 的 `-128` 取反后分别超出最大值 `7`、`127`，按现有 SSFP
转换规则裁剪。仅这些边界值产生一个 scale 步长的误差，其他值保持反量化结果。
checkpoint 的 packed words、scale 和原始文件保持不变。

使用原 SSFP 内核重新运行相同的 16-token 贪心测试，ALIGNED 输出为
“我是词元星火研发的认知智能大模型，专注于通过自然语言”，不再重复输出“决”。
180 个投影中的负 scale 数量为 `0/58,982,400`；FAST 仍输出
“您好，我是词元星火研发的认知智能大模型，专注于通过”。
此次测试验证转换及生成执行，未进行完整精度评测；日志保存在
`work_dirs/spark25_ssfp_scale_fixed/<eval-type>/`。
旧 HMONNX 产物需从量化 checkpoint 重新导出，才能包含修正后的权重和 scale。

通过脚本查看实际权重的负 scale 分布：

```bash
CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/xh_token_spark_2_5/debug_scripts/spark_tokens_xh2_generate.py \
  --model-dir work_dirs/xh_token_spark_2_5_4b_autoround_quant \
  --config configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml \
  --eval-type quanted_aligned --diagnose-ssfp-scales \
  --work-dir work_dirs/spark25_ssfp_scale_fixed/quanted_aligned \
  --no-think --max-new-tokens 2
```

修复前的诊断证据和临时实验保存在 `work_dirs/spark25_aligned_diagnosis/`；
其中临时修改内核的实验只用于定位原因，正式修复遵循内核的非负 scale 约定。

### `wrap` / `quanted_disable` 的重复特殊 token

补测时发现两种模式在“您好”后重复生成 `<｜start▁of▁sentence｜>`，普通解码会隐藏这些
特殊 token。第一个非有限输出出现在第 3 层的浮点 FlashAttention 单 token decode。
该 SDK 路径使用物理长度 2048 的 KV cache，但此次有效 KV 长度只有 22；
3D 分段计算中的空分段提前返回，未写入以 `torch.empty` 分配的中间输出。
合并阶段读取了这些未初始化值，`0 * NaN` 仍为 NaN，导致后续 logits 异常。

已将 xhquanttool 的
`backend/hmops/fused_ops/flash_attention/fp/_flash_attention_fp_triton.py` 中
`segm_out` 的分配改为 `torch.zeros`。CUDA 回归用释放后的 NaN 内存复现触发条件，
检查有效 KV 长度 22/129 的 decode 输出有限且与 PyTorch SDPA 一致。
复测后，两种模式均恢复上表中的可读输出；FlashAttention 仍按 YAML 启用。

## vLLM 服务评测

评测客户端、128K 套件、数据下载和 IFEval 评分入口位于 [eval/](eval/README.md)。
客户端通过已有服务运行，不加载本地权重；运行命令、数据准备与续跑说明见该目录文档。
