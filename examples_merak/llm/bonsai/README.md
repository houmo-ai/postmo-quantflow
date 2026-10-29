# Bonsai Phase A

Bonsai 是独立的模型适配，继承现有 Merak Qwen3.5 的配置、模型和 runtime，
不修改 Qwen3.5 的源文件或公共 wrapper 注册。模型类型为
`BonsaiForConditionalGeneration`，统一接口 ID 为 `bonsai-27b-phase-a`。

实现位于 `xhmodel_merak/xh_llm/models/bonsai/`。MLX 命名、浮点参数转换、
Embedding 逆旋转、在线 Hadamard 和整数码加载均由该目录承担。
xhquant 只负责已有通用 SSFP 量化和 ONNX 导出。

## 转换

从 ModelZoo 根目录执行，输入 HF 目录仅用于复制配置与 tokenizer，不读取基座权重：

```bash
python examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py \
  --bonsai-dir /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  --hf-model-dir /path/to/matching/Qwen3.8-27B \
  --output-dir work_dirs/bonsai_phase_a
```

转换程序不需要初始化 xhquant 后端。未传 `--hf-model-dir` 时，如果 Bonsai 源目录
存在 `config.json`，就复制其中的配置与 tokenizer。只做单层权重检查时可以使用
`--include-name language_model.model.layers.0.mlp.gate_proj`，该产物不能用于整模构建。

产物包含：

- `quant_weight.pt`：HF 参数和各 Linear 的整数码；Embedding 已离线逆旋转。
- `phase_a_aux.pt`：signs、K64 scales 和共享的 forward Hadamard bank。
- `phase_a_metadata.json`：v2 完整性、分组和显式 SSFP4 载荷位宽。
- `config.json`、tokenizer 文件：用于构造原生模型和最终发布。

## 构建和运行

```bash
PYTHONPATH="$PWD" python examples_merak/llm/bonsai/build.py \
  --converted-dir work_dirs/bonsai_phase_a \
  --output-dir releases/bonsai_phase_a \
  --device cuda:0
```

对应 workflow 为 `configs_merak/workflows/xh2a/llm_models/bonsai/bonsai_27b_phase_a.yaml`。
它只消费已经转换的完整 checkpoint，不运行 GPTQ/AutoRound；未签名的浮点 Linear
沿用 W8 SEFP，签名投影和 Hadamard 投影按声明使用 SSFP4。拆分 Q/K/V 时同时保留整数码
与位宽。模型包装使用独立 registry，普通 Qwen3.5 的注册和默认前向保持原样。

构建产物通过统一接口运行：

```python
from xhmodel_merak.model_inference import inference_model

result = inference_model(
    artifact_dir="releases/bonsai_phase_a",
    inputs={"prompt": "17 + 25 等于多少？", "max_new_tokens": 32},
    device="cuda:0",
)
print(result.output_text)
```

当前接入是 text-only、standard export，不提供视觉、LoRA 或推测解码入口。
Phase A 仍是展开的 SSFP4 兼容格式，`pq2_weight=false`；它不是最终 DDR PQ2 packed 格式。

## 验证

```bash
python -m pytest \
  -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_bonsai_conversion.py \
  benchmark_test/ci_test_transformer5.16.0/test_bonsai_phase_a_contract.py
```

真实权重的逐层对照入口为同目录 `verify_phase_a_vs_bonsai.py`。
历史 27B 验证记录见 `docs/xh2_pq2_0_bonsai_adapt_notes.md`；历史结果不能替代新入口的整模验证。

新入口已用包含 linear/full attention 的两层小模型，实测完成加载、包装、FX 图执行、
prefill/decode HMONNX 导出与单卡生成 2 token；这验证接口链路，不代表随机小模型具备回答能力。
当前本机环境的统一 build CI 受已登记依赖版本不匹配阻塞（Transformers 5.5.0 对 5.16.1、
onnxsim 0.6.5 对 0.6.2，并缺少 sentencepiece），尚未重跑真实 27B。
