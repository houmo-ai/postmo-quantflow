# Macaron-V1-Tall

Macaron 使用 Qwen3.5-MoE 的量化、导出和加载流程，模型类型为
`Qwen3_5MoeForConditionalGeneration`。不再使用导出后按 ONNX 节点名注入 LoRA
的私有流程。普通 Linear LoRA 沿用 Qwen3.5 实现；PEFT 的
`experts.gate_up_proj`、`experts.down_proj` 在模型包装后绑定到 MoeBlock，
由 SDK 量化并导出专家 A/B 权重。

配置位于 `configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/`：

- `macaron_v1_tall_w4a8.yaml`：不带 LoRA 的 W4A8 主模型。
- `macaron_v1_tall_w8a8.yaml`：不带 LoRA 的 W8A8 主模型。
- `macaron_v1_tall_lora.yaml`：W4A8 主模型、W8 LoRA 权重。

LoRA 使用通用的 `export.model.lora.path` 列表；需要三个 adapter 时填三个目录，
没有固定 L0–L3 数量限制。每个目录提供 `adapter_config.json` 和 safetensors 权重。
专家 LoRA 的激活精度继承 MoeBlock；当前执行使用 ALIGNED 模式。

```yaml
export:
  model:
    model_type: Qwen3_5MoeForConditionalGeneration
    lora:
      path: [/models/loras/L0, /models/loras/L1, /models/loras/L2]
      w_schema: {bits: 8, fp_mode: ssfp, hidden_bit: false}
```

`macaron_workflow.py --lora-paths /models/loras/L0 /models/loras/L1 /models/loras/L2`
通过 `AutoLLMWorkflow` 执行公共工作流。`--export-from-quanted-model` 可以复用已有
量化 checkpoint；adapter 路径独立传入，不要求保存在 checkpoint 内。

产物遵循 Qwen3.5 的布局：根目录是基础模型，`lora/<adapter-name>/` 中保存每个
adapter 的完整 Prefill/Decode 视图，根 metadata 的 `lora_adapters` 记录各视图。
`hmonnx_demo.py --export-dir <产物目录>` 加载全部已导出 adapter，
`--adapters L0 L2` 可选择其中几个。加载不依赖原始 adapter 或 checkpoint 目录。
`hf_demo.py` 提供 Transformers + PEFT 参考推理。

小模型 CI 覆盖真实 AutoRound 量化、三个 adapter、产物搬迁和加载推理。
专家 LoRA 的独立测试覆盖单 token/多 token、PEFT 权重布局和 W4/W8 量化传递。
