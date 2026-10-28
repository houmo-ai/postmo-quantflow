# Xing4.0-29B-A4B

质量评测和与官方参数的对齐记录见
[状态文档](../../../docs/xing4_0_29b_a4b_w4a8_status_20261013.md)。
下面的统一接口构建 XH2a 导出产物；质量评测使用 `hm_eval` 的 BF16、GPTQ 和
QuantGraph 后端，导出成功本身不代表质量达标。

```bash
python -m examples_merak.unified_api.catalog xing4.0-29b-a4b
python -m examples_merak.unified_api.build \
  --model-id xing4.0-29b-a4b \
  --model-dir /data01/llm_models/Xing4.0-29B-A4B \
  --quant-output-dir work_dirs/xing4_0_29b_a4b_absorbed_quant \
  --output-dir work_dirs/xing4_0_release --device cuda:0
python -m examples_merak.unified_api.generate \
  --artifact-dir work_dirs/xing4_0_release \
  --prompt '简要介绍量子计算的基本原理。' --max-new-tokens 64 --device cuda:0
```

构建输出目录必须不存在。省略 `--quant-checkpoint` 时，使用 workflow 的
GPTQ W4/G64 routed experts、W8 attention/dense/shared experts 校准流程。
此 recipe 只登记现有 standard 导出及文本输入；未登记 MTP、LoRA 或分层导出。
公开 checkpoint 精度为 `wmix/amix`。各区域由 Xing 在导出前和打包时校验，
位宽证据记录在构建清单的 `artifact.checkpoint_precision_regions`；公共 checkpoint
检查器继续核对来源、配置、权重容器和张量。生产端保存时保留源配置的 head/RoPE 字段表示。

默认布局为 `mla_weight_layout: xh2_absorbed_v1`：GPTQModel 从原始浮点权重
完成 MLA 吸收，再校准和量化融合后的 Linear。checkpoint 保存布局、变换版本及
源权重 dtype；ModelZoo 保留融合 Linear 的整数载荷，不重复吸收。
加载、再次保存和导出需要配套的 GPTQModel Xing MLA 实现。

复用新版 checkpoint 时，用 `--quant-checkpoint` 指向它以跳过量化。
导出已有 checkpoint 时，`--model-dir` 也可指向同一 checkpoint，直接使用其配置作为来源；
旧的配置序列化与原始模型配置不一致时使用此方式，无需修改源文件或重新量化。
旧产物按 `hf_unabsorbed` 单独处理，不能通过改 metadata 升级成新版。
新布局要求吸收时权重已物化，默认 `offload_to_disk=false`。
Q-LoRA、原生 attention scale、RoPE 排列及 K/V 独立量化和缓存均保留。
