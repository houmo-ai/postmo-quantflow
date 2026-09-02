# 多模态与多组件边界

## 视觉 token 档位

当前 Qwen3.5 的视觉入口按 patch/token 容量组织，已经不采用固定宽高配置。
[visual_token_gears](../../../xhmodel_merak/xh_llm/models/qwen3_5/visual_token_gears.py)
负责档位选择、patch padding、position/RoPE 输入和 manifest；
[视觉导出](../../../xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_vision_model.py)
记录各档图的相对路径。

图容量、有效 patch 数、merge 后 image token 数是不同量，不能拿一个长度填所有字段。
runtime 选择能容纳有效 token 的最小档位，并保持 padding mask、位置和输出恢复顺序。
这是 Qwen3.5 当前实现的契约，不要求其他 VLM 照抄档位列表或位置编码。

历史 [visual token tests](../../../tests/qwen3_5/test_visual_token_gears.py)
涵盖几何/位置、padding mask、分组恢复和产物搬移，尚未纳入当前 CI 的专项覆盖。
当前四个 Transformer block CI 是文本层导出，不能作为视觉档位或完整多模态验证证据。

## Host 与图的分工

多模态 LLM、语音流水线和扩散模型都要按真实数据与状态边界拆图，不能只按模态名称分包。
例如 [Kokoro](../../../xhmodel_merak/xh_other_model/models/kokoro/)
有 Host 输入准备、静态图和 runtime 调度；模型页维护实际组件表和数据契约。

稳定记录应包括每条边的输入输出、shape/dtype、有效长度、组件调用顺序、
跨 step/chunk 状态和 metadata 路径。文本规范化、媒体解码、采样循环等是否留在 Host，
应由具体模型与部署成本决定；前处理不能仅因能写成 tensor 运算就默认放进 NPU 图。

新芯片优先复用这些语义边界；如果搬移算子跨越 Host/NPU，明确边界 tensor 的变化与传输成本，
在当前 CI 增加接口或小图回归，再单独记录目标执行证据。
