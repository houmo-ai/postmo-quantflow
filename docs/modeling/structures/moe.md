# MoE：路由语义与权重载荷

MoE 适配需同时保留 expert 顺序、gate/up/down 投影、top-k/归一化规则和量化表示。
尺寸一致只能说明 tensor 能拼接，不能证明路由或权重含义正确。

[Gemma4 `_pack_defused_experts_to_moeblock`](../../../xhmodel_merak/xh_llm/models/gemma4_series/_llm_model_impl.py)
把分离的 expert 权重交给 `xhquant.nn.MoeBlock`。GPTQModel 的反量化 Linear
可能同时带有浮点 `weight` 和用 int8 容器存储的 4-bit `quant_weight`；
打包时两种载荷都必须保留。`tensor.dtype == int8` 不能直接推导量化位宽为 8。
源权重移到 meta 用于控制内存，后续阶段不能再假定源模块仍持有可读的实际载荷。

历史 [weight pack test](../../../tests/gemma4/test_moe_quant_weight_pack.py)
覆盖 expert 轴、量化值范围和源权重释放，尚未成为当前 CI 的独立契约。
当前 [Gemma4 MoE block export](../../../benchmark_test/ci_test_transformer5.16.0/test_gemma4_26b_a4b_block_export.py)
覆盖固定 checkpoint 的真实导出；它不能代替所有路由组合的数值对齐。

新结构或新设备涉及 expert 排列、打包布局、group size、scale/zero-point 时，
在当前 CI 用区分得出 expert 顺序的小权重和输入校验载荷与输出，再检查对应目标导出。
模型包拥有路由数学和权重解释；目标打包格式/算子实现通过工具链边界适配，
不要为每个芯片复制一套模型或把某个 MoE 模型的激活函数写成通用要求。
