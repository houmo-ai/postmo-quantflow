# 归一化：保留权重与有效区域语义

## Qwen 的两种 RMSNorm 不是同一种权重解释

当前 [hybrid adapters](../../../xhmodel_merak/xh_llm/models/qwen3_5/_hybrid_gated_delta_net.py) 中：

| 实现 | 权重转换与执行 |
| --- | --- |
| `HybridRMSNormMixin` | 上游为 zero-centered 权重；构造 `xhquant.nn.RMSNorm` 后复制 `weight + 1` |
| `HybridRMSNormGatedMixin` | 复制 `weight`，归一化后乘 `silu(gate)` |

两者都保留实际 epsilon、device 和 dtype。不能因为名字都含 RMSNorm 就共用一套权重变换，
也不能把 gate 移到归一化前。新模型需核对其上游公式，而非沿用 Qwen 的 `+1`。
当前 Qwen block CI 走真实模型导出，但没有分别锁定这两种权重转换的独立 CPU 断言。

## 固定长度 padding 会改变归一化结果

[Kokoro `_MaskedTemporalRMSNorm`](../../../xhmodel_merak/xh_other_model/models/kokoro/graph.py)
对固定 temporal length 构造 RMSNorm，Host 提供有效区域对应的 scale；
它用于保持 masked AdaIN 的有效长度语义。padding 为零并不自动保持均值/方差的分母。

历史 [Kokoro test](../../../tests/kokoro/test_kokoro_support.py) 包含有效长度变化、
Host scale 互逆、各 stage 长度和 FP16 长轴均值的断言，尚不属于当前 CI。
新增回归在当前 CI 内用短/满有效长度的小 tensor 对齐参考公式；涉及图改写再补对应导出路径。
这是结构与数值约束；是否由硬件原生支持，需要目标 SDK 的独立证据。
