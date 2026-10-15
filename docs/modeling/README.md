# 算子、结构与目标设备知识

这里沉淀模型适配中不容易从接口名推断的事实：为什么这样写、什么条件下成立、
如何验证，以及什么时候可以去掉 workaround。架构职责见 [Merak 架构](../architecture/merak.md)。
这是一份按需查阅的知识库，不要求每次任务通读。

## 从问题进入

| 遇到的问题 | 入口 |
| --- | --- |
| 量化算法、模型适配和权重格式应改 ModelZoo 还是 GPTQModel | [跨仓库开发边界](../architecture/quantization-boundary.md) |
| 模型注册与 wrapper 注册、`_setup`/`graph_forward`、frontend 选择 | [图表达边界](operators/graph-expression.md) |
| FlashAttention 精度、QKV 布局、mask 图表达 | [Attention 算子](operators/attention.md) |
| Ling MLA 先融合再量化、K/V 独立 SEFP 指数域与 cache | [Ling MLA 量化边界](structures/ling-mla-quantization.md) |
| RMSNorm 权重、gating、padding 对归一化的影响 | [归一化算子](operators/normalization.md) |
| prefill/decode、混合状态、投机验证回滚 | [Cache 与状态](structures/cache.md) |
| DeepSeek-V4 Flash 0731 DSpark 组件、W4/W8、共享 head 与回滚 | [DSpark](structures/deepseek-v4-dspark.md) |
| expert 权重、路由与量化载荷 | [MoE](structures/moe.md) |
| 视觉 token 档位、多图与 Host 边界 | [多模态与多组件](structures/multimodal.md) |
| Qwen-Image-2.1 的特征边界、共享调制、前缀 KV 与浮点诊断 | [接入记录（进行中）](structures/qwen-image21.md) |
| Qwen-Image-2.1-Turbo 的 8 步采样、模型身份与部署验证 | [Turbo 接入记录](structures/qwen-image21-turbo.md) |
| M50、YH、XH3 如何选择已有路径或扩展 | [目标设备索引](targets/README.md) |

算子页拥有图表达和数值语义，结构页描述组合关系与状态，目标页只记录设备差异和证据入口。
同一个约束只维护一份正文，其余页面链接它；模型特例留在模型实现或其 README。

## 怎样增加一条知识

条目按语义命名，如 `masked-softmax`、`hybrid-cache`，避免用某次目录迁移的名称。
小结可以留在现有页；独立主题再新增文件并更新上面的入口。
只记录与结论相关的条件，包含：

| 内容 | 要回答的问题 |
| --- | --- |
| 结论与触发条件 | 哪个算子/结构、哪条导出路径、什么 shape/layout/精度下，需要什么写法？ |
| 原因与边界 | 是模型数学、前端图表达、量化工具链还是设备/runtime 的约束？哪些情况不能照搬？ |
| 证据 | 实现路径与符号；当前 CI 用例及 impact 声明，或明确的覆盖缺口 |
| 验证记录 | 仓库/SDK/runtime 版本、target、输入和配置、命令、比较指标与结果；未测写明未测 |
| 失效条件 | 上游/SDK 修复后，经过什么回归可删除改写？ |

**状态必须带阶段。** `有实现` 是代码事实；`已验证` 需注明是 CPU 数值、图导出、
HMONNX 执行还是真机结果；`不支持` 需有明确拒绝或失败证据；没有证据写 `待验证`。
Mock 构造参数通过不代表 backend 支持所有参数组合，旧测试文件存在也不代表当前 CI 会执行。

新增回归一律进入 [当前 CI](../../benchmark_test/ci_test_transformer5.16.0/README.md)，
遵循用例与同名 `.impact.json` 的接入规格。优先覆盖真实接口的语义，必要时补小图导出；
不能把新测试继续写进历史 `tests/`，也不能只在文档中写“已覆盖”。
本库对历史用例的链接只用于解释既有设计；相应断言尚未接入当前 CI 的地方会明确标注。

## 本次整理的证据范围

初始条目核对基线为 `ee4aa01`（2026-09-06）。记录来自当前实现和测试源码，
不是一次 M50/YH/XH3 真机认证；SDK 版本相关的限制只在有版本证据时写成确定结论。
代码重构时更新引用，保留仍成立的语义；新设备复用结构页并补目标页的差异，
不复制整套模型说明或替未知设备填写“同 XH2a”。
