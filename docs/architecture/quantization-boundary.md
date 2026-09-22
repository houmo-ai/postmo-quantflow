# ModelZoo、GPTQModel 与 xhquant 的开发边界

ModelZoo 负责把模型接入部署工作流，GPTQModel/AutoRound 负责生成可消费的量化权重，
xhquant 负责图和算子的量化、目标实现与 HMONNX 导出。下面区分职责要求与现有实现；
并非所有模型已经使用统一 recipe，也不把全部“量化”都归到 GPTQModel。
整体调用关系见 [Merak 架构](merak.md)。

## 功能归属

| 工作 | ModelZoo | GPTQModel / 对应量化后端 |
| --- | --- | --- |
| 用户入口与配置 | workflow YAML、`quant/export` 编排、路径解析；把部署需求映射为公开 recipe 参数 | 维护 recipe/API 及参数语义、算法默认值、实际执行和执行记录 |
| 量化侧模型适配 | 选择模型、变体和所需精度，校验与后续部署的兼容性 | HF 加载、量化模块树/遍历顺序、激活采集、MoE 校准适配、量化时权重变换 |
| 校准与求解 | 提供数据来源、样本数、长度、seed 等明确要求 | 构造有效校准样本并执行 GPTQ 求解、AutoRound 优化、量化侧内存/offload 调度 |
| 量化产物 | 消费 checkpoint；按已有格式解包/反量化，再转换为部署 wrapper 所需结构 | 维护 `qweight/qzeros/scales/g_idx` 等格式语义、打包、HF 配置与必要辅助文件，保证自身可重载 |
| 部署适配 | graph wrapper、RoPE/mask/cache、图切分与档位、Host processor、metadata、HMONNX runtime 编排 | 提供可表达的量化精度/格式；算法能力不直接宣称为芯片能力 |
| 正确性证据 | 参数接入契约、产物消费、导出与部署数值/状态语义 | 量化算法精度、校准行为、保存/重载一致性、量化后模型验证 |

xhquant 承担表中部署阶段的算子实现、图量化和 target lowering；ModelZoo 选择并调用它。
例如 GPTQ 的权重 bits 与 FlashAttention 的 Q/K/V/score/probability 精度不是同一个开关。
新增 M50/YH/XH3 路径首先调整部署配置、图实现和 runtime 契约；
当权重量化能力或产物格式也需要变化时，再扩展量化库的对应接口。

## 两边都做模型适配，按用途决定修改位置

- **为了正确量化上游模型**：新增量化模块注册、校准 hook、expert 遍历/拆分、
  rotation 数学变换或通用 checkpoint 写入，归 GPTQModel/对应后端。
  这些能力应能由量化库独立调用，不反向导入 ModelZoo workflow 或依赖 Merak 示例目录。
- **为了正确部署已有产物**：tensor/state ABI、cache 更新、图可追踪的 norm/attention、
  将已量化 expert 转成 `MoeBlock` 输入，以及 Host/NPU 边界，归 ModelZoo 与 xhquant。
  产物读取/转换是必要的消费适配，不因其中出现 `gptqmodel` 就要求整段迁到量化库。
- **为了匹配用户要求**：ModelZoo 的 quant adapter 保留字段转换、资源解析、结果封装和
  部署兼容性检查。例如 Qwen 的 rotation 与 MTP 组合由消费侧提前校验；
  rotation 的算法实现和量化产物正确性仍由量化侧负责。

量化质量异常可先在量化库保存/重载后复现；若原产物正常而导出后出现误差，
再沿 ModelZoo 的反量化/结构转换和 xhquant 图路径定位。修复应落在首次破坏契约的位置。

## 跨仓库以 API 和产物交接

输入使用公开 recipe 的命名参数，包含模型、算法、精度、校准和执行要求；
部署方案选择的值由 ModelZoo 显式传入，算法默认值由后端维护，避免两侧默认值悄悄分叉。
必要的结构/模态信息可以传递，不把整份 Merak workflow 配置变成量化库的隐式依赖。

recipe 的输出目录交给 ModelZoo 的 [QuantResult](../../xhmodel_merak/xh_llm/workflows/result.py)，
再由 `export()` 消费；GPTQModel 的 recipe result 与 `QuantResult` 是不同类型。
量化产物配置描述量化格式和模型，HMONNX metadata 描述部署图与状态，职责分别维护。
兼容性记录至少关联双方版本、量化格式/参数和验证结果；不通过目录名字猜格式或能力。

**签名兼容不等于参数生效。** 参数新增/改名应同步核对 adapter、recipe、实际后端与测试；
不靠过滤未知 kwargs 或成功返回来掩盖不匹配。当前 Gemma recipe 将部分兼容参数记录到
`provenance.ignored_compat_args`，`export_subgraphs` 等也可仅作为执行记录；
这些字段被接受，不代表后端已执行相应量化策略或 HMONNX 导出。
新增实际能力时，后端需要落实参数行为，消费侧需要校验所要求的能力并给出相应证据。

## 当前实现与需要收敛的地方

| 现有路径 | 当前事实与后续归属 |
| --- | --- |
| [Qwen3.5 quant adapter](../../xhmodel_merak/xh_llm/models/qwen3_5/quant_adapter.py) | GPTQ 调用 `gptqmodel.recipes.qwen35:quantize_qwen35`；AutoRound 调用 `qwen35_autoround:quantize_qwen35_autoround`，adapter 封装 `QuantResult` |
| [Gemma4 quant adapter](../../xhmodel_merak/xh_llm/models/gemma4_series/quant_adapter.py) | GPTQ 使用 recipe 并检查签名；已有 AutoRound mode1 仍映射到外部脚本 CLI，不能写成所有路径均已统一到 Python recipe |
| [公共 GPTQ Linear 转换](../../xhmodel_merak/xh_llm/base_llm_model.py)、[反量化转换](../../xhmodel_merak/xh_llm/_dequant_converter.py) | 将量化载荷转为可导出的权重/模块并保留所需量化信息，是 ModelZoo 的消费适配；格式真值由生产端定义 |
| [Ling 量化适配](../../xhmodel_merak/xh_llm/models/ling_3_flash/quant_adapter.py) | GPTQ 与 AutoRound 都调用 `gptqmodel.recipes.ling3_flash:quantize_ling3_flash`；后端先吸收浮点 MLA，再量化、保存新布局。ModelZoo 复用整数载荷，保留 K/V 独立 SEFP cache；边界见 [专项说明](../modeling/structures/ling-mla-quantization.md) |
| [Laguna 兼容模块](../../xhmodel_merak/xh_llm/models/laguna/gptqmodel_compat.py) | 补 GPTQModel 注册表的 `register_laguna_gptqmodel()` 是量化侧接入遗留；同文件中的 `restore_gptqmodel_moe_structure()` 是消费侧结构恢复，两者不按文件名一并迁移 |

新增接入优先使用已存在的公开 recipe；缺少能力时在量化库实现接口，再由 ModelZoo 对接。
迁移旧兼容逻辑需先有可用的后端版本与消费回归，不能仅因文档划了边界就删除仍在用的修补。
各路径的迁移状态以上表为准，不能由某个 recipe 已接通推断其他模型也已完成。

## 主输出 head 的校准与复用

Qwen3.5/Qwen3.6 仅在 export head 小于 8 bit 时要求独立校准，并校验
`quant.head.bits` 与导出 head 位宽一致。现有 W8 YAML 无需新增字段；W8 head
允许保留浮点，由导出侧沿用 RTN SEFP。此规则不依赖主体是否已经量化。

原生 HF 输入使用原有完整 recipe；主体与 head 的阶段划分是 GPTQModel 内部实现，
不增加 workflow scope。完整流程最终输出一个 checkpoint。tied head 先复制为独立
Linear，保留 embedding 数值；保存时同步解除配置绑定，重载不得重新共享。

已有 GPTQModel 产物沿用现有跳过量化入口。workflow 保留原 YAML 的 head 配置：
兼容的 packed head 直接复用原路径；浮点 head 通过 GPTQModel 补校准，并 warning
提示同级 `<输入目录名>_head_w{bits}` 新路径及后续复用方式。输出已存在时不覆盖。
packed head 不符合要求时拒绝自动二次量化。是否兼容由量化库检查实际 head 张量与
metadata，ModelZoo 不通过目录名称猜测格式。

head 算法默认跟随主体生产算法；AutoRound 产物的 GPTQ 格式标签不代表 GPTQ 算法。
采集 head 输入时使用已量化主体的实际前向、原始 MoE 路由与 final norm。
GPTQModel 负责解绑定、校准、分块内存控制和 checkpoint 保留；ModelZoo 只负责
配置映射、quant/export 一致性、路径选择和结果封装。跳过主体不能跳过低位宽 head
校验；直接调用 export 也检查实际输入产物。

本地 W5 使用 8 bit 存储槽位、5 bit 数值范围。GPTQModel 提供 `logical_bits=5`，
ModelZoo 将其与 packed 张量传给 xhquant，图精度按逻辑位宽选择，解包按存储位宽
进行。该契约不引入官方 `gptq_p` 布局。用法见
[Qwen MoE README](../../examples_merak/llm/qwen3_5_moe/README.md#qwen36-35b-a3b-head-w4--w5)。

## Qwen MoE 自动混合精度

`quant.auto_mixed_precision` 是可选的 recipe 输入。ModelZoo 仅做字段/数据路径映射，
搜索、候选数值近似、整模 KL/NLL 评估、静态分配和最终量化由 GPTQModel 完成。
搜索以 block 的 attention 主投影、shared expert 为组，W4/W5/W8 可选；非共享专家
由 `quant.moe` 固定配置。搜索采用文本和可选图像回答的 RTN 筛选，加既有 AutoRound/GPTQ 最终校准，
不把筛选阈值解释为最终模型或多模态任务精度承诺。

图像 JSONL 解析、原生 processor/视觉编码/M-RoPE、回答位置掩码和缓存由 GPTQModel
完成；ModelZoo 仅解析 `image_dataset` 等资源路径。文本和图像分别满足 KL/PPL 预算，
不能合并平均值掩盖某一模态退化。视觉和 embedding 未参与搜索，可缓存原生语言模型的
真实输入；缓存必须以未缓存的原生图像前向验证。最终校准数据仍单独配置。

`auto_mixed_precision` 与 `head.bits` 属于同一份 quant 配置。一次 recipe 先搜索并锁定
block 位宽，恢复浮点权重后执行主干 AutoRound/GPTQ，再用已量化主干的输出激活
校准浮点 head。最终输出目录包含主干、head 和位宽表，内部临时主干由 recipe 清理；
ModelZoo 不把它拆成多个配置或要求用户手工补量化。Head 默认继承主干算法。

producer 在 checkpoint metadata 和 `auto_mixed_precision.json` 中记录逻辑位宽，
head 补量化须保留分配。消费侧校验 producer 的位宽/packed 张量契约并为每个目标
投影生成导出配置，W5 不得被默认 W8 覆盖；已有 checkpoint 不因跳过量化而跳过
分配校验。原 YAML 不开启该字段时保持旧行为。用法见
[Qwen MoE README](../../examples_merak/llm/qwen3_5_moe/README.md#自动-block-混合精度)。

## Qwen MoE MTP 专家补量化

`quant.mtp` 是显式启用项，旧 YAML 不变。GPTQModel recipe 负责从传入主模型
（可包含已量化主干和 head）采集 post-norm hidden，与 next-token embedding
对齐，再经过 MTP 融合 FC、attention 和 norm，校准非共享专家的三矩阵。
算法默认沿用 head 补量化的 `inherit` 识别规则：AutoRound 主体使用 AutoRound
联合重建，GPTQ 主体使用 true-sequential GPTQ。固定 W4/G64/int_sym_xh，
不参与主干混合精度搜索。

生产端以独立 `mtp_quantization` 元数据及 canonical GPTQ V2 专家载荷保存到
新目录，主干和主 head 的张量与配置不变。已量化 MTP 按真实载荷校验后复用。
ModelZoo 只读取该契约、保留整数码、绑定匹配的导出位宽；MTP 其余主干沿用
RTN SEFP W8。已打包主 head 必须读取其实际载荷，不能对 untied head 回退到
embedding。producer 的 teacher-forcing 验证与 runtime 接收率分别记录。

## Qwen target-owned head

Qwen3.5 dense/MoE（包括同架构的 Qwen3.6/3.8）和 Qwen3-Next 共用
`xh_llm/shared_head.py` 的配置及 checkpoint 消费规则。主模型 export 的
`lm_head` 是唯一位宽来源；低于 W8 时仍须匹配 `quant.head.bits`。
MTP、DFlash、DFlash2 继承该值；显式旧 draft 位宽或 head 节点覆盖不一致则报错。
Qwen3-Next 当前仅有 MTP 接入，这条共享规则不意味着新增 DFlash 架构支持。

导出时先完成主模型 head 的量化，draft 在 PTQ 前绑定其固定量化算子，复用完整
整数码、scale/指数、对称零点语义及权重格式。W8 也不再对 draft head 独立 RTN。
不同图保留独立算子元数据，共享载荷；非共享专家、attention 的量化规则不由 head 改写。
低位宽 GPTQ head 读取实际 packed 载荷，只有 tied 浮点模型允许 embedding 回退。
裁剪词表导致 head 形状不一致时拒绝共享，不隐式替换为另一份 head。

新产物携带 `spec_decode.shared_head` v1（owner=target、logical_bits）。V2 验证入口
及 vLLM Merak 插件检查主图/draft 的格式和张量一致后绑定同一存储，并在执行前检查
地址，不能以数值相等替代共享。没有该字段的旧产物保持原加载语义。

标准 W5＋MTP 及混合精度＋W5＋量化 MTP 入口见
[Qwen MoE README](../../examples_merak/llm/qwen3_5_moe/README.md#共享-head-与-mtp-标准入口)。
Qwen 与 Gemma4 的权重共享范围不同，见
[推测解码权重所有权](speculative-weight-ownership.md)。

## 验证边界

ModelZoo 的新增测试统一位于 [当前 CI](../../benchmark_test/ci_test_transformer5.16.0/README.md)，
遵循同名 impact 声明及最小真实模型规则；GPTQModel 内部算法测试遵循其自身仓库规范。
跨仓库接入参考当前 [recipe contract](../../benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py)
及 [impact 声明](../../benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.impact.json)：
它们检查 YAML → kwargs、真实签名、dry-run 生成命令与后端 parser、workflow 接口及契约失配。
这不执行量化，也不证明所有接受的参数已生效。

需要验证产物格式时，优先由小 config 构造模型，保留受影响模块并检查保存/重载与消费转换；
必要的真实量化 checkpoint 复用现有固定 block 产物。ModelZoo PR CI 不重复完整量化算法验证。
ModelZoo selector 只看本仓库变化；单独升级 GPTQModel 不会凭空产生 ModelZoo 的 impact 命中，
跨仓库接口或依赖版本变化时要显式运行相关契约与必要的消费测试。

本仓库源码核对基线：`ee4aa01`。GPTQModel 侧核对 `d083cf0f` 的
`gptqmodel/recipes/{qwen35,gemma4}.py`、`gptqmodel/models/{base,loader,writer}.py`、
`gptqmodel/models/definitions/gemma4.py` 与 `gptqmodel/quantization/gptq.py`（路径均相对 GPTQModel 仓库）。
这是当前项目使用的源码事实，不把公开上游包名或版本号当作这些 recipe 一定存在的保证。
