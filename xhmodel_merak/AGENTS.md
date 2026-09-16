# Merak

接入模型或修改公共机制前阅读 [架构与扩展边界](../docs/architecture/merak.md)。
具体算子、结构、目标设备的约束从 [建模知识](../docs/modeling/README.md) 按需查阅。

- 预置模型对外使用 `model_build` 的 `list_models`、`describe_model`、`build_model`
  和 `model_inference.inference_model`；接入方式见 [接口开发约定](model_integrations/README.md)。
  底层 workflow 使用 `xhmodel_merak.workflows.AutoWorkflow`，具体行为由 family 承担。
- 模型在 `model_integrations/<family>/plugin.py` 自注册；公共注册器、构建器和加载器
  不新增按模型名分支。注册阶段只声明元数据，不加载权重、Torch、SDK 或修改进程环境。
- 新模型必须随实现提交 recipe，不得等后续补统一接口。新架构包、新主模型类型和新的
  对外权重规格都适用；同一模型的精度、上下文档位或推测解码使用 recipe 参数/variant，
  不为每个 YAML 重复创建模型 ID。完整示例见 [recipe.py](../examples_merak/unified_api/recipe.py)。
- 底层注册使用字面量 `@register_llm_model("Type")` / `@register_other_model("Type")`，
  `master`、`force` 用布尔关键字，不使用别名、动态名称或运行时循环注册。
  recipe 的 workflow 必须指向实际主模型；辅助组件在所属模型包内用 `master=False`，
  不冒充独立模型。新 family 要有 `__init__.py`，只有 `plugin.py` 文件不代表已注册。
- 接入时运行 [recipe 门禁](../benchmark_test/ci_test_transformer5.16.0/test_recipe_coverage.py)，
  并补真实功能测试及同名 impact 声明。门禁反查主模型和目录，存量豁免只允许删除；
  新权重规格复用已有架构时，还必须由模型测试检查新 `model_id` 的参数、来源约束和执行链路。
- 量化接入遵循 [跨仓库职责](../docs/architecture/quantization-boundary.md)：adapter 转换配置、校验部署兼容性，
  GPTQModel/AutoRound 承担权重量化与产物写入；ModelZoo 消费产物并适配导出。
- 模型语义、processor、可复用 runtime 属于模型包；示例负责用户输入与编排。
  新代码不反向依赖示例或历史实现；既有迁移遗留不作为新模板。
- 配置、导出图与 runtime 共同维护 tensor/state ABI；调整其中一侧时核对另两侧。
- 已注册模型修改配置、组件、精度边界、预处理或产物布局时，同步核对 recipe、build/inference
  adapter、统一接口示例和当前 CI。没有 Git 文本冲突不代表接口仍兼容。
  adapter 复用模型包的 processor/runtime，不复制一套分词、切句或数值逻辑。
- `describe_model` 默认只描述可修改的构建参数；`full=True` 才给完整元数据。
  新参数必须贯通描述、解析校验、实际执行和测试；不暴露无效开关，不静默忽略请求。
- `output_dir` 是最终交付根目录，不再套发布目录；省略时在当前目录按发布名加日期创建。
  不覆盖已有目录。prefill/decode、视觉档位和 mtp_*/dflash_* 等组件平铺，运行资源随包保留。
  从运行元数据和 ONNX external_data 引用确定交付文件，不把整个导出工作区写入 Manifest。
  临时图、逐层导出工作文件、评测报告和日志不随构建交付。推理允许在包内新增普通缓存或
  临时文件，不要求目录与 Manifest 完全相等；已登记文件仍检查完整性，路径安全规则不变。
- 分层导出由 `VariantRecipe.export_modes` 登记，第一项为默认模式。公共 `low_memory`
  参数只能选已登记模式，不能直接透传用户环境变量；executor 必须在结束或异常时恢复环境。
  不得仅因模型继承某个方法或接受开关就声称验证通过，须验证主图回填、draft 图、状态 ABI
  和峰值内存。LoRA 等不支持的组合在解析阶段拒绝，不要导出一半才报错。
- 低内存导出沿用同一模型语义，只改变权重加载和导出调度。占位模块不持有真实权重；
  与真实模块共用静态输出布局，明确 Q/K/V cache 的顺序、数量及 recurrent state 是否返回。
  prefill、单 token decode、多 token verify 分别验证；不对 FX Proxy 求长度或靠截断掩盖不一致。
- 主图回填同时处理可执行图和量化元数据。SDK 元数据未记录 tuple 索引时，必须在 FX
  索引仍可用的阶段解析成显式端口，不能按节点顺序或跨图名称猜接线。改动先跑当前 CI 中
  的小 GDN 导出/拼装/推理及 tuple 边界用例，再跑真实权重；小用例通过不等于整模型通过。
- 组件图精度通过 recipe 的 `deployment_precision_bindings` 声明实际 QuantScheme 路径、
  可选值及共用配置的组件；默认值取所选 variant/context 的 YAML。覆盖只改基础 `quant_type`，
  保留节点/算子例外，不改已打包权重。构建 Manifest 与最终有效配置必须一致，推理时复核。
- `quant_methods` 是可执行的量化流程；`quant_checkpoint_methods` 是导出器可读取的已有权重来源。
  复用权重不运行量化、不要求安装量化生产工具；仍检查来源、位宽、模型身份、张量及资源完整性。
- checkpoint 的 embedding 命名、配置中声明的附加层由 family 的布局字段表达；公共检查器
  不按模型名硬编码，也不把 Q/K、MLA 等内部 norm 当成 hidden-size norm。
  base 构建允许携带已声明的辅助层，不代表统一接口已经支持该模型的 MTP 导出或推理。
- SDK 依赖和能力必须对应真实接口，并在实际使用阶段检查；不把计划中的 SDK 开关
  登记为所有阶段的硬依赖。运行时缓存与交付文件分开，规则见 [接口接入约定](model_integrations/README.md)。
- 新芯片优先复用结构契约，通过目标配置和 xhquant 算子适配表达差异。
  不复制整套模型/workflow，也不把未实现的目标登记为已支持。

## Kokoro / M50 路径的 LSTM

- 当前 Kokoro `XH2a` 配置使用单层、batch-1、`batch_first` 的双向 LSTM 改写。
  前向分支保持输入顺序；反向分支复制原模型的反向权重，用有效前缀 index 执行
  `Gather → forward LSTM → Gather`；恢复时间顺序后，沿特征维按 `[前向, 反向]` 拼接。
- index 由 Host 按本次真实 token / frame 长度分别生成，只反转有效前缀，不翻转 padding，
  不固化为导出样例的长度。它是普通 Gather 输入，不是 LSTM 私有参数，也不是 `sequence_lens`。
  调用方保留输出 mask；当前改写不承诺最终 `h/c` 与变长 LSTM 相同，不直接用于流式状态。
- 浮点参考 ONNX 保留标准 forward LSTM；xhquanttool 保留 LSTM 语义到量化阶段，
  Kokoro 显式要求导出阶段 `decompose_lstm=True`，发布 HMONNX 不得残留 LSTM。
  不在模型侧手工展开递归或重排量化运算顺序；SDK 已有三种方向及标准有效长度的实现，
  不把 Kokoro 的写法强制套用到所有模型，不把 native 图表达等同于 M50 真机支持。

实现见 [graph.py](xh_other_model/models/kokoro/graph.py) 和
[quant_export.py](xh_other_model/models/kokoro/quant_export.py)；
导出回归见 [test_kokoro_release.py](../benchmark_test/ci_test_transformer5.16.0/test_kokoro_release.py)。

新增回归用例进入 [当前 CI](../benchmark_test/ci_test_transformer5.16.0/AGENTS.md)，
并声明被测实现、配置及共享依赖的 impact 范围。
