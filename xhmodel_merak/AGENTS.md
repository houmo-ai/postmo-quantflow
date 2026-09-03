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
