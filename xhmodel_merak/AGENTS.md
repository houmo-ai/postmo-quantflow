# Merak

接入模型或修改公共机制前阅读 [架构与扩展边界](../docs/architecture/merak.md)。
具体算子、结构、目标设备的约束从 [建模知识](../docs/modeling/README.md) 按需查阅。

- 对外 workflow 使用 `xhmodel_merak.workflows.AutoWorkflow`，具体行为由 family 承担。
- 量化接入遵循 [跨仓库职责](../docs/architecture/quantization-boundary.md)：adapter 转换配置、校验部署兼容性，
  GPTQModel/AutoRound 承担权重量化与产物写入；ModelZoo 消费产物并适配导出。
- 模型语义、processor、可复用 runtime 属于模型包；示例负责用户输入与编排。
  新代码不反向依赖示例或历史实现；既有迁移遗留不作为新模板。
- 配置、导出图与 runtime 共同维护 tensor/state ABI；调整其中一侧时核对另两侧。
- 新芯片优先复用结构契约，通过目标配置和 xhquant 算子适配表达差异。
  不复制整套模型/workflow，也不把未实现的目标登记为已支持。

新增回归用例进入 [当前 CI](../benchmark_test/ci_test_transformer5.16.0/AGENTS.md)，
并声明被测实现、配置及共享依赖的 impact 范围。
