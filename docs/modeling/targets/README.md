# 目标设备：名称、路径与证据

设备名、计算设备、量化格式和已验证能力分开记录。
`cuda:0` 是执行导出的计算设备；配置里的 target/chip_arch 是部署工具链的选择。
模型能在 CUDA eager 运行，不说明其图可在目标芯片运行。

| 用户侧名称 | 当前仓库可证明的入口 | 适配说明 |
| --- | --- | --- |
| M50 / Merak | README 将其与天璇架构及 XH2a/YueHui 关联；主要 workflow 配置使用 `XH2a` | [M50](m50.md) |
| YH | LLM builder 接受 `YueHui`；旧 Merak 基础配置存在该标识 | [YH](yh.md) |
| XH3 | 当前 Merak LLM builder 没有此分支，未找到对应 Merak workflow/CI 用例 | [XH3](xh3.md) |

证据：[README_merak](../../../README_merak.md)、
[LLM builder](../../../xhmodel_merak/xh_llm/builder.py)、
[当前 CI](../../../benchmark_test/ci_test_transformer5.16.0/README.md)。
M50 的产品称呼不在这里自动归一化成一个 SDK 枚举；使用时以具体 workflow/SDK 为准。

同一算子的支持记录应关联 target、SDK/runtime 版本、shape/layout、精度、执行模式与验证阶段。
`有注册入口`、`已导出`、`HMONNX 数值通过`、`真机通过` 是不同证据。
当前没有覆盖三种目标的统一能力表；未登记的组合是待验证，不继承其他目标的结论。

新增目标时补一页差异并链接已有算子/结构页。代码改动遵循
[多芯片扩展边界](../../architecture/merak.md)，在现有 family/SDK 接口上落实，
需要回归时把用例及 impact 声明放到当前 CI。
