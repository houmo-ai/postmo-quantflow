# XH3：接入边界与待确认项

当前 [Merak LLM builder](../../../xhmodel_merak/xh_llm/builder.py)
只接受 `XH2a` / `YueHui`，没有 XH3 分支；当前 Merak 配置和活动 CI 也没有对应实例。
这说明本仓库接入尚未建立，不是对 XH3 硬件本身能力的判断。

| 需要落实的边界 | 当前结论 |
| --- | --- |
| SDK target 标识、版本与注册 | 待确认；不能猜 `XH3`/`xh3` 字符串或静默映射到 XH2a |
| 算子 dtype/layout、量化/打包及图 lowering | 待确认；不继承 M50/YH 的限制或能力 |
| HMONNX 导出、metadata 与 cache/runtime ABI | 待确认；应复用模型语义并显式适配实际差异 |
| 当前 CI 的 XH3 回归 | 尚无；新增用例及 impact 声明仍归属当前 CI 目录 |

后续接入先选择一个代表性结构和已确认 SDK 路径，通过小图/接口建立证据，
再扩展设备配置与模型覆盖。沿用 [Merak 的职责边界](../../architecture/merak.md)：
模型定义语义，目标 adapter/工具链选择图实现，runtime 遵守已导出的状态契约。
不为尚未确认的能力建立空实现，也不要求为了预留 XH3 先重写现有模型。
