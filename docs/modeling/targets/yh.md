# YH：已有名称入口，能力按路径验证

[LLM builder](../../../xhmodel_merak/xh_llm/builder.py) 的 `register_llm_model()` 和
`get_model_class()` 接受 `YueHui`，与 XH2a 使用同一个模型 registry；
[旧 Merak 基础配置](../../../configs_merak/xh2a/_base_/yuehui_base.py) 也使用 `chip_arch = "YueHui"`。
这证明入口存在，不证明所有已注册模型或所有 XH2a 量化组合均支持 YH。

在 LLM 配置中核对 `export.model.chip_arch` 与安装的 SDK 标识。
other-model loader 仅要求 `export.target_device` 是非空字符串，
因此仅把它改成 `YueHui` 并通过 YAML 校验也不是后端支持证据。

适配时复用已有 [结构契约](../README.md)，逐项核对 SDK 算子 API、精度、layout、
cache 更新与 HMONNX runtime。发现差异时记录具体模型/配置、SDK 版本与图节点，
把改写放在目标配置或 adapter，避免给整个模型 family 写宽泛的芯片特例。

当前活动 CI 没有专门的 YueHui 配置/用例；算子支持范围、性能和真机结果均待验证。
新增 YH 回归仍进入 `benchmark_test/ci_test_transformer5.16.0/`，并声明实际覆盖的输入，
不新建另一条测试目录或从 XH2a 的通过记录推导 YH 通过。
