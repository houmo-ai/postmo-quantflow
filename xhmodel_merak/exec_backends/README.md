# 推理后端扩展

本目录将公共推理接口映射到具体 runtime，并提供 session 创建、图准备和能力声明。
公共模型层通过后端对象调用这些行为，不增加 `backend == "tcim"` 一类名称判断。
职责边界见 [Merak 架构](../../docs/architecture/merak.md#推理后端边界)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| [factory.py](factory.py) | 环境配置解析、创建后端对象、兼容 facade |
| [backend.py](backend.py) | `ExecutionBackend`、按路径延迟加载接口、V2 session 创建辅助函数 |
| [builtin_backends.py](builtin_backends.py) | 显式组合内置后端的接口映射、别名和策略 |
| [hmonnx_backend.py](hmonnx_backend.py) | HMONNX 声明、legacy/V2 选择与 session 创建、图准备委托 |
| [tcim_backend.py](tcim_backend.py) | TCIM 声明、能力限制和预编译 HMM 图准备策略 |
| [tcim_hmonnx.py](tcim_hmonnx.py) | TCIM 的 session、loader、cache 具体适配实现 |

声明模块可以定义策略函数，但不能在模块顶层导入 Torch、SDK、runtime 或加载模型。
`CLASSES` 保存导入路径字符串；实际依赖在接口解析或执行策略时加载。

## 选择与生命周期

`XHMODEL_INFERENCE_BACKEND` 默认或空值为 `hmonnx`，忽略首尾空白和大小写。
`tcim_runtime`、`tcim_lite` 是 `tcim` 的别名；未知值报错。

HMONNX 的 `ENABLE_HMINFERENCE_V2` 接受 `1/true/yes/on`，否则使用 legacy 路径。
TCIM 始终使用 V2 适配接口；这不表示它支持 HMONNX 的 CUDA Graph 或运行时图转换。

组合 runtime 在初始化时调用一次 `get_backend()`，将返回值传给子组件：

```python
from xhmodel_merak.exec_backends import get_backend
from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel

backend = get_backend()
prefill = HMONNXModel("prefill/model.onnx", backend=backend)
decode = HMONNXModel("decode/model.onnx", backend=backend)
cache_type = backend.resolve_class("CacheTensor")
```

示例路径需替换为所选后端实际可加载的产物。`ExecutionBackend` 复制接口映射，
固定后端名称及 legacy/V2 选择；能力与行为仍由策略模块提供，不应在运行中修改策略声明。
基础 LLM 已将 prefill、decode、loader 和基础 KV cache 绑定到同一对象。
自定义模型新增或替换 cache、loader、视觉/投机组件时，也需要传递该对象，不能假设自动继承绑定。

兼容 facade（如 `HMONNXInferenceV2`、`CacheTensor`）仍在调用时读取环境，
其 `isinstance` / `issubclass` 判断也依赖当前选择。已绑定的 runtime 应使用
`backend.resolve_class()` 获取具体类型，避免运行期间更改环境导致实现混用。

## 新增内置后端

1. 增加 `<name>_backend.py`，提供下表中的声明和策略。名称及别名使用非空小写字符串。
2. 将真实 session、loader、cache 适配放在独立实现模块，在 `CLASSES` 中显式映射。
3. 在 `builtin_backends.py` 导入声明模块并加入 `_BUILTINS`。三张注册表会统一生成，
   不需要修改工厂分派，也没有自动目录扫描或 entry points 加载。
4. 检查名称和别名不重复、不相互遮蔽；当前字典组装不会主动拒绝冲突。
5. 补充当前 CI 用例及 impact 声明，再验证实际 SDK 和目标运行环境。

| 声明/策略 | 契约 |
| --- | --- |
| `NAME` | 后端规范名称 |
| `ALIASES` | 别名元组，无别名时为 `()` |
| `CLASSES` | `{公共接口名: (绝对模块路径, 实现类名)}` |
| `SUPPORTS_CUDA_GRAPH` | 所选 runtime 接口是否支持 CUDA Graph |
| `SUPPORTS_AUTO_OFFLOAD` | 所选 runtime 接口是否支持 auto offload |
| `use_inference_v2()` | 返回布尔值；在 `get_backend()` 时求值并保存 |
| `prepare_graphs(prefill, decode, *, prepare, **options)` | 返回可加载的 `(prefill_path, decode_path)` |
| `create_session(backend, path, **options)` | 返回 `(session, device)` |

能力声明描述 runtime 接口，不代表某芯片在所有模型、精度和 shape 下均支持该能力。
不支持的能力不要通过接口名称或芯片名称推断。

### 接口映射

当前内置后端提供以下公共接口，具体映射可参考两个 `*_backend.py`：

- Session：`HMONNXInference`、`HMONNXGoldenInference`、`HMONNXGraphGoldenInference`、
  `HMONNXCUDAGraphInference`、`HMONNXInferenceV2`、`HMONNXInferenceConfig`。
- Loader：`LLMHMONNXLoader`、`MultiHMONNXLoader`。
- Cache：`CacheTensor`、`HybridCacheTensor`。

`HMONNXGrapInference` 是工厂保留的 `HMONNXGraphGoldenInference` 别名，无需额外映射。
映射存在不等于能力已实现；不支持的操作应在对应实现中明确拒绝。
缺失接口会在解析时报 `ValueError`，注册阶段不会验证所有接口是否完整或实际可用。

### 图准备策略

`options` 包含 `enable_parallel_linear_fusion`、`enable_page_attention` 和
`convert_unfused_sliding_kv_cache`。`prepare` 是公共 LLM 提供的现有图派生回调：
先进行 parallel-linear fusion，再进行 page-attention 转换，并返回派生路径。

HMONNX 策略调用该回调。TCIM 使用已经编译的 HMM，直接返回输入路径；
请求 fusion 或 page attention 时明确报错，不调用 HMONNX 图转换。
新后端在自己的策略中处理这些要求，不在公共模型构造函数中增加特判。

### Session 策略

`options` 包含 `onnx_graph`、`enable_cuda_graph`、`enable_auto_offload`、
`enable_golden`、`device_map` 和 `layer_infos`。
返回的 session 必须满足实际调用方的执行、设备、状态、golden 和 cache ABI；
只提供同名类或让构造成功不算完成接入。

若实现兼容现有 V2 配置与构造协议，可复用 `create_v2_session()`：
它通过绑定后端解析配置和 session 类型，按能力限制 CUDA Graph / offload，
有预加载图时调用 `from_onnx_graph()`，否则按路径构造。
该辅助函数要求调用者提供全部上述选项，不替代具体后端的能力校验。

保持当前兼容行为：TCIM 的 CUDA Graph / offload 请求被关闭；
不支持的图转换和 golden 操作明确报错。变更这些行为需单独说明并补回归。

## 验证

新增测试统一放在 [当前 CI 目录](../../benchmark_test/ci_test_transformer5.16.0/README.md)。
已有 [后端契约测试](../../benchmark_test/ci_test_transformer5.16.0/test_tcim_backend_contract.py)
覆盖选择、别名、延迟导入、扩展映射、legacy/V2、能力开关、图准备及基础 LLM 生命周期。
测试及同名 `.impact.json` 应覆盖新声明、实现和实际消费方；不要修改全局 policy 扩大覆盖。

在配置好 SDK 的环境中，从仓库根执行：

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

重点覆盖环境变量在对象创建后变化、prefill/decode 共用选择、cache 类型一致性，
以及不支持操作的错误路径。契约测试可替换昂贵的 SDK 执行边界，但必须运行真实策略与编排。
这些测试不等于实机验证；还需用实际产物核对输入输出、cache 更新、设备放置及数值结果。
