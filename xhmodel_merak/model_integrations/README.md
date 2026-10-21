# 统一接口接入约定

用户通过 `model_id` 找模型，调用 `build_model` 得到可搬移的目录，再把目录交给
`inference_model`。接入代码负责把这几个动作接到模型已有实现上，不另写一套模型。
调用示例见 [unified_api](../../examples_merak/unified_api/README.md)。

## 新模型放哪里

同一个执行契约能复用就放进现有 family；确实不同才新建目录：

```text
model_integrations/<family>/
  __init__.py    # 保持轻量
  plugin.py      # 声明模型、参数、组件、依赖和 adapter
  build.py       # 整理导出结果、打包资源、校验产物
  inference.py   # 校验输入，调用模型 processor/runtime，返回结果
  artifact.py    # 构建与加载共用的模型产物校验，有需要才加
```

`plugin.py` 提供唯一的 `PROVIDER_ID` 和 `register(catalog)`，用
`register_recipe`、`register_adapter` 注册；新增产物格式才调用
`register_artifact_format`。目录会自动发现，不往中心注册表追加模型名单。
不要反向导入示例，也不要把另一 family 的内部实现当公共接口。

注册阶段只用轻量声明。adapter 的 `target` 写模块路径字符串，等构建或推理时加载。
`list_models()`、`describe_model()` 在没有 Torch、SDK、模型权重时也应可用，
不能靠注册模块下载资源或改环境变量。环境设置必须在执行阶段限定作用域并恢复。

可选 `config_resolver` 是轻量规划 hook，接收 `recipe`、`variant`、`model_dir`、
`sources`、`workflow`、`quant_checkpoint`、`precision`，返回 Workflow 点路径替换。
只读来源配置，不加载权重或执行校准。先应用 family 策略，再解析用户的组件精度覆盖；
已有 checkpoint 的模型身份、实际载荷和格式由执行阶段继续验证。
Qwen3-Next 复用 checkpoint 构建 MTP 时，MTP 权重必须包含在同一 checkpoint 中；
不从原始模型拼接 draft，避免旋转后的主模型与未旋转 MTP 混用。

### 组件精度与已有权重

`VariantRecipe.export_modes` 登记导出实现，第一项是默认值。只有实际实现分层导出的变体
才能列出 `low_memory`；仅有环境变量开关不算支持。公共 `low_memory=True/False` 选择模式，
不传时按变体默认值，不继承父进程开关。Qwen 的普通 LoRA 登记 `standard` 和 `low_memory`；LoRA + MTP 仍只登记
`standard`。专家 LoRA 复用 Qwen3.5 与 SDK MoeBlock 的量化导出，Macaron 只提供
配置和 recipe，不另建导出后注入流程。Ling 默认
`low_memory`，Gemma4-31B 的 full/MTP 经真实权重低内存构建与推理验证，默认 `low_memory` 且仍可选 `standard`；其余 Gemma 只登记 `standard`。这不是推理低内存选项，也不是实测通过标记；
新增模式需验证最终主图无 placeholder 残留、draft 图完整、推理状态 ABI 和峰值内存。
Qwen3.8 的 DFlash2 公开变体、`speculative.mode` 和 source key 均为 `dflash2`；
Qwen3.5／3.6 的 DFlash 保持 `dflash`。两者不能互作别名，规划时校验 draft 架构；
产物的组件名、依赖 profile 和推理返回 mode 保留该区别，底层复用 `dflash_config`。
投机解码的低内存导出必须沿用主图的多步验证配置，不能只调用普通 `set_decode()`。
逐层子图也要保留每个验证步的卷积缓存和 recurrent state；切回 prefill 时恢复配置，
避免把验证步数或中间输出标志带到 prefill。需要覆盖主图配置交接、子图拼装和真实 MTP/DFLASH 推理。

`precision` 描述 checkpoint 权重，`deployment_precision` 选择导出图的基础 QuantScheme，
两者不互相改写。例如 `deployment_precision={"visual": "w8a16h1_sefp"}` 只调整视觉图；
未指定组件沿用所选 variant/context YAML 的配置，节点、算子级精度例外保持不变。

family 用 `DeploymentPrecisionBinding` 声明组件、实际 `export.model.*.quant_scheme` 路径和
可选值。prefill/decode 共用一个 schema 时写在同一个 binding，调用方改其中之一就同时生效，
不能传两个冲突的值。audio/video 的继承配置要在改父配置前固定下来，不能顺带改变未请求的组件。
没有 binding 的组件不接受覆盖。新增选择须经过实际导出和数值验证，不能只改描述文字。

固定尺寸且可选择图精度的 pipeline 可在 recipe 声明 `artifact_name_template`，
用一个 `{deployment_precision}` 占位符生成产物名。规划和加载均使用已解析的图精度；
相同精度去掉 `_sefp` 后缀，不同组件精度用 `wmix_amix`。不要把 checkpoint 的
`precision` 用来给这种产物命名，也不要在公共规划器中增加模型名称分支。
模板与 `fixed_artifact_name` 互斥，防止同一 recipe 出现两个命名契约。

`quant_methods` 只列本模型能运行的量化流程；`quant_checkpoint_methods` 列导出器能读取的
已有 checkpoint 来源。传 `quant_checkpoint` 时默认从元数据识别来源，不重跑量化。
显式传了 `quant_method` 就检查是否匹配。来源识别不替代张量、位宽和模型身份校验。
配置比较只接受可证明的等价序列化：例如两边 `rope_parameters.partial_rotary_factor`
明确一致时，允许量化配置在外层重复写相同值；缺失证据或数值冲突仍拒绝，不猜隐式默认值。
组合模型的子模型 checkpoint 可由 family 的 `checkpoint_config_projection` 提供同一原生配置
视图后校验身份与行为；完整 Host 配置与运行资源仍按原始模型打包。
上游只改了子配置的类型名时，在 family recipe 的 `config_type_aliases` 声明
`(完整字段路径, 原类型名, 新类型名)`，例如 Qwen 的 `vision_config.model_type`。
必须用该模型的原生配置类证明两者构造出相同配置，并覆盖错误类型和尺寸变化的拒绝测试。
不能忽略所有 `model_type`，也不能把模型名称分支放进公共校验器；源权重检查和产物打包共用这份声明。

checkpoint 的命名差异通过 `checkpoint_embedding_suffixes` 声明；主层后还有 MTP 层时，
用 `checkpoint_auxiliary_layers_field` 指向原始模型配置里的数量字段。主层仍必须完整，
超出声明数量的层仍拒绝。base 导出允许保留这些辅助权重，不等于登记了 MTP 能力。
公共尺寸检查只把直接属于 decoder 层的残差流 norm 与 hidden_size 对齐；
Q/K norm、MLA 内部 norm 等形状由具体模型加载和导出检查，不能一概当成 hidden_size。

### 依赖与运行时缓存

`dependencies` 是唯一包版本声明，按 common、build、inference、quantization、checkpoint_build 分阶段，
变体可补充自己的依赖。公开档案、requirements 导出和 worker 前置检查使用同一合并结果。
`worker_capabilities` / `quant_capabilities` 只检查运行入口能力，不另行声明包版本。
能力函数必须在支持的 SDK 中真实存在，并验证调用结果；不要为接口假设一个尚未实现的
SDK 函数或环境开关，也不要让纯推理的可选优化阻断构建。
视频文件还需实际解码验证，不能只检查解码包是否安装。TorchCodec 与 PyTorch 必须
遵循[官方兼容表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)；
例如 PyTorch 2.8 对应 TorchCodec 0.6/0.7，不是 0.10。视频解码依赖不应阻断文本或图片调用。

构建只交付运行必需文件；推理允许在交付目录新增普通缓存或临时文件，不把它们补进 Manifest。
首次和再次推理都校验已登记文件的存在、大小及摘要，但不要求目录文件清单与 Manifest 完全相等。
新增文件不放宽包内路径、软链接和特殊文件的安全检查，也不能覆盖已登记的模型文件。
统一入口不设置 SDK 摘要缓存开关，不依赖额外的 SDK 缓存能力；缓存读写和权重去重由 SDK 负责。

## 新模型怎样才算接入完成

实现、recipe 和测试在同一个变更里交付。只有底层 `register_llm_model`、
`register_other_model` 或一个能导出的脚本，不算对外接入完成。

1. 在模型包实现主模型、workflow 和 runtime。底层注册用字面量装饰器，
   如 `@register_llm_model("MyModelForCausalLM")`；不要用动态名称、别名或循环注册。
   `master`、`force` 使用布尔关键字。辅助组件放在所属模型包，用 `master=False`。
2. 提供能实际执行的 Workflow YAML。`export.model.model_type` 指向 LLM 主模型，
   `export.model.type` 指向 other-model 主模型，只能选一个。
3. 在 family 的 `plugin.py` 注册 `ModelRecipe`。已有 family 直接扩展；新目录要有
   `__init__.py`、唯一 `PROVIDER_ID` 和 `register(catalog)`，不能只放一个 plugin 文件。
4. 复用适用的 build/inference adapter；有新行为才实现新 adapter。组件、精度、资源、
   来源约束、输入输出必须写实际支持范围，没做的能力不登记。
5. 提供统一接口调用示例、当前 CI 用例和同名 `.impact.json`，覆盖模型实现、配置、
   plugin、adapter 和共享依赖。至少验证查询、参数拒绝、构建与加载契约及实际数值/ABI。

新架构、新主模型类型、新的对外权重规格都需要 recipe。同一模型增加精度、上下文档位、
MTP/DFLASH 或 LoRA，扩展已有 recipe 的参数或 variant；不为每份调试 YAML 注册一个模型。
视觉编码器、声码器等内部组件随主模型交付，不单独伪装成用户模型。

### 可以照着写的例子

[recipe.py](../../examples_merak/unified_api/recipe.py) 展开了 Ling 的完整声明，包含
`ModelRecipe`、`VariantRecipe`、延迟加载的 inference adapter 和 `register(catalog)`。
它复用现有 autoregressive 构建/产物契约，没有重写模型或导出器。

```bash
python -m examples_merak.unified_api.recipe
```

示例只创建独立 catalog，打印完整说明，不导出模型，也不修改 `list_models()` 的正式结果。
CI 会核对示例与实际 Ling recipe 一致，防止示例过期。接入其他模型时，改成自己的模型 ID、
Workflow、精度、组件、资源和来源约束；不要沿用 Ling 的数值配置。
非自回归模型参考 [Kokoro plugin](kokoro/plugin.py)；构建/推理差异仍留在 family 内。

### CI 具体拦什么

每次 CI（包括空 diff 和文档修改）都跑 `test_recipe_coverage.py`：

- 从两套模型目录和生产用 AST 注册扫描器建立实现清单，不导入模型、Torch 或 SDK。
- 从真实自动发现的 recipe 读取所有 variant/context Workflow，反查主模型类型。
  新模型目录没有 recipe、已有目录新增主模型但没 recipe、Workflow 不存在或目标无效，都失败。
- 目录扫描不依赖装饰器，所以忘写底层注册也过不了；动态注册被明确拒绝，不静默漏掉。
- 检查不依赖 Git 历史，只有源码和 changed-path 文件的 CI checkout 也能执行。

例如新增 `xh_llm/models/my_model/` 却只写了导出测试，会报：

```text
New Merak models require a registered ModelRecipe:
  xh_llm/my_model/
  xh_llm/my_model:MyModelForCausalLM
```

`recipe_legacy_gaps.json` 冻结的是本门禁引入前的缺口：40 个主模型类型，以及 38 个未被
recipe 使用的目录（包括被新实现替代的旧目录）。目录项不是整个 family 的通配豁免；
即使已有目录在清单中，新加主模型类型也会失败。清单只减不增；旧模型接入或删除后，
CI 会要求删掉失效项。不要为过 CI 添加例外，更不要登记没有实现的占位 recipe。

这个门禁证明的是“实现已接入目录”，不是数值正确。同一架构新增一套权重，静态检查无法
判断它是否应成为新的产品模型，必须由该模型的测试检查新 `model_id` 和来源约束。
功能覆盖仍由原 impact 门禁检查；recipe 门禁不认领整片模型目录来代替真实回归。

本地先跑接入检查，再预览受影响用例并执行 CI：

```bash
python -m pytest -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_recipe_coverage.py -q
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

### 这里参考了什么

[Transformers 的 check_repo](https://github.com/huggingface/transformers/blob/main/utils/check_repo.py)
从模型实现反查 Auto 注册和测试，避免“只检查名单里的模型”。我们沿用这个检查方向。
[timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_registry.py)
把注册和默认配置放在模型附近；[vLLM 的插件注册](https://docs.vllm.ai/en/latest/contributing/model/registration/)
支持字符串延迟加载。这里保留 family 自注册和轻量元数据，不照搬它们的中心模型表。
实现独立编写，没有复制第三方代码。

## 参数怎么加

可选的具名输入目录由 `VariantRecipe.optional_sources` 声明，通过统一
`build_model(sources={...})` 传入；省略不构建该组件。它与 `required_sources`
互斥，未知名称仍拒绝。family 的 config resolver 负责验证组件身份和互斥选择，
并将实际目录绑定到 workflow；来源指纹和发布仍由公共构建事务维护。
FLUX 的 `prompt_upsampler`、`upsampler_bundle` 和 `image_bundle` 是组合构建示例。

先区分用户选择和部署约束。可选精度、档位、来源和资源放进 recipe；只有一种合法
取值的设置不必变成开关。新的通用参数需要确实有跨模型含义，模型私有参数不要塞进
公共入口的任意 `**kwargs`。

`describe_model(model_id)` 返回可修改构建参数的说明，包括默认值、选项和约束；
它不是能直接展开成 `build_model(**result)` 的参数字典。
`full=True` 还返回任务、部署精度、组件、输入输出 schema 等完整说明。
推理输入由 `input_schema` 描述，不与构建参数混用。
非投机/LoRA 的 profile 与已注册 variant 同名时，选择该 variant；其他 profile 保留其原有档位含义。

加参数时一起改声明、描述、解析校验、配置映射、实际调用和测试。传下去了不等于
生效了，测试需要检查消费方实际收到的值和行为。unsupported 值明确报错，不能过滤掉
未知参数、吞异常或偷偷换实现。构建计算设备 `device` 也不是部署芯片。

## 构建与产物

目录和对外交付索引统一遵循 [产物布局与 export_meta_info 契约](../../docs/architecture/artifact-layout.md)。
`validate_artifact` 除运行信息外返回 `export_entries`，公共发布流程校验并写入根目录
`export_meta_info.json`，随后纳入 Manifest；不能按文件后缀推断 HMONNX/ONNX。
内部运行元数据原先占用该文件名时，发布器将其迁到 `runtime_meta_info.json`，
加载端从 Manifest 的 `artifact.metadata_file` 取路径。LoRA 每个 checkpoint 单独描述实际图和资源。
公共发布器同时收集随包运行/配置 JSON，以 `type: json` 登记。LoRA family 返回
`export_json_roots={checkpoint: 包内相对目录}` 指定 JSON 归属；该构建期字段不写入 Manifest。

指定 `output_dir` 就直接交付到该目录，不在里面再套一层发布名。省略或传 `None` 时，
在当前工作目录创建 `<发布名>_<YYYYMMDD>`，实际路径由 `BuildResult.output_dir` 返回。
目标必须尚不存在，父目录必须存在；自动命名重名也报错，不覆盖、不偷偷追加随机后缀。

`prefill/`、`decode/`、视觉档位和 `mtp_*/dflash_*` 按已选模型变体平铺在最终目录。
Manifest、有效配置、运行 metadata、HF 配置/tokenizer/processor、embedding 和图引用的权重
是交付文件。自回归打包按运行 metadata 和 ONNX external_data 引用收集文件；HF 资源按配置、
词表、模板及其显式本地依赖选择，不按 `.json/.py` 后缀整批复制。临时子图、日志、评测报告、
工作区权重和空目录不进入交付包。清理只发生在本次持有的 staging，不修改用户源模型。

公共 executor 管事务、来源指纹、依赖、Manifest 和发布；family adapter 只处理模型差异：

| hook | 负责什么 |
| --- | --- |
| `normalize_export` | 将 workflow 的实际产物整理到构建暂存根；不能越界或覆盖已有文件 |
| `package_runtime` | 补齐推理必需的资源；workflow 已导出的资源直接复用 |
| `sanitize_artifact` | 整理模型特有的可搬移元数据，不改数学或图 ABI |
| `validate_artifact` | 检查组件、精度、图档位、资源及可加载性，返回真实 metadata 路径 |

模型路径从 exporter metadata 读取，不从旧文件名、目录层级或用户的 `output_dir`
猜。共用同一份 build/load 校验，避免“能构建、不能加载”。路径必须在包内且进入
Manifest 文件清单；权重 external_data 的引用也必须完整。

`normalize_export` 返回旧名称时，公共发布器会统一改名；如果模型已按发布规则命名且
要求保持图与 external_data 原样，返回 `None`。Kokoro 就属于后者：只提平外层
release 目录，内部日期前缀、文件名和 ONNX location 保持不动。实际发布前缀读
运行 metadata 的 `release_prefix`（统一发布后为 `runtime_meta_info.json`），不能拿目录名或 recipe 的固定标识去拼文件。

`build_model` 不顺便生成 Golden，也不声称精度验证通过。转换失败或校验失败必须
终止发布；不能把失败的可选图、残留权重或日志包装成可用模型。

## 推理与精度

adapter 只校验参数、解析资源、调用模型包，再包装任务结果。文本处理、切句策略、
Host/NPU 调度、cache 和数值实现由模型包维护。不要为了统一接口悄悄更改输入语义。
用完 runtime 需要释放资源，异常路径也一样。

Qwen 的统一推理使用 packed W4 与 HMONNX Inference V2，MTP/DFlash 也不例外。
投机解码沿用原状态机，但 target prefill/verify 和 draft 图必须经过共享 initializer
加载器，再创建 V2 session；只设置环境变量、实际仍创建旧 session 不算接通。
同值权重共享与 W4 打包在加载时完成，不能靠提高显存预算掩盖重复驻留。
CUDA Graph 是独立的执行优化，不是 V2 或 packed W4 的前提；默认不因此开启。
V2 session 创建时固定执行设备，不支持旧 session 的动态换卡、手填显存预算和
resource_tight_mode 卸载重建；这些组合须在加载前明确拒绝，不能静默忽略。

例如 Kokoro 当前是四对 T/F 档位、三张 W16A16 NPU 图和一张 CPU FP32 相位图。
相位图从 `f0` 开始，不是从 `phase_increments` 开始；HMONNX 只接受展开 LSTM。
统一入口和模型示例共用 `prepare_text`，都拒绝英文、未知音素和超长输入，不自动切句。
不能把 CPU FP32 边界合回 NPU，也不能复活 native 或旧 graph_mode 开关。

底层 `dump_golden` 支持某模型，不等于统一推理入口支持该模型的 Golden。
只有输出收集、目录与覆盖语义都接好并测过，才能注册 `golden_components`。
当前 Kokoro 全档串联 Golden 仍由模型 workflow 单独执行。

## 模型升级时一起检查

从 source 推导出的字段（例如 DFlash 的 noise token）属于 planner 的
`config_replacements`，用户请求才进入 `config_overrides`。产物校验要核对正确的
计划字段；测试至少让真实 resolver 生成一次计划再传给 validator，不能仅手填 mock
计划，否则两侧把字段放错位置也可能一起通过。

修改模型实现或合入远端更新时，逐项核对：配置默认值、组件及顺序、I/O 名称和
shape/dtype、Host/NPU 精度边界、路由与溢出策略、资源路径、依赖版本、Golden 语义。
然后更新 recipe、两侧 adapter、示例和测试，不以“rebase 无冲突”作为完成依据。

不维护开发阶段旧调用和旧产物的兼容层。契约不兼容时修改相应 recipe/格式版本并
给出明确的重新导出提示；版本号不能替代字段和文件校验。

新增回归都放进 [当前 CI](../../benchmark_test/ci_test_transformer5.16.0/README.md)，
并带同名 `.impact.json`。模型包、配置、adapter 和共享 helper 的变化都要选中真实
消费它们的用例。至少覆盖：默认值和拒绝路径、构建到加载、搬移后无源目录依赖、
失败不发布、受影响的实际数值/ABI 边界。用小真实模块验证数值；只在测试编排时替换
无关的昂贵导出或会话，并说明没覆盖什么。示例能运行、mock 测试通过都不是精度结论。

### 依赖声明

在 `ModelRecipe.dependencies` 声明公共、构建、推理依赖，在
`VariantRecipe.dependencies` 声明补充项。使用轻量 `DependencySpec`，不要读取当前
环境或导入模型推导声明。公开 profiles 合并已注册量化方法及 checkpoint 加载依赖，
保留旧 worker 预检字段；必要依赖审计与安装验证分别记录，未验证不能标为可安装。
详细字段、发布阻塞项及示例见 [依赖契约](../../docs/architecture/model-dependencies.md)。
