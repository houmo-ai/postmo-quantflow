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

先区分用户选择和部署约束。可选精度、档位、来源和资源放进 recipe；只有一种合法
取值的设置不必变成开关。新的通用参数需要确实有跨模型含义，模型私有参数不要塞进
公共入口的任意 `**kwargs`。

`describe_model(model_id)` 返回可修改构建参数的说明，包括默认值、选项和约束；
它不是能直接展开成 `build_model(**result)` 的参数字典。
`full=True` 还返回任务、部署精度、组件、输入输出 schema 等完整说明。
推理输入由 `input_schema` 描述，不与构建参数混用。

加参数时一起改声明、描述、解析校验、配置映射、实际调用和测试。传下去了不等于
生效了，测试需要检查消费方实际收到的值和行为。unsupported 值明确报错，不能过滤掉
未知参数、吞异常或偷偷换实现。构建计算设备 `device` 也不是部署芯片。

## 构建与产物

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
`export_meta_info.json.release_prefix`，不能拿目录名或 recipe 的固定标识去拼文件。

`build_model` 不顺便生成 Golden，也不声称精度验证通过。转换失败或校验失败必须
终止发布；不能把失败的可选图、残留权重或日志包装成可用模型。

## 推理与精度

adapter 只校验参数、解析资源、调用模型包，再包装任务结果。文本处理、切句策略、
Host/NPU 调度、cache 和数值实现由模型包维护。不要为了统一接口悄悄更改输入语义。
用完 runtime 需要释放资源，异常路径也一样。

例如 Kokoro 当前是四对 T/F 档位、三张 W16A16 NPU 图和一张 CPU FP32 相位图。
相位图从 `f0` 开始，不是从 `phase_increments` 开始；HMONNX 只接受展开 LSTM。
统一入口和模型示例共用 `prepare_text`，都拒绝英文、未知音素和超长输入，不自动切句。
不能把 CPU FP32 边界合回 NPU，也不能复活 native 或旧 graph_mode 开关。

底层 `dump_golden` 支持某模型，不等于统一推理入口支持该模型的 Golden。
只有输出收集、目录与覆盖语义都接好并测过，才能注册 `golden_components`。
当前 Kokoro 全档串联 Golden 仍由模型 workflow 单独执行。

## 模型升级时一起检查

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
