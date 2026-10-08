# xh2modelzoo

## 仓库约定

- 发布产物遵循 [统一产物布局与集成清单](docs/architecture/artifact-layout.md)：ASR 用 `encode`，
  解码用 `decode`，视觉用 `visual`；组件按角色与实际档位平铺。统一 build 必须生成
  根目录 `export_meta_info.json`（组件/资源的包内相对路径与 type，LoRA 按 checkpoint 分组），
  随包运行 metadata、档位和 HF 配置等 JSON 以 `type: json` 描述路径，不内嵌内容；
  改布局须同步维护加载端及当前 CI。

- 当前实现、配置、示例分别在 `xhmodel_merak/`、`configs_merak/`、`examples_merak/`。
  历史根目录由 [CI policy](benchmark_test/ci_test_transformer5.16.0/impact_policy.json) 冻结。
- vLLM Merak 插件位于 `vllm_merak/`；后续插件开发、修复和集成在本仓库进行，
  不再向旧的独立 `vllm-merak` 仓库增加改动。插件仍保留独立安装入口。
- 修改 Merak 的入口、分层或模型接入时，先读 [Merak 架构](docs/architecture/merak.md)。
  保持 workflow 编排、模型语义、图表达和 runtime 的职责边界。
- 修改已注册模型或统一接口时，同时遵循 [接口接入约定](xhmodel_merak/model_integrations/README.md)；
  模型升级需要一起核对 recipe、构建、加载、示例和测试，不能只解文本冲突。
- 新增 Merak 模型必须在同一变更中提供可发现的 `ModelRecipe`、可用 workflow、统一接口
  示例及当前 CI 用例/impact 声明；只有底层注册或导出脚本不算接入完成。
  [recipe 门禁](benchmark_test/ci_test_transformer5.16.0/test_recipe_coverage.py) 每次 CI 都运行，
  存量缺口清单只减不增，不用新增豁免或 `master=False` 规避公开主模型的接入要求。
- 涉及量化接入或权重格式时，遵循 [ModelZoo / GPTQModel 开发边界](docs/architecture/quantization-boundary.md)。
- 算子、结构和芯片适配知识从 [建模知识索引](docs/modeling/README.md) 按需读取；
  设备支持结论必须带适用配置和验证证据，不能由芯片名称或单个模型类推。
- 涉及 MLA/权重吸收或 cache 共享时，先核对 [Ling MLA 约束](docs/modeling/structures/ling-mla-quantization.md)：
  融合须在权重量化前完成；SEFP 指数域不同的 K/V 不得因浮点 latent 相同就共享缓存。
- 涉及 Kokoro / M50 路径的 LSTM 时，先读 [Merak 的 LSTM 约定](xhmodel_merak/AGENTS.md)：
  有效前缀索引属于模型图；标准 LSTM 语义、量化和最终展开属于 xhquanttool，不能混作芯片能力限制。
- **新增测试统一写到 `benchmark_test/ci_test_transformer5.16.0/`，不向 `tests/`、
  `test/` 或其他测试目录增加用例。** 补充已有行为的回归也遵循此约定。
  先读该目录的 [AGENTS.md](benchmark_test/ci_test_transformer5.16.0/AGENTS.md) 和
  [CI 设计及接入规格](benchmark_test/ci_test_transformer5.16.0/README.md)。
- 引入第三方实现遵守 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 测试覆盖完整、执行快：优先最小真实模型和小 tensor，保留触发条件，覆盖相关契约与边界。

## 验证与审查

CI 入口：`./benchmark_test/ci_test_transformer5.16.0/run.sh`。
先用同目录 `select_tests.py --format report` 确认用例及选择原因；
新增用例同时维护同名 `.impact.json`，不要靠修改全局 policy 扩大或绕过覆盖边界。

存在 `.codegraph/` 时，先用可用的 CodeGraph 工具定位代码及影响范围；
未覆盖的源码、配置和文档再用 `rg` / 直接读取。MCP 安装和本机配置由使用者管理。
代码 review 使用 `ocr`，带 `--audience agent` 和实际需求背景；
范围不明确先 `--preview`。审查提交可用 `ocr review --repo . --commit HEAD --audience agent`。
复核工具结论，报告实际验证范围、失败原因和未执行项。

提交遵守 [飞书提交规范](https://houmo.feishu.cn/wiki/B7Yeww5MKixxaSkqaJ7cOlF5n2c)。

## 维护这些指引

`AGENTS.md` 只保留仓库特有约定与阅读入口，工具专属文件只指向它。
架构决策写入架构文档，算子/结构/目标差异写入建模知识，具体使用命令留在对应 README。
重构时同步更新受影响的链接和结论，不把通用编程流程、旧迁移步骤或整份 skill 复制进指令。
