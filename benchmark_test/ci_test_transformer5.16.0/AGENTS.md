# 当前 ModelZoo CI

先读 [README.md](README.md) 的 Coverage policy、Onboard a model 和测试设计边界。
本目录是所有新增测试的归属地，包括 CPU 契约、算子/结构回归和真实导出。

- 用例使用 `test_*.py` 或 `*_test.py`；与同名 `<test-stem>.impact.json` 一起维护。
  声明格式见 [_impact_manifest.py](_impact_manifest.py)，选择逻辑见 [select_tests.py](select_tests.py)。
- impact 的 `patterns` 是仓库相对的正向 Python `fnmatch` 模式，覆盖真正执行的源文件、
  YAML、入口和共享 helper。一个输入命中的所有声明都会生效；显式声明会取代该输入的
  import/family 推断，必须为每个需要回归的使用方补齐声明。
- 普通模型接入扩展相邻声明，不修改全局 `impact_policy.json`。
  冻结目录或无覆盖的受保护代码会失败；`--all` 也不能绕过。
- 新模型必须有 recipe，`test_recipe_coverage.py` 是 always-run 门禁：从模型实现反查
  已发现的 recipe 和真实 workflow，不依赖 Git 历史。它只检查接入完整性，不替代功能覆盖；
  不给它声明 `models/**`、`configs_merak/**` 这类宽泛 impact，以免压掉真实消费方测试。
  `recipe_legacy_gaps.json` 是固定存量缺口，只减不增；接入或删除旧模型时同步删除失效项。
- **覆盖完整、执行快**：优先小 tensor 或由小 config 构造的最小真实模型，保留被测的
  attention/norm/expert/cache 结构，用参数化覆盖相关边界、错误输入与状态转换。
  不 mock 被测行为，不为减少层数删掉触发条件；具体取舍见 README 的 Test design。
- 必需的真实模型导出复用现有 block harness 和 SHA-256 固定的
  checkpoint。recipe 契约不重跑 AutoRound；四个 Transformer block 用例只导出，
  不生成 golden。BGE 兼容用例的量化/golden 是保留特例。
- marker 只描述类别；实际触发依赖 selector 和 impact 声明。
  不用无条件 skip、弱化断言或扩大映射掩盖缺失覆盖。

从仓库根运行 `./benchmark_test/ci_test_transformer5.16.0/run.sh`。
选择预览和单文件调试命令见 README；保留 runner 的 `-c` 与 `--confcutdir`，
避免继承父目录旧 benchmark fixture。文档改动也要通过快速 policy gate。
