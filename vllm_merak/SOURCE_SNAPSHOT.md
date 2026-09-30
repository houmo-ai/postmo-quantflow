# vLLM Merak 迁移记录

本子项目是可独立安装的 vLLM 插件 `vllm-merak`。现在，
`xh2modelzoo/vllm_merak` 是其开发、安装、文档和 CI 的唯一维护源。
代码继续使用 vLLM 插件 entry point；它不是 ModelZoo inference 插件，
ModelZoo 基础包也不会自动导入它。

## 迁移来源

- 原仓库：`quantization/vllm-merak`
- 迁移时的原远端：`git@10.10.1.58:quantization/vllm-merak.git`
- 原分支：`main`
- 原提交：`b5ee5d5`
- 快照日期：`2026-09-21`
- 快照范围：原工作树中的所有已跟踪文件，以及所有未被忽略的未跟踪文件；
  不包括 `.git`、缓存、构建产物、本地环境、权重和其他被忽略的产物。
- 添加本来源记录前的快照文件数：133

原工作树包含尚在进行中的 Merak Cache/Runner 和 WeMM Embedding 修改。
迁移时有意将这些修改一起纳入，使 ModelZoo 内的插件与当时实际生效的源码工作树一致，
而不只是与原仓库最后一次提交一致。由于目标工作树中已经存在与迁移无关的用户修改，
本次没有直接合并原仓库 Git 历史；上述原提交和原工作树状态共同构成本次迁移的审计边界。

完成逐字节源码快照后，ModelZoo 新增了本来源记录和仓库的 Apache-2.0
`LICENSE`，将已经废弃的 TOML license table 写法规范化为等价的 SPDX 表达，
并完成了下述运行切换。原工作树包含未提交修改，因此继续保留为回退副本；
它不再是当前生效的 editable 安装源或维护入口。

## 快照后的迁移修改

- 新增经过验收的运行环境 profile 和环境检查器，用于核对包版本、editable
  源码路径、vLLM entry point，以及 `vllm_merak` 的实际 import 位置。
- 新增 `tools/install_from_modelzoo.sh`；它使用 `--no-deps` 只安装本子项目，
  因此不会替换运行环境中对 ABI 敏感的 Torch/vLLM 软件栈。
- 使用 `tools/install_vllm_cuda_xhquant.sh` 替换原 CPU-target 构建路径。
  该脚本要求显式提供 `VLLM_SOURCE`，从与源码精确匹配的提交和 CUDA variant
  安装官方 CUDA 扩展（并提供源码构建回退路径）；如果
  `_C_stable_libtorch` 或 CUDA repetition-penalty kernel 无法正确加载并执行，
  脚本会直接失败。
- 更新部署文档和示例，使用 `${MODELZOO_ROOT}/vllm_merak`，不再使用与
  ModelZoo 并列的独立仓库路径。
- 修复离线生成 smoke 示例，将 `max_model_len` 和 `max_num_seqs` 传给
  `vllm.LLM`；原示例虽然解析了这两个参数，但没有继续传递，导致参数被静默丢弃。
- 为独立插件包、entry point、运行交付文件、构建辅助脚本和 smoke 参数新增
  ModelZoo CI 契约与 impact 声明。
- 新增失败即停止的 PageAttention ABI 检查，以及真实 HMONNX release smoke
  测试矩阵；该矩阵覆盖 greedy、temperature/top-k/top-p，以及
  repetition/presence/frequency penalty。
- 新增轻量 Qwen3.5-9B FlashAttention 视觉导出 profile，用于同时验证文本和
  图文生成；迁入代码继续保留原独立仓库按 metadata 选择视觉或纯文本 runtime
  的行为。
- 修复 WeMM Hybrid Cache 契约：Merak 会关闭不安全的 Prefix Cache，提供
  vLLM 要求的 Mamba block-size sentinel，并且只 padding Attention Cache 页，
  同时保持 Mamba Cache Spec 不变。

## 运行边界

需要 HMONNX vLLM 后端时，应将本目录作为独立包安装。必须使用带检查的安装脚本，
避免依赖解析替换对 ABI 敏感的运行软件栈：

```bash
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh
```

本包保留以下 vLLM entry point：

```toml
[project.entry-points."vllm.general_plugins"]
merak = "vllm_merak:register_plugins"
wemm = "vllm_merak.native_wemm:register"

[project.entry-points."vllm.platform_plugins"]
merak = "vllm_merak.platform:register"
```

原生 Float/GPTQ vLLM 与 Merak HMONNX vLLM 应在独立进程和相互兼容的环境中运行。
原生 WeMM 应显式选择 `VLLM_PLUGINS=wemm`；HMONNX Worker 应显式选择
`VLLM_PLUGINS=merak`，并使用
`config_format="merak_llm"` 和 `load_format="merak_llm"`。

`wemm` 是快照完成后新增的 general plugin。它将原生浮点 WeMM 模型实现保存在
ModelZoo 的插件目录中，不再依赖本机 vLLM 源码树里的私有修改；`merak` 的
HMONNX 平台、配置解析和 loader 入口保持独立。

## 迁移验证

- `[verified/已验证]` 在 import 边界，除 `pyproject.toml` 外的 132 个原始源码文件
  均与迁移时实际生效的原工作树逐字节一致。`pyproject.toml` 最初唯一的变化是
  等价的 SPDX license 规范化；快照后有意进行的其他修改已在上文列出。
- `[verified/已验证]` 隔离 wheel 构建产物包含 Apache-2.0 许可证和两个原有的
  vLLM entry point；安装到隔离目标目录后，entry-point 元数据仍然完整保留。
- `[verified/已验证]` 本地 `vllm` 环境已从旧独立工作树切换到 ModelZoo 路径。
  `direct_url.json`、发现到的三个 entry point 和 `import vllm_merak` 现在均指向
  本子项目。
- `[verified/已验证]` 设置 `VLLM_PLUGINS=merak` 时，vLLM 会选择
  `vllm_merak.platform.MerakPlatform` 并注册 `MerakConfigParser`；设置
  `VLLM_PLUGINS=""` 时不会选择该插件。HMONNX 服务 profile 现在要求使用
  CUDA-target vLLM host，不再接受 CPU-target wheel。
- `[verified/已验证]` 在 Python 3.12.14、vLLM 0.23.0+cpu、Torch
  2.11.0+cu129 和 Transformers 5.13.0 环境下，从新旧两份源码树运行完整插件
  测试集得到相同结果：639 passed、17 failed。因此，这 17 个失败被记录为迁移前
  已存在的兼容性/测试债务，而不是迁移回归。
- `[verified/已验证]` 当前 ModelZoo impact 选择执行完成，包含新增插件包门禁和
  原工作树中已有的 WeMM 修改；结果为 264 passed、14 个 deprecation warning。
- `[verified/已验证]` 一个满足 PageAttention 契约的 WeMM-2B 导出产物通过迁移后
  的插件完成了真实的 320-word embedding 请求。平台自动关闭了不安全的 Prefix
  Cache，并返回归一化的 2048 维 embedding；运行过程中没有隐藏的原生模型回退。
- `[verified/已验证]` 使用有意缩小的校准 profile（`nsamples=2`、`seqlen=128`、
  `iters=2`）导出了完整的 32 层 Qwen3.5-9B W4/G64 HMONNX 产物。静态检查在
  Prefill 和 Decode 图中均发现 8 个有效 FlashAttention 节点，直接 HMONNX 文本
  生成也成功输出 token。随后，CUDA-target vLLM Merak release 门禁成功加载转换后
  的 PageAttention 图，并分别在 greedy、temperature/top-k/top-p 和
  repetition/presence/frequency-penalty 三种场景中生成 4 个 token。
- `[verified/已验证]` 2026-09-22，release acceptance 包装脚本完成端到端
  验证：环境/entry-point 检查、`pip check`、FlashAttention ABI、HMONNX Loader
  实例化、PageAttention Cache 分配，以及三种采样场景均通过；该次验证使用纯文本
  smoke 产物，纯文本 runtime 仍属于迁入基线能力。
- `[verified/已验证]` 2026-09-29，在独立 worktree 中使用完整五档视觉
  Qwen3.5-9B W4A8 产物完成真实 vLLM Merak 验收。同一引擎完成 greedy、
  temperature/top-k/top-p、repetition/presence/frequency penalty 和图文生成，
  四类请求均生成 4 个 token；实际 import 路径指向该 worktree。
- `[verified/已验证]` 2026-09-29，Qwen3.5/Qwen3-Next 聚焦回归 41 项、
  WeMM 插件与评测适配回归 26 项、ModelZoo 插件交付契约 10 项、WeMM 当前
  CI 契约 143 项均通过。插件完整测试为 `612 passed, 44 failed`；原工作目录同环境
  基线为 `613 passed, 46 failed`，当前变更没有新增失败，剩余失败集中在迁移快照中
  已存在的工具路径、Gemma4 harness、Runner API 和服务脚本文档债务。
- `[verified/已验证]` 2026-09-21，`vllm` 服务环境中的
  `python -m pip check` 返回 `No broken requirements found.`。ModelZoo 和
  xhquanttool 的导出依赖已隔离到 `export` extra 中；`runtime_profiles.toml`
  记录了经过验收的 CUDA-target 运行环境，以及该环境要求的模块和 sampler-kernel
  ABI。

## 旧源码退役边界

源码所有权和当前生效的 editable 安装已经迁移到 ModelZoo。旧独立工作树继续作为
可恢复的回退副本保留，直到其中的未提交修改被正式纳入 ModelZoo 的正常 Git 历史。
删除旧工作树不属于本次迁移步骤的范围。
