# vLLM Merak 迁入 ModelZoo：修改与问题处理记录

> 记录日期：2026-09-28
> ModelZoo：`/data01/home/jing.huang/xh2modelzoo`
> 迁入后的插件根目录：`/data01/home/jing.huang/xh2modelzoo/vllm_merak`
> 当前 Python 包目录：`/data01/home/jing.huang/xh2modelzoo/vllm_merak/vllm_merak`
> 原独立工作树（保留用于回退和对照）：`/data01/home/jing.huang/vllm-merak`
> 来源提交：`b5ee5d52ba9d71a159e1fbc8b7cc4a696469784f`

## 1. 结论与边界

本次工作的准确含义是：

1. 将 `vllm-merak` 源码和维护责任迁入 `xh2modelzoo`；
2. 保留它原有的 **vLLM 插件**运行方式；
3. 使插件在新的源码位置可独立安装、可被 vLLM 发现、可加载 HMONNX，并能完成真实推理验收；
4. 隔离 ModelZoo 导出环境和 vLLM 服务环境，避免两套依赖相互覆盖。

它**不是**将 vLLM Merak 改造成 ModelZoo inference 后端，也不是让 ModelZoo 主包在导入时自动加载 Merak。运行关系仍是：

```text
HM-Eval / 调用方
    -> vLLM
    -> vLLM entry point 发现 vllm-merak 插件
    -> Merak Config / Loader / Worker / Runner
    -> HMONNX Runtime
```

当前运行迁移和真实模型验收已经完成，`xh2modelzoo/vllm_merak/` 的第一阶段迁移
已进入 ModelZoo Git 历史。原生浮点 WeMM 插件适配已在独立 worktree 中完成修改
和验证，并整理到第二阶段 WeMM 支持提交。

原独立工作树仍保留，未删除，继续作为回退副本。

## 2. 迁移工作的完成步骤

| 步骤 | 实施内容 | 目的/结果 |
| --- | --- | --- |
| 1. 固化来源 | 将原工作树（包括当时未提交的 Cache/Runner、WeMM 改动）完整快照到 `xh2modelzoo/vllm_merak`；在 `SOURCE_SNAPSHOT.md` 记录仓库、提交、日期和范围 | 明确迁移基线，保证来源可追溯；迁移的不是只有旧仓库最后一次提交，而是当时实际运行的工作树 |
| 2. 建立独立插件子项目 | 保留独立 `pyproject.toml`、包名 `vllm-merak` 和 `vllm_merak/` Python 包 | 源码归 ModelZoo 管理，但插件仍可独立安装，不与 ModelZoo 主包强绑定 |
| 3. 保留 vLLM 插件入口 | 保留 Merak 的 general/platform entry point，并新增独立的原生 WeMM general entry point | `merak` 负责 HMONNX，`wemm` 负责原生浮点 WeMM；两条运行路径可由 `VLLM_PLUGINS` 分开选择 |
| 4. 切换安装来源 | 新增 `tools/install_from_modelzoo.sh` 和环境检查器；在 `vllm` 环境从新目录 editable 安装 | 使 `direct_url.json`、`import vllm_merak` 和三个 entry point 都指向 ModelZoo 内的新目录 |
| 5. 隔离依赖 | ModelZoo 和专用 xhquanttool 工作树把量化/导出依赖移入 `export` extra；插件使用 `--no-deps` 独立安装 | 防止服务环境为满足 ModelZoo 元数据而替换 vLLM 所需的 Torch、Transformers、ONNX 等 ABI 敏感依赖；使 `pip check` 干净 |
| 6. 固化 CUDA vLLM 环境 | 增加 CUDA-target vLLM 安装脚本和 `runtime_profiles.toml` | 避免 CPU-target vLLM 缺少 `_C_stable_libtorch` 或 CUDA logits-sampling kernel；固定已验收的 Python/Torch/CUDA/vLLM 组合 |
| 7. 修复真实运行契约 | 对 Hybrid Cache、Prefix Cache 和 PageAttention 设备映射做处理 | 解决“可以 import，但真实 HMONNX 模型启动或生成失败”的问题 |
| 8. 增加真实验收 | 增加 PageAttention ABI 检查、release smoke、轻量 Qwen3.5 FlashAttention 导出配置和 ModelZoo CI 契约测试 | 验证 Loader、Cache 分配、HMONNX 加载和真实采样，而不只验证包能导入 |
| 9. 更新文档和示例 | 将部署文档中的旧兄弟仓库路径改为 `${MODELZOO_ROOT}/vllm_merak`；修复 smoke 参数转发 | 统一新维护入口，避免示例解析参数后没有真正传给 `vllm.LLM` |
| 10. 保留回退边界 | 不删除 `/data01/home/jing.huang/vllm-merak` | 在新目录正式提交前保留可恢复基线 |
| 11. 迁入原生 WeMM | 将本机 vLLM 中的 WeMM 模型实现迁到 `vllm_merak/model_executor/model_loader/wemm_embedding_native.py`，增加 `wemm` general plugin、原生示例和契约测试 | 官方浮点 checkpoint 无需私改 vLLM 源码或 `hf_overrides`；末尾 embedding token 与视频帧组逻辑由模型插件维护 |

## 3. 遇到的问题及解决方式

| 问题 | 原因 | 解决方式 | 效果 |
| --- | --- | --- | --- |
| 安装后仍可能使用旧仓库源码 | editable 安装会在 `direct_url.json` 中记录源码路径；仅复制源码不会自动切换已安装包 | 从 `xh2modelzoo/vllm_merak` 重新 editable 安装；检查 `direct_url.json`、`import vllm_merak.__file__` 和三个 entry point | Python 与 vLLM 均从新目录加载插件 |
| vLLM 环境执行 `pip check` 报依赖冲突 | ModelZoo 主包原本把完整量化/导出依赖写入基础包元数据；这些版本约束与 vLLM 服务栈不同 | ModelZoo 和 vLLM 专用 xhquanttool 的基础依赖置空，将完整工具链移到 `[project.optional-dependencies].export`；服务环境用 `--no-deps` 安装 | 服务环境只安装源码包边界，不强行解析导出工具链；迁移验收时 `pip check` 返回无损坏依赖 |
| CPU-target vLLM 缺 `_C_stable_libtorch` 或采样算子 | Python 包可导入不代表 C++/CUDA 扩展与当前 Torch ABI 匹配 | 使用与 vLLM 源码提交、Torch 和 CUDA variant 匹配的 CUDA-target 构建；安装脚本实际执行 repetition-penalty kernel | 在真实采样前即可发现 ABI 错配；greedy、top-k/top-p、penalty 均完成实测 |
| vLLM 会尝试统一 Attention 与 Mamba 的 Cache 页大小 | 混合模型中 Attention 使用导出时固定的 64-token 逻辑页，而 Mamba 状态页的物理尺寸不同；通用统一逻辑会错误放大 Attention 逻辑 block 或尝试 padding Mamba | 保持 Attention 逻辑 block 为 64；只通过 `AttentionSpec.page_size_padded` 扩大 Attention 的物理分配，Mamba Spec 不变；在 Worker 子进程容量规划前再次注册兼容逻辑 | HMONNX 仍按导出契约解释 block，同时 vLLM 获得统一的物理页分配尺寸 |
| Prefix Cache 的全局 block ID 不能直接作为 Mamba 紧凑状态槽位 | vLLM Prefix Cache 的 scheduler block ID 与 Merak 按请求持有的 Linear/Mamba 状态索引不是同一个命名空间；长文本跨 64-token 页时会越界 | 对当前 WeMM/相关单请求 Hybrid 契约关闭 Prefix Cache；设置 `mamba_block_size=max_model_len`、`mamba_cache_mode="none"`，并要求 `max_num_seqs=1` | 保留 PageAttention，但不复用不安全的跨请求 Mamba 状态；避免长文本跨页索引越界 |
| 调度容量不等于真实物理 Cache | vLLM 调度器需要的是逻辑/虚拟 token 容量，实际分配则由 Attention block 数和 Mamba 请求状态槽位共同决定 | 保留 `kv_capacity.py` 的几何与容量计划：虚拟预算单独返回给调度器；Attention 按 `num_blocks` 分配，Mamba 按 `max_num_seqs` 分配；最终配置再次校验 | 避免把虚拟容量直接当物理 tensor 数量导致过量分配或越界 |
| HMONNX 绑定 PageAttention 前找不到正确设备 | 图在绑定 Cache 前仍可能暴露为 FlashAttention；旧 runtime 还可能延迟初始化，并把解释器元数据留在 CPU | Runner 在绑定前同时扫描 PageAttention/FlashAttention；必要时初始化 legacy session；当实际 fallback 为 CUDA 而所有元数据为 CPU 时以运行设备为准 | Cache tensor 在正确 CUDA 设备分配，避免将 CPU tensor 传给 Triton PageAttention |
| Qwen3.5 metadata 声明了无效视觉图 | 视觉配置存在时必须能解析到实际 HMONNX 文件 | 校验非空 `visual_config.hmonnx` 及对应文件；根级 `visual_config` 缺失时继续选择原有纯文本 runtime | 无效视觉产物在启动前失败，纯文本产物保持原独立仓库行为 |
| 示例接收参数但运行时未生效 | `max_model_len`、`max_num_seqs` 被 argparse 解析后未传给 `vllm.LLM` | 在 `examples/vllm_merak_generate.py` 中显式转发 | smoke 的长度、单请求 Hybrid 约束与命令行一致 |
| 只做 import 测试无法覆盖真实问题 | 插件发现成功不代表 HMONNX 图、PageAttention、Cache 和 sampler 可运行 | 新增环境检查、ABI 检查、真实模型 release smoke 和 CI 契约测试 | 验收从“可安装”提升为“可启动、可分配、可生成” |

## 4. 当前 `vllm_merak/vllm_merak` 源码差异明细

### 4.1 对照方法

当前 ModelZoo 中的 Python 包与保留的原独立工作树进行逐文件对照：

```text
旧：/data01/home/jing.huang/vllm-merak/vllm_merak
新：/data01/home/jing.huang/xh2modelzoo/vllm_merak/vllm_merak
```

排除 `__pycache__` 和 `.pyc` 后，当前只有以下 6 个 Python 源文件存在内容差异：

```text
model_executor/model_loader/qwen3_5.py
model_executor/model_loader/wemm_embedding.py
platform.py
utils.py
worker_v1/merak_model_runner_v1.py
worker_v1/merak_worker_v1.py
```

下表中的行号以 2026-09-28 当前文件为准；后续代码变化时应优先按函数/类名定位。

### 4.2 六个差异文件的逐项说明

| 范围 | 文件及当前位置 | 修改内容 | 为什么修改/解决什么问题 |
| --- | --- | --- | --- |
| 迁移真实验收修复 | `platform.py:176`，`_disable_merak_prefix_caching` | 统一关闭 Prefix Cache，并设置 `mamba_block_size=max_model_len`、`user_specified_mamba_block_size=False`、`mamba_cache_mode="none"` | 避免 Prefix Cache 的全局 block ID 被当作请求内 Mamba 状态槽位；同时满足 vLLM 在关闭 Prefix Cache 后仍要求 Mamba block size 存在的校验 |
| 迁移真实验收修复 | `platform.py:217`，`_install_merak_hybrid_cache_page_compatibility` | 包装 vLLM 的页大小统一逻辑；只给较小的 Attention Spec 设置 `page_size_padded`，不修改 Mamba Spec | 保持 HMONNX PageAttention 的 64-token 逻辑 block，不让通用 Hybrid Cache 对齐逻辑破坏导出契约 |
| 迁移真实验收修复 | `platform.py:258`，`register_kv_cache_spec` | 注册 HMFP Cache Spec 前安装上述 plugin-wide 页兼容逻辑 | 保证 Cache 分组/容量规划采用 Merak 的混合页契约 |
| 迁移真实验收修复 | `platform.py:400`、`:414`、`:446`，`check_and_update_config` | WeMM、contiguous-single-request 和 Gemma4 固定环形 Cache 场景统一调用 Prefix Cache 关闭函数；Hybrid Linear 要求 `max_num_seqs=1` | 防止不安全的跨请求状态复用，并把支持边界改为显式失败而不是运行中越界 |
| 迁入时已存在并保留 | `platform.py:483`，`update_block_size_for_backend` | 不调用 vLLM 通用 Hybrid block 对齐；固定 Merak 首选逻辑 block | 该保护在迁入快照中已经存在；作用是避免 vLLM 把逻辑 Attention block 改成 Mamba 页宽。不是复制目录后才新增的代码 |
| 迁移真实验收修复 | `worker_v1/merak_worker_v1.py:288`，`determine_available_memory` | Worker 子进程开始容量规划前再次调用 `register_kv_cache_spec()` | vLLM V1 可能在新进程创建 Worker；仅在父进程 Platform hook 注册不够，stock 页统一逻辑可能抢先运行 |
| 迁入时已存在并保留 | `worker_v1/merak_worker_v1.py:288-388` 及 `kv_capacity.py` | 分开计算 scheduler 虚拟预算、Attention 物理 block 和 Mamba 请求状态槽位；初始化时校验最终几何 | “调度容量不等于真实物理缓存”的主体机制在迁入快照中已经存在；迁移后新增的是在规划前确保页兼容 hook 生效 |
| 迁移真实验收修复 | `worker_v1/merak_model_runner_v1.py:221`，`_page_or_flash_attention_node_map` | PageAttention 尚未改写时扫描 FlashAttention 节点；当 runtime 实际在 CUDA、旧元数据全为 CPU 时使用 CUDA fallback | 解决绑定 PageAttention Cache 前无法确定层数/设备，或错误在 CPU 分配物理页的问题 |
| 迁移真实验收修复 | `worker_v1/merak_model_runner_v1.py:302`，`_hmonnx_attention_node_map` | 支持 legacy HMONNX session 延迟初始化；支持从 `node_modules` 按 attention 顺序建立稳定身份 | 兼容旧 runtime 的图暴露方式，避免误判为 0 个 Attention Cache 层 |
| 迁移真实验收修复 | `worker_v1/merak_model_runner_v1.py:415`，`_get_kv_cache_layer_devices` | Prefill/Decode/单图路径均改用 Page-or-Flash 映射 | 确保三种图组织方式使用同一设备发现规则 |
| 迁入时已存在并保留 | `worker_v1/merak_model_runner_v1.py:2572`，`_publish_finalized_attention_block_size`；`:2603`，`_wrap_sefp_kv_cache_handles` | 从最终 per-layer Attention Spec 读取逻辑 block 和物理布局，不重新使用已被 vLLM 改写的全局 `cache_config.block_size` | 解决 Hybrid 分组后全局值可能变成 Mamba block size，而 Attention 真实页仍为 64 的问题；该改动属于迁入快照中的既有工作树内容 |
| 迁入时已存在并保留 | `model_executor/model_loader/qwen3_5.py`、`config.py` | 根据根级 `visual_config` 选择多模态或 `Qwen3_5MerakTextForCausalLM` runtime；声明视觉图但文件无效时失败 | 保持原独立仓库 Qwen3.5 行为；迁移验收额外使用完整视觉产物验证图文生成 |
| 迁移真实验收修复 | `model_executor/model_loader/wemm_embedding.py:71`，`_install_wemm_hybrid_cache_page_compatibility` | 删除 WeMM 文件内的重复页兼容实现，转调 `platform.py` 的 plugin-wide 实现 | 同一 Hybrid 页契约由平台层统一安装，避免只在 WeMM loader 导入顺序正确时才生效 |
| 迁移后的 WeMM 功能改动，非搬仓库本身 | `model_executor/model_loader/wemm_embedding.py:47`，`_ensure_wemm_embedding_token`；`:96`，`WeMMEmbeddingMultiModalProcessor` | 追加 `<embedding>` pooling anchor，并补齐视频 prompt 的视觉边界 token | 这是迁移后为 WeMM 输入/池化协议继续做的精度与功能适配；不应记为源码迁址所必需的改动 |
| 迁移后的 HMONNX V2/HM-Eval 改动，非搬仓库本身 | `utils.py:75`，`_use_v2_enabled`；`:89`，`get_hmonnx_runtime_kwargs`；`:120`，`should_preserve_hmonnx_runtime_device_map` | 统一解析 `MERAK_USE_V2`/`ENABLE_HMINFERENCE_V2`；非 offload 时固定当前 CUDA device；V2+auto-offload 时保留多卡 map | 这是后续把 HM-Eval HMONNX 后端统一到 vLLM Merak、默认 V2 时增加的运行参数处理，不属于最初迁址/插件安装工作 |

### 4.3 迁入快照中已经存在、并非迁址后新增的内容

原独立工作树在迁入时不是干净提交，以下内容被**原样纳入快照**。记录它们是为了说明来源，而不是声称迁移过程中重新实现了一次：

| 文件 | 迁入时已有内容 | 迁入原因 |
| --- | --- | --- |
| `vllm_merak/config.py` | WeMM 元数据、architecture/runner 配置解析 | 保持迁入前实际在用的模型配置能力 |
| `vllm_merak/worker_v1/merak_model_runner_v1.py` | pooling task、最终 Attention Spec、SEFP Cache handle 和 Linear/Mamba 状态处理 | 保持迁入前实际 Cache/Runner 行为，不退回仓库旧提交 |
| `vllm_merak/worker_v1/merak_worker_v1.py` | generate/pooling 支持和虚拟/物理容量规划 | 保持迁入前实际 Worker 行为 |
| `vllm_merak/model_executor/model_loader/wemm_embedding.py` | WeMM 模型注册、pooler、多模态 processor、Mamba 状态复制协议 | 使当时正在开发的 WeMM 能随实际工作树一并迁入；该文件在原工作树中当时尚未被 Git 跟踪 |
| `tests/test_merak_kv_capacity.py`、`tests/test_merak_model_runner_v1.py`、`tests/test_platform_kv_cache_dtype.py`、`tests/test_sefp_kv_cache.py` | 对上述 Cache/Runner 修改的测试 | 避免迁移只带实现、不带测试 |
| `tests/test_wemm_embedding_support.py`、`examples/vllm_merak_embedding.py`、`docs/wemm_embedding_merak_deploy.md` | WeMM 测试、示例和部署说明 | 迁移完整功能闭环；这些文件当时同样属于原工作树未提交内容 |

## 5. Python 包外的迁移改动

### 5.1 `xh2modelzoo/vllm_merak` 子项目

| 文件/目录 | 改动 | 原因 |
| --- | --- | --- |
| `SOURCE_SNAPSHOT.md` | 新增迁移来源、基线、验证结果和退役边界记录 | 追溯“从哪里迁、迁了什么、验证到什么程度” |
| `LICENSE` | 增加 Apache-2.0 许可证文件 | 使独立 wheel/源码包携带完整许可证 |
| `pyproject.toml` | 保留包名并增加原生 WeMM entry point；license 改为 SPDX 表达；声明 license file 和 Python 3.12 范围；插件依赖保持为空 | 保留 vLLM 插件机制；避免插件安装器改写服务环境的 ABI 栈 |
| `__init__.py`（子项目外层） | 增加外层 import 转发 | 兼容从 ModelZoo 根目录直接解析子项目时的包边界 |
| `runtime_profiles.toml` | 固化已验收 CUDA-target 环境的包版本、模块和 sampler ABI 要求 | 环境验证有明确基准，不依赖口头约定 |
| `tools/install_from_modelzoo.sh` | 从当前插件根目录执行 editable `--no-deps` 安装，随后运行环境检查 | 一条命令完成安装来源切换和 fail-closed 校验 |
| `tools/check_runtime_environment.py` | 检查 Python/包版本、`pip check`、editable 路径、import 路径、三个 entry point 和所需模块 | 防止“命令成功但仍导入旧仓库”或环境 ABI 不完整 |
| `tools/install_vllm_cuda_xhquant.sh` | 安装/构建与源码、Torch、CUDA 匹配的 CUDA-target vLLM，并验证扩展和采样 kernel | 解决 CPU-target 或 ABI 不匹配导致的运行期失败 |
| `tools/install_vllm_cpu_xhquant.sh` | 明确源码、环境和编译器路径并增加 `pip check` | 保留 CPU 构建辅助，但 release HMONNX 服务不以它作为最终验收环境 |
| `tools/check_hmonnx_attention_abi.py` | 检查 HMONNX 产物的 Flash/PageAttention 契约 | 在启动服务前发现图结构不满足 PageAttention 要求 |
| `tools/run_hmonnx_release_smoke.py` | 真实加载 HMONNX，执行 greedy、top-k/top-p 和 penalty 请求 | 覆盖模型加载、Cache、生成与特殊 logits sampling |
| `tools/run_release_acceptance.sh` | 串联环境检查、Attention ABI 和真实 smoke | 形成失败即停止的统一发布门禁 |
| `test_profiles/qwen3_5_9b_flash_attention_smoke.yaml` | 使用 `nsamples=2`、`seqlen=128`、`iters=2` 的轻量完整层导出配置，并保留 96/196/384/704/1536 五档视觉图 | 快速生成满足 Flash/PageAttention 与视觉输入契约的 Qwen3.5 测试产物 |
| `examples/vllm_merak_generate.py` | 把 `max_model_len`、`max_num_seqs` 真正传入 `vllm.LLM`，默认单请求 | 防止参数被静默丢弃，并符合当前 Hybrid Linear 状态所有权约束 |
| `examples/vllm_server_start.sh` | 根据脚本自身位置解析 `PLUGIN_ROOT` | 从任意当前目录启动都能找到迁入后的插件 |
| `README.md` | 增加独立安装、环境隔离、插件启停、HMONNX 加载和验收说明 | 将 ModelZoo 内的新路径设为唯一维护入口 |
| `docs/*deploy*.md` 等部署文档 | 旧的并列仓库路径改为 `${MODELZOO_ROOT}/vllm_merak`，并引用带检查的安装脚本 | 避免后续人员再次从旧目录安装 |

### 5.2 ModelZoo 主仓及关联工作树

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `xh2modelzoo/pyproject.toml` | 基础 `dependencies=[]`；完整量化/导出依赖移入 `project.optional-dependencies.export` | ModelZoo 源码包可进入 vLLM 服务环境，但不会要求安装另一套 Torch/ONNX/Transformers 版本 |
| `xhquanttool-vllm/pyproject_torch28.toml` | 同样将导出依赖与服务基础安装隔离 | 避免 xhquanttool 元数据再次令 `pip check` 失败；这是 vLLM 专用工作树，不代表普通导出环境不再需要这些依赖 |
| `xh2modelzoo/README.md` | 说明 `.[export]` 和 vLLM Merak 独立插件入口 | 明确导出环境与服务环境的安装方式 |
| `benchmark_test/ci_test_transformer5.16.0/test_vllm_merak_plugin_package.py` | 新增包结构、entry point、工具和参数转发契约测试 | 防止后续改包或改脚本破坏插件交付 |
| `.impact.json` | 将插件契约测试纳入相关变更的 CI 影响范围 | 修改插件或关联代码时自动触发门禁 |
| `vllm_merak/tests/test_qwen35_image_capability.py` | 保留纯文本 runtime、视觉图、视觉档位和文件完整性用例 | 迁移不改变原独立仓库的 Qwen3.5 能力边界 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/workflow.py` | 导出后将 `architectures` 写成 `WeMMEmbeddingMerakForEmbedding`，并设置 `runner_type=pooling` | 让统一 artifact 自描述，vLLM Merak 自动选择正确模型和 pooling runner |
| `benchmark_test/.../wemm_embedding_support_test.py` | 增加 WeMM 导出配置落盘测试 | 防止产物再次退回通用 CausalLM 配置 |

## 6. 插件机制保留情况

迁入后的 `pyproject.toml` 仍保留：

```toml
[project.entry-points."vllm.general_plugins"]
merak = "vllm_merak:register_plugins"
wemm = "vllm_merak.native_wemm:register"

[project.entry-points."vllm.platform_plugins"]
merak = "vllm_merak.platform:register"
```

因此：

- `VLLM_PLUGINS=merak`：启用 Merak 平台、配置解析器和 HMONNX Loader；
- `VLLM_PLUGINS=wemm`：在 CUDA 原生 vLLM 上注册浮点 WeMM，并只在显式选择时将官方配置中的 `Qwen3_5ForConditionalGeneration` 映射到 WeMM 类；
- 其他 Float/GPTQ：按模型需要选择插件，不启用 `merak` 平台；
- HMONNX artifact 通过 `config_format="merak_llm"`、`load_format="merak_llm"` 选择 Merak 的配置解析与加载路径；
- ModelZoo 主包不会自动 import 或启动 vLLM Merak。

## 7. 已完成的验证与证据

迁移记录中的历史验收结果如下：

1. 除最初规范化的 `pyproject.toml` 外，迁移边界内 132 个原始文件在快照时逐字节一致；原工作树的未提交内容也被纳入。
2. 构建出的独立 wheel 包含 Apache-2.0 许可证和 vLLM 插件 entry point。
3. `direct_url.json`、entry point 和 `import vllm_merak` 均已验证指向 `xh2modelzoo/vllm_merak`。
4. `VLLM_PLUGINS=merak` 能选择 `MerakPlatform` 并注册 `MerakConfigParser`；关闭插件时不选择 Merak。
5. 新旧源码树的完整插件测试结果相同：`639 passed, 17 failed`；17 项被认定为迁移前已存在的兼容性/测试债务，不是迁移回归。
6. ModelZoo impact 测试结果：`264 passed, 14 warnings`。
7. 真实 WeMM-2B PageAttention 产物完成 320-word 请求，返回归一化 2048 维 embedding，无原生模型 fallback。
8. 轻量纯文本 Qwen3.5-9B W4/G64 验证通过；2026-09-29 又使用完整五档视觉 Qwen3.5-9B 产物验证：vLLM Merak 同一引擎完成 greedy、temperature/top-k/top-p、repetition/presence/frequency penalty 和图文生成，四类请求均成功生成 4 个 token。
9. CUDA-target vLLM Merak 成功加载转换后的 PageAttention 图，并完成 greedy、temperature/top-k/top-p、repetition/presence/frequency penalty 三类请求。
10. 2026-09-22 的 release acceptance 完成环境、entry point、`pip check`、ABI、Loader、Cache 分配和采样全链路验证。
11. 2026-09-21 的 vLLM 服务环境执行 `python -m pip check`，结果为 `No broken requirements found.`。
12. 2026-09-29 在隔离的 `vllm_wemm_native_dev` 环境和 GPU 5 上验证原生浮点
    WeMM-2B：文本、动态图像和 4 帧视频均输出 2048 维有限向量。与官方
    Transformers 对同一输入的输出余弦相似度依次为 `0.999795`、`0.999391` 和
    `0.997787`。

### 7.1 2026-09-29 Qwen3.5 视觉产物迁移验收

- 独立 worktree：`/data01/home/jing.huang/xh2modelzoo_qwen35_visual_only`
- 分支：`feat/vllm-merak-migration`
- 当前状态：第一阶段迁移已经提交；原生浮点 WeMM 插件修改和测试已整理到第二阶段提交。
- 迁移验收过程中曾试验“缺少根级 `visual_config` 时直接失败”的收紧策略；
  提交前已撤销该实验，恢复原独立仓库的 `Qwen3_5MerakTextForCausalLM` 和
  专用纯文本 runtime，不将视觉契约变化纳入迁移提交。
- 恢复原契约后重新执行 Qwen3.5、Qwen3-Next、精确架构分发及平台配置回归，
  结果为 `66 passed`；ModelZoo 插件交付契约仍为 `10 passed`。
- 使用
  `work_dirs/qwen3_5_9b_flash_visual_export/hmquant_xh2_qwen3_5_9b_w4a8_256_2k_mpe256k_visualm96_196_384_704_1536_20260928`
  在 GPU 4 完成真实验收。解析结果为 `Qwen3_5MerakForCausalLM`、
  `runner_type=generate`、`image_input_supported=true`；同一引擎中的 greedy、
  temperature/top-k/top-p、penalty 和图文请求均成功生成 4 个 token。
- 同一 worktree 使用 WeMM-2B HMONNX 产物完成真实文本 embedding 回归，输出
  2048 维 `float32` 向量，L2 norm 为 `1.00000012`。
- Qwen3.5/Qwen3-Next 聚焦回归为 `41 passed`，WeMM 插件和评测适配回归为
  `26 passed`，ModelZoo 插件交付契约为 `10 passed`，WeMM 当前 CI 契约为
  `143 passed`。
- 插件完整测试结果为 `612 passed, 44 failed`；原工作目录在相同环境中的基线为
  `613 passed, 46 failed`。当前修改没有新增失败；剩余失败集中在迁移快照已有的
  工具路径、Gemma4 harness、Runner API 和服务脚本文档债务，不属于本次
  插件迁移或 WeMM 适配。
- ModelZoo CI impact selector 因 `pyproject.toml` 变化选择了 full 模式（82 个
  测试文件），但在收集阶段被当前 `xhquant_22` 环境缺少
  `xhquant.quantization.yuehui.yuehui_qconfig` 阻断，具体涉及 CosyVoice3 和
  Qwen3-ASR 两个 YueHui 契约测试。相同命令在原始 checkout 上得到相同的两个
  收集错误，因此这是测试环境基线，不是本 worktree 引入的回归；已独立通过的
  本次直接相关 ModelZoo 测试仍为 `143 passed` 和插件交付契约 `10 passed`。

注意：测试通过是“迁移时/验收时”的证据。环境或依赖升级后，应重新执行下一节命令，不应只沿用历史结论。

## 8. 后续如何复核迁移是否仍然有效

在目标 `vllm` Conda 环境中执行：

```bash
cd /data01/home/jing.huang/xh2modelzoo

# 1. 从 ModelZoo 内的新位置安装插件，并检查运行环境
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh

# 2. 只复查，不重新安装
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh --check-only

# 3. 依赖元数据必须干净
/path/to/vllm-env/bin/python -m pip check

# 4. 对真实 HMONNX release 做最终验收
VLLM_PLUGINS=merak \
PYTHON_BIN=/path/to/vllm-env/bin/python \
CUDA_VISIBLE_DEVICES=0 \
  vllm_merak/tools/run_release_acceptance.sh \
  /path/to/HMONNX_MODEL_DIR 512
```

还应人工确认：

```bash
/path/to/vllm-env/bin/python - <<'PY'
import importlib.metadata
import vllm_merak

print("import:", vllm_merak.__file__)
for group in ("vllm.general_plugins", "vllm.platform_plugins"):
    for ep in importlib.metadata.entry_points(group=group):
        if ep.name == "merak":
            print(group, ep.value, ep.dist.locate_file(""))
PY
```

判定标准：

- import 路径和 editable `direct_url.json` 指向 `/data01/home/jing.huang/xh2modelzoo/vllm_merak`；
- `merak` general/platform 与 `wemm` general entry point 均存在且值正确；
- `pip check` 无 broken requirements；
- 打开插件时选择 Merak，关闭插件时走原生 vLLM；
- 真实 HMONNX 模型完成 PageAttention Cache 分配和三类采样；
- 日志中没有隐藏的 Transformers/原生模型 fallback。

## 9. 当前遗留事项和提交建议

1. 第一阶段 `vllm_merak` 迁入已经进入 ModelZoo Git 历史；原独立工作树继续保留作对照。
2. 原生浮点 WeMM 插件、MMEB-V3 调用切换、示例、契约测试和本文档作为后续 WeMM
   支持提交，避免继续依赖本机 `/data01/home/jing.huang/vllm` 中的私有源码修改。
3. 正式提交前重新运行插件契约测试、静态检查和文本/图像/视频真实 smoke。

## 10. 一句话回顾

本次迁移完成了 **源码归属迁移、vLLM 插件入口保留、editable 安装来源切换、服务/导出依赖隔离、Hybrid Cache 与 HMONNX 真实运行契约修复、CUDA ABI 与真实模型验收、CI/文档交付**。第一阶段迁入已经提交；原生浮点 WeMM 插件支持纳入第二阶段提交继续评审。
