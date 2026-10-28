# vLLM Merak 插件

`vllm_merak` 是 **ModelZoo 仓库内唯一维护的 vLLM Merak
源码**。它仍是独立安装的 vLLM 插件，不会并入 ModelZoo inference，也不会让
ModelZoo 的量化、导出环境强制安装 vLLM。

运行边界：

- WeMM Float：原生 vLLM 进程，设置 `VLLM_PLUGINS=wemm`；
- 其他 Float/GPTQ：原生 vLLM 进程，按模型需要选择插件；
- HMONNX：vLLM Merak 进程，设置 `VLLM_PLUGINS=merak`，并使用
  `config_format="merak_llm"`、`load_format="merak_llm"`；
- 两类进程使用独立环境，避免插件平台选择和 Torch/xhquant ABI 相互污染。

Merak 启动时通过源码中的字面量注册声明发现 loader，向 vLLM 注册延迟导入路径。
配置解析和模型加载只导入选中的 loader；未选中模型的 Transformers 模块不参与启动。
新增 loader 的 `MERAK_MODEL_REGISTRY.register_module` 名称须使用字符串或字符串列表字面量。

## 目录与维护边界

```text
xh2modelzoo/
└── vllm_merak/
    ├── pyproject.toml             # 独立 Python 包及 vLLM entry points
    ├── runtime_profiles.toml      # 已验收服务环境
    ├── vllm_merak/                # 插件实现
    ├── tests/                     # 插件完整回归用例
    ├── examples/                  # 离线推理和服务入口
    ├── tools/                     # 安装、环境验收及诊断工具
    ├── docs/                      # 部署、设计与迁移记录
    └── SOURCE_SNAPSHOT.md         # 原仓库迁移来源与验收记录
```

后续插件代码、测试、文档和版本都在此目录修改；旧的独立
`vllm-merak` 工作树仅作为迁移回退副本，不再作为安装或开发入口。
项目目录与 Python 包同名；外层 `__init__.py` 只负责将从 ModelZoo 根目录发起的
`import vllm_merak` 定位到内层 `vllm_merak/` 包，安装和发布仍由本目录的
`pyproject.toml` 管理。

完整迁移过程、修改边界与验收结果见
[vllm_merak 迁移修改记录](docs/vllm_merak迁移修改记录.md)。

## 运行环境

量化/导出环境与 vLLM 服务环境必须分开。插件包故意不在
`project.dependencies` 中声明 Torch、vLLM、xhquant 等二进制栈，安装时必须
使用 `--no-deps`，防止 pip 替换已经按 CUDA/Torch ABI 配好的 wheel 或扩展。
具体且可机器检查的版本组合见 [runtime_profiles.toml](runtime_profiles.toml)。

本次环境验收使用：Python 3.12、vLLM 0.23.x CUDA-target、Torch
2.11.0+cu129、Transformers 5.13.0、ONNX 1.22.0、xhquanttool 0.9.2。ModelZoo 与 xhquanttool 的完整
量化/导出依赖分别放在各自的 `export` extra 中，服务环境的 editable 安装不会
再声明另一套 Torch/Transformers/ONNX 约束。

Merak 对 vLLM 暴露 CPU-like 调度平台，但 logits sampling 实际运行在 CUDA。
因此服务环境必须使用 CUDA-target vLLM，并提供
`vllm._C_stable_libtorch` 与 `torch.ops._C.apply_repetition_penalties_`；仅安装
CUDA Torch 搭配 CPU-target vLLM 不能满足 repetition/presence/frequency penalty。
当前公共依赖与运行 ABI 记录在 `runtime_profiles.toml`，最终以环境检查器和
`python -m pip check` 同时通过为准。

## 从 ModelZoo 安装

先在独立的 vLLM 服务环境中安装与当前 Torch ABI 匹配的 vLLM、xhquanttool，
并以 editable 方式安装 ModelZoo；随后从 ModelZoo 根目录执行：

```bash
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh
```

该命令会：

1. 从 `xh2modelzoo/vllm_merak` 安装插件，且不解析依赖；
2. 校验关键包版本；
3. 校验 `vllm-merak` 与 `xhmodel-zoo` 的 editable 源码路径；
4. 校验三个 vLLM entry point；
5. 校验 `import vllm_merak` 实际解析到 ModelZoo 内的源码。

仅复查现有环境：

```bash
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh --check-only
```

安装 CUDA-target vLLM 时必须显式传入源码树，避免把插件目录误装成 vLLM：

```bash
VLLM_SOURCE=/path/to/vllm \
CONDA_ENV=vllm \
  vllm_merak/tools/install_vllm_cuda_xhquant.sh
```

脚本默认从 vLLM 官方 wheel 仓库取得与源码 `HEAD` **同提交**、与当前 Torch
CUDA 版本同 variant 的预编译扩展；可设置 `VLLM_USE_PRECOMPILED=0` 改为本地
源码编译，此时只编译当前可见 GPU 的计算能力。两种路径都会在安装后立即实际
执行 stable sampler 的 repetition-penalty CUDA 算子，而不只是检查模块存在。
完成后必须执行：

```bash
/path/to/vllm-env/bin/python -m pip check
PYTHON_BIN=/path/to/vllm-env/bin/python \
  vllm_merak/tools/install_from_modelzoo.sh --check-only
```

## 插件发现验证

安装脚本完成后，可独立确认元数据：

```bash
python - <<'PY'
from importlib.metadata import entry_points

for group in ("vllm.general_plugins", "vllm.platform_plugins"):
    print(group, [(ep.name, ep.value) for ep in entry_points(group=group)])
PY
```

必须包含：

```text
vllm.general_plugins  merak -> vllm_merak:register_plugins
vllm.general_plugins  wemm  -> vllm_merak.native_wemm:register
vllm.platform_plugins merak -> vllm_merak.platform:register
```

修改 `pyproject.toml` entry points 后必须重新安装插件，单纯修改源码不会刷新
site-packages 中的 entry-point 元数据。

## WeMM 原生浮点推理

`wemm` general plugin 在插件包中提供原生浮点 WeMM 模型，并复用 vLLM
Qwen3.5 的语言模型、视觉塔和 pooling runner。模型差异集中在
`model_executor/model_loader/wemm_embedding_native.py`：

- 每条输入只保留一个末尾 `<embedding>` pooling token；
- 最后有效 token 的 hidden state 使用 FP32 L2 归一化；
- 视频的 `[T,H,W]` 网格按官方实现展开成 `T` 个 `[1,H,W]` 帧组；
- 视频 prompt 保留逐帧视觉边界。

官方 WeMM `config.json` 的 architecture 仍是
`Qwen3_5ForConditionalGeneration`。只有显式设置 `VLLM_PLUGINS=wemm` 时，插件才
把这个 source architecture 映射到原生 WeMM 类；插件自动发现但未显式选择时，
不会覆盖普通 Qwen3.5。因而无需修改官方模型目录，也无需传入 `hf_overrides`。

文本、图像或视频 smoke：

```bash
cd /path/to/xh2modelzoo/vllm_merak
CUDA_VISIBLE_DEVICES=0 \
python examples/vllm_wemm_embedding.py \
  --model /path/to/WeMM-Embedding-2B \
  --text "Represent this input for retrieval."
```

图像增加 `--image /path/to/image.jpg`；视频增加
`--video /path/to/video.mp4 --video-frames 4`。示例在导入 vLLM 前固定
`VLLM_PLUGINS=wemm`，并直接使用官方 checkpoint 与 chat template。

## Qwen3-VL-Reranker HMONNX 评分

已支持 Qwen3-VL-Reranker-2B 的 W8 和 AutoRound W4 发布包，通过 pooling runner
执行文本、图片和图文相关性评分。支持 Query 带图、多图、动态视觉与 DeepStack、
32K 及分块调度；单引擎使用 `max_num_seqs=1`。离线评分、在线 `/rerank`、
旧产物准备方式和实际验证范围见 [部署说明](docs/qwen3_vl_reranker_merak_deploy.md)。

## HMONNX 推理

模型目录必须包含：

```text
golden_meta_info.json
merak_config.json
hf_config/config.json
```

`merak_config.json` 至少声明：

```json
{
  "architectures": ["MerakForCausalLM"],
  "config_format": "merak_llm",
  "load_format": "merak_llm",
  "xh_model": {
    "model_type": "hmonnx",
    "meta_info": "golden_meta_info.json"
  }
}
```

离线 smoke：

```bash
cd /path/to/xh2modelzoo/vllm_merak
VLLM_PLUGINS=merak \
CUDA_VISIBLE_DEVICES=0 \
python examples/vllm_merak_generate.py --model /path/to/hmonnx-model
```

OpenAI 服务：

```bash
cd /path/to/xh2modelzoo/vllm_merak
VLLM_PLUGINS=merak \
CUDA_VISIBLE_DEVICES=0 \
bash examples/vllm_server_start.sh /path/to/hmonnx-model
```

`MerakConfigParser` 会读取 `hf_config/config.json` 的官方架构名，通过
`MERAK_MODEL_REGISTRY` 解析实际 Merak loader；`MerakForCausalLM` 只是入口
占位值，不是绕过模型架构检查的 fallback。

## 回归与 CI

插件自身完整测试保留在本目录：

```bash
cd /path/to/xh2modelzoo/vllm_merak
python -m pytest -q tests
```

ModelZoo 活跃 CI 会对本目录的每次变更运行包边界、entry point、环境交付文件
和源码卫生契约：

```bash
cd /path/to/xh2modelzoo
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

真实 HMONNX 模型启动、首 token、连续请求、KV/cache 边界和无原生 fallback
仍属于发布前 runtime 验收，不能用元数据或 mock 测试替代。仓库提供轻量
Qwen3.5-9B FlashAttention 配置和真实采样门禁：

```bash
# 量化参数仅用于 runtime smoke，不作为精度基线：nsamples=2、iters=2、seqlen=128。
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir /path/to/Qwen3.5-9B \
  --config-path vllm_merak/test_profiles/qwen3_5_9b_flash_attention_smoke.yaml \
  --quant-output-dir work_dirs/qwen3_5_9b_flash_smoke_quant \
  --export-output-dir work_dirs/qwen3_5_9b_flash_smoke_export \
  --device cuda:0 --quick-test

VLLM_PLUGINS=merak CUDA_VISIBLE_DEVICES=0 \
  vllm_merak/tools/run_release_acceptance.sh \
  /path/to/exported-model 512
```

该门禁依次检查服务环境/entry point/`pip check`、Flash/PageAttention 图 ABI，
再连续执行 greedy、temperature/top-k/top-p 和三种 penalty 参数；任何一个请求
没有生成 token 都会返回非零状态。Qwen3.5 smoke 配置保留完整视觉子图，发布验收
同时覆盖纯文本请求和至少一个图文请求。
