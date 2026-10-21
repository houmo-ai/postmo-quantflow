# FLUX 3 Klein optional prompt component validation

Date: 2026-10-19. Task worktree: `/data01/home/jing.huang/xh2modelzoo-flux3-klein`,
base `b55b9dee`. Checkpoint moved, without a compatibility symlink, to
`/data01/datasets/flux-3-image-klein-prompt-upsampler`. The checkpoint's download
record pins Hugging Face `black-forest-labs/flux-3-image-klein-pup` revision
`3f3edb52b2c55de0d49feab28598b68f7f8e4830`. Its architecture is Qwen3.5-9B;
reference inference protocol is BFL commit `e53f4ea697fe45db8f38fca91a90f5623f7f8f37`.

## Environment and graph evidence

Validation used the task-local `work_dirs/flux3_env` interpreter with PyTorch
2.8.0+cu128, Transformers 5.16.1 and the installed xhquanttool, on RTX 6000D.
The task environment's `bin` and its base conda `bin` were placed on `PATH`,
and the worktree was placed on `PYTHONPATH`. GPU ordinals 5 and 6 ran task exports
and inference. These are SDK GPU-reference results, not NPU-board measurements.

| Stage | Result and retained evidence |
| --- | --- |
| Original BF16 upsampler | Real generation/editing completed, `work_dirs/pup_validation/native_report.json` |
| Independent component build | Three real graphs, `work_dirs/prompt_upsampler_w8a16/upsampler_component.json`; `work_dirs/pup_build.log` |
| Raw-source public composite build | Loadable seven-graph public bundle, `work_dirs/runtime_packages/flux3_klein_with_upsampler`; `work_dirs/pup_composite_build.log` |
| Reuse public composite build | Reused core and prompt graphs; `work_dirs/pup_reuse_build.log`; final bundle relocated to `work_dirs/runtime_packages/flux3_klein_optional_portable` |
| Real generation/editing | 1024 square, seed 42, 16 steps each; finite latents and exact direct-caption parity, `work_dirs/pup_pipeline_validation/report.json` |
| Core precision combinations | W16A16 and W8A8 core composites also passed both 16-step cases and exact direct-caption parity; `work_dirs/pup_pipeline_validation_w16/report.json`, `work_dirs/pup_pipeline_validation_w8/report.json` |
| Public CLI after relocation | Offline run, `work_dirs/pup_portable_cli.log`; PNG byte content equals the original composite run |
| Prompt Golden | Synthetic and real-reference cases for all three graphs, `work_dirs/prompt_upsampler_w8a16/upsampler_golden_report.json` |
| Golden I/O validation | Six graph steps, all input/output shapes checked and all outputs finite, `work_dirs/pup_final_io_validation.log` |
| Artifact reuse identity | Core graph/data files and all six prompt graph/data files match SHA256, `work_dirs/pup_artifact_parity.json` |
| Forged physical contracts | Language length, KV length and group-size changes rejected, `work_dirs/pup_guard_report.json` |
| Official/packaged effective inputs | Token IDs, masks, image patches/grid, M-RoPE positions and deltas are exact for generation/editing, `work_dirs/pup_input_parity.json` |
| Focused contracts and recipe coverage | 135 passed; `work_dirs/pup_contracts_release.log` |
| Missing optional component | Public `UNSUPPORTED_COMBINATION` before session loading; `work_dirs/pup_missing_component_release.log` |

The prompt graphs use W8A16H1 SEFP/group 64 independently of FLUX precision.
Prefill accepts `[1,256,4096]`, Decode `[1,1,4096]`; each has 16 full-attention
K/V cache inputs of `[1,4,8192,256]`. The 24 linear-attention layers use
`GDRChunkScan` in Prefill and `GDRRecurrentScan` in Decode. The one visual gear
accepts `[1,4096,1536]` patches and produces at most 1024 merged image tokens.

The original four core graphs are byte-identical to the previously validated
W8A16 bundle, so their existing synthetic/real Golden is retained under
`work_dirs/artifacts/flux3_klein_w8a16`. No old Golden or SDK cache is copied into
the clean public composite package. New prompt Golden resides separately in the
independent component. Golden collection uses one connected V2 call per graph
and case, including actual state produced by prior forwards; normal generation
continues through V2 after each capture.

## Executed public commands

Run from the task worktree with the environment described above:

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/runtime_packages/flux3_klein_w8a16 \
  --prompt-upsampler /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/flux3_klein_with_upsampler
```

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/runtime_packages/flux3_klein_w8a16 \
  --upsampler-bundle work_dirs/runtime_packages/flux3_klein_with_upsampler \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/flux3_klein_composed_reuse
```

The second output was renamed to `flux3_klein_optional_portable` before running:

```bash
HF_HUB_OFFLINE=1 python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_optional_portable \
  --prompt-upsampling \
  --prompt 'A red ceramic teapot on a wooden table beside a window.' \
  --output work_dirs/pup_pipeline_validation/portable_cli_generate.png \
  --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.validate_upsampler \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_with_upsampler \
  --reference work_dirs/real_validation/native_generate.png \
  --output-dir work_dirs/pup_pipeline_validation --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.unified golden-upsampler \
  --artifact-dir work_dirs/prompt_upsampler_w8a16 \
  --image work_dirs/real_validation/native_generate.png --device cuda:0
```

The independent build preceding these commands called the same `build_component`
used by `unified build-upsampler`. The raw-source public build reran the actual
Qwen35 workflow; the reuse build performed no neural graph export.

## Numerical/behavior boundaries

For the real t2i sample, the upsampler generated 259 tokens and FLUX consumed 320
tokens including template and official padding. Editing generated 222 tokens,
with 240 FLUX tokens. W8 prompt captions differ in some details from BF16 greedy
captions, while preserving the tested subject/change constraints; no claim of
token-identical quantized-versus-native decoding or benchmark quality is made.
For both samples, passing the exact generated caption directly to FLUX produced
bit-identical final latents and PNG pixels. Repeated request state did not leak
into the next request. The portable public CLI reproduced the t2i PNG exactly.

`mm_token_type_ids` is removed at the processor/wrapper boundary because the
deployed Qwen wrapper computes M-RoPE from IDs/grid metadata. Actual token IDs,
image patches and grids remain the official processor outputs.
The raw and packaged processor tensors, and the native/deployed M-RoPE position
IDs and deltas, were separately checked for exact equality on both real inputs.
Presence penalty applies to generated tokens only. Expanded text is decoded and retokenized by
FLUX, never passed as Qwen3.5 hidden states. Capacity 1280 remains in force;
no silent truncation to it is introduced. Bbox and multiple-reference requests
remain outside this integration's public support.

The same prompt component was composed with the existing W16 and W8 core
packages through the reuse build. All six cases (three core precisions times
generation/editing) passed finite-latent and bit-exact direct-caption checks.
Additional public packages are `work_dirs/runtime_packages/flux3_klein_with_upsampler_w16`
and `work_dirs/runtime_packages/flux3_klein_with_upsampler_w8`. Their executed
build/validation logs are `work_dirs/pup_matrix_w16.log` and `work_dirs/pup_matrix_w8.log`.

## Regression scope

`select_tests.py --format report` selects 57 test targets for the shared optional
source declaration and FLUX changes. The original core four-graph build,
relocation, generation, editing and precision guards passed for all three core
precisions. The final catalog/resolver/upsampler/recipe-coverage run passed 135 tests.

The expanded run is not wholly green in this environment. Failures in other
model flows have been reproduced from the unchanged `b55b9dee` worktree at
`/data01/home/jing.huang/xh2modelzoo-flux3-pup-baseline`; reports are retained in
`work_dirs/pup_baseline_*.log`. Examples include the missing external GPTQModel
`deepseek_v4_dspark` module and pre-existing quantization CUDA failures. Those
other model implementations and the external GPTQModel repository were not
modified. The complete expanded run is retained in `work_dirs/pup_ci.log`.

The completed selected run collected 1039 tests: **1012 passed, 27 failed**,
with eight warnings, in 4322.31 seconds. All 27 failed node IDs are present in
the unchanged-baseline failure reports; `work_dirs/pup_regression_audit.json`
records zero unmatched failures. This is not reported as a green global CI run.
The task-owned baseline worktree was removed after retaining those reports.

All 18 changed Python files passed Ruff checks and format verification, and
both staged and unstaged diffs passed whitespace checks. `ocr` was unavailable
in this environment; the staged implementation was reviewed directly.

<a id="multireference-multigear"></a>

## 2026-10-21：多参考、多视觉档位及实际上下文容量

本节记录后续扩展；前面的单参考、单视觉图结果保留为当时的验证范围。
任务目录仍为 `/data01/home/jing.huang/xh2modelzoo-flux3-klein`，增量基线为
`04b19a4bd98b30788f6fde409f21a90f544fc5a3`。本轮不修改原始模型和 SDK。

组件 schema v2 从实际 Qwen3.5 metadata 读取上下文 C、prefill P 和视觉档位，
并复核语言图、KV 端口和各视觉图的物理 shape。schema v1 单视觉图组件仍可加载。
默认扩写器为两张语言图加三张视觉图，与 FLUX 四张核心图组合共九图。
每张参考独立选择最小可容纳视觉档位，按附件顺序合并；增加参考数量不会逐数量导出图。

| 配置 | 上下文 C | Prefill P | 合并视觉 token 档位 | 独立组件 |
| --- | ---: | ---: | --- | --- |
| 默认 | 8192 | 256 | 256 / 512 / 1024 | `work_dirs/prompt_upsampler_multigear` |
| 自定义验证 | 4096 | 128 | 256 / 512 / 1024 | `work_dirs/prompt_upsampler_multigear_4k` |

这两套均通过真实 Qwen3.5 权重导出，不是 ONNX 契约测试中的端口 fixture。
各视觉图 patch 输入分别为 `[1,1024,1536]`、`[1,2048,1536]`、`[1,4096,1536]`。
语言 Prefill 输入 `[1,P,4096]`，Decode `[1,1,4096]`；两者各含 16 个
`[1,4,C,256]` 的全注意力 K/V 端口，及 24 层线性注意力状态。
扩写精度保持 W8A16H1 SEFP、group 64，独立于三种 FLUX 核心精度。

官方参考图策略保持不变：一张最高 1024²、两张每张最高 724²、三至十张每张最高
512²，另做 EXIF、白底透明合成、32 对齐、LANCZOS 和 JPEG-q92 处理。
FLUX VAE 仍从原始附件独立处理。因此扩容 FLUX 的参考面积，不要求扩写器同步扩容。
扩写器输入长度 L 包含模板、文字和所有图片；本次生成预算为
`min(组件生成上限, C - L)`。默认生成上限 3584；输入没有剩余空间及未正常生成停止符
均报错。FLUX 仍独立检查 caption 的文本容量 1280。

默认三档覆盖官方常见面积，不是数量限制或经实测得出的最优档位数。
五档 `[64,128,256,512,1024]` 已覆盖清单/端口契约测试；本轮真实导出、Golden 和
完整推理验收针对三档，不将五档契约测试作为五档真实后端验收。

### 执行入口

以下命令从任务目录执行。使用 `work_dirs/flux3_env/bin/python`，并将该环境与其基础
conda 的 `bin` 放入 `PATH`、仓库放入 `PYTHONPATH`，限制 tokenizer/OpenMP 线程。
导出和推理使用 RTX 6000D 的 GPU HMONNX V2；这些结果不代表 NPU 板端性能。

```bash
python -m examples_merak.aigc.flux3_klein.unified build-upsampler \
  --model-dir /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --output-dir work_dirs/prompt_upsampler_multigear --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.unified build-upsampler \
  --model-dir /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --config work_dirs/upsampler_multigear/context4096.yaml \
  --output-dir work_dirs/prompt_upsampler_multigear_4k --device cuda:0
```

自定义配置由默认 YAML 仅将 `context_max_length` 改为 4096、
`prefill_chunk_length` 改为 128。导出日志分别为 `export_8k.log`、`export_4k.log`，
任务状态记录含完整 argv 和耗时，均位于 `work_dirs/upsampler_multigear/`。

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/runtime_packages/flux3_klein_adaptive_w8a16 \
  --upsampler-bundle work_dirs/prompt_upsampler_multigear \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/flux3_klein_multigear_w8a16
```

W16/W8 的组合构建分别替换核心目录、精度和输出目录。
4k 验证组合使用 `_4k` 扩写组件，输出 `flux3_klein_multigear4k_w8a16`。

```bash
python -m examples_merak.aigc.flux3_klein.validate_upsampler \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_multigear_w8a16 \
  --reference work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --reference work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --reference-counts 1,2,4,8 --steps 16 --device cuda:0 \
  --output-dir work_dirs/upsampler_multigear/validation_w8a16
```

```bash
python -m examples_merak.aigc.flux3_klein.unified golden-upsampler \
  --artifact-dir work_dirs/prompt_upsampler_multigear \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png --device cuda:0
```

Golden 主请求使用八附件串联执行，采集实际调用的 prefill、decode、256 视觉图。
512/1024 图使用显式校准视图补齐，分别为 704×736 与 1024×1024；记录视图而不将其
称为原始八附件的官方尺寸处理。合成及真实来源分别检查每图全部输入/输出 shape、dtype、
输出有限值。语言状态来自实际相连调用，不独立伪造各层状态。

### 官方输入与实际容量验证

原始 BF16 Qwen3.5 的双图、八图扩写均正常结束，视觉特征有限，证据为
`work_dirs/upsampler_multigear/native_multi/report.json`。
导出组件在 0/1/2/4/8 图请求上，原始与包内 processor 的 token IDs、mask、patches、
grid 均逐值一致；原生与部署端 M-RoPE 的位置和 delta 逐值一致。
两套上下文配置均正常生成，八图重复请求的完整 caption 一致。

| 参考数 | 输入 token | 每图视觉 token / 所选档 | C=8192 的生成预算 | C=4096 的生成预算 |
| ---: | ---: | --- | ---: | ---: |
| 0 | 753 | 无 | 3584 | 3343 |
| 1 | 1634 | 1024 / 1024 | 3584 | 2462 |
| 2 | 1580 | 484 / 512 | 3584 | 2516 |
| 4 | 1640 | 256 / 256 | 3584 | 2456 |
| 8 | 2672 | 256 / 256 | 3584 | 1424 |

此表使用相同的多图合成请求检查各参考数量；完整 FLUX 的单图改色请求使用另一条提示词，
因此其输入 token 数不同。八图输出 300 个 token，正常在 1424 的实际预算内结束；
不要求固定预留 3584 才允许推理。两套结果见
`component_check_8k_retry/report.json`、`component_check_4k_retry/report.json`。

实际 W8A16H1 HMONNX 视觉输出与原始 BF16 对比，双图逐图余弦为
0.996411～0.997112，八图为 0.994834～0.997570，最大 RMSE 分别为
0.022083、0.027006。量化 caption 与原生 caption 有细节差异，不声明逐 token 相同。
八附件重复使用两张红/蓝茶壶图，验证数量、顺序、位置和状态，不等同于八主体画质评估。

### 完整级联与交付验收

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 三种核心精度的多图级联 | verified | `validation_{w16,w8a16,w8}/report.json`，各含文生图及 1/2/4/8 参考，共 15 个样例 |
| 4096 / 128 的完整级联 | verified | `validation_4k/report.json`，文生图及八图，共 2 个样例 |
| 扩写 caption 接口一致性 | verified | 全部 17 个样例均为 1024²、seed 42、16 步，扩写路径与直接使用同一 caption 的完整 latent 和 PNG 逐值一致 |
| 请求隔离 | verified | 各最终报告记录切换请求后八图扩写 caption 完全复现 |
| 包搬迁后的公开 CLI | verified | `portable_cli.status.json`、`portable_cli_edit_8.png`；离线八图与 W8A16、4k 包 PNG 逐像素一致 |
| 两套扩写组件 Golden | verified | 各 5 图 × 合成/真实 2 例，共 20 组，shape/dtype 和输出有限值均通过 |
| 组合包完整性 | verified | `artifact_audit.json`，四套组合包各 9 图、44 个 Manifest 文件，逐文件 SHA256 与外部权重闭包通过 |
| 核心图及 Golden 来源 | verified | 四图及外部权重与相应 adaptive 核心字节一致；Golden 复用来源逐图登记，不宣称新八图逐算子采集 |
| 扩写图与 Golden 来源 | verified | 组合包的五图及外部权重与对应独立组件字节一致，使用其本轮新采集的 Golden |
| 最终增量 CI | verified | `ci_complete.xml`、`ci_complete.log`，181 passed；包含真实小模型三种精度的核心导出及统一接口 |
| 五档实际导出、十图级联、NPU 板端及正式画质 benchmark | unsupported | 本轮未执行，不从三档、八附件 GPU 功能验证推断这些范围 |

以上相对证据路径均位于 `work_dirs/upsampler_multigear/`；Golden 报告例外，保存在
各独立扩写组件根目录的 `upsampler_golden_report.json`。
总验收索引为 `work_dirs/upsampler_multigear/acceptance.json`。

完整纯推理组合产物如下；每套包含 FLUX 四图和扩写器五图、两套 processor、embedding、
系统提示词、运行 metadata、根清单及 Manifest，部署无需原始 checkpoint：

| 核心精度 / 扩写上下文 | 产物目录 | Manifest 文件总量约 |
| --- | --- | ---: |
| W16A16 / 8192 | `work_dirs/runtime_packages/flux3_klein_multigear_w16` | 39.12 GiB |
| W8A16 / 8192 | `work_dirs/runtime_packages/flux3_klein_multigear_w8a16` | 30.87 GiB |
| W8A8 / 8192 | `work_dirs/runtime_packages/flux3_klein_multigear_w8` | 30.87 GiB |
| W8A16 / 4096 | `work_dirs/runtime_packages/flux3_klein_multigear4k_w8a16` | 30.86 GiB |

容量配置均保持 FLUX `max_references: 8`、参考总预算 8704；两图每张约 1 MP，
更多图由 FLUX 既有预算策略动态缩小。扩写器按自己的官方面积规则独立处理。

验收曾因会话中断停止。续跑使用 `resume_validation.py` 保留原日志中已完成全部断言的
样例，继续执行其余样例，最后增加跨请求的八图复现。W16/W8A16 的双图通过后，将四图、八图分配至独立 GPU
运行；原续跑进程在完整双图断言通过后主动结束。`parallel_*` 保存各用例的完整报告，
`collect_split_validation.py` 将通过的样例及原日志摘要汇总。命令、设备与耗时保留于
`*.{job,status}.json`；不将主动结束的进程报告为退出码 0。没有将只保存了 PNG、未通过对照的中断样例
计入完成结果。4k Golden 的十组数据已在中断前采集，使用以下命令完成全量 I/O 检查并
补写报告；原始连通采集日志保留在 `golden_4k.log`：

```bash
python work_dirs/upsampler_multigear/finish_golden_4k.py
```

最终 CI 从以下正式入口重跑，中断的 `ci_release` 不作为通过证据：

```bash
./benchmark_test/ci_test_transformer5.16.0/run.sh \
  --junitxml=work_dirs/upsampler_multigear/ci_complete.xml
```

七个改动 Python 文件通过 Ruff check/format，文档本地链接及 Bash/Python 代码块通过
静态检查，两种 `git diff --check` 通过。环境无 `ocr`，对组件清单、物理容量、
多图顺序、状态重置、Golden hook 恢复及公开构建边界进行了人工审查。
