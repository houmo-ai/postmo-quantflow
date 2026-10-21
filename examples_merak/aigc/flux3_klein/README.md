# FLUX 3 Image Klein

统一模型 ID：`flux-3-image-klein`。输入为完整场景描述和可选参考图，输出 RGB。
本接入面向官方 BF16 蒸馏 checkpoint（flow-map），不把 Base、FP8 或 NVFP4 权重当作等价来源。
提示词扩写器是可选组件：默认直接接收完整 caption；构建包含扩写器且推理显式启用时，也可输入短请求。
接入官方扩写器请参阅[接入步骤、推理示例与实现引用](#prompt-upsampler)。

## 阅读入口

| 查阅内容 | 位置 |
| --- | --- |
| 适配范围、实现文件、提交及验收证据 | [完整适配记录](../../../docs/modeling/structures/flux3-image-klein.md#adaptation-overview) |
| 模型语义、Host/计算图边界与数值处理 | [模型解释](../../../xhmodel_merak/xh_other_model/models/flux3_klein/EXPLANATION.md) |
| 构建、精度与组件组合 | [环境与构建](#环境与构建) |
| 四张核心计算图及 token 容量 | [核心计算图与容量配置](#flux-核心计算图与容量配置) |
| 图像缩放、2048 输入及输出配置 | [参考图预处理](#参考图预处理)、[分辨率配置](#输入与输出分辨率配置) |
| 两图各 1 MP、更多图共享预算及扩容 | [参考图数量与容量选择](#reference-capacity) |
| 扩写器构建、多图、视觉档位及上下文 | [扩写器指南](#prompt-upsampler) |
| 已验收运行包及三种精度分别打包 | [产物与打包](#artifacts-and-packaging) |

## 环境与构建

在仓库根目录、已准备好 recipe 所列依赖的 Python 环境执行。
验证使用 Torch 2.8.0+cu128、Transformers 5.16.1 与 xhquanttool；无需安装官方应用的 vLLM/Gradio。
SDK 需包含大 head FlashAttention、CPU FP32 分块 attention 与 bounded-unfold Conv 实现。

```bash
mkdir -p work_dirs/artifacts
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --output-dir work_dirs/artifacts/flux3_klein_w8a16 \
  --precision w8a16_sefp --device cuda:0
```

可选精度为 `w16a16_sefp`、`w8a16_sefp`、`w8a8_sefp`。每种精度须分别构建。
独立权重量化阶段为 skipped，SDK 在 export 中执行 PTQ；group size 沿用 SDK 的固定 64。
源模型、校准和图导出在 CPU，当前权重 PTQ 模块临时使用指定 GPU。
`output_dir` 为最终交付根目录，必须不存在；不覆盖已有发布包。

统一 CLI/`build_model` 读取的基础配置为
`configs_merak/workflows/xh2a/other_models/flux3_klein/flux3_klein_1mp_w16.yaml`。
修改容量或尺寸时，修改该文件的 `export` 字段，保留 `model` 等其他配置。
`--precision` 单独选择图精度；仅修改旁边的 W8A16/W8 YAML 不会改变统一入口的容量。

### 可选参数：`--image-bundle`

该参数接收一个已导出的、未包含提示词扩写器的 FLUX 核心产物目录，用于减少重复导出。
不传时，从 `--model-dir` 指定的原始权重完整构建四张图；传入时按以下条件处理：

| 本次构建配置 | 行为 |
| --- | --- |
| 容量和精度相同，或仅参考数量上限不同 | 直接使用四张计算图及运行资源，生成构建配置和 Manifest |
| 仅参考总容量和参考数量上限变化 | 使用文本/VAE 图及资源，重新量化导出 DiT |
| 目标/参考校准尺寸、目标或单参考容量、VAE 画布、面积预算、文本档位或精度不同 | 不使用此参数，执行完整构建 |

例如，使用匹配的核心计算图构建：

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/artifacts/flux3_klein_w8a16 \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/flux3_klein_adaptive_w8a16
```

构建仍会校验原始模型路径，输出目录必须不存在。输出产物包含独立推理所需的全部资源，
推理时无需访问 `--image-bundle` 指定的目录。部署环境需使用本仓库对应版本的运行库。

## 文生图与编辑

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/artifacts/flux3_klein_w8a16 \
  --prompt 'A realistic photograph of a red ceramic teapot on a wooden table beside a window, soft daylight, glossy glaze and gentle shadows.' \
  --height 1024 --width 1024 --steps 16 --seed 42 \
  --output work_dirs/flux3_generate.png --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/artifacts/flux3_klein_w8a16 \
  --prompt 'The same ceramic teapot as in the reference, with a blue glossy glaze, on the same wooden table beside the window, preserving the soft daylight and scene.' \
  --image work_dirs/flux3_generate.png \
  --height 768 --width 1376 --steps 16 --seed 42 \
  --output work_dirs/flux3_edit.png --device cuda:0
```

公开 Python API 为 `model_build.build_model` 和 `model_inference.inference_model`，与 CLI 使用相同链路。
结果 `stats.steps` 为实际步数。蒸馏版不接受 CFG/negative prompt；两个时间输入由 Host 根据
checkpoint `alpha` 和请求步数计算。随机噪声排列、初始 BF16 舍入、FP32 持久状态及 BF16 更新增量
遵循参考实现。换尺寸、换参考图的请求不共享学习状态。

## FLUX 核心计算图与容量配置

| 图 | 边界 |
| --- | --- |
| `text_encode` | Qwen3-VL 的八层特征拼接，保留官方文本 padding query 的特征 |
| `vae_encode` | RGB 画布 → posterior mean、2×2 patchify、运行统计归一化后的 128 通道 latent 画布 |
| `denoise` | 三个 mode 的独立前置层及联合层，当前/下一步时间 → 目标速度预测 |
| `vae_decode` | 逆归一化、unpatchify 与 VAE decoder → RGB 画布 |

默认配置在 `configs_merak/workflows/xh2a/other_models/flux3_klein/`：

| 项目 | 默认值 |
| --- | ---: |
| 默认目标、校准参考尺寸 | 1024×1024 |
| VAE 物理画布 | 1536×1536 |
| 目标 token 容量 | 4352 |
| 单参考/参考总 token 容量 | 4352 / 8704 |
| 文本物理容量 | 1280 |
| 最大参考图数量 | 8 |
| 联合容量（派生） | 14336 |

真实 neural 宽高按 16 对齐，目标/参考 token 数为 H×W/256。
高宽、单图面积、参考总面积、数量和文本长度分别检查。容量内可改变宽高，不增加图。
默认最多八张参考图，共享 8704 个参考 token。单张最高 1048576 像素、每边不超过 1536；
两张各 1 MP 无需额外缩小，更多图片按实际总量动态缩小。超过数量上限、文本/目标容量，
或图像过小而无法完成 16 像素对齐时明确失败。
参考总容量增加时重新导出 DiT；文本容量或 VAE 画布改变时需要重建相应组件。

官方包配置声明 `max_tokens=8192`；本接入锁定参考实现的 4096 截断上限。
在其之下，部署容量 1280 计入 chat template 和官方 80-token 补齐；超出部署容量时报错，
不静默截断到 1280。

## 参考图预处理

单图遵循 EXIF 修正 → RGB → 等比 LANCZOS 缩小 → 居中 reflect/symmetric 填充。
多图先估算每张在单图面积/边长上限内的处理尺寸；若总 token 不超过参考总容量，保留这些尺寸。
超限时采用有上限的均分：小图保持原有预算，其余预算由大图共享；从 EXIF 修正后的原图
直接做一次等比 LANCZOS 缩小，再中心裁剪到 16 网格。不会拉伸成正方形，也不会放大小图。
对齐可能留下少量未使用预算，不会为填满容量反复缩放。
默认输出尺寸跟随第一张最终处理后的参考图；
单图且未指定任何输出边长时，结果裁回填充前区域，因此最终 PNG 宽高可以不是 16 倍数。

部署默认将单图面积预算从官方 4MP 收紧到 1MP，边长限制收紧到画布范围；该差异显式记录在配置。
总预算缩放是静态部署的附加策略：官方多图只限制每张约 1 MP，不共享固定的参考 token 上限。
公开推理结果的 `stats.geometry.references_hw` 给出实际处理后的高宽；需要固定 1024×1024
输出时显式传入 `--height 1024 --width 1024`，避免输出尺寸随第一张参考的缩小而变化。
官方反射填充属于有效图像。额外部署补零属于无效画布，卷积每层屏蔽；GroupNorm 仅统计
有效矩形，VAE attention 先 Gather 有效二维位置再执行 FlashAttention，然后恢复网格。

## 输入与输出分辨率配置

原始参考图尺寸和实际参与计算的尺寸分别处理。默认配置可接收 2048×2048 原图，但会按
1 MP 及共享预算自动缩小；若接受这种缩小，无需重新导出。默认档位输出 1024×1024 时，
推理直接指定 `--height 1024 --width 1024` 即可，也无需重新导出。

下面两组配置分别以默认 YAML 为基础，只替换列出的 `export` 字段，其他字段保留。
改变 VAE 画布或目标/单参考容量时，执行完整构建，不传 `--image-bundle`。

### 单张参考图保留 2048×2048，输出 1024×1024

2048×2048 为 4194304 像素，对应 `2048 × 2048 ÷ 256 = 16384` 个图像 token。
同时扩大 VAE 画布、单参考容量、参考总容量和单参考面积预算：

```yaml
export:
  reference_size: [2048, 2048]
  canvas_size: [2048, 2048]
  reference_capacity: 16384
  reference_total_capacity: 16384
  single_reference_max_pixels: 4194304
```

`reference_size` 是构建时的参考校准尺寸，不会强制把任意输入拉伸成该尺寸。
此方案保留默认 `output_size: [1024, 1024]` 和 `target_capacity: 4352`。

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --output-dir work_dirs/artifacts/flux3_klein_ref2048_out1024_w8a16 \
  --precision w8a16_sefp --device cuda:0
```

将 `/path/to/reference_2048.png` 替换为实际的单张参考图路径，编辑时显式指定输出尺寸：

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/artifacts/flux3_klein_ref2048_out1024_w8a16 \
  --image /path/to/reference_2048.png \
  --prompt 'A realistic photograph of the same subject as the reference image, preserving its identity, proportions and composition, with soft natural daylight.' \
  --height 1024 --width 1024 --steps 16 --seed 42 \
  --output work_dirs/flux3_ref2048_out1024.png --device cuda:0
```

不能省略这里的输出宽高：单参考编辑默认跟随处理后的参考图尺寸，可能尝试生成 2048×2048，
而本方案仍使用 1024 档位的目标容量，超限会报错。

### 生成 2048×2048 输出

将默认目标尺寸和 VAE 画布设为 2048×2048，目标容量至少设为 16384：

```yaml
export:
  output_size: [2048, 2048]
  canvas_size: [2048, 2048]
  target_capacity: 16384
```

参考图面积和参考总预算可保持默认；输出 2048 不要求参考图也为 2048。

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --output-dir work_dirs/artifacts/flux3_klein_out2048_w8a16 \
  --precision w8a16_sefp --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/artifacts/flux3_klein_out2048_w8a16 \
  --prompt 'A realistic photograph of a red ceramic teapot on a wooden table beside a window, soft daylight, glossy glaze and gentle shadows.' \
  --height 2048 --width 2048 --steps 16 --seed 42 \
  --output work_dirs/flux3_out2048.png --device cuda:0
```

编辑时可增加 `--image <参考图路径>`。若同时要求单张参考保留 2048×2048 且输出 2048×2048，
合并以上两组 YAML 字段，再完整构建；无需修改 Python，`joint_capacity` 会自动派生。
仅在当前 1024 档位推理时指定 `--height 2048 --width 2048` 不会扩大图容量，会报错。

以上输入保留配置针对单张参考。当前多参考预处理每张仍有 1 MP 的代码上限；若要求多张
参考每张都保留 2048×2048，仅修改 `multi_reference_max_pixels` 不够，还需调整
`xhmodel_merak/xh_other_model/models/flux3_klein/pipeline.py` 中的多参考面积上限，并设置足够的
单参考/参考总容量。2048 档位会增加计算和运行内存需求，需另行完成导出、真实推理和 Golden
验证；这里提供配置与使用方法，不将其列为已经验收的档位。

## 多参考图推理

所有参考共用同一张 VAE encoder，按附件顺序逐张编码；latent 拼接后交给同一个 DiT。
第四个位置坐标为 -1、-2、…，目标和参考的空间坐标规则不变。各张图可有不同宽高。
多图没有单图 reflect padding 的输出裁回步骤，未指定尺寸时跟随第一张处理后的参考图。

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_adaptive_w8a16 \
  --prompt 'Create a photograph of the red teapot from the first image on the left and the blue teapot from the second image on the right, together on a wooden table in soft daylight.' \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --height 1024 --width 1024 --steps 16 --seed 42 \
  --output work_dirs/flux3_dual_edit.png --device cuda:0
```

这里直接传入完整双图 caption；如需扩写简短请求，使用包含扩写器的组合产物并增加
`--prompt-upsampling`，接入方式见[扩写器指南](#prompt-upsampler)。
原生对照和双图级联验证使用以下命令；第二个命令分别替换为三种精度的包即可。

```bash
python -m examples_merak.aigc.flux3_klein.validate_dual native \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --output-dir work_dirs/dual_reference_native --steps 16 --seed 42 --device cuda:0
```

```bash
python -m examples_merak.aigc.flux3_klein.validate_dual deployed \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_adaptive_w8a16 \
  --baseline-dir work_dirs/dual_reference_native \
  --output-dir work_dirs/dual_reference_w8a16 --device cuda:0
```

验证逐步比较原生 BF16 latent 与部署结果，并检查交换参考顺序会影响结果、切换双图/单图/
无参考和尺寸后重复请求保持一致、统一公开接口与底层级联输出逐像素一致。

<a id="reference-capacity"></a>

## 如何选择四张参考图各 1 MP 和更多参考图

`max_references` 是请求数量上限，`reference_total_capacity` 是 DiT 的物理参考预算，
`multi_reference_max_pixels` 是每张参考的面积上限，三者分别生效。动态缩放始终由包内
容量驱动，不需要为三图、四图、八图分别修改 Python。当前协议最多十张参考。

| 需求 | `max_references` | `reference_total_capacity` | 计算图容量变化 |
| --- | ---: | ---: | --- |
| 默认：两张各 1 MP，三至八张共享预算 | 8 | 8704 | 保持默认容量 |
| 最多四张，每张都能达到 1 MP | 4 或更大 | 17408 | 扩大 DiT 参考容量 |
| 最多十张，共享当前预算、接受动态缩小 | 10 | 8704 | 不改变计算图容量 |
| 最多八张，每张都能达到 1 MP | 8 | 34816 | 扩大 DiT 参考容量 |

配置入口见“环境与构建”。例如四张各 1 MP，修改基础 YAML 的 `export`（其余字段保留）：

```yaml
export:
  max_references: 4
  reference_total_capacity: 17408
  multi_reference_max_pixels: 1048576
```

`joint_capacity` 自动派生为 23040，无需手动设置。然后使用全新输出目录执行：

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/flux3_klein_4ref_1mp_w8a16
```

上述命令从原始权重完整构建。需要减少重复导出时，可按“环境与构建”的适用条件传入
`--image-bundle`；只改变参考总容量时，构建器仅重新量化导出 DiT。
更多参考且需要每张 1 MP，可按 `reference_total_capacity = 4352 × 张数` 预留容量；
若只增加数量上限、接受自动缩小，保持 8704 即可。修改配置后需重新构建，并使用构建输出
目录推理，不直接修改发布目录中的 metadata/Manifest。各新增档位均需完成真实输入验证
和对应 Golden；配置示例表示构建方法，不代表该档位已经验收。多图直接提供完整 caption，
扩写需使用包含组件的组合产物，并满足扩写器自身的视觉及上下文预算。

多参考图共享容量时，当前分配和缩放策略如下：

1. 先计算各图的正常处理尺寸：每张不超过 1 MP、单边不超过 1536，保持宽高比，并中心裁剪到 16 像素对齐。
2. 统计 token：每张为 高 × 宽 ÷ 256。总量不超过参考容量时，不再额外缩小。
3. 超限才分配预算：按各图所需 token 从小到大处理。小图需要的预算少，就保留它的需求；剩余预算由其他大图共享。
4. 按最终预算缩放：从原图直接进行一次等比 LANCZOS 缩小，再中心裁剪到 16 像素对齐，不拉伸、不放大小图。

例如共享容量为 8704，输入一张只需 64 token 的小图和三张各需 4096 token 的大图：
小图保留：64 token
三张大图各分配：(8704 - 64) ÷ 3 = 2880 token
这里分配的是预算上限。最后做尺寸取整和 16 像素对齐后，实际 token 可能略少；当前不会为了填满剩余容量再反复调整。

## 四张参考图各 1 MP 的档位只传两张时的开销

输入同样两张 1024×1024 参考、目标 1024×1024、文本 160 token 时：

| 项目 | 当前共享预算包 | 四张各 1 MP 的大档位 |
| --- | ---: | ---: |
| 实际 VAE 编码次数 | 2 | 2 |
| 实际参考 token | 8192 | 8192 |
| 实际有效联合长度 | 12448 | 12448 |
| 参考物理容量 | 8704 | 17408 |
| 联合物理容量 | 14336 | 23040 |

大档位只传两图不会额外编码空图，文本编码和相同尺寸的 VAE 解码也不增加。
有效 token 会 Gather 到前面，FlashAttention 接收实际长度，padding 不作为有效条件参与
交互；但这不等于全部算子都跳过 padding。当前输入/QKV 投影、MLP、归一化、RoPE、
拼接/Gather 等仍面对更大的物理张量，会增加计算、缓冲区和内存读写，且每个去噪步都会发生。
参考分支物理长度为原来的 2 倍，联合部分约 1.61 倍；这不是端到端耗时或总显存的实测倍率。

动态预算方案保持物理容量不变，只提高允许数量。同样两图的处理像素和有效长度不变，
DiT 没有扩容带来的额外计算，仅增加少量 Host 预算判断；实际多传图片会增加 VAE 调用，
有效 token 增多时也可能增加注意力工作量。需要兼顾双图效率与四张各 1 MP 时，可保留
两个 DiT 档位，由应用按所需参考 token 总量选择，当前示例不自动路由档位。

## 动态多图验证

以下命令用八个参考附件检查 2/3/4/8 图的完整推理、模式切换后的复现、显式预处理一致性，
以及公开 API 一致性。示例重复两张已有图片，以验证数量与容量；不作为八主体画质 benchmark。
如替换图片、caption、seed 或步数，请移除 `--dual-baseline-dir` 或重新生成对应双图基线。
`dit_io/` 记录八图请求首步的连通顶层 I/O，区别于逐算子 Golden。

```bash
python -m examples_merak.aigc.flux3_klein.validate_multi \
  --artifact-dir work_dirs/runtime_packages/flux3_klein_adaptive_w8a16 \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_generate.png \
  --image work_dirs/pup_pipeline_validation/hmonnx_edit.png \
  --dual-baseline-dir work_dirs/dual_reference_w8a16 \
  --output-dir work_dirs/adaptive_references/validation_w8a16 \
  --steps 16 --seed 42 --height 1024 --width 1024 --device cuda:0
```

## Golden 与数值对照

```bash
python -m examples_merak.aigc.flux3_klein.unified golden \
  --artifact-dir work_dirs/artifacts/flux3_klein_w8a16 \
  --prompt 'A blue ceramic teapot on the same wooden table, preserving the scene and soft daylight.' \
  --image work_dirs/flux3_generate.png --height 768 --width 1376 \
  --steps 1 --seed 42 --device cuda:0
```

Golden 通过真实级联 runtime 采集。每次运行先用 synthetic reference 覆盖四张图，保存各图首次
调用为 `step_0`；提供真实参考图时，再保存真实案例为 `step_1`。后续去噪步仍实际运行但不重复
保存逐算子数据，因此 Golden 命令可设为 1 步。已有任一 `step_*` 时拒绝覆盖。
`golden_meta_info.json` 记录案例、实际几何和调用数；Golden 不在普通 build 中自动生成。
SDK 生成的图链接会转为包内普通硬链接，保留公共 Manifest 的禁止软链接约束。

```bash
python -m examples_merak.aigc.flux3_klein.validate native \
  --model-dir /data01/datasets/flux-3-image-klein \
  --output-dir work_dirs/real_validation --device cuda:1
python -m examples_merak.aigc.flux3_klein.validate deployed \
  --artifact-dir work_dirs/artifacts/flux3_klein_w8a16 \
  --output-dir work_dirs/real_validation --device cuda:0
```

对照固定原生生成图作为编辑参考，保存每步 latent、RGB PSNR 和 cosine/RMSE，检查重复请求及公开接口。
已有完整 paired 结果时，可将 `deployed` 改为 `checks` 单独复验轨迹门禁、公共接口与请求隔离。
这些是样本级数值检查，PSNR 不代表画质 benchmark 成绩；GPU HMONNX Golden 不是 NPU 板端性能证明。
实际执行证据和适用范围维护于 [适配记录](../../../docs/modeling/structures/flux3-image-klein.md)。

<a id="prompt-upsampler"></a>

## 可选的官方提示词扩写器

### 选择接入方式

官方扩写器已经接入，可直接使用以下构建和推理入口，无需另外编写模型适配代码。
运行流程为：短提示词与可选参考图 → Qwen3.5 扩写 caption → FLUX 自身重新分词并
提取文本特征 → FLUX 四张计算图完成生成。构建时加入组件和推理时启用扩写是两个独立步骤。

| 已有资源 | 构建方式 | 推理时操作 |
| --- | --- | --- |
| FLUX 与扩写器的原始权重 | `build --prompt-upsampler <扩写权重目录>` | 对组合产物使用 `infer --prompt-upsampling` |
| 匹配的 FLUX 核心产物、扩写器原始权重 | 上述命令增加 `--image-bundle <核心产物目录>` | 同上 |
| 已导出的独立扩写组件 | `build --upsampler-bundle <组件目录>`；可同时指定匹配的 `--image-bundle` | 同上 |

下文依次提供构建、CLI/Python 推理、Golden 和验证示例。
实现位置与此前执行结果见本节末尾的[接入实现与验证记录](#prompt-upsampler-references)。
扩写支持文生图和多参考编辑；完整 caption 的请求仍可省略扩写开关。

FLUX 核心仍由四张计算图组成。提供 `sources.prompt_upsampler` 后，会额外加入官方
Qwen3.5-9B 扩写组件，复用现有 Qwen3.5 工作流导出 `prefill`、`decode`，
以及每个视觉档位对应的一张 `visual_m*` 图。所有神经网络前向均使用打包后的 HMONNX，推理时无需访问
FLUX 或扩写器的原始权重目录。

扩写器使用 W8A16H1 SEFP 精度，独立于 FLUX 核心精度。默认 prefill 分块为 256、
上下文为 8192，视觉档位为 256/512/1024 个合并后 token，分别对应 1024/2048/4096
个 patch token。这些是默认导出配置，不是 FLUX 包装层固定的容量上限；配置方法见下文。
当前提供文生图及多参考编辑扩写，不公开 bbox/layout 接口。

### 构建包含扩写器的产物

扩写器原始权重目录为 `/data01/datasets/flux-3-image-klein-prompt-upsampler`；
若下载到其他位置，替换下列命令中的路径，保留权重包自带的 processor 和 `system_prompts.json`。
从原始权重构建包含扩写器的完整产物：

```bash
python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --prompt-upsampler /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/flux3_with_upsampler
```

不传 `--prompt-upsampler` 时，仅构建 FLUX 核心。若要复用已构建的核心，传入
`--image-bundle <existing-core-bundle>`，其精度、文本和 VAE 档位须与本次构建配置一致；
仅参考总容量变化时自动重新导出 DiT。
组合构建只复制运行所需文件，不修改原产物。也可以单独构建一次扩写器，再与不同的
FLUX 核心产物组合：

```bash
python -m examples_merak.aigc.flux3_klein.unified build-upsampler \
  --model-dir /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --output-dir work_dirs/flux3_prompt_component --device cuda:0

python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/flux3_core \
  --upsampler-bundle work_dirs/flux3_prompt_component \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/flux3_composed
```

`--prompt-upsampler` 与 `--upsampler-bundle` 互斥，输出目录必须尚不存在。
独立扩写组件通过 `upsampler_component.json` 记录计算图和资源清单；组合后的公开构建
产物还在根目录提供完整的 `export_meta_info.json` 和构建 Manifest，覆盖全部语言/视觉/FLUX 图、
两套独立的 processor、embedding 和系统提示词。

公开 API 也支持相同的组件选择：

```python
from xhmodel_merak.model_build import build_model

result = build_model(
    model_id="flux-3-image-klein",
    model_dir="/data01/datasets/flux-3-image-klein",
    output_dir="work_dirs/flux3_with_upsampler",
    sources={
        "prompt_upsampler": "/data01/datasets/flux-3-image-klein-prompt-upsampler",
        # 可选："image_bundle": "work_dirs/flux3_core",
    },
    deployment_precision={"image": "w8a16_sefp"},
    device="cuda:0",
)
```

### 调整扩写器上下文、prefill 和视觉档位

默认扩写 workflow 为
`configs_merak/workflows/xh2a/other_models/flux3_klein/prompt_upsampler.yaml`。
可以修改该文件，或复制后通过独立构建的 `--config` 指定。相关字段位于 `export.model`：

```yaml
export:
  model:
    # 仅列出需要调整的字段，其他模型及量化设置保留。
    context_max_length: 8192
    prefill_chunk_length: 256
    visual_config:
      image_token_gears: [256, 512, 1024]
      image_token_capacity: 1024
```

`context_max_length` 是导出语言图的真实上下文/KV 容量；`max_pe_length` 是位置编码范围，
不能代替它。`prefill_chunk_length` 是分块长度，不是整条提示词的长度限制。
每张图按 processor 输出的实际视觉 token 数选择最小可容纳档位，再按附件顺序拼接；
不是给两图、四图、八图分别导出视觉图。一个 1024 档位也能逐张处理多图。

默认三档覆盖官方单图、双图和更多参考图的常见尺寸；不是档位数量限制。若小图较多，
可配置 `image_token_gears: [64, 128, 256, 512, 1024]`，让小图选择更小的图、减少补齐计算。
五档会额外增加两张视觉计算图及其导出、存储和 Golden 验证工作，不会要求一次推理执行全部五档。
通用 Qwen3.5 的五档 `[96,196,384,704,1536]` 面向另一种输入分布，不应直接照搬：
当前常见的 256、484、1024 token 图片会分别落到 384、704、1536 档，补齐量反而增加。

修改后使用全新目录重新构建扩写组件，再组合到 FLUX 核心：

```bash
python -m examples_merak.aigc.flux3_klein.unified build-upsampler \
  --model-dir /data01/datasets/flux-3-image-klein-prompt-upsampler \
  --config configs_merak/workflows/xh2a/other_models/flux3_klein/prompt_upsampler.yaml \
  --max-new-tokens 3584 --device cuda:0 \
  --output-dir work_dirs/flux3_prompt_multigear

python -m examples_merak.aigc.flux3_klein.unified build \
  --model-dir /data01/datasets/flux-3-image-klein \
  --image-bundle work_dirs/runtime_packages/flux3_klein_adaptive_w8a16 \
  --upsampler-bundle work_dirs/flux3_prompt_multigear \
  --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/flux3_multigear_with_upsampler
```

组件清单从实际导出 metadata 读取上下文、prefill 和全部视觉档位，加载时再检查物理图
输入及 KV shape；直接修改 metadata 不会扩大图容量。默认扩写组件含两张语言图和三张
视觉图，与 FLUX 核心组合共九图。只保留一个视觉档位时仍为七图，已有单档位组件可继续加载。

FLUX 与扩写器从原始附件分别处理图像。扩写器保留官方规则：一张最高约 1024²、
两张每张约 724²、三至十张每张约 512²，之后进行 32 对齐及 JPEG-q92 处理。
FLUX 改为四张或八张各 1 MP 时，不会要求扩写器也使用同样的面积；沿用上述规则且
现有视觉/上下文容量够用时，可复用扩写组件，只构建 FLUX 所需档位并重新组合。

### 推理时启用扩写

推理时仅在显式启用后执行扩写。因此，已有完整 caption 的请求可以复用同一套产物，
无需加载扩写器的神经网络组件：

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/flux3_with_upsampler --prompt-upsampling \
  --prompt 'A red ceramic teapot on a wooden table beside a window.' \
  --output work_dirs/expanded_generate.png --device cuda:0

python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/flux3_with_upsampler --prompt-upsampling \
  --image work_dirs/expanded_generate.png \
  --prompt 'Change the teapot to blue and keep everything else unchanged.' \
  --output work_dirs/expanded_edit.png --device cuda:0
```

使用 `inference_model` 时，在 `inputs` 中设置 `"prompt_upsampling": True`。
返回值 `stats.caption` 记录实际传给 FLUX 的完整 caption，`stats.upsampler` 记录
扩写器的输入/输出 token 数和扩写耗时。组件缺失、扩写结果为空或未正常结束、上下文
超限及 FLUX 文本容量超限都会明确报错，不会静默截断文本或切换到原生神经网络模型。
省略该选项或设为 `False` 时，直接使用传入的完整 caption。

```python
from PIL import Image
from xhmodel_merak.model_inference import inference_model

result = inference_model(
    artifact_dir="work_dirs/flux3_with_upsampler",
    expected_model_id="flux-3-image-klein",
    inputs={
        "prompt": "Change the teapot to blue and keep everything else unchanged.",
        "images": ["/path/to/reference.png"],  # 文生图改为 []，并提供生图提示词。
        "prompt_upsampling": True,
        "height": 1024,
        "width": 1024,
        "steps": 16,
        "seed": 42,
    },
    device="cuda:0",
)
Image.fromarray(result.image).save("work_dirs/expanded_edit_api.png")
print(result.stats["caption"])
print(result.stats["upsampler"])
```

### 多参考图扩写与验证

在已组合的多参考图产物上，重复 `--image` 即可启用多图扩写：

```bash
python -m examples_merak.aigc.flux3_klein.unified infer \
  --artifact-dir work_dirs/flux3_multigear_with_upsampler --prompt-upsampling \
  --image /path/to/reference_1.png --image /path/to/reference_2.png \
  --prompt 'Combine the reference objects into one scene, preserving their colors and identities.' \
  --height 1024 --width 1024 --steps 16 --seed 42 \
  --output work_dirs/expanded_multi.png --device cuda:0
```

参考数量受组合产物的 `max_references` 限制，当前默认最多八张，参考协议最多十张。
实际可用数量还受扩写器上下文、每图视觉容量和 FLUX 文本容量影响。
`stats.upsampler` 返回 `reference_count`、逐图 `image_tokens` 和 `visual_gears`，
以及实际 `context_length` 和本次 `max_new_tokens`，便于检查路由与预算。

```bash
python -m examples_merak.aigc.flux3_klein.validate_upsampler \
  --artifact-dir work_dirs/flux3_multigear_with_upsampler \
  --reference /path/to/reference_1.png --reference /path/to/reference_2.png \
  --reference-counts 1,2 --steps 16 --device cuda:0 \
  --output-dir work_dirs/flux3_multigear_validation
```

提供更多 `--reference` 后可用 `--reference-counts 1,2,4,8` 检查多个数量。
验证包含文生图、各数量编辑、精确复用扩写 caption 的 latent/PNG 对照，以及重复扩写的状态隔离。

### 协议与长度预算

扩写协议遵循 BFL 版本 `e53f4ea697fe45db8f38fca91a90f5623f7f8f37`：使用权重包
自带的文生图/编辑系统提示词，关闭 thinking，采用贪心生成；仅对已生成 token 施加
presence penalty，文生图为 1.5，编辑为 0；默认输出 token 上限为 3584，使用官方的两个
停止 token。文生图时将最终目标画布尺寸传给扩写器。编辑参考图保持附件顺序，在送入
原始 processor 前，按 BFL 流程处理 EXIF 方向、将透明区域以白底合成为 RGB、使用
LANCZOS 缩放并对齐到 32 像素，再经过 JPEG 质量 92 的编解码。
FLUX 的 VAE 输入则独立从原始参考图进行预处理。

扩写器输出的是文字，不直接传递 FLUX 条件编码所需的 hidden states。FLUX 使用自身
的 processor 对文字重新分词，再提取八层文本特征。当前部署文本容量仍为 **1280，
包含聊天模板及补齐至 80 的倍数所需的 token**；官方文本编码器的 **4096** 是另一项
文本处理上限。扩写器上下文和输出预算分别独立，不能互相替代。
运行时使用实际导出的上下文 C，计算 `max_new_tokens = min(组件生成上限, C - 输入 token 数)`。
输入统计包含模板、用户文字和全部图片；无剩余空间直接报错。动态收紧后仍必须正常生成停止符，
不返回因预算耗尽而未完成的 caption。修改 FLUX 的参考容量不会自动改变其文本容量 1280。

### Golden 与串联验证

串联流程的 Golden 需要显式采集，普通构建不会生成：

```bash
python -m examples_merak.aigc.flux3_klein.unified golden \
  --artifact-dir work_dirs/flux3_with_upsampler --prompt-upsampling \
  --prompt 'Change the teapot to blue and keep everything else unchanged.' \
  --image work_dirs/expanded_generate.png --device cuda:0
```

该命令记录四张 FLUX 图的合成样本与真实参考图样本，并为每个样本记录扩写器串联执行
中的一次 prefill、decode 和实际选中视觉图调用；
未经过的视觉档位使用显式校准视图补齐各档位的连通采集。后续 token 仍执行实际计算图，但不会全部
转储其中间层输出。Golden 采集会拒绝覆盖已有步骤。

独立扩写组件也可单独采集全部语言和视觉图的 Golden；若核心产物未变化，可保留核心已有的 Golden：

```bash
python -m examples_merak.aigc.flux3_klein.unified golden-upsampler \
  --artifact-dir work_dirs/flux3_prompt_component \
  --image work_dirs/expanded_generate.png --device cuda:0

python -m examples_merak.aigc.flux3_klein.validate_upsampler \
  --artifact-dir work_dirs/flux3_with_upsampler \
  --reference work_dirs/expanded_generate.png \
  --output-dir work_dirs/flux3_upsampler_validation --device cuda:0
```

验证会执行 16 步文生图和编辑，分别比较“启用扩写器”与“直接复用该次扩写得到的完整
caption”两条路径的完整 latent 和 PNG。当前 W16A16、W8A16、W8A8 三种核心精度均已
通过 1024×1024 文生图及 1/2/4/8 参考图的完整对照；4096 上下文、128 prefill 的扩写
组件也已通过文生图及八图对照，共 17 个样例。参考图使用红/蓝茶壶，八附件重复这两张图，
用于验证数量、接口和状态，不代表八主体画质 benchmark 或 NPU 板端性能。

<a id="prompt-upsampler-references"></a>

### 接入实现与验证记录

当前验收覆盖官方 BF16 输入与视觉特征对照、两套上下文的五张扩写图导出、九图组合构建、
搬迁后离线八图 CLI、全部扩写图 Golden，以及三种 FLUX 核心精度的文生图/多图级联。
具体命令、样例、结果和产物范围见
[多参考与多档位验证](../../../docs/modeling/structures/flux3-klein-prompt-upsampler-validation.md#multireference-multigear)。
各阶段的对照方法与完整日志索引保留在
[分阶段接入记录](../../../docs/modeling/structures/flux3-klein-prompt-upsampler-validation.md)，
每项结论均对应其实际构建档位及输入样例。

| 查阅目的 | 实现或示例 |
| --- | --- |
| CLI 参数与公开 API 调用方式 | [unified.py](unified.py) |
| 短提示词/参考图消息、官方图处理、解码与长度检查 | [upsampler.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/upsampler.py) |
| 独立语言/多视觉图构建、组件校验与组合打包 | [upsampler_build.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/upsampler_build.py) |
| 扩写器精度、prefill、上下文和视觉档位 | [prompt_upsampler.yaml](../../../configs_merak/workflows/xh2a/other_models/flux3_klein/prompt_upsampler.yaml) |
| caption 接入 FLUX 的位置与四张核心计算图采样流程 | [pipeline.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/pipeline.py) |
| 扩写与直接使用同一 caption 的完整对照 | [validate_upsampler.py](validate_upsampler.py) |
| 扩写组件全部图的连通 Golden | [upsampler_golden.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/upsampler_golden.py) |

模型内部职责和 tensor/state 接口另见[模型解释：可选官方扩写器](../../../xhmodel_merak/xh_other_model/models/flux3_klein/EXPLANATION.md#可选官方扩写器)。

<a id="artifacts-and-packaging"></a>

## 已验收产物与打包

以下目录相对于仓库根目录，均为本次实际构建、验证的运行包。FLUX 核心由四张计算图组成；
扩写组合由四张核心图、两张语言图和三张视觉图组成，共九张计算图。
两类运行包的默认参考图数量上限均为八张。参考图数量与计算图数量相互独立。

| FLUX 核心精度 | 核心运行包 | 带扩写器的组合运行包 |
| --- | --- | --- |
| W16A16 | `work_dirs/runtime_packages/flux3_klein_adaptive_w16` | `work_dirs/runtime_packages/flux3_klein_multigear_w16` |
| W8A16 | `work_dirs/runtime_packages/flux3_klein_adaptive_w8a16` | `work_dirs/runtime_packages/flux3_klein_multigear_w8a16` |
| W8A8 | `work_dirs/runtime_packages/flux3_klein_adaptive_w8` | `work_dirs/runtime_packages/flux3_klein_multigear_w8` |

表中扩写器统一采用 W8A16H1 SEFP、8192 上下文、256 prefill 和 256/512/1024 视觉档位。
4096 上下文、128 prefill 的验证组合包为
`work_dirs/runtime_packages/flux3_klein_multigear4k_w8a16`。
上下文、prefill 和视觉档位以各包实际 metadata 及物理图为准。

每个运行包包含相应计算图及外部权重、processor/tokenizer、embedding、运行 metadata、
`export_meta_info.json`、有效配置和 Manifest。部署使用对应版本的本仓库运行库与包内
`environment.json` 所列依赖，无需原始 checkpoint。三种精度各自独立部署。

三套核心包分别打包的命令如下；每个压缩包保留完整模型目录：

```bash
cd /data01/home/jing.huang/xh2modelzoo-flux3-klein/work_dirs/runtime_packages

tar -czf flux3_klein_adaptive_w16.tar.gz flux3_klein_adaptive_w16
tar -czf flux3_klein_adaptive_w8a16.tar.gz flux3_klein_adaptive_w8a16
tar -czf flux3_klein_adaptive_w8.tar.gz flux3_klein_adaptive_w8
```

需要分发扩写能力时，将命令中的目录及压缩包名称替换为表中对应的
`flux3_klein_multigear_*`。仅核心包接收完整 caption；组合包可显式启用扩写。

核心 Golden 位于 `work_dirs/artifacts/flux3_klein_adaptive_{w16,w8a16,w8}`；
两套扩写组件及其 Golden 位于 `work_dirs/prompt_upsampler_multigear` 与
`work_dirs/prompt_upsampler_multigear_4k`。验收索引分别为
`work_dirs/adaptive_references/acceptance.json` 和
`work_dirs/upsampler_multigear/acceptance.json`。具体输入、复用来源及验证范围见
[核心适配记录](../../../docs/modeling/structures/flux3-image-klein.md#adaptation-overview)和
[多图扩写验收](../../../docs/modeling/structures/flux3-klein-prompt-upsampler-validation.md#multireference-multigear)。
