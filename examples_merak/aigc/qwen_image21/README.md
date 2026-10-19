# Qwen-Image-2.1 / Turbo 构建与推理

## Qwen-Image-2.1-Turbo

Turbo 使用独立的模型 ID `qwen-image-2.1-turbo`，共享普通版的六图实现与容量规则。
先在 Merak 构建/推理环境中安装兼容的 xhquant SDK，再从本仓库根目录运行。
以下路径均为部署示例，需替换为实际仓库和官方 Turbo 权重目录；
Diffusers 和 Hub 使用与普通版相同的固定版本：

```bash
cd /path/to/xh2modelzoo
python benchmark_test/ci_test_transformer5.16.0/prepare_dependencies.py \
  --model-id qwen-image-2.1-turbo --packages diffusers huggingface-hub --install
python -m examples_merak.aigc.qwen_image21.unified \
  --model-id qwen-image-2.1-turbo \
  --model-dir /path/to/Qwen-Image-2.1-Turbo \
  --artifact-dir work_dirs/qwen_image21_turbo/release_w8a16 \
  --quant-type w8a16_sefp --device cuda:0 \
  --prompt 'A red ceramic teapot on a wooden table.' \
  --seed 42 --output work_dirs/qwen_image21_turbo/generated.png
```

构建目录必须不存在。省略 `--quant-type` 使用 W16A16，W8A8 使用 `w8a8_sefp`；
全部组件从 Turbo checkpoint 新鲜导出，不复用普通版的量化图。
本地来源目录中的 ModelScope 缓存 `.msc` 若不可读，会导致构建来源哈希失败；
此时应提供只含 `model_index.json`、`processor`、`scheduler`、`text_encoder`、`transformer`、`vae`
的可读副本。SDK 需要包含 `b514c978`（2026-10-09）的 Qwen Image 1MP 支持，
本次使用 `develop` 的 `b4c7f631`；请核对[接入记录中的 SDK 条件及验证范围](../../../docs/modeling/structures/qwen-image21-turbo.md)。
已有包推理时省略 `--model-dir` 和 `--quant-type`，保留 `--model-id qwen-image-2.1-turbo`。
编辑增加 `--image reference.png`，更改输出尺寸增加 `--height 768 --width 1376`。
支持的尺寸、单参考图、RGBA 输出与普通版一致，均受发布包容量限制。

当前依赖与普通版一致：Diffusers 0.41.0、Hub 1.33.0、Transformers 5.16.1；
请通过上面的 recipe 准备命令安装。本地既有 `xhquant_55` 的历史环境验证记录位于
`work_dirs/qwen_image21_turbo_validation_20261012/xhquant_55_environment/`，
迁移到最新 develop 后的验证记录另见接入记录。

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

build = build_model(
    model_id="qwen-image-2.1-turbo",
    model_dir="/path/to/Qwen-Image-2.1-Turbo",
    output_dir="work_dirs/qwen_image21_turbo/api_release",
    device="cuda:0",
    deployment_precision={"image": "w8a16_sefp"},
)
result = inference_model(
    artifact_dir=build.output_dir,
    expected_model_id="qwen-image-2.1-turbo",
    device="cuda:0",
    inputs={"prompt": "A red ceramic teapot.", "seed": 42},
)
assert result.stats["steps"] == 8
```

Turbo 的默认 8 步来自 checkpoint 的非均匀 `sample_sigmas`，仅把普通版设为 `steps=8`
不能复现它的调度。该序列与 scheduler 配置随包保存，搬移后仍可独立推理。
显式 `sigmas`（CLI 的 `--sigmas`）覆盖默认序列，须严格递减、在 `(0,1]` 内且不包含终点零。
指定 `steps` 不改变 Turbo 的固定序列；`stats["steps"]` 始终报告实际步数。
普通版仍默认 20 步。当前固定 Diffusers 版本的原生对照需显式传入 checkpoint sigma，
部署 runtime 已直接支持随包读取，细节与实际验证范围见[Turbo 接入记录](../../../docs/modeling/structures/qwen-image21-turbo.md)。

配置入口沿用 `configured`，Turbo 必须选用 `qwen_image21_turbo_1mp_*.yaml`，
其中 `export.model_id` 明确标记 Turbo。公开构建会拒绝把 Turbo checkpoint 当作普通版发布。

完整权重 W8A16 六图包已通过 GPU HMONNX 生成、编辑、多尺寸、CFG 与重复请求检查，
位于 `work_dirs/qwen_image21_all_configs_export_20261013/qwen_image21_turbo_1mp_w8a16/`。
seed42、checkpoint 八步、CFG1 的 1024×1024 文生图与 768×1376 编辑，
对原生 FP16 的 RGB PSNR 分别为 36.18/31.96 dB；
`work_dirs/qwen_image21_turbo_validation_20261012/` 下的
`fp16_w8a16_comparison.png` 和 `real_report.json` 保存图片对照及数值。
W16A16/W8A16/W8A8 均通过六图真实小模型测试；完整权重的本次数值结论限于 W8A16 和上述样本。

## 普通版与浮点诊断

已接入 Merak 统一构建与推理；浮点诊断仍可用于排查数值问题。
结构、验证范围和产物记录见[接入记录](../../../docs/modeling/structures/qwen-image21.md)。

使用用户指定的 `xhquant_55_tf_latest_20260828` 环境与物理 GPU 1。
初始环境的 Diffusers 不含此模型。先把 recipe 声明的正式发布版及 Hub 版本装入同一个环境：

```bash
conda activate xhquant_55_tf_latest_20260828
python benchmark_test/ci_test_transformer5.16.0/prepare_dependencies.py \
  --model-id qwen-image-2.1 --packages diffusers huggingface-hub --install
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
python examples_merak/aigc/qwen_image21/float_probe.py \
  --dtype bfloat16 --size 512 --steps 20 --output work_dirs/qwen_image21/bf16_run
python examples_merak/aigc/qwen_image21/float_probe.py \
  --dtype float16 --size 512 --steps 20 --output work_dirs/qwen_image21/fp16_run
```

安装命令从 recipe 读取 `diffusers==0.41.0` 与 `huggingface-hub==1.33.0`，由当前 Python 的 pip 安装；
不再依赖 `work_dirs` 下的源码或临时 Hub `PYTHONPATH`。CI 也在其既有环境准备阶段执行同一命令。
官方 0.41.0 安装包包含此模型，要求 `huggingface-hub>=1.32.0,<2.0`；0.40.0 不包含此模型。
此前导出和数值验证使用 Diffusers 0.41.0.dev0、Hub 1.31.0、Transformers 5.16.1。
正式版的 DiT/RoPE 和采样器源码与该次固定源码相同，VAE 的差异仅为注释和返回类型标注；
此轮依赖调整只做静态检查，未重跑本文记录的模型验证。

输出目录必须不存在。报告记录实际版本、模块边界最大幅值、首个非有限张量、
每步 latent 和最终图片；两种 dtype 均从 seed 42 的同一份 FP32 噪声出发。
`passed` 仅表示该样本的监测点未发现非有限值，**不代表画质验收通过**。
FP16 同时拦截 `aten.clip` 与 `aten.clamp`，避免上游裁剪掩盖 Inf；
报告中 `fp16_pre_clamp.calls` 应至少包含 `32 × steps` 次 DiT block 裁剪。
它不监测所有融合 kernel 内部中间值，也不证明所有输入均不溢出。

`--image path.png` 检查图像条件路径；`--no-cache` 关闭前缀缓存用于对照。
当前提示固定为红色茶壶，正式画质验收还需扩充提示、种子、参考图与输出尺寸。
日志中的实际耗时包含详细监测同步开销，不应作为推理性能指标。

诊断现在会校验 FP16 裁剪前的观测次数；少于 `层数 × 步数` 时判为诊断失败，
防止监测漏点被误报为“无溢出”。精度对照应从 checkpoint 分别按目标 dtype 加载，
不要对已加载模型反复调用 `.to(dtype)`：该调用也会改变原本保持 FP32 的浮点 buffer，
从而混入额外的数值变化。

## 统一构建与推理

沿用上面的固定依赖和 `CUDA_VISIBLE_DEVICES=1`，从仓库根调用：

```bash
python -m examples_merak.aigc.qwen_image21.unified \
  --model-dir weights/Qwen-Image-2.1 \
  --artifact-dir work_dirs/qwen_image21/release \
  --prompt 'A white porcelain teapot on a wooden table.' \
  --steps 20 --seed 42 --output work_dirs/qwen_image21/generated.png
```

`--model-dir` 会调用 `build_model(model_id="qwen-image-2.1", ...)`，目标目录必须不存在。
已有包推理时省略它；`--artifact-dir` 可指向搬移后的目录，推理无需原始 HF 权重。
参考图编辑加 `--image reference.png`。默认配置输出 1024×1024 RGBA、最多一张参考图；
参考图按比例拟合到容量范围，并将高、宽对齐到 32，同一份实际尺寸图像进入视觉编码器和 VAE。
输出可加 `--height 768 --width 1376`。默认画布为 1536×1536，实际面积上限为
4352 个 latent token（像素面积 4352×256），同时受画布每条边的上限约束。
例如 1024×1088、768×1376、1376×768、672×1536 均能复用同一个包。
文本容量由 tokenizer 实测系统模板、512 token 提示预算和视觉容量推导，超长提示报错。
CFG 加 `--negative-prompt '' --guidance-scale 2`；大于 1 的 guidance 必须显式给出负向提示词。
已有包不接受 `--quant-type`，精度从包内 Manifest 读取；该选项仅在同时指定 `--model-dir` 时用于构建。
当前 1MP W16A16/W8A16/W8A8 六图包已构建完成，位于
`work_dirs/qwen_image21/standard_rmsnorm_1mp_20261009/artifacts/` 的对应精度目录；
仅修改现有 VAE ONNX 的 RMSNorm 属性、runtime metadata 和 Manifest，没有重新导出权重。
三精度真实 VAE 均通过脏 padding 排除、换尺寸和重复请求检查；decoder 复用了各自之前保存的 DiT latent。
横评图为同目录上一级的 `fp16_w16_w8a16_w8a8_1mp_comparison.png`，图中明确标注此验证范围。
W8A16 另已通过完整多尺寸生成、编辑、CFG 隔离和重复请求；1024² 文生图 / 高768宽1376 编辑
对原生 FP16 的 RGB PSNR 为 40.52/41.74 dB，seed42、20 步、CFG1。
该完整链路横评为 `w8a16_full_pipeline_comparison.png`。
完整报告为 `validation_summary.json`，实际执行范围为 GPU 1 上的 HMONNX V2 golden。
本次使用 SDK 六文件补丁 `standard_rmsnorm_1mp_20261009/sdk_qwen_image21_standard_rmsnorm_1mp.patch`，
版本、算子核对和上一版横评记录见接入记录。
完整链作为 `image` 组件发布，内部仍按 text_encode、visual、denoise、VAE 角色平铺。
公开默认 recipe 为 XH2a W16A16 SEFP，统一接口可选 W8A8：

```bash
python -m examples_merak.aigc.qwen_image21.unified \
  --model-dir weights/Qwen-Image-2.1 \
  --artifact-dir work_dirs/qwen_image21/release_w8 \
  --quant-type w8a8_sefp \
  --prompt 'A white porcelain teapot on a wooden table.' \
  --steps 20 --seed 42 --output work_dirs/qwen_image21/generated_w8.png
```

对应 `build_model(..., deployment_precision={"image": "w8a8_sefp"})`。
W8A16 使用 `--quant-type w8a16_sefp`，或
`build_model(..., deployment_precision={"image": "w8a16_sefp"})`。
它将 Linear/Conv 权重设为 SEFP 8 bit、激活设为 SEFP 16 bit，cache 接口仍为 FP16；
W8A16 与 W8A8 都需要独立导出，不能修改已有包的 metadata 来切换精度。
对应配置为
[`qwen_image21_1mp_w8a16.yaml`](../../../configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_1mp_w8a16.yaml)。
Manifest、最终有效配置与 runtime metadata 都记录实际精度，发布和加载会交叉校验。
W8A8 配置见
[`qwen_image21_1mp_w8.yaml`](../../../configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_1mp_w8.yaml)，
也可用下方 `configured` 入口构建。未登记 W4、低内存导出模式或统一 Golden。
源 checkpoint 以 FP32 加载到 CPU，校准前向、图 trace 和 HMONNX 导出都在 CPU。
SDK 只在权重 PTQ 时将当前模块放到选定 GPU，随后恢复 CPU；图转换使用独立副本，
并释放量化后无用参数，避免整套 DiT 及副本同时驻留 GPU。
原生 FP16/BF16 诊断、部署 SEFP 精度和芯片实机运行是不同验证范围。

此前 72 图版本搬移后经公开接口完成文生图和编辑，旧 HMONNX 及其权重已按要求清理。
历史图片和报告仍保留；报告为
`work_dirs/qwen_image21/public_validation_report.json`；当时 CI 761 项通过。
实际执行为 GPU 1 上的 HMONNX V2，未做 NPU 板端验证。

## 两张完整 DiT 与自定义构建尺寸

新构建将整个 DiT 导出为 causal 文本/non-causal 图像两张图，整包六张学习组件图。
公开入口要求完整包 schema 3 和 DiT schema 4，旧发布包需重新构建。
两图都有全部层、投影、时间 MLP 和共享调制，attention 均使用 SDK FlashAttention。
实际 target token 补齐到容量后推理，输出裁回实际长度；RoPE 和时间调度使用实际尺寸。
条件按原始文本/参考图顺序分段跑完整层栈，t 固定为 0；编辑模板通常需要文本、参考图、文本三段。
后续复用图像图处理目标，SDK KVCache 原地覆盖实际条件之后的 [P,P+N) 区域，不输出整份 cache。
两图只使用实际 past/current 长度控制可见范围；请求和正负 CFG 分支各自持有 cache。
VAE 接口和 mask 均为四维 NCHW，原通道归一化使用现有 SDK RMSNorm，epsilon=1e-8，无需扩展算子。
最新三精度包与本次复测材料位于 `work_dirs/qwen_image21/standard_rmsnorm_1mp_20261009/`；
`segmented_1mp_20261009/` 保留修改前的构建与完整链路报告。

默认统一构建使用 W16 YAML，并可通过 `deployment_precision` 选择 W8A8。
要构建其他容量，复制对应 YAML 并设置精度、output_size、reference_size、canvas_size、
target_capacity、reference_capacity、visual_patch_capacity、prompt_budget、max_references 和 max_position。
不要指定 feature_capacity/prefix_capacity，这两项由 tokenizer 自动推导。

```bash
python -m examples_merak.aigc.qwen_image21.configured \
  --model-dir weights/Qwen-Image-2.1 \
  --config /path/to/my_qwen_image21.yaml \
  --output-dir work_dirs/qwen_image21/custom_build
```

输出包在 custom_build/qwen_image21_export。直接 workflow 产物携带 export_meta_info.json
和 runtime_meta_info.json，不生成统一 build_model 的 Manifest；用模型运行库加载：

```python
from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import load_runtime, read_bundle_metadata

root = 'work_dirs/qwen_image21/custom_build/qwen_image21_export'
meta = read_bundle_metadata(root)
runtime = load_runtime(root, device='cuda:0')
image, latents = runtime.generate('A white teapot.', height=768, width=1376, steps=20)
image.save('custom.png')
```

配置中尺寸顺序为高、宽，实际输入输出必须对齐到 32。视觉按紧凑 patch 容量复用；
VAE 在固定画布计算，每层卷积输入输出、重采样和残差路径都有对应分辨率的 mask。
VAE attention 将有效二维位置收集到连续前缀，FlashAttention 排除 padding 后再恢复二维网格。
修改容量需要重新构建；容量内改变实际尺寸不增加图数量。

### 尺寸需求与容量规划

当前 Turbo W8A8 包约束的是**导出容量**，不是官方模型每边只能生成 1024。
`output_size` 是默认输出/导出校准尺寸，`reference_size` 是导出校准参考尺寸，
真正的运行边界由 token capacity 和 `canvas_size` 共同决定。
尺寸数组始终是 `[height, width]`；1024×1024 的像素面积约为 1MP，2048×2048 约为 4MP。

| 需求或尺寸 | 当前 1MP 包行为 |
| --- | --- |
| 输出 1024×1088、1344×768 | 对齐、画布和 token 数均满足，无需扩容 |
| 输出 2048×512 | 4096 token 在预算内，但 2048 边超过画布，需要改画布并重新构建 |
| 输出 2048×2048 | 需要 16384 target token，同时扩大画布 |
| 上传 2048×2048 参考图 | 可以接收文件；当前 runtime 会先缩到 1024×1024 后编码 |
| 模型实际使用完整 2048×2048 参考图 | 扩参考 latent、visual patch 容量和画布，重新构建 |
| 只把生成结果放大后保存为更大的 PNG | 可以使用后处理缩放；不增加去噪生成时的原生细节 |

当前 `target_capacity=4352`，所以实际面积上限为 `4352*256=1,114,112` 像素；
每边仍不能超过 1536，且高、宽都须是 32 的倍数。`1536×1536` 虽在画布内也超过 token 预算。
参考图的拟合同时约束边长、面积和 32 对齐，同一份拟合后图像进入 visual 和 VAE encoder。
接受大图片文件与保留其原始分辨率是两个不同的服务承诺。

令 `B` 为 prompt_budget，`R` 为 max_references，`T` 为真实 tokenizer 测出的模板 token，
`align64(x)` 表示向上取整到 64。当前 VAE 压缩率为 16、DiT patch_size 为 1，
单图的 target/reference latent token 与视觉 merge 前 patch 数均为 `H*W/256`；
视觉 2×2 merge 后送入文本特征编码器的视觉 token 数为 `visual_patch_capacity/4`。

```text
feature_capacity  = align64(T + B + R * visual_patch_capacity / 4)
dit_text_capacity = align64(T + B)
prefix_capacity   = align64(T + B + R * reference_capacity)
image_tokens      = max(target_capacity, reference_capacity)
cache_tokens      = prefix_capacity + target_capacity
```

`T` 来自 `capacity.derive_text_capacities`，不能把 29 当成所有 tokenizer/多参考场景的常数。
feature_capacity、prefix_capacity 和 dit_text_capacity 由构建器推导，不手填进 YAML。
DiT 图像分段一次处理一张参考或目标，因此 image_tokens 取最大值；联合 K/V cache 则需要
容纳全部前缀和目标，所以 cache_tokens 是相加。两者不能混用。

| 字段 | 只增大输出 | 要保留更大的单张参考图 | 原因 |
| --- | --- | --- | --- |
| output_size | 更新默认/校准尺寸 | 可保持 | 不等同于最大输出尺寸 |
| target_capacity | 面积超预算时增大 | 可保持 | 目标 latent 上限 |
| reference_size | 可保持 | 更新校准尺寸 | 大参考路径需要对应校准样本 |
| reference_capacity | 保持 | 增大 | 参考 VAE latent 与 DiT 前缀预算 |
| visual_patch_capacity | 保持 | 增大，且必须不小于 reference_capacity | 视觉 merge 前 patch 预算 |
| canvas_size | 任一输出边超画布时增大 | 任一参考边超画布时增大 | 当前 VAE encode/decode 共用画布 |
| feature_capacity | 保持 | 自动增大 | 视觉 token 进入 Qwen3-VL 特征编码 |
| prefix_capacity | 保持 | 自动增大 | DiT 前缀包含参考 latent |
| image_tokens | 必要时自动增大 | 必要时自动增大 | max(target, reference) |
| cache_tokens | 自动增大 | 自动增大 | 前缀与目标联合空间 |
| prompt_budget、max_references | 保持 | 保持 | 分辨率变化不代表文字变长或图片数量变多 |
| dit_text_capacity | 保持 | 保持 | 上述文字预算和模板不变 |
| max_position | 通常保持 | feature 容量/实际 mRoPE 位置超限时才增加 | 文本特征 RoPE 表上限，不是 DiT cache 长度 |
| 层数、heads、head_dim、latent channels、Turbo sigma、精度 | 保持 | 保持 | 容量扩展不改变模型架构或调度 |

以下四份 W8A8 示例放在 `capacity_configs/`，仅作客户需求拆分和配置检查，
不登记为统一 recipe 的正式 profile。它们**尚未通过完整权重的新容量导出、运行、精度或板端验收**。
数值按当前本地 Turbo tokenizer 的 T=29、B=512、R=1 推导：

| YAML 文件 | 输出 H×W / 参考 H×W | target / reference / visual | feature / prefix / cache |
| --- | --- | --- | --- |
| 当前发布包（对照） | 1024² / 1024² | 4352 / 4352 / 4352 | 1664 / 4928 / 9280 |
| [output_2048_reference_1024.yaml](capacity_configs/output_2048_reference_1024.yaml) | 2048² / 1024² | 16384 / 4352 / 4352 | 1664 / 4928 / 21312 |
| [output_1024_reference_2048.yaml](capacity_configs/output_1024_reference_2048.yaml) | 1024² / 2048² | 4352 / 16384 / 16384 | 4672 / 16960 / 21312 |
| [output_2048_reference_2048.yaml](capacity_configs/output_2048_reference_2048.yaml) | 2048² / 2048² | 16384 / 16384 / 16384 | 4672 / 16960 / 33344 |
| [output_512x2048_reference_1024.yaml](capacity_configs/output_512x2048_reference_1024.yaml) | 512×2048 / 1024² | 4352 / 4352 / 4352 | 1664 / 4928 / 9280 |

四例的 dit_text_capacity 都保持 576，max_position 都保持 32768。
输入输出都是 2048² 时 cache_tokens=33344 大于 32768 并不自动要求增加 max_position：
后者约束的 feature_capacity 此时为 4672；实际 mRoPE 索引仍需在运行中检查。
本节状态：配置校验和本地真实 tokenizer 规划为 `verified`；新档位完整权重导出、运行、精度及
板端证据为 `blocked`（本次未执行）；现有 ComfyUI 对这些未正式发布的新档位为 `unsupported`。

从图的静态边界看，只增大输出的示例改变两张 VAE 图和两张 DiT 图，visual 和 text_encode
边界不变。只加宽画布的示例仅改变两张 VAE 图边界。增大参考图的示例改变全部六图边界。
即使参考图仍为 1024，增大共用 canvas 也会改变 VAE encoder 的输入、mask 和输出形状。
目前 exporter 统一重新导出六图；此边界分析不是手工拼接旧图或改 metadata 的操作指南。

`capacity_plan.py` 是只读检查工具，只加载本地 tokenizer 和配置，不读模型权重、不导出图、
不进行 GPU 推理。它复用真实的容量校验、tokenizer 模板计数、参考图拟合及 DenoiseGear：

```bash
CUDA_VISIBLE_DEVICES="" python -m examples_merak.aigc.qwen_image21.capacity_plan \
  --model-dir /path/to/Qwen-Image-2.1-Turbo \
  --config \
    configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_turbo_1mp_w8.yaml \
    examples_merak/aigc/qwen_image21/capacity_configs/output_2048_reference_1024.yaml \
    examples_merak/aigc/qwen_image21/capacity_configs/output_1024_reference_2048.yaml \
    examples_merak/aigc/qwen_image21/capacity_configs/output_2048_reference_2048.yaml \
    examples_merak/aigc/qwen_image21/capacity_configs/output_512x2048_reference_1024.yaml \
  --reference-input 2048 2048 \
  --report work_dirs/qwen_image21_capacity/plan.json
```

报告状态是 `configuration_checked_not_exported`；changed_graph_boundaries 仅比较尺寸与位置表边界，
不能证明量化数值、算子可执行性或芯片能力。报告使用实际模型层数、heads、head_dim 计算的
FP16 逻辑 K/V 大小，每个 CFG 分支约为当前 4.53 GiB、单侧 4MP 的 10.41 GiB、双侧 4MP 的
16.28 GiB。它不包括权重、SEFP 内部存储、临时 buffer、VAE 和框架开销，不能当作峰值显存。
固定 canvas 增大时，即使运行较小图片，VAE 的静态 tensor 和计算开销也可能增加。

客户确定需求后，在已有 Merak 环境中、选择空闲 GPU，用新目录执行真实导出。例如只增大输出：

```bash
python -m examples_merak.aigc.qwen_image21.configured \
  --model-dir /path/to/Qwen-Image-2.1-Turbo \
  --config examples_merak/aigc/qwen_image21/capacity_configs/output_2048_reference_1024.yaml \
  --output-dir work_dirs/qwen_image21_custom_output2048 \
  --device cuda:0
```

设备编号需要与 CUDA_VISIBLE_DEVICES 对应。output-dir 必须不存在。
这会创建 `work_dirs/qwen_image21_custom_output2048/qwen_image21_export`，不会修改原来的 1MP 包。
成功导出后，使用前文的 `bundle.load_runtime`，例如：

```python
from pathlib import Path
from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import load_runtime, read_bundle_metadata

root = Path("work_dirs/qwen_image21_custom_output2048/qwen_image21_export")
metadata = read_bundle_metadata(root)
runtime = load_runtime(root, device="cuda:0")
image, latents = runtime.generate(
    "A red ceramic teapot on a wooden table.",
    height=2048, width=2048, seed=42, guidance_scale=1.0,
)
image.save(root.parent / "generated_2048.png")
```

编辑时传入 `images=[PIL.Image.open(...).convert("RGBA")]`；若需要保留 2048² 参考细节，
须先使用 reference_2048 的配置构建，而不是只改变运行时输出大小。
验收应检查六图 I/O、precision、跨组件 metadata、每步有限值、尺寸边界、参考图实际拟合尺寸、
换尺寸后重跑的 cache 隔离、Golden 以及同输入的官方/target-float/W8A8 精度对照，再执行板端验证。

**当前 ComfyUI 还有两层独立限制，不能只把新目录填给现有 worker：**

- 节点 INPUT_TYPES 中 width/height 的 max 仍为 1536，需要与正式新包的画布契约同步。
- `configured` 产物没有统一 Manifest；现有 worker 调用 `inference_model`，要求正式发布包。
  统一 recipe 目前只登记 `1mp-c4352`。客户交付新档位时，要先扩展正式 profile/构建发布契约、
  生成和验证 Manifest，再让 worker 加载，并进行实际 ComfyUI 出图回归；本示例未实现该扩展。

因此应按客户常用场景交付经验证的多个档位，而不是不断把一个通用包扩大。
大文件输入可以选择显式缩放；要求保留原始细节时再增加参考容量。
多参考图、长文本、4K 生成和 NPU 峰值资源是独立验收项，不由上述配置检查推断支持。

## Turbo W8A8 的 ComfyUI 接入

`comfyui_nodes/` 提供 `MerakQwenImage21TurboW8A8` 节点，显示名为
`Qwen Image 2.1 Turbo · Merak W8A8`。它接收 prompt、width、height、seed，
可选一张 `reference_image`，输出标准 ComfyUI `IMAGE`，直接连接 `SaveImage`。
文本编码、视觉编码、VAE 编解码、DiT 都使用发布包中的 HMONNX 图；
Host 只处理 tokenizer/embedding lookup、几何/RoPE、随机噪声和调度器。
采样使用 checkpoint 的八个 sigma，CFG 固定为 1；不另设无效的 steps 控件。

| 范围 | 状态 |
| --- | --- |
| Turbo `w8a8_sefp` 六图、GPU HMONNX V2、1024×1024 文生图 | verified |
| 单参考图编辑，自动经过量化 visual 和 VAE encode | verified |
| 同包内 1344×768 输出、清除 ComfyUI 缓存后的固定 seed 重跑 | verified |
| BF16 原生工作流和 W8A8 节点共存 | verified；使用独立 Python 后端环境 |
| 官方 float → target float → W8A8 的精度归因及大规模评估 | blocked；本接入不作该精度结论 |
| 当前 1MP 包的 2048×2048、多参考图、LoRA、NPU 板端执行 | unsupported |

宽、高均对齐 32；当前包要求每边不超过 1536，且
`height * width / 256 <= 4352`。因此 1536×1536 也超出容量。
参考图会按原 runtime 规则拟合到参考容量，响应记录实际高、宽。
W8A8 描述 Linear/Conv2d 的 SEFP 权重和激活位宽；图边界仍使用 FP16/整数几何输入，
不表示每个输入 tensor 都是 INT8，也不表示已有 NPU 性能优化。

ComfyUI 与 Merak 的 PyTorch/SDK 环境需要隔离。ComfyUI 节点只使用 Torch、NumPy、Pillow
及标准库 HTTP 客户端，实际推理由本机 `127.0.0.1:8190` 的 worker 执行。
先准备两个独立环境：Merak 环境安装本仓库及兼容 SDK，按本文开头的 recipe 命令准备依赖；
ComfyUI 环境按其自身 requirements 安装，无需安装 Merak SDK。
`verify_comfyui` 还使用 aiohttp，运行前应在执行验证脚本的环境中确认该依赖可用。
源码提交不包含模型权重；worker 需要完整的已发布 W8A8 模型包，不能直接加载官方 BF16 权重目录。

在 **Merak 推理环境**、仓库根目录运行。GPU 0 仅为示例，需选择实际空闲设备；
设置 CUDA_VISIBLE_DEVICES 后，`cuda:0` 指可见设备中的第一张卡：

```bash
cd /path/to/xh2modelzoo
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.qwen_image21.comfyui_worker \
  --artifact-dir /absolute/path/to/published_turbo_w8a8_bundle \
  --device cuda:0 --port 8190
```

这里的包必须有统一 build 生成的 Manifest，model_id 为 `qwen-image-2.1-turbo`，
deployment precision 为 `w8a8_sefp`。每次生成都经过现有 `inference_model` 校验
Manifest、文件摘要、模型身份、runtime metadata 和依赖，不绕过统一加载契约。
本地 2026-10-15 验证使用 `xhquant_55` 的 Python 3.12、Torch 2.8、Diffusers 0.41.0、
Transformers 5.16.1；ComfyUI 使用独立 Python 3.10、Torch 2.12 环境。
这是复用现有环境的验证，未宣称 recipe 依赖档案已通过全新环境安装。

在另一个终端的 Merak 环境安装节点并生成一个可修改尺寸的工作流：

```bash
cd /path/to/xh2modelzoo
export COMFYUI_DIR=/path/to/ComfyUI
mkdir -p "$COMFYUI_DIR/custom_nodes" "$COMFYUI_DIR/user/default/workflows"
ln -s "$PWD/examples_merak/aigc/qwen_image21/comfyui_nodes" \
  "$COMFYUI_DIR/custom_nodes/comfyui_merak_qwen_image21"
python -m examples_merak.aigc.qwen_image21.comfyui_workflow \
  --output-dir work_dirs/comfyui_turbo_w8a8/workflows
cp work_dirs/comfyui_turbo_w8a8/workflows/qwen_image21_turbo_w8a8.json \
  "$COMFYUI_DIR/user/default/workflows/"
```

软链接命令用于首次安装；已有同名节点时先核对其指向。仓库移动后需更新软链接。
然后激活 **ComfyUI 自己的环境**，启动或重启 ComfyUI；示例与 worker 共用同一张 GPU：

```bash
cd /path/to/ComfyUI
CUDA_VISIBLE_DEVICES=0 python main.py --listen 127.0.0.1 --port 8188
```

浏览器刷新后加载 `qwen_image21_turbo_w8a8.json`，填写 prompt 后运行。
远程部署时，通过 VS Code Ports 或 SSH 将服务器的 8188 端口转发到本机，
再访问 `http://127.0.0.1:8188`。8190 是两个本机进程之间的通信端口，无需转发到浏览器。
编辑时添加 `LoadImage`，把 IMAGE 输出连到 W8A8 节点的 `reference_image`；
VAE 及视觉编码已经在节点内部执行。通过 width/height 修改尺寸，不需要再生成另一份工作流。
改变 worker 端口时，在 ComfyUI 启动环境设置
`MERAK_QWEN_IMAGE21_URL=http://127.0.0.1:<port>`；服务和客户端均只允许 IPv4 loopback。
若所有节点同时出现数字控件过窄、布局或文字异常，可先保存工作流，再强制刷新浏览器；
仍有异常时，可在设置中关闭 `VueNodes.Enabled`、切回经典节点渲染后刷新。
这是前端兼容性排查步骤，不要求分发本机 `comfy.settings.json`。

worker 串行处理请求，调用现有公开 API 按请求加载、释放 sessions。
节点会在请求前释放 ComfyUI 原生模型的显存，worker 在请求结束后释放 CUDA allocator 缓存，
便于与 BF16 工作流共用 GPU。请求仍有权重加载和摘要校验开销：
本次文生图推理约 18 秒、端到端约 71 秒；编辑推理约 26 秒、端到端约 89 秒。
这些是本机 GPU 模拟数据，不是芯片性能结果。ComfyUI 缓存命中会省略后端执行；
取消 UI 队列也不会立即终止一个已经提交的 worker 请求。

### 验证和调用关系

```bash
python -m examples_merak.aigc.qwen_image21.verify_comfyui \
  --comfy-root /path/to/ComfyUI --name t2i --clear-cache \
  --report work_dirs/comfyui_turbo_w8a8/comfy_t2i.json
python -m examples_merak.aigc.qwen_image21.verify_comfyui \
  --comfy-root /path/to/ComfyUI --name edit --clear-cache \
  --reference reference.png --prompt 'Edit <image1>. Make the object blue.' \
  --report work_dirs/comfyui_turbo_w8a8/comfy_edit.json
```

reference 是已存在于 `ComfyUI/input` 的文件名。验证命令检查真实 ComfyUI history 成功状态、
八步采样统计、参考图计数、PNG 尺寸/颜色/非空内容，以及 PNG 内的 API prompt 和可导入工作流。
验证脚本遇到 ComfyUI 缓存命中会明确失败，提示先清除缓存，而不把缓存当成新推理。
`--clear-cache` 要求队列空闲，仅清除执行缓存，不删除历史和已保存图片。

| 源码入口 | 调用与契约 |
| --- | --- |
| `comfyui_nodes/__init__.py` | 注册唯一节点类和显示名；不导入 SDK 或 Merak |
| `comfyui_nodes/nodes.py:INPUT_TYPES/run` | 定义 widgets 和可选 IMAGE；检查 batch=1、RGB/RGBA、有限值；BHWC float 图转 PNG，释放原生模型显存，再调用 client；返回 CPU BHWC float IMAGE |
| `comfyui_nodes/client.py:encode_image/decode_image/generate` | Pillow PNG 与 base64 无损往返；urllib 请求本机服务；检查返回模型和量化身份，透传明确错误 |
| `comfyui_worker.py:validate_request` | 拒绝未知字段、无效 prompt/seed、超出导出画布或 token 容量的尺寸；解码可选参考图 |
| `comfyui_worker.py:ImageWorker` | 验证 Turbo/W8A8 身份；临时参考 PNG 交给公开 `inference_model`；输出图、实际 steps/尺寸/参考计数和时延；finally 释放分配器缓存 |
| `comfyui_worker.py:create_server/main` | 标准库单请求 HTTPServer；启动读取 bundle metadata；`GET /info`、`POST /generate`；输入错误 400，实际推理错误 500，不回退到 BF16 |
| `comfyui_workflow.py:build_workflow/main` | 生成同一份两节点 UI workflow 和等价 API prompt；可选参考 socket 默认断开，seed 固定 |
| `verify_comfyui.py:verify/main` | aiohttp 提交和 websocket 等待；读取 history，再用 Pillow/NumPy 校验真实 PNG 和工作流元数据，保存证据 |

模型调用链复用 `model_inference/api.py` → `model_integrations/qwen_image21/inference.py`
→ `bundle.load_runtime` → `pipeline.ImageRuntime.generate`，没有复制去噪/编辑实现。
bundle 根据包内相对路径加载 processor、token embedding 和六个 HMONNX session。
pipeline 为每次调用新建 scheduler、随机数生成器和 denoise request；
`segment_runtime.py` 按文本/参考图顺序建立前缀，DiT sessions 用 request-owned CacheTensor
原地覆盖固定 target 区域。正、负分支的 cache 不共享；此节点固定 CFG=1，仅有条件分支。
跨请求的 sessions 由公开 API 释放，因此不会沿用上次参考图或 target cache。

本地验证材料在任务 worktree 的 `work_dirs/comfyui_turbo_w8a8/`：
`graph_audit.json` 保存六图真实 I/O 和加权算子位宽，`comfy_*.json` 保存 API/history/PNG 路径，
`repeat_comparison.json` 保存清缓存后重跑的逐像素比较，`ci.log` 保存当前 CI 结果。
recipe 仍未登记统一 Golden，不能把本次 UI 接入表述为新增了统一 Golden API。
附加验证使用已有最小真实 Turbo/W8A8 六图用例完成构建、搬移、删除源权重后推理、编辑与重跑，
并将 SDK Golden 输出重定向到独立目录；六图各自生成算子级 Golden。
生产 worker 保持 SDK `enable_golden=False`：该 SDK 版本的默认 dump 目录与图目录冲突，
不能直接在发布包上开启。`synthetic_golden_report.json` 记录独立目录和六图文件数。
`real_golden/` 保存完整权重编辑请求的六图首调用逻辑 tensor I/O 及调用次数，
其中 CacheTensor 内部 SEFP 状态不属于可搬移的 replay ABI，不将这些快照宣称为板端 Golden。
该诊断使用原始 RGBA 参考图；ComfyUI `LoadImage` 输出 RGB，两者输入 alpha 不同，
对应图像差异不能用作量化精度指标。以上诊断不修改共享发布包中的权重和发布元数据。
