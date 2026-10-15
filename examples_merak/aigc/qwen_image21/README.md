# Qwen-Image-2.1 / Turbo 构建与推理

## Qwen-Image-2.1-Turbo

Turbo 使用独立的模型 ID `qwen-image-2.1-turbo`，共享普通版的六图实现与容量规则。
本机已在既有 conda 环境 `xhquant_55` 中补齐构建和推理依赖，直接使用正式仓库
`/data01/home/jing.huang/xh2modelzoo` 和 `/data01/home/jing.huang/xhquanttool`。
从仓库根运行；Diffusers 和 Hub 使用与普通版相同的固定版本：

```bash
conda activate xhquant_55
cd /data01/home/jing.huang/xh2modelzoo
python benchmark_test/ci_test_transformer5.16.0/prepare_dependencies.py \
  --model-id qwen-image-2.1-turbo --packages diffusers huggingface-hub --install
python -m examples_merak.aigc.qwen_image21.unified \
  --model-id qwen-image-2.1-turbo \
  --model-dir /data01/datasets/Qwen-Image-2.1-Turbo \
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
    model_dir="/data01/datasets/Qwen-Image-2.1-Turbo",
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
