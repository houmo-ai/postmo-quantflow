# DepthSplat：2v workflow、6v ONNX 与视频 demo

当前支持状态、最新精度及已知限制见 [DepthSplat Model Card](model_card.md)。

本目录包含从真实 RE10K 数据、上游权重到 ONNX、HMONNX 推理和视频的入口。
2v 256×256 接入正式 `ModelRecipe` / OtherModel workflow；6v 512×960 使用独立
ONNX CLI，尚未注册为统一 recipe。两者均不包含编译或芯片实机执行。
模型权重、上游源码、数据集和生成的视频需自行放到工作目录，不随 patch 发布。

| 阶段 | 仓库入口 |
| --- | --- |
| 真实数据 / 相机准备 | `demo.py prepare` |
| 2v / 6v checkpoint → raw ONNX | `export_onnx.py` |
| 2v 量化 / artifact 打包与推理 | `workflow.py` / `unified.py` |
| 2v 浮点 / HMONNX Gaussian 对比 | `evaluate.py` |
| 6v Host 相机、Gather 修复、PTQ、推理 | `onnx_pipeline.py` 的对应子命令 |
| 2v / 6v raw 或 Gaussian → 视频与图像精度 | `demo.py render` |

以下命令均从仓库根目录执行。CLI 只负责参数和编排，算法位于
`xhmodel_merak/xh_other_model/models/depthsplat/`。

## 环境与外部材料

导出和渲染使用安装了上游依赖的独立 Python（记为 `$FLOAT_PY`）；量化与 HMONNX
使用仓库及 xhquanttool 对应环境的 Python（记为 `$QUANT_PY`）。本次实际验证分别使用
Python 3.10 和 3.12，CUDA/A100。不要把两套环境的 site-packages 混用。
两套环境都需满足本仓库基础依赖（含 xhquanttool / accelerate，模型包注册会导入它们）。
从源码运行本仓库时，将仓库根目录加入 `PYTHONPATH`，或按仓库规范安装包。

导出需要 torch/torchvision、onnx、onnxruntime、OmegaConf、dacite、einops、e3nn、
jaxtyping、beartype、Pillow；渲染还需上游 CUDA `diff_gaussian_rasterization` 扩展、
scikit-image 和 imageio-ffmpeg。已验证的上游依赖版本为 dacite 1.8.1、e3nn 0.5.1、
jaxtyping 0.2.33、beartype 0.18.5、imageio-ffmpeg 0.6.0。
依赖只在执行对应阶段时导入，不因渲染而加载完整 encoder 或 checkpoint。

源码须由调用者准备为下列固定版本。导出 metadata 中 `expected_source_commit`
记录预期版本；解压源码包不含 git 元信息，脚本不把该值当作自动核验结果。

固定外部版本：

- DepthSplat：`https://github.com/cvg/depthsplat`，commit `2dad25a7b9ba7c5537aba08463ef88d8511a06be`。
- DINOv2：`https://github.com/facebookresearch/dinov2`，commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- CUDA renderer：`https://github.com/dcharatan/diff-gaussian-rasterization-modified`，
  commit `1250c420ebb945f0dce9945086e22faab9157c92`；按上游要求在渲染环境构建，
  包括其 GLM 子模块。renderer 是外部依赖，本仓库不分发其实现；使用前核对该依赖的许可。
- 数据：`https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/re10k_720p_test_subset.zip`，
  SHA-256 `3372815fdfe878a8a030ad2515dd0dbb9a14e596aa5c25285f2e9a31c2464392`。
- 权重下载前缀：`https://huggingface.co/haofeixu/depthsplat/resolve/main/`。

| Profile | 权重文件 | SHA-256 |
| --- | --- | --- |
| 2v | `depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` | `cfeab6b10077ad9bd4097b82b029737221bfbbbd3b4dfca5fc0feaf860276506` |
| 6v | `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth` | `f8ddd845a76a2eabf79621b9757bddfb2817c28e886a6af9ad5f3c8bbd8603d6` |

先设置自己的路径；输出可全部放在 `work_dirs` 下：

```bash
export FLOAT_PY=/path/to/float-env/bin/python
export QUANT_PY=/path/to/quant-env/bin/python
export SOURCE=/path/to/depthsplat
export DINO=/path/to/dinov2
export DATA=/path/to/re10k_720p_test_subset/test
export SMALL_WEIGHT=/path/to/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
export BASE_WEIGHT=/path/to/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth
export OUT="$PWD/work_dirs/depthsplat-demo"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export USE_TRITON_MATMUL=0
export USE_FAST_TRITON_MATMUL=0
mkdir -p "$OUT"
```

两个 MatMul 开关针对当前 SDK 的大 batch Triton launch 限制，保持 HMONNX
量化仿真语义。已验证 SDK 源 pyproject 版本 0.9.2（运行时 banner 为 0.4.0）。

## 2v：导出、量化与完整轨迹视频

两个场景分别用于校准与独立评测。上下文采用官方 `evaluation_index_re10k.json`。
prepare 默认保留全部目标帧索引；网络一次预测场景 Gaussians，再按目标相机渲染轨迹。

```bash
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py prepare \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 000c3ab189999a83 \
  --profile 2v --context-indices 9,126 --output "$OUT/2v-calibration.npz"
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py prepare \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 6ca4076fadf3200e \
  --profile 2v --context-indices 66,170 --output "$OUT/2v-evaluation.npz"
"$FLOAT_PY" examples_merak/cv/depthsplat/export_onnx.py \
  --source-dir "$SOURCE" --dino-dir "$DINO" --checkpoint "$SMALL_WEIGHT" \
  --profile 2v_256x256 --calibration-sample "$OUT/2v-calibration.npz" \
  --sample "$OUT/2v-evaluation.npz" --output-dir "$OUT/2v-bundle" --device cuda:0 --verify
"$QUANT_PY" examples_merak/cv/depthsplat/unified.py \
  --model-dir "$OUT/2v-bundle" --artifact-dir "$OUT/2v-artifact" \
  --sample "$OUT/2v-bundle/2v-evaluation.npz" \
  --output "$OUT/2v-gaussians.npz" --device cuda:0
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py render \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 6ca4076fadf3200e \
  --profile 2v --context "$OUT/2v-evaluation.npz" \
  --gaussians "$OUT/2v-gaussians.npz" --output-dir "$OUT/2v-video"
```

`unified.py` 构建时要求 artifact 目录尚不存在；已有产物时省略 `--model-dir`。
仅跑 workflow 量化可使用：

```bash
"$QUANT_PY" examples_merak/cv/depthsplat/workflow.py \
  --model-dir "$OUT/2v-bundle" --output-dir "$OUT/2v-workflow" --device cuda:0
"$QUANT_PY" examples_merak/cv/depthsplat/evaluate.py \
  --model-dir "$OUT/2v-bundle" --artifact-dir "$OUT/2v-artifact" \
  --report "$OUT/2v-tensor-metrics.json" --outputs-dir "$OUT/2v-comparison" --device cuda:0
```

量化在 OtherModel workflow 的 export 阶段完成（配置 `quant: null`），不使用 GPTQ。
`evaluate.py` 输出浮点和量化 Gaussian NPZ，分别交给 `demo.py render --gaussians`
即可比较两路视频/PSNR/SSIM；tensor 相似度不能代替渲染精度。

## 6v：双尺度 ONNX、量化与视频

采用官方 `re10k_ctx_6v_video.json` 的六个上下文，图像 512×960；base 模型为
vitb、两尺度、upsample_factor=4、lowest_feature_resolution=8、scale_max=0.1，
near/far=0.5/200。原始图的邻居 GatherElements 会触发当前 SDK 的错误融合，
需先执行固定 B=1 的 48 处等价 Gather 改写。

```bash
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py prepare \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 000c3ab189999a83 \
  --profile 6v --context-indices 0,55,111,166,222,278 --output "$OUT/6v-calibration.npz"
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py prepare \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 6ca4076fadf3200e \
  --profile 6v --context-indices 0,54,109,163,218,273 --output "$OUT/6v-evaluation.npz"
"$FLOAT_PY" examples_merak/cv/depthsplat/export_onnx.py \
  --source-dir "$SOURCE" --dino-dir "$DINO" --checkpoint "$BASE_WEIGHT" \
  --profile 6v_512x960 --calibration-sample "$OUT/6v-calibration.npz" \
  --sample "$OUT/6v-evaluation.npz" --output-dir "$OUT/6v-bundle" --device cuda:0 --verify
"$QUANT_PY" examples_merak/cv/depthsplat/onnx_pipeline.py prepare-inputs \
  --sample "$OUT/6v-calibration.npz" --output "$OUT/6v-calibration-inputs.npz"
"$QUANT_PY" examples_merak/cv/depthsplat/onnx_pipeline.py prepare-inputs \
  --sample "$OUT/6v-evaluation.npz" --output "$OUT/6v-evaluation-inputs.npz"
"$QUANT_PY" examples_merak/cv/depthsplat/onnx_pipeline.py normalize \
  --onnx "$OUT/6v-bundle/encoder.onnx" --output "$OUT/6v-gather.onnx" \
  --sample "$OUT/6v-calibration-inputs.npz" --sample "$OUT/6v-evaluation-inputs.npz"
"$QUANT_PY" examples_merak/cv/depthsplat/onnx_pipeline.py quantize \
  --onnx "$OUT/6v-gather.onnx" --calibration "$OUT/6v-calibration-inputs.npz" \
  --output-dir "$OUT/6v-hmonnx" --geometry16 --device cuda:0
"$QUANT_PY" examples_merak/cv/depthsplat/onnx_pipeline.py infer \
  --hmonnx "$OUT/6v-hmonnx/model.onnx" --sample "$OUT/6v-evaluation-inputs.npz" \
  --output "$OUT/6v-quantized-raw.npz" --device cuda:0
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py render \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 6ca4076fadf3200e \
  --profile 6v --context "$OUT/6v-evaluation.npz" \
  --raw "$OUT/6v-quantized-raw.npz" --output-dir "$OUT/6v-video"
```

`--geometry16` 只将 8 个几何 MatMul 两输入改为 SEFP a16a16，其余保持 W8A8；
省略该参数可复现默认 W8A8 对照。SEFP16 的 FP16 输出 / FP24 累加不等于 IEEE FP32。
HMONNX 必须与同目录 `model_external_data` 一起保存和搬移。
相机准备只使用当前真实图像、内外参、near/far，不加载浮点 encoder。
推理不依赖浮点结果；可选 `--reference` 仅用于记录输出误差。

## 视频与精度输出

`demo.py render` 接收已保存的 Gaussian 或 raw 网络输出，不重新执行/替换网络预测。
raw 输入使用 profile 对应的 FP32 GaussianAdapter，然后使用官方 CUDA renderer。
浮点基线可直接使用 exporter 的 `*-raw-onnx.npz`（`--verify` 时生成）：

```bash
"$FLOAT_PY" examples_merak/cv/depthsplat/demo.py render \
  --source-dir "$SOURCE" --dataset-dir "$DATA" --scene 6ca4076fadf3200e \
  --profile 6v --context "$OUT/6v-evaluation.npz" \
  --raw "$OUT/6v-bundle/6v-evaluation-raw-onnx.npz" --output-dir "$OUT/6v-float-video"
```

2v 同理替换 profile、context 和 raw 文件为 2v。量化渲染时增加
`--baseline-metrics "$OUT/6v-float-video/metrics.json"` 可生成同轨迹对照。
`--verify` 对每个导出样本执行 PyTorch/CPU ONNX 输出对比并记录误差；
浮点验证使用相对 L2 阈值，不声称所有元素逐点完全一致。

目标 RGB 解码和渲染按批次执行；`--max-frames N` 可用于短片验证，省略则渲染全部目标。
输出 MP4、逐帧 PSNR/SSIM/MSE 及汇总报告；指标在视频压缩之前计算，汇总同时排除上下文帧。
播放帧率不代表模型推理速度。`--baseline-metrics` 可比较同一轨迹的浮点渲染指标。

## 必须保留的 FP32 边界

- 原始图像 resize/crop 与内参同步调整；原始内外参、near/far 保持 FP32。
  6v 相机中心距离、邻居索引、两尺度内外参逆矩阵由 Host 生成；邻居索引用整数。
- **2v**：候选逆深度、像素反投影、相对位姿、投影除法和 GridSample 坐标归一化
  全部在 `preprocess.py::prepare_matching_grid` 中以 CPU FP32 完成。
  ONNX 输入是 images、near、far、matching_grid；GridSample 插值仍在网络中。
- **6v**：matching_grid 仍在图内。第二尺度依赖第一尺度真实网络预测，不能提前
  用浮点参考预测生成 grid。完全外置需要进一步分段执行，当前没有声称完成。
- 网络输出 depths/raw_gaussians 后，FP32 完成 opacity/offset、深度反投影、
  Gaussian 世界位置、scale 激活/裁剪、四元数归一化、协方差与世界旋转、SH 颜色/旋转。
  正式 2v runtime 使用 `postprocess.py`；视频工具使用外部上游 FP32 adapter。
- 最终 Gaussian rasterization 是独立浮点渲染阶段，不在量化网络中。

2v exporter 生成 schema 3 bundle，配置包含 `depthsplat_matching_grid_fp32_v1` 和
`depthsplat_gaussian_fp32_v1`，以及显式 `host_input_specs`。样本包含原始 FP32
images/intrinsics/extrinsics/near/far 与三组 SH 旋转矩阵；matching_grid 在运行时生成。
旧 schema 1 完整图 / schema 2 仅后处理图兼容保留；不要将旧全图 W8A8 当作精度可用方案。
仅将送入 HMONNX 的副本转换为图端口 dtype，不能先降精度再转回 FP32 冒充原始相机。

## 实测范围与限制

数据仅为官方 RealEstate10K 720p **两个场景**：000c3ab189999a83（279 帧，校准）和
6ca4076fadf3200e（274 帧，独立评测）。未跑完整 RE10K benchmark，也未评测独立 DL3DV
数据集；6v checkpoint 名称包含 re10kdl3dv 不表示本次用 DL3DV 测试。

| 测试范围 | 浮点 PSNR / SSIM | 量化 PSNR / SSIM |
| --- | --- | --- |
| 2v：4 组上下文、30 次目标评测（29 个唯一目标帧） | 22.782822 / 0.803825 | 22.627123 / 0.795782 |
| 2v：完整 274 帧 | 20.836807 / 0.787326 | 20.696139 / 0.779145 |
| 6v：完整 274 帧，默认 W8A8 | 22.528789 / 0.783371 | 17.415854 / 0.530522 |
| 6v：完整 274 帧，8 个几何 MatMul a16a16 | 22.528789 / 0.783371 | 22.011380 / 0.749921 |

排除输入帧后：2v 272 帧 PSNR 下降 0.139135 dB；6v a16 268 帧下降 0.512932 dB，
平均 MSE 增加 9.23494%。若验收门槛是 MSE 增幅 ≤5%，6v 当前仍未通过。
6v 两场景 553 帧浮点视频已验证，量化对照只在第二个场景进行。
8 节点 A/B 对照图结构和全部 1209 个 initializer 一致，仅其两输入 mantissa 属性改变。
这些结果支持几何精度是主要误差来源之一，不能证明全部残差均来自几何或 GridSample。

SDK 当前只使用首个校准样本传播 shape，多个 NPZ 不代表拟合了多样本激活 scale；
SEFP 共享指数由运行时输入决定。artifact metadata 的 `accuracy_verified: false`
表示构建成功不代表完整精度验收通过。
CI 包含 Host 几何、输入/产物契约、ONNX 邻居改写、CLI 和统一构建/搬移/推理回归，
每个用例附对应 impact 声明。完整 2v 用例从内部 CI 资产库自动下载 SHA-256 固定的
真实 ONNX 与校准/评测样本；无需手动设置 `DEPTHSPLAT_CI_BUNDLE`。离线覆盖路径须含
完整发布清单（含许可和来源），并通过同样校验。见
[CI 资产说明](../../../benchmark_test/ci_test_transformer5.16.0/fixtures/depthsplat/README.md)。
精度结论见 [QNT-3](http://jira.houmo.ai/browse/QNT-3)。
