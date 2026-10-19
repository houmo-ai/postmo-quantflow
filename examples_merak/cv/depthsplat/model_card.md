# DepthSplat Model Card

更新日期：2026-10-20。验证代码：`a363399dae356569b8932e7d774b8ea7deb1c12a`，
[Gerrit 46782 / Patch Set 2](http://gerrit.houmo.ai/c/houmoquantization/xh2modelzoo/+/46782/2)。

**状态：HMONNX 推理与真实场景视频精度已验证；完整数据集验收尚未完成。**
2v 在本次单场景、排除输入帧后的平均 MSE 增幅为 **0.7056%**；
6v 几何 a16a16 为 **9.2344%**。若门槛为平均 MSE 增幅不超过 5%，
2v 当前场景通过，6v 不通过。此判断不代表逐帧通过、芯片实机通过或正式发布。

本卡为覆盖两种接入路径的说明性模型卡，不作为 `merak_delivery run-workflow` 的 YAML 输入。
精度数值及模型/样本来源快照见 [accuracy_2026-10-20.json](model_cards/accuracy_2026-10-20.json)。

## 模型与用途

DepthSplat 从多视图 RGB 图像及相机参数预测深度和 3D Gaussian 场景表示，
再用外部 Gaussian renderer 在目标相机视角生成图像/视频。
当前实现用于多视图场景重建、新视角合成与量化精度评估；没有验证实时性能、
动态场景泛化或安全关键用途。视频播放帧率不等于网络推理速度。

| 项目 | 2v small | 6v base |
| --- | --- | --- |
| 输入图像 | `[1,2,3,256,256]` | `[1,6,3,512,960]` |
| 档位 | 单尺度、2 个上下文视图 | 双尺度、6 个上下文视图、ViT-B |
| 仓库接入 | 正式 ModelRecipe / OtherModel workflow / 统一 build、inference | 独立 ONNX 导出、量化与推理 CLI |
| recipe ID | `depthsplat-gs-small-re10k-256x256-view2` | 尚未注册统一 recipe |
| 量化配置 | 神经网络 W8A8 + Host FP32 前后处理 | W8A8；改善版对 8 个几何 MatMul 使用 SEFP a16a16 |
| near / far | 0.5 / 100 | 0.5 / 200 |
| Gaussian scale 上限 | 3.0 | 0.1 |
| 实测后端 | A100 上 HMONNX ALIGNED + 浮点 CUDA renderer | 同左 |
| SDK target | XH2a | XH2a |
| 芯片编译/实机 | 未测试 | 未测试 |

模型参数量、FLOPs、稳态吞吐、端到端延迟及峰值内存没有形成受控测量，本卡不填推测值。
两档模型的分辨率和 checkpoint 不同，绝对 PSNR 不宜直接作为同等条件下的优劣比较。

## 来源与固定版本

上游代码：[cvg/depthsplat](https://github.com/cvg/depthsplat)，
commit `2dad25a7b9ba7c5537aba08463ef88d8511a06be`。
DINOv2 commit：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
外部 CUDA renderer commit：`1250c420ebb945f0dce9945086e22faab9157c92`。
本地归档的版本需由使用者核验；export metadata 的 `expected_source_commit` 表示预期版本。

| 档位 | Checkpoint | SHA-256 |
| --- | --- | --- |
| 2v | `depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` | `cfeab6b10077ad9bd4097b82b029737221bfbbbd3b4dfca5fc0feaf860276506` |
| 6v | `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth` | `f8ddd845a76a2eabf79621b9757bddfb2817c28e886a6af9ad5f3c8bbd8603d6` |

本仓库中的上游衍生代码遵循 [DepthSplat MIT 许可](../../../licenses/DepthSplat/LICENSE)，
其余本仓库实现按各文件头声明。权重、数据及外部 renderer 不随代码分发，
它们的许可需分别核对，不能由上游模型代码的 MIT 许可推定。

## 输入、输出与 FP32 边界

输入由真实 RGB 图像、裁剪/缩放后匹配的归一化内参、相机到世界坐标外参、near/far 构成。
原始相机与图像保留 FP32，只将送入 HMONNX 的副本转换为对应端口 dtype。

- **2v 前处理**：Host CPU FP32 生成候选深度、相机反投影/相对位姿、透视投影和
  GridSample 归一化采样坐标。ONNX 输入为 images、near、far、matching_grid；
  grid shape 为 `[2,8192,64,2]`。GridSample 插值仍在网络中。
- **6v 前处理**：Host FP32 计算邻居选择和两尺度相机逆矩阵，邻居索引为整数。
  两尺度 matching_grid 仍在网络内；第二尺度依赖第一尺度预测，不能提前用浮点参考预测替代。
- **神经网络输出**：depths 为 `[1,V,H*W,1,1]`，raw_gaussians 为 `[V,37,H,W]`。
- **后处理**：FP32 执行 opacity/offset、深度反投影、Gaussian 世界位置、scale 激活/裁剪、
  四元数归一化、协方差及世界旋转、SH 颜色与旋转。2v 统一 runtime 使用仓库 NumPy 实现；
  6v 视频工具使用上游 CPU FP32 GaussianAdapter。
- **Gaussian 输出**：means `[1,N,3]`、covariances `[1,N,3,3]`、harmonics `[1,N,3,9]`、
  opacities `[1,N]`，`N=V*H*W`。外部 CUDA renderer 使用目标相机生成 RGB。

6v 的 SEFP a16a16 采用 FP16 输出 / FP24 累加，**不等于 IEEE FP32**。
48 处邻居 GatherElements 的等价 Gather 改写用于规避当前 SDK 融合错误；
两场景浮点对齐及新旧图对比已验证，未修改共享 SDK。

## 精度评测协议

数据为官方 `re10k_720p_test_subset.zip`，SHA-256：
`3372815fdfe878a8a030ad2515dd0dbb9a14e596aa5c25285f2e9a31c2464392`。

| 场景 | 用途 | 轨迹帧数 |
| --- | --- | ---: |
| `000c3ab189999a83` | 量化校准，与精度评测分离 | 279 |
| `6ca4076fadf3200e` | 本卡精度评测 | 274 |

- 输入全部是真实图像与相机参数。2v 上下文帧为 66、170；6v 为 0、54、109、163、218、273。
- 每路渲染全部 274 帧；主要结论排除上下文，分别统计 272 / 268 个目标帧。
- 浮点基准为相同 checkpoint 导出的 FP32 ONNX。2v 本轮重新执行浮点与 HMONNX 推理；
  6v 浮点 raw 复用此前通过导出校验的 ONNX 输出，本轮重跑默认/a16 HMONNX 与三路完整渲染。
- 两路使用相同后处理和渲染器，没有将浮点网络预测注入量化网络结果。
- PSNR、SSIM、MSE 均对真实目标 RGB，在视频编码前计算，再对目标帧求平均。
  PSNR 单位 dB；SSIM 使用窗口 11、高斯权重、data_range=1。
- MSE 增幅定义为 `100 * (mean(MSE_quant) / mean(MSE_float) - 1)`。
  它不是逐帧 MSE 百分比增幅的平均，也不能用 PSNR dB 差直接换算成“精度损失百分比”。
- “平均 MSE 增幅 ≤5%”是条件性比较口径，尚未确认其为完整业务验收规则。

这些帧来自同一个评测场景且彼此相关；未评测完整 RE10K benchmark 或独立 DL3DV 数据集。
6v checkpoint 名称中的 `re10kdl3dv` 不是本次测试数据来源的证明。

## 当前精度

### 主要结果：排除输入帧

| 方案 | 帧数 | 浮点 PSNR | 量化 PSNR | ΔPSNR | 浮点 SSIM | 量化 SSIM | 平均 MSE 增幅 | 条件性 5% 门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2v W8A8 + Host FP32 | 272 | 20.769293 | 20.630158 | -0.139135 dB | 0.786159 | 0.777982 | +0.7056% | 当前场景通过 |
| 6v 默认 W8A8 | 268 | 22.506683 | 17.409664 | -5.097019 dB | 0.782417 | 0.528760 | +191.2993% | 不通过 |
| 6v 几何 a16a16 | 268 | 22.506683 | 21.993755 | -0.512928 dB | 0.782417 | 0.748993 | +9.2344% | 不通过 |

| 方案 | 浮点平均 MSE | 量化平均 MSE |
| --- | ---: | ---: |
| 2v | 0.0190915211 | 0.0192262319 |
| 6v 默认 | 0.0063437840 | 0.0184794009 |
| 6v a16a16 | 0.0063437840 | 0.0069295914 |

### 完整 274 帧（含输入帧）

| 方案 | 浮点 PSNR / SSIM | 量化 PSNR / SSIM | 平均 MSE 增幅 |
| --- | --- | --- | ---: |
| 2v | 20.836807 / 0.787326 | 20.696139 / 0.779145 | +0.7087% |
| 6v 默认 | 22.528785 / 0.783370 | 17.415854 / 0.530522 | +190.6647% |
| 6v a16a16 | 22.528785 / 0.783370 | 22.011380 / 0.749921 | +9.1755% |

### 局部退化与结论

- 2v 有 **87/272** 帧的单帧 MSE 增幅超过 5%；最差目标帧 149，PSNR 下降
  **0.547579 dB**、MSE 增加 **13.437840%**。平均通过不代表逐帧通过。
- 6v 默认有 **268/268** 帧超过该单帧门槛；a16a16 为 **215/268** 帧。
  a16a16 最差目标帧 55，PSNR 下降 **1.176727 dB**。
- 几何精度提升显著改善 6v，但剩余误差仍需定位；不能断言所有误差都来自几何或 GridSample。
- 这是固定上下文生成场景后，对各目标相机独立渲染的端到端误差，不是视频逐帧递归累积误差。
- **推荐验证路径**：2v 用于当前已覆盖档位的后续集成验证；6v a16a16 用于继续定位与改善精度。
  两者均需更多独立场景及明确业务指标后再做发布判断。

## 运行入口与复现

完整安装要求、固定资产和可复制命令见 [README](README.md)。
所有路径均从仓库根目录执行；模型权重、数据、ONNX/HMONNX 和视频放在工作目录。

| 阶段 | 入口 |
| --- | --- |
| 真实样本与目标轨迹准备 | `demo.py prepare --profile 2v` / `6v` |
| 从本地 checkpoint 导出 ONNX 并对齐 | `export_onnx.py --profile 2v_256x256` / `6v_512x960 --verify` |
| 2v 量化 | `workflow.py`；配置 `configs_merak/workflows/xh2a/other_models/depthsplat/depthsplat_xh2a_w8a8.yaml` |
| 2v 正式构建、推理 | `unified.py` |
| 2v 浮点/HMONNX 张量对比、保存 Gaussian | `evaluate.py --outputs-dir ...` |
| 6v 相机准备、图改写、量化、推理 | `onnx_pipeline.py prepare-inputs / normalize / quantize / infer`；改善版量化加 `--geometry16` |
| 浮点或量化视频与 GT 图像指标 | `demo.py render --profile 2v` / `6v`，输入 `--gaussians` 或 `--raw` |

SDK 环境需设置 `USE_TRITON_MATMUL=0` 和 `USE_FAST_TRITON_MATMUL=0`，
使用已验证的 HMONNX native MatMul 路径。HMONNX 的 `model.onnx` 与 `model_external_data`
需一起保存。量化/推理与上游导出/渲染依赖环境按 README 分别准备。

原始证据目录：`work_dirs/depthsplat-adaptation/accuracy-2026-10-20/`。
包含 `2v/commands.sh`、`6v/commands.sh`、模型/输入 SHA-256、推理日志、逐帧指标、
五路完整视频、汇总报告及精度曲线。该目录为本地生成产物；
随本卡保留的 JSON 快照使当前结果和来源可随代码审阅，但不替代重跑所需的数据和权重。

## 未完成验证与已知限制

1. 未覆盖完整 RE10K、独立 DL3DV、更多独立场景或不同输入档位；未计算 LPIPS。
2. 6v 双尺度采样坐标未全部移到 Host FP32，尚未接入正式统一 recipe。
3. 未做芯片编译/实机测试、吞吐/延迟/内存基准；A100 仿真成功不代表目标芯片通过。
4. SDK 当前只使用首个校准样本传播 shape，SEFP 共享指数由运行时输入决定；
   多个校准 NPZ 不代表完成多样本激活 scale 拟合。
5. CI 已提供相邻 impact 声明、`gaussian-scene` 注册断言和真实 2v
   构建/搬移/重新加载推理用例；CI 完整模型输入采用内部镜像与固定 SHA-256 校验。
   这些回归验证接口与执行链路，不能替代上文独立场景的图像精度评测。
6. 本卡保留为验证记录，不标记 `accuracy_passed` 或 `released`。后续更新应保留指标口径、
   场景划分、checkpoint、模型产物哈希和验证代码版本。
