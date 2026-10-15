# Qwen-Image-2.1-Turbo 接入记录

模型 ID 为 `qwen-image-2.1-turbo`。本次以 `/data01/datasets/Qwen-Image-2.1-Turbo`
的本地完整 checkpoint 为来源；它包含独立的 text_encoder、transformer、VAE、processor
和 scheduler，并非必须合并 LoRA 才能使用的发布形式。
构建和推理命令见[示例 README](../../../examples_merak/aigc/qwen_image21/README.md)。

## 与普通版共用的结构

本地配置声明 QwenImage21Pipeline、QwenImage21Transformer2DModel、
AutoencoderKLQwenImage21 与 Qwen3VLForConditionalGeneration。
DiT 为 32 层、32 heads、head_dim=128、context_in_dim=4096、in/out_channels=64、
patch_size=1、axes_dims_rope=[16,56,56]、causal_condition=true，
与普通版的图结构和缓存端口相同。两者共享 family `qwen_image21`、构建和推理 adapter，
从各自 checkpoint 独立导出六图，产物名称及 runtime model_id 区分版本。
六图、容量、RGBA 与 cache 语义详见[普通版接入记录](qwen-image21.md)。

Turbo processor 使用嵌套的 `processor_config.json`，文本权重也不采用普通版的四分片布局。
构建沿用 AutoProcessor/原生模型的 `from_pretrained`，不假设旧 processor 文件名或分片数量。

## 采样属于发布契约

`model_index.json` 保存的默认 sigma 为：

```json
[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
```

同时 scheduler 的 `use_dynamic_shifting=false`、`shift_terminal=null`、`shift=1`。
因此不能通过普通版的线性 8 步及动态 shift 来代替 Turbo 调度。
Host runtime 按显式 `sigmas`、checkpoint `sample_sigmas`、按 `steps` 生成线性序列的
优先级选择调度；scheduler 自行添加终点零。sigma 严格递减且在 `(0,1]` 内。
有 checkpoint 序列时，单独指定 `steps` 不改变实际序列；公开结果报告实际执行步数。

构建前校验模型身份与 scheduler，完整包要求随包保存 pipeline 配置资源。
`export_meta_info.json` 将其以包内相对路径、`type=json` 登记；Manifest 收录并校验文件。
加载端再次校验 sampling、模型身份与部署精度，拒绝缺失 Turbo 序列的包及错误模型身份。
历史普通版 schema 3 包可省略 model_id 和 pipeline_config，仍走原来的 steps 语义。

当前 recipe 与已合入的普通版保持一致，固定 Diffusers 0.41.0、Hub 1.33.0 和 Transformers 5.16.1。
下述首次完整模型验证使用 Diffusers commit `8d3c30bfda9b511c00992f40cff4170a5502814d`
及 Hub 1.31.0；当时的原生 pipeline 不自动使用 sample_sigmas，因此数值对照显式传入
checkpoint 的 sigma。部署 runtime 自己读取随包序列，不依赖原生 pipeline 执行学习网络。

## 验证环境与范围

首次验证使用 RTX 6000D，Torch 2.8.0+cu128，独立 Python 3.12 环境。
随后已把构建与推理依赖安装到既有 conda 环境 `xhquant_55`（Python 3.12.11），
正式运行路径为 `/data01/home/jing.huang/xh2modelzoo` 与 `/data01/home/jing.huang/xhquanttool`，
无需临时 worktree 的 Python 环境或 `PYTHONPATH`。
本地数据根目录存在不可读的 ModelScope 缓存 `.msc`，公开构建的完整来源哈希会因此失败。
本次只将 model_index.json 和五个组件目录复制到可读的验证目录，再由公开 build_model 构建；
不修改原始权重目录，不跳过来源哈希校验。使用相同目录时也需要移除不可读缓存或提供可读副本。

SDK 需要包含上游 `b514c978ecad53c85047854fc179252c2621c0b1`（2026-10-09，
`enable Qwen Image 1MP golden`）。本次验证版本为 `develop` 的
`b4c7f631c9afddfb1a9745b4a1d401e042e64476`（2026-10-11）。
首次验证使用独立 SDK worktree；随后正式 SDK 目录的 `develop` 分支已从
`09de920a` 快进到同一验证版本 `b4c7f631`。
该提交新增三个文件，并修改三个调用处：

- `xhquant/backend/hmops/fused_ops/flash_attention/fp/_flash_attention_fp_torch_opt.py`：CPU FP32 校准的分块 attention。
- `xhquant/backend/hmops/fused_ops/flash_attention/xh2/_flash_attention_xh2_large_head.py`：大 head XH2 attention。
- `xhquant/backend/xh2a/ir/operators/_conv_bounded_unfold.py`：控制 VAE 卷积临时展开内存。
- `xhquant/backend/hmops/fused_ops/flash_attention/fp/flash_attention_fp.py`：CPU attention 调用入口。
- `xhquant/backend/hmops/fused_ops/flash_attention/xh2/_flash_attention_xh2_triton.py`：大 head 分派。
- `xhquant/backend/xh2a/ir/operators/conv.py`：分块卷积调用入口。

VAE encoder/decoder 的单头维度为 768/1152，不能拆成多个独立 head 替代。
新 SDK 的 D=128/768/1152 CUDA 探针及现有 CPU attention、卷积、大 head HMONNX 回归
共 31 项通过。模型接口、资源、调度和共享 recipe 契约 396 项通过。
CPU 导出测试 14 项通过，包括普通版/Turbo 三种精度的六图真实小模型导出、搬移、
生成、编辑、显式 sigma 覆盖、缓存隔离和重复请求。上述测试均在本次新 SDK 下执行。

完整权重 W8A16 已通过公开 `build_model` 从可读来源新鲜导出六图、发布与搬移验证。
六图所有 Linear/Conv 均检查为 weight/activation mantissa 8/16；学习组件全部走 HMONNX V2。
原生对照按 FP16 从同一 checkpoint 加载，显式传入 checkpoint sigma，
使用相同 seed42、通道优先的初始 FP32 噪声和 CFG1。
文生图提示词为 `A red ceramic teapot on a wooden table, soft window light, photographic detail.`，
编辑提示词为 `Make the teapot blue.`，参考图为原生文生图。
原生两组逐步 latent 均已保存且有限，部署 runtime 也逐步校验 latent 有限性。
最终输出 RGBA；RGB PSNR 不含 alpha 通道。

| 场景 | 输出高×宽 | 实际步数 | 对原生 FP16 的 RGB PSNR |
| --- | --- | --- | --- |
| 文生图 | 1024×1024 | 8 | 36.18 dB |
| 单图编辑 | 768×1376 | 8 | 31.96 dB |

完整包另通过 1376×768 竖图、512×512 CFG2，以及换尺寸/CFG 后重复生成原图的逐字节一致性检查。
请求 `steps=20` 或 `steps=1` 都实际执行相同 checkpoint 八步，生成结果一致。
GPU HMONNX 的首次执行包含算子初始化/编译，报告中的耗时不作为芯片性能结果。

验证材料位于仓库 `work_dirs/qwen_image21_turbo_validation_20261012/`：
完整包已移动到 `work_dirs/qwen_image21_all_configs_export_20261013/qwen_image21_turbo_1mp_w8a16/`；
原目录的 `artifact_relocation.json` 记录移动关系，`real_report.json` 保存逐步数值与精度核对，
`fp16_w8a16_comparison.png` 展示两组原生/部署图片，`environment.json` 记录依赖版本，
`contracts_latest_sdk.log`、`export_latest_sdk.log`、`sdk_latest_regression.log` 为通过日志。

`xhquant_55_environment/` 保存正式 conda 环境的依赖安装清单、安装前后快照、
导入检查、构建与推理验证记录。该环境使用 TorchAO 0.15.0，解决原有 0.14.1 缺少
Diffusers 所需 `FqnToConfig` 的导入问题；本模型不使用 TorchAO 的 C++ 扩展。
同一正式环境的 429 项相关测试通过，包含 396 项契约、31 项 SDK 回归和
普通版 W16A16 / Turbo W8A16 两项真实小模型六图构建、搬移与推理。
完整权重包的 1024×1024 文生图与 768×1376 编辑均实际执行八步，
两组 RGBA 输出均与首次验证环境逐像素一致；现有 GPTQModel 导入也通过。
上述验证时 NumPy 按 recipe 与 GPTQModel 固定为 2.2.6；SDK 包元数据的 ≥2.3.3 要求
仍造成一条 `pip check` 声明冲突，未通过修改依赖元数据隐藏该差异。

## 最新 develop 的迁移验证（2026-10-15）

独立 Turbo 变更基于 `develop` 的 `581de30743f6dec3bcd9b726327332312bea541b`，
其祖先已包含普通版 Qwen-Image-2.1 的最终合入版本。保留普通版官方 CPU feature encoder 的
加载方式，示例沿用 `examples_merak/aigc/qwen_image21/`，统一接口测试按当前 CI 的
每模型文件组织；不恢复上游已删除的集中测试清单。

在既有 `xhquant_55` 中，将 Diffusers 和 Hub 更新到当前 recipe 的 0.41.0 / 1.33.0；
其余已安装包保持原版本。现有完整权重 W8A16 包的 1024×1024 文生图、768×1376 编辑
均通过公开接口实际执行 checkpoint 八步，RGBA 输出与首次验证图片逐像素一致。
记录位于 `work_dirs/qwen_image21_turbo_submission_20261015/runtime_report.json`，
依赖变更记录为同目录的 `environment_changes.json`。

增量 CI 选择 982 项：结合失败修正后的复测，980 项通过。普通版/Turbo 三种精度的
六图构建、搬移与推理矩阵，以及两者各自的统一流程均通过。两项其他模型的环境失败
在同一 `develop` 基线上重现：Gemma E2B 要求 datasets 3.6.0 和 torchcodec 0.7.0，
现有环境为 datasets 4.8.5 且缺少 torchcodec；Pi0.5 用例缺少 lerobot。完整 CI 未全绿。
三项由验证临时目录引起的失败已在仓库外短路径复测通过；workflow impact 声明按普通版与
Turbo YAML 分开后，选择器、recipe 与统一矩阵检查复测通过，仅保留上述 Pi0.5 环境失败。
完整日志、复测结果及基线证据位于同目录的 `validation_summary.json`、`ci.log`、
`ci.xml`、`gates.log`、`baseline_gemma.log`、`baseline_pi05.log`。

此记录仅覆盖实际执行的配置和样本。完整权重 W16A16/W8A8 的精度对照、NPU 板端编译、
芯片速度优化、ComfyUI、额外 LoRA/蒸馏变体与大规模画质评估仍需分别验证。
