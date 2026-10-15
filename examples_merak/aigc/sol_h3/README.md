# Sol-H3：4 步 H3 + 3 步 LTX-2.5

独立实现位于 `xhmodel_merak/xh_other_model/models/sol_h3/`，公开 recipe 为 `sol-h3`。
本目录使用新的 workflow、产物和 runtime。完整结构、数值改写和证据见
[Sol-H3 建模记录](../../../../docs/modeling/structures/sol-h3.md)。

默认公开档位 `all-dense-384p-768p-5s` 在一个包中支持五种模式：672×384、124 帧 H3 粗生成，随后 1344×768、121 帧 LTX 精修，24 FPS。
论文宣传中的 360p/720p 对应这里使用的作者对齐尺寸。注意当前 Merak 配置的两阶段 attention 都是 dense；
4/3 步蒸馏权重、AdaLN 折叠、逐时间步精确文本 KV 缓存和数值补偿已接入，本次发布仅开放 Dense。

## 当前构建选项与新增组件

当前默认构建使用固定五角色 H3 布局，以及一张最大长度 LTX DiT；两者均使用 SDK/编译器的
广播 key mask 接口。需要本变更配套 SDK 和 hmcc 前端补丁；CUDA simulator 验证不等于芯片编译验收。
新图 ABI 需要重新导出，不能用下述历史包的验证记录替代当前构建验收。

Sol-H3 导出不再运行完整 H3/LTX 去噪采样，使用固定种子、有界且尺寸正确的 latent
作为下游组件的导图示例；GPTQ/AutoRound 的真实量化校准保持不变。完整音视频生成
需在发布后独立验收，导出成功不代替生成验证。Native H3/LoRA 的构建流程未改。
此优化不追溯到已启动导出的进程，历史大包须按各自源码快照和验收记录判断。

```yaml
export:
  norm_fp16: true
  ltx_reference_conditioning:
    enabled: true
    encoder_norm_eps: 5.960464477539063e-8
```

`norm_fp16` 只改变 H3/LTX DiT 的 RMSNorm 累加路径；Qwen、文本 refiner 和 VAE 保持宽归约。
首尾参考一致性默认开启，新增一张共享 W16A16 LTX 图像 encoder：
`[1,3,1,768,1344] → [1,128,1,24,42]`，残差流缩放32。
I2VA/L2VA/FL2VA 分别编码首图、末图、两图；T2VA/Ref2VA 不把参考素材解释为目标首尾帧。
LTX DiT 统一使用17136个视频token，新增 `first_role:int32[1]` 与
`video_valid_length:int32[1]`。首图固定首1008个目标token；末图追加1008个guide，参与attention，
三步结束后丢弃。无末图时有效长度16128，dummy guide保持零且不能影响目标或音频。
目标最后latent仍参与更新，不能用末图覆盖它所对应的八帧。

`enabled:false` 构建不带图像encoder的16128-token目标档，保留原生首target marker；
它只关闭LTX约束，H3原有参考条件继续生效。两种构建的ABI由metadata明确区分。
`encoder_norm_eps` 默认是 FP16 最小正值 `2^-24`，对缩放后的 PixelNorm epsilon 取此下限；
该完整 encoder 的37处均使用普通宽累加 RMSNorm，residual scale仍为32。
这是抬高 epsilon 的有损方案，与 `finfo.tiny=2^-14` 不同；原本更大的 epsilon 不会被降低。
显式 `null` 仅用于保留尺度补偿语义的有界 RMS 诊断。修改此值必须重导 encoder 并同步产物清单，不能只改字段。

encoder、DiT、decoder按阶段顺序加载与卸载；运行时不导入作者LTX源码或读取源checkpoint。

## 历史已验证发布包（2026-10-24）

下列旧包的证据仅对应当时的图和代码：

```text
/data01/datasets/Sol-H3-assets/releases/unified_w4_rms1024_reflectpad_20261024/relocated/bundle
```

该包包含 19 张图：参考 VAE 使用 `split_v1` 三档布局，Qwen ViT 使用
1024/1280/65536 patch 三档。默认 H3 残差缩放为 /1024，使用 epsilon=`2^-24` 的普通 RMSNorm。
本轮先重新导出全部组件，再补导三档 ViT；最终包保留本轮的 16 张非 ViT 图与三张新 ViT 图。
下文旧单画布、单 ViT 和独立包记录仅供追溯，不能作为当前代码的推理输入。

2026-10-24 已完成此包的五模式全权重 CUDA simulator 验证。每个模式均通过统一
`inference_model` API，在独立 mount namespace 中屏蔽 12 个原始权重/构建目录，
从搬迁后的包生成 1344×768、121 帧、24 FPS 视频及 32 kHz 双声道音频。
Ref2VA 同时覆盖参考图片、视频和音频；普通条件图实际使用 1024 档 ViT，
Ref2VA 实际使用 1024/65536 档，1280 档另有真实视频输入的数值对照。
所有音频输出有限且无削波；五模式抽帧未见明显白边或灾难性画面异常，
T2VA 存在不可可靠辨读的生成字幕样文字。抽帧检查不代表逐帧、听感或口型同步验收。

证据位于上述发布目录的上两级：`audit_bundle.json`、`validation/<mode>/acceptance.json`、
`validation/<mode>/visual_review.json`；视频为 `validation/<mode>/video.mp4`。
本轮只核对 HMCC 源码的 Pad reflect 解析/分派，没有执行编译器。
独立 CPU ASR 结果见 `validation/audio_review.json`：五模式均识别出完整句子，
差异仅为“后摩”的同音字；原始 CER 为 T2VA 25%、其余模式 12.5%，不据此宣称人工听感或口型同步通过。

运行需要包含本轮 RMS、reflect 和 ViT 分档改动的 ModelZoo 源码，以及 SDK Gerrit
47132 PS3（`5674c7b1`）的完整源码树。发布目录的 `modelzoo.patch` 记录相对于
ModelZoo `6cd4830e` 的增量；仅使用此前 Gerrit 46780 的旧补丁版本不包含本轮改动。
运行不需要 GPTQModel、AutoRound、作者 Sana/LTX/upscaler 源码或原始权重。

以下步骤使用已经配置好的运行环境，不执行安装。已验证环境为 Python 3.12、
Torch/Torchaudio 2.8.0 的 CUDA 12.8 版本、Transformers 5.16.1、Diffusers 0.41.0、
NumPy 2.2.6、PyAV 15.1.0，以及 SDK 的运行依赖；图像/视频预处理还需要
Pillow 11.3.0 和 Torchvision 0.23.0。包内 `environment.json` 和
`environments/inference.txt` 是候选直接依赖声明，并非完整安装锁。
本次复用现有环境，未进行新环境安装验证。只推理不要安装 ModelZoo 的 `[export]` extra：
其 NumPy 范围与本包验证版本不同。

SDK 首次执行部分算子会 JIT 编译，因此除兼容的 NVIDIA 驱动外，还需匹配的 CUDA
工具链（含 `nvcc`）、C++ 编译器、PATH 中可用的 `ninja` 和可写的编译缓存目录。
这里运行的是 CUDA HMONNX simulator，不是芯片部署命令。80 GiB 是本次验证使用的
GPU 容量，不是模型的最低显存要求。以下为旧包的显存记录：在该卡上限制 PyTorch allocator 为 46 GiB，
为 CUDA context 等非 PyTorch 开销预留 2 GiB，模拟 48 GiB 的显存预算；
这不是物理 48 GiB 显卡实测，allocator 限额也不限制全部设备内存分配。

该预算下，FL2VA 首尾帧模式使用 1024-token prompt，已完成 1344×768、121 帧音视频
生成及媒体校验。PyTorch allocated/reserved 峰值分别为 21.0354/23.8965 GiB，
NVML 采样峰值为 24.3916 GiB。最大参考条件的 Ref2VA 也已完成全流程及媒体解码校验，
allocated/reserved 峰值为 25.2932/31.8125 GiB，NVML 采样峰值为 32.3096 GiB。
采样峰值不保证捕获瞬时峰值；上述是对应输入配置的预算验证，不是最低显存承诺。
上述显存数据来自简化 text refiner RMSNorm 之前的发布包，不能替代本轮新图的验证。

从本变更的 ModelZoo 仓库根目录运行；将 SDK 路径替换为对应补丁的本机 checkout，避免误用原有
editable 安装指向的另一份源码：

```bash
PYTHONPATH=/path/to/xhquanttool-patched:$PWD \
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir /data01/datasets/Sol-H3-assets/releases/unified_w4_rms1024_reflectpad_20261024/relocated/bundle \
  --prompt "A woman speaks clearly in Mandarin: 后摩智能，我的最爱。" \
  --seed 42 --output /outputs/sol-h3.mp4
```

普通单图参考增加 `--image /inputs/person.png`；尾帧用 `--last-image`，首尾帧同时提供
二者。Ref2VA 用 `--references '[{"kind":"image","path":"/inputs/person.png"}]'`，
也可按包内预算添加 video/audio 条目。媒体文件由本次请求提供，不读取构建时校准媒体。
包可整体复制到新目录，再修改 `--artifact-dir`；不要只复制 ONNX 或修改已冻结包内 JSON。

## 当前导出契约

Qwen 文本残差及 embedding 使用 /16；普通 SDK RMSNorm 的 epsilon 同步除以 256。
H3 context projection 在权重中补偿输入尺度，Host 不再额外除以 16。H3 主干默认使用 /1024
及普通 RMSNorm（epsilon 为 `2^-24`）；DiT累加精度由上述 `norm_fp16` 控制，Attention V/512 与 FFN 乘积补偿保留。
该主干 RMS 设置是有损精度方案，不影响 Qwen、text refiner 和 Q/K norm 的 epsilon。

可在自定义 workflow YAML 的 `export.h3_precision` 中配置（保留其余字段）：

```yaml
h3_precision:
  residual_scale: 1024.0
  rms_norm_mode: plain
  rms_norm_eps: 5.960464477539063e-8
```

恢复稳定 RMSNorm 时改为 `rms_norm_mode: stable`、`rms_norm_eps: null`；
`residual_scale` 可保持 1024，或设 4096 复现旧主干表示。stable 使用源模型的 epsilon/scale²，
通过最大值重标定和追加坐标限制倒平方根，不覆盖源 epsilon。配置在导出前校验，随包记录。
plain 的 epsilon 是缩放后特征的实际值，支持 FP32 表示，但 FP32 舍入后须不小于
`1/65504²`，因为当前 SDK 把倒平方根结果窄化到 FP16。默认 `2^-24` 对应上限 4096。
平方及归约继续使用 FP32；epsilon 能保存并不代表其倒平方根一定安全。
自定义 YAML 的构建命令见下文；统一 API 的默认 recipe 使用同一 plain 配置。

统一包使用 schema 5，单模式包使用 schema 4。五模式 H3 Dense 图统一使用 10 个输入：`video_tokens, audio_tokens, text_features, step, cos, sin, valid_length, row_roles, pack, unpack`。
每个模态槽位向上对齐 256；pack 将有效行放到 attention 前缀，unpack 还原槽位，输出裁回原生长度。
LTX 的四组 RoPE（共 8 个 cos/sin 张量）作为共享输入，随包保存在 `hf_config/ltx_rope.npz`。
加上 `video, audio, step`，去噪图共 11 个输入，避免展开后的 RoPE 常量在每层重复占用显存。
全部部署图使用 `.onnx` 后缀，算子仍为 HMONNX；图数随模式、文本分组及 VAE 档位变化。
加载时校验包内精度策略与组件契约；缺少当前 RMS 策略字段的旧包需要重新导出，不能混用不同文本尺度的组件。

统一包共用 Qwen 语言特征图；视觉图按原生 processor 预算导出 patch 容量档位。
默认三档为 1024、1280、65536，分别覆盖普通图、参考视频的单个时间组及最大参考图。
运行时按实际 `H×W` patch 数选最小可容纳档，每个视频时间组独立编码后按原顺序拼接；
所有档位使用非因果 FlashAttention，并输出 image features 和三路 DeepStack 特征。
容量向上对齐 256，不改变 processor 的图片尺寸和送入 LLM 的有效 token 数；
自定义配置若推算出相同容量则合并档位，单模式只导出所需容量。
参考 VAE 默认分为普通图片、Ref 图片、Ref 视频三个 encoder 档位，
LTX 视频 decoder 仍只有一个图。
其余音频、桥接和 LTX 组件也共用。H3 保留 `base`、`ref` 两条权重分支，各自包含文本 refiner
和 DiT；前四种模式使用 base，Ref2VA 使用 ref。包内 `mode_profiles` 记录每种模式的输入约束、
容量和分支绑定。运行时根据 `image`、`last_image`、`references` 选择模式，每次请求新建 KV cache。

Ref 参考 VAE 合图使用各分辨率的有效区域 mask，GroupNorm 排除补齐区域。空间边界
恢复原生 reflect，避免 zero padding 在条件视频中引入细白边；metadata 记录
`spatial_padding_mode="reflect"`。Ref 档每个尺度携带真实有效高宽的反射索引。
普通图片档导出显式 `Pad(mode=reflect) → Conv2d`，不依赖 Conv 的反射属性；
Ref 档用 Gather 在有效区域边界反射，不能替换成对整张 768 画布的固定 Pad。
时间维保持原生因果零边界。LTX decoder 本来使用空间零边界，每次卷积前屏蔽补齐区域，
保留原生 tile 拼接。这两处 mask 不改变原生预处理尺寸。

生产参考 encoder 默认三档：普通首/尾图 `1×384×672`、Ref 图片 `1×768×768`、
Ref 视频 `17×768×768`。普通图名 `h3_reference_encode_base_t1_h384_w672`，
仅接受 `pixels`，使用原生 GroupNorm，不带 mask、不补到768、不做 Host 空间切片；
尺寸随 YAML `draft_size`。Ref 图名后缀为 `ref_t1_h768_w768` / `ref_t17`，保留空间
mask，三档均保留原生空间反射边界。Ref 档位按配置和原生切片推算，边界 tile 共用最大画布。
`reference_vae_layout="split_v1"` 标识新布局，`reference_vae_base` 描述普通图；
`reference_vae_tiles` / `reference_vae_masks` 仅描述 Ref 路由与 shape、levels、padding。
请求模式选择角色，不能仅按尺寸区分。统一包仍为 schema 5，同包覆盖五模式；
旧单画布、缺失新布局标识或旧无 mask 参考包须重新导出。单模式构建也使用新布局。
视频 T=17 是原生固定 clip 长度，不是图内动态上限。Host 将长视频切成17帧片段；
不足17帧的末段复制末帧补齐，再按原生 `token_drop` 裁后验输出。不会改成任意T图，
也不会把视频尾片当作图片档位执行。

## 本地构建输入

`--model-dir` 使用轻量 source-index 目录，例如 `/data01/datasets/Sol-H3-merak-source`，
其中包含清单、Qwen 原生配置和校准输入索引。
避免将实验输出或虚拟环境放入该目录，公开 build 会对整个输入目录做内容指纹检查。
权重和作者源码可以留在 `/data01/datasets` 的原位置。清单文件为 `checkpoint-paths.json`：

```json
{
  "h3_model": "/data01/datasets/MiniMax-H3",
  "dense_lora": "/data01/datasets/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
  "ref2va_lora": "/data01/datasets/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
  "transformer": "/absolute/path/ltx-2.5-transformer.safetensors",
  "refiner_lora": "/absolute/path/ltx-2.5-distilled-lora.safetensors",
  "output_video_vae": "/absolute/path/ltx-2.5-conv-video-vae.safetensors",
  "audio_vae": "/absolute/path/ltx-2.5-audio-vae.safetensors",
  "h3_upscaler_checkpoint": "/absolute/path/h3-upscaler.safetensors",
  "adapter_dir": "/absolute/path/h3-ltx-adapter",
  "sana_source": "/data01/datasets/Sol-H3-assets/reproduction/Sana",
  "ltx_source": "/data01/datasets/Sol-H3-assets/reproduction/LTX-2",
  "upscaler_source": "/data01/datasets/Sol-H3-assets/reproduction/upscaler-source.py",
  "refiner_prompt_cache": "/data01/datasets/Sol-H3-assets/reproduction/fixed-prompt-int8-a100.pt"
}
```

W4 构建和 Qwen 量化 checkpoint 复用都要求 source-index 根目录同时放置 Qwen3-VL
的原生 `config.json`，内容必须与清单 `h3_model` 下的 `text_encoder/config.json` 一致：

```bash
cp /data01/datasets/MiniMax-H3/text_encoder/config.json \
  /data01/datasets/Sol-H3-merak-source/config.json
```

这只是复制配置到构建索引，不移动或修改原始权重。构建只读并校验两份 JSON，不会自动生成
或修正它们；显式 W16 构建不要求此配置镜像。

实际下载路径已记录在 `/data01/datasets/Sol-H3-assets/checkpoint-paths.json`，新的清单目录已创建。
构建只读本地文件，不下载 Hugging Face 权重。外部源码只在构建时加载，导出的 HMONNX 不依赖这些源码。
默认统一包还需要下文的 `calibration_inputs.json`，以及 H3 目录中的 `transformer_ref`。

构建源码版本：Sana `62da34b14a53d7289e82fa56692f34e96e089bee`，LTX-2
`d151147788a9284cca791edc6ce898007e727fe6`，upscaler companion
`d7c01b9011f2e8439493f6c02c29995a27df276f`。不要将上游 Gemma 的 Transformers 5.14.1 环境覆盖到 Merak
的 Transformers 5.16.1 环境；本构建使用预先生成的固定提示词 cache。上游源码与模型权重的许可证随原下载保留。

## 构建和推理

默认 workflow 将 Qwen3-VL 实际使用的前 50 层导出为一张 `h3_feature_00_50.onnx`，
`text_layers_per_graph=50`。两条 H3 分支各自保留独立的文本 refiner。
统一和单模式 profile 均使用 AutoRound W4/group64 checkpoint，Qwen 语言图为 W4A8，视觉图为 W8A8。

从仓库根运行，`CUDA_VISIBLE_DEVICES` 限定单卡，进程内统一使用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.sol_h3.build \
  --model-dir /data01/datasets/Sol-H3-merak-source \
  --quant-checkpoint /models/qwen3-vl-autoround-w4g64 \
  --output-dir /outputs/sol-h3-all

CUDA_VISIBLE_DEVICES=1 python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir /outputs/sol-h3-all \
  --prompt "A woman in detailed Chinese animation style speaks clearly in Mandarin: 后摩智能，我的最爱。" \
  --seed 42 --output /outputs/medusa.mp4
```

Qwen 语言特征图采用 `w4a8h0_ssfp`；其余图默认部署精度是 `w8a8_sefp`：Linear/Conv 的权重与输入均为 SEFP8，FlashAttention 的 Q/K/V/P 均为 8 位，score 使用 SDK 的 FP16 路径。
图接口和浮点常量是 FP16，索引是 int32；RMSNorm 保留宽累加，Linear/Conv 使用 FP24 累加。
Host 使用 FP32 solver 状态、音频预处理和 tile 混合。时间步由 int32 `step` 输入选择，不能修改步数后复用原表。

单输入通道的 Conv1d/Conv2d 保持原始图结构，由 SDK 直接分派到 VP 分支，无需广播输入。
例如参考音频 encoder 首层 `[64, 1, 7]` 保持 `in_channels=groups=1, out_channels=64`；
原有 `groups=Cin=Cout>1` 的 depthwise VP 分支继续保留。
这类卷积使用 SDK 的 VP 分支及原始 FP16 系数；节点上的 SEFP 属性属于 TE 分支，不能据此称其为 W8 计算。
图审计通过 `vp_fp16_convolutions` 单独报告这些节点。GPU simulator 的分支验证不代表芯片加速比。

产物包含根目录 `export_meta_info.json`、`model_build_manifest.json`、`runtime_meta_info.json`，
各角色的 HMONNX、独立 JSON 配置、tokenizer 和必要的 NumPy 资源。产物可以搬迁；运行不读取原始 checkpoint。

## 历史记录：2026-10-22 完整 W4 统一候选包（当前代码不兼容）

该历史包位于 `/data01/datasets/Sol-H3-assets/releases/unified_w4_20261022/bundle`，
完整权重公开构建和搬迁已完成，共 15 个图，实际占用约 85.83 GiB。
Qwen 语言前 50 层为 AutoRound W4A8/group64，视觉为 W8A8；prompt 上限 1024，
公共 KV 容量 18944，参考 VAE 与 LTX video decoder 各一个 mask 图。
隐藏源目录后的公开 API 已完成五种模式的 1344×768、121 帧有声视频。
SDK VP 逐次 FP16 舍入修复后，五模式公开 API 已重新完成验收，记录在 `validation_vp_fp16/`。
同一 runtime 的 T2VA→Ref2VA→T2VA 固定输入与 seed 对照通过：前后 T2VA 编码前的
RGB 与 PCM 均逐位相同，也与独立公开 API 的 T2VA 相同，见
`runtime_switch_vp_fp16/raw_equality.json`。该结论不代表所有图转换均无量化误差。
早期有损视频摘要比较曾失败；中间张量重放逐位一致，同一原始 RGB 跨进程编码可产生
不同的解码后摘要。旧基线未保存全部原始 RGB，不能将全部历史差异归因为编码。
单通道 VP 的消费前端 mantissa 解析问题尚未闭环，不能宣称该包已可编译部署。
旧五个独立包保留至新包完整验收。

上述历史构建日志曾执行 LTX 三步，为后续 VAE 导出产生真实 latent。
当前 Sol-H3 构建只导出组件，使用 CPU 上确定性、有界的契约示例连接各导出阶段，
不回放 Qwen、参考编码器、桥接、音视频编码器或 H3/LTX 采样管线。
真实权重的 AdaLN 表、固定 prompt 文本 KV、RoPE 等必要导出常量仍正常生成；
导出器内部的图捕获和 weight RTN、显式请求的 GPTQ/AutoRound 权重量化也保留。
生成与质量验收通过发布包的独立推理入口执行，不由导出自动启动。

## 历史记录：完整 W4 五个独立包（当前代码不兼容）

发布根目录：`/data01/datasets/Sol-H3-assets/releases/w4a8_five_modes_20261022`。
`bundles/{t2va,i2va,l2va,fl2va,ref2va}` 是五个可独立搬迁的完整包，图数分别为
13、15、15、15、65。所有组件均由当前代码重新导出；语言权重复用已验证的完整
AutoRound W4/group64 checkpoint，未复用旧实验的组件图。

五包均通过公开 `build_model` 构建，搬迁后通过公开 `inference_model` 生成。
生成进程的 mount namespace 隐藏原始权重、量化 checkpoint、作者源码、构建输入及旧实验目录，
进入推理前逐项确认这些目录为空；共享源权重仍保留在宿主机。
实际序列化图审计确认：每包 Qwen 前 50 层的 350 个 Linear 为 W4A8，
条件模式视觉图的 117 个 Linear 为 W8A8。单通道 Conv 使用 SDK 原生 VP 分支。

验证媒体位于 `validation/<mode>/medusa.mp4`，均为 1344×768、121 帧、24 fps，含
32 kHz 双声道音轨，解码数据有限且非静音。Ref2VA 同时使用一张图、一段视频和一段音频。
各模式的 `acceptance.json`、`inputs.json` 和 `frames.jpg` 保存完整验收、输入及抽帧。
独立 Qwen3-ASR 未提供预期文本，五段均识别到“后摩智能，我的最爱”的同音句；原始文字
“后魔”或“厚膜”保存在 `transcripts.json`。抽帧与 ASR 不代替完整运动、口型或主观音质评估。

发布目录的 `README.md` / `verification.json` 汇总模型、视频、测试和清理结果。
`deduplication.json` 记录实际 SHA-256 核对后共享的相同文件；硬链接不引入包外路径依赖。
23 个旧模型及实验输出目录已删除，清理期间磁盘可用空间增加约 1.218 TiB。
五包共享相同文件后合计实际占用约 105.78 GiB；独立复制单包时需按该包完整逻辑大小预留空间。
清理结果见 `cleanup_receipt.json`；历史小型回执已归档到 `previous_receipts/`，
早期 /256、W8 及常量 RoPE 的数值结论仍见建模记录。

## 验证

当前测试均在 `benchmark_test/ci_test_transformer5.16.0`，包括原生 H3/文本/LTX/codec 数值对照、hybrid LoRA
完整合并、静态卷积、产物契约和长序列 `P×V` 溢出回归。
仅导出用例禁止构建编排调用 HMONNX 推理，并核对完整图集、资源和五模式 ABI。
独立生成用例在发布、搬迁及删除源权重后，才执行所有学习组件、4+3 次更新和音视频 mux。
CI 用已安装的 Diffusers/Transformers 与真实 PyTorch 小组件构造权重，不依赖作者源码目录或 release 大权重。
加载边界使用这些小组件，图的 forward、SDK 导出、搬迁删源和五模式推理真实执行；这不认证作者 checkpoint builder 或原生 LTX-2.5 数值等价。

```bash
CUDA_VISIBLE_DEVICES=3 ./benchmark_test/ci_test_transformer5.16.0/run.sh \
  -k sol_h3
```

完整 W4 五模式验收见上节，相关 CI 日志位于同一发布目录。
历史 BF16/W8 对照执行 H3 四步加 LTX 三步，不包含 H3 独立高分辨率约 49 步，
不能用来证明级联方案与原始 H3 的生成质量相当。
独立 wheel 安装与核心模块导入已通过。独立环境的 T2VA、I2VA、L2VA 已完成，
编码前 RGB 与 PCM 与对应源码环境结果逐位相同；后续扩展验证按用户要求停止。
这不等于独立安装后的五模式验收全部完成，记录见发布目录 `validation_installed/`。

`*_unscaled_diagnostic` 与 `*_schema2_prompt_diagnostic` 是旧诊断产物，schema-3 runtime 会拒绝加载。
正式档位的数值、视频质量和真机性能分别记录验证状态；GPU HMONNX 执行不等于 XH2a 真机性能认证。

## Qwen3-VL 的 GPTQ / AutoRound 接口

ModelZoo adapter `sol_h3.quant_adapter.quantize_conditioner` 调用生产端已注册的
`gptqmodel.recipes.quantize_qwen3_vl`，支持 `method="gptq"` 和 `"autoround"`。
输入为原生 H3 presentation 的 tensor 字典列表，保留图像/视频 grid、像素和 token 类型，
通过 `torch.save(samples, calibration_file)` 提供。不可先加聊天模板或只留下文本。

```python
from xhmodel_merak.xh_other_model.models.sol_h3.assets import SourceAssets
from xhmodel_merak.xh_other_model.models.sol_h3.quant_adapter import quantize_conditioner

result = quantize_conditioner(
    SourceAssets.read("/models/sol-h3-inventory"),
    "/outputs/qwen3-vl-w4",
    {"method": "autoround", "calibration_file": "/calibration/h3.pt",
     "bits": 4, "group_size": 64, "seed": 42, "iters": 200},
    layers=50, device="cuda:0",
)
```

结果包含 `output_dir` 和 provenance。`SolH3Workflow.quant` 使用相同 adapter；
设置 workflow 的 `quant` 为上述字典后，`export` 消费返回的 QuantResult。
导出复用 Merak 的 GPTQ packed 权重转换，语言特征图使用 W4A8 SSFP，
其余图沿用原有精度，图清单逐图记录 `quant_type`。
公开 `build_model(model_id="sol-h3")` 默认执行 AutoRound W4；在 model_dir 提供
`calibration.pt`（原生 tensor 样本列表）。使用 `quant_checkpoint="/models/qwen-w4"` 可复用
完整 GPTQ 或 AutoRound checkpoint，构建前验证 Qwen 组件的身份、格式、层数及载荷。
显式传入 `quant_method=None` 保留原始浮点 checkpoint → W8 图的对照路径。
统一包及五个单模式的 `*_w4.yaml` 均显式要求 W4，直接 workflow 缺少相应 checkpoint 时会报错，不静默回退 W8。

生产端必须包含 Qwen3-VL recipe、DeepStack 校准修复及随仓 AutoRound 的 dynamic override 修复。
消费端复用 Merak 的 `prepare_loaded_module`，对 v2 只补齐格式标记，整数载荷保持不变。
小模型联调覆盖两个算法的真实量化、重载及三层 W4A8 HMONNX 导出；
完整 32B、前 50 层 AutoRound W4 checkpoint 已通过重载与 Ref2VA 导出验证；
统一包完整权重构建已完成，五种模式的生成和跨模式切换验收进度见上文；
不能用小模型或旧独立包结果代替新包验收。

## 文本特征的分块 prefill

默认 workflow 使用以下配置：

```yaml
export:
  mode: all
  prompt:
    max_tokens: 1024
  qwen3_vl:
    prefill_chunk_length: 256
  text_layers_per_graph: 50
```

`prefill_chunk_length` 是每次执行的 token 数，`prompt.max_tokens` 是用户提示词上限，
导出得到的 `text_capacity` 是整段条件的容量，
两者不能互相替代。每个分组图顺序处理所有 chunk，保存全部有效 hidden states，
随后进入下一层分组；不会只取最后一个 token，也不会对输出做 embedding pooling。
各层独立使用 SDK `LLMCacheV2`，每次请求新建缓存，末尾 padding 不计入有效长度。
缓存容量同时向上对齐到 256 和 chunk 的公倍数；图分组仍由 `text_layers_per_graph` 控制。

产物记录 `text_prefill_chunk_length`、`text_cache_shape` 以及各特征图覆盖的层范围，
加载端核对 cache 的实际图输入尺寸。旧的整段特征图产物继续按其原有 ABI 加载。
默认 YAML 的提示词上限为 1024；可以修改 `prompt.max_tokens`，导出会核对原生
`max_position_embeddings` 并记录 Qwen、KV cache 和 DiT 容量。参考媒体容量规划已有
原生 processor 对照测试；通过下述 profile 选择五种模式。

## 模式、输入上限与 YAML 选择

unified API 默认构建五模式共用包，也保留每个模式的独立 workflow：

| 模式 | `build_model(profile=...)` | 推理媒体输入 |
| --- | --- | --- |
| 五模式共包 | `all-dense-384p-768p-5s`（默认） | 按下述媒体输入自动选择 |
| T2VA | `dense-384p-768p-5s` | 无 |
| I2VA | `i2va-dense-384p-768p-5s` | `image` |
| L2VA | `l2va-dense-384p-768p-5s` | `last_image` |
| FL2VA | `fl2va-dense-384p-768p-5s` | `image`、`last_image` |
| Ref2VA | `ref2va-dense-384p-768p-5s` | 有序 `references: [{kind, path}, ...]` |

默认 YAML 在 `configs_merak/workflows/xh2a/other_models/sol_h3/`。
首尾帧数量由模式固定为 1/1/2，按目标画布预处理，不使用 `references` 容量配置。
Ref2VA 默认配置如下；尺寸数组均为 `[height, width]`：

```yaml
export:
  mode: ref2va
  profile: ref2va-dense-384p-768p-5s
  prompt:
    max_tokens: 1024
  qwen3_vl:
    prefill_chunk_length: 256
    # 不覆盖像素预算，沿用 checkpoint 的 processor 配置
  references:
    image:
      max_count: 1
      vae_short_edge: 2048
    video:
      max_count: 1
      max_duration_seconds: 5
    audio:
      max_count: 1
      max_duration_seconds: 5
```

默认不覆盖 Qwen processor 的像素预算。本地官方 checkpoint 的 image `longest_edge` 为
16777216，video 为 25165824；修改模型来源后以其实际 processor 配置为准。
`image_max_pixels` / `video_max_pixels` 仍可在自定义 YAML 显式覆盖。
当前固定版本的 video processor 把视频预算用于整段采样视频的 `T×H×W`，不是每帧面积。
实际 token 数使用原生 patch、merge、时间采样和取整规则计算。
例如 120 帧、384×672 的参考视频按 2 fps 采样为 10 帧，官方预算下保持画面尺寸，
时间 patch 2、空间 patch 16、merge 2，产生 `5×12×21=1260` 个 Qwen token。

Ref2VA 图像 VAE 默认沿用官方短边 2048，无额外面积压缩，宽高比在 1:4 到 4:1 内，
因此最大画布是 2048×8192（或转置），按 32 对齐。Qwen 接收同一张原生预处理图像，
再按其 processor 规则处理，单图最多 16384 个 Qwen token。
参考视频 VAE 原生策略沿用生成画布的短边和面积约束；本 profile 生成画布仍为 384×672，
没有因恢复参考图片的官方短边而改变生成尺寸。VAE tile 拆分也继续沿用现有原生策略。
音频不进入 Qwen 视觉编码器，其 presentation 标签进入文本序列，音频 latent 进入 DiT。

导出自动记录 `runtime_meta_info.json.capacity_plan`：

- `text_tokens`：提示词 token 上限＋所有媒体 Qwen token 上限＋原生标签、时间戳开销。
- `text_capacity`：`text_tokens` 向上对齐到 256。
- `kv_capacity`：同时对齐到 256 和 `prefill_chunk_length` 的公倍数。
- `video_capacity` / `audio_capacity`：生成段 token＋参考媒体 VAE token，再分别向上对齐到 256。
- `dit_capacity`：上述 text、video、audio 三项之和；DiT 输入使用有效行压紧的布局。

默认 prompt 为 1024、恢复官方参考预算并按 256 对齐后，使用本地官方 H3 tokenizer/processor 和五份默认 YAML 推算的静态上限如下
（不是每个请求的实际 token 数）：

| 模式 | Qwen 总容量 | KV 容量 | DiT 总容量 |
| --- | ---: | ---: | ---: |
| T2VA | 1,024 | 1,024 | 11,008 |
| I2VA / L2VA | 1,536 | 1,536 | 11,776 |
| FL2VA | 1,792 | 1,792 | 12,288 |
| Ref2VA | 18,944 | 18,944 | 54,784 |

旧 `w4a8_five_modes_20261022` 包仍为 prompt 512；新统一候选包已按上述预算重新导出。

Ref2VA 的未对齐文本上界为 `1024+16384+1350+64=18822`。
共包的公共 Qwen KV 因而取 18944。base 分支共用 text/video/audio 槽位
1792/9984/512，ref 分支使用 18944/34560/1280。单模式表中的容量仍适用于独立导出。
`*_tokens` 保留原生数量，图像/视频尺寸和实际有效长度不因容量对齐而改变。

Ref2VA 包含可变宽高比的保守 VAE 容量，以及视频可能携带的音轨。
更换 tokenizer、processor 或上限后以重新导出的 capacity_plan 为准。

增加 chunk 只改变 Qwen 分块执行和 cache 对齐，不增加 prompt 或 DiT 的允许长度。
参考图同时贡献 Qwen 特征 token 和图像 VAE token，两项都要计入 DiT。
提示词按 tokenizer token 计数，超限报错而不截断。文件音轨按自身 stream duration 去除
不足一个编码帧的解码尾部补齐（例如 5 秒 AAC 的尾帧 padding），不按配置上限或视频
时长截短音轨。无有效时长、达到或超过一个编码帧的差异及内存音频对象不作该裁剪。
数量、有效原始时长在原生缩放和裁剪前检查；
单模式包要求输入匹配该模式；统一包按媒体字段选择已登记模式。修改上限后需要重新导出，不能只修改已发布包的 JSON。

原生上限继续生效：图像最多 9、视频最多 3、独立音频最多 3；单段音视频 2–15 秒，
视频和音频分别有累计 15 秒限制。默认 YAML 的上限更小。视频自带音轨不占独立音频数量，
但受音频时长限制，导出也会为每个视频预留音轨容量。参考 VAE 使用原生 tile 拆分；
单模式包按实际档位导出，统一包用 mask 合图。统一图的 GroupNorm 只统计有效区域，
各次卷积前清除无效区域；当前统一图采用有损的空间零边界，不宣称与原生 reflect 等价。

### 校准输入与构建

条件模式需要真实媒体，由原生 processor 准备尺寸、token 布局和导出样例。
字段仍名 `export.calibration_inputs`，不替代独立的 GPTQ/AutoRound 权重量化校准。
自定义 YAML 可填写：

```yaml
calibration_inputs:
  image: /calibration/first.png
  last_image: /calibration/last.png
```

这是 FL2VA 示例；I2VA 只提供 image，L2VA 只提供 last_image；Ref2VA 提供 references。
Ref2VA 开启视频容量时必须提供校准视频，开启音频容量（含视频音轨）时必须提供校准音频。
校准图片不固化为运行时参考图。Ref2VA 源清单还需要 `ref2va_lora`，H3 目录需要
`transformer_ref`；融合复用现有 PEFT 加载器与 LoRA alpha/rank 规则。

使用 unified API 的默认 YAML 时，在 model_dir 放置 `calibration_inputs.json`，按模式保存媒体：

```json
{
  "i2va": {"image": "media/first.png"},
  "l2va": {"last_image": "media/last.png"},
  "fl2va": {"image": "media/first.png", "last_image": "media/last.png"},
  "ref2va": {"references": [
    {"kind": "image", "path": "media/person.png"},
    {"kind": "video", "path": "media/motion.mp4"},
    {"kind": "audio", "path": "media/voice.wav"}
  ]}
}
```

该 JSON 中相对路径以 model_dir 为基准；自定义 YAML 中的媒体建议使用绝对路径。
YAML 显式 calibration_inputs 优先。校准清单参与统一构建输入指纹；校准媒体不随产物加载。
当 YAML 的 `mode: all` 时，`export.calibration_inputs` 也使用上述按模式组织的字典；
构建从 FL2VA 和 Ref2VA 校准样本分别准备 base/ref 分支，共享编码图只导出一次。

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

build_model(model_id="sol-h3", model_dir="/models/sol-h3-inventory",
            profile="fl2va-dense-384p-768p-5s", output_dir="/outputs/sol-h3-fl2va", device="cuda:0")
inference_model(artifact_dir="/outputs/sol-h3-fl2va", device="cuda:0",
                inputs={"prompt": "A woman in Chinese animation style smiles gently", "image": "/inputs/first.png",
                        "last_image": "/inputs/last.png", "output_path": "/outputs/result.mp4"})
```

CLI 同样支持 `build --profile ...`、`generate --image ... --last-image ...`，
Ref2VA 使用 `generate --references '[{"kind":"image","path":"/inputs/person.png"}]'`。
要改变上限，复制完整默认 YAML 并编辑，再选择 workflow：

```bash
python -m examples_merak.aigc.sol_h3.build \
  --model-dir /models/sol-h3-inventory --workflow /configs/my_ref2va.yaml \
  --output-dir /outputs/custom-ref2va
```

该命令复用 `AutoWorkflow.from_config(...).quant()/export()`；产物位于输出目录的
`sol_h3_export/`，可用 `SolH3Runtime` 加载。统一 build 额外生成 Manifest；
直接 workflow 产物不能冒充统一 API 的发布包。

五模式已用最小真实模型通过公开 `build_model` / `inference_model`：构建、产物搬迁、
删除源权重后生成 121 帧带音频视频。条件模式实际选择各自 profile YAML，并读取校准媒体清单。
条件 DiT 使用十输入 ABI，固定参考前缀、仅更新生成行。完整权重五模式的独立验收见上节；
小模型和完整权重的 GPU simulator 验证均不代表 XH2a 真机性能。
