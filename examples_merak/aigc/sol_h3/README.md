# Sol-H3：4 步 H3 + 3 步 LTX-2.5

独立实现位于 `xhmodel_merak/xh_other_model/models/sol_h3/`，公开 recipe 为 `sol-h3`。
本目录使用新的 workflow、产物和 runtime。完整结构、数值改写和证据见
[Sol-H3 建模记录](../../../docs/modeling/structures/sol-h3.md)。

默认公开档位 `all-dense-384p-768p-5s` 在一个包中支持五种模式：672×384、124 帧 H3 粗生成，随后 1344×768、121 帧 LTX 精修，24 FPS。
`all-dense-384p-768p-15s` 档位另导出一套五模式包，H3/LTX 帧数为 `362/361`。
另有 `all-dense-384p-768p-portrait-5s` 和 `all-dense-384p-768p-portrait-15s`
竖屏档位，H3 画布为 384×672，LTX 输出为 768×1344（宽×高），帧数分别与对应横屏档位相同。
当前包使用 schema 9，支持运行时选择 LTX 音频策略；模型位置及本轮验证见下文。
论文宣传中的 360p/720p 对应这里使用的作者对齐尺寸。注意当前 Merak 配置的两阶段 attention 都是 dense；
4/3 步蒸馏权重、AdaLN 折叠、逐时间步精确文本 KV 缓存和数值补偿已接入，本次发布仅开放 Dense。

## Ref2VA 默认精修与音频策略

使用当前 5 秒／15 秒 schema 9 包时，Ref2VA 默认采用 `ltx_guide_policy="midguide"`
和 `ltx_audio_mode="joint"`。前两步 LTX 分别使用同一次 H3→LTX bridge 视频 latent 的
中段时间格作 clean guide：5 秒为 `4`、`10`，15 秒为 `12`、`30`；第三步撤掉 guide。
仍然实际运行三次 LTX，所有目标视频 token 三步都更新，guide 仅作条件。它复用现有图的
1008-token 槽，guide 来自生成的 latent，不对输入参考图额外运行图像编码；不需要重导
模型、更换权重或增加 ONNX 图。

统一接口的 `inputs["ltx_guide_policy"]` 与 CLI 的 `--ltx-guide-policy` 均可选：

| 值 | 行为 |
| --- | --- |
| 省略参数 | Ref2VA 使用 `midguide`；T2VA、I2VA、L2VA、FL2VA 保持原有 guide 行为 |
| `midguide` | 显式启用上述方案，只支持有静态 guide 槽及输入 RoPE 的 5 秒／15 秒 Ref2VA 包 |
| `none` | Ref2VA 回退无中段 guide 的原三步；其他四种模式保持原有 guide 行为 |

选择 `none` 不改变音频策略。音频由 `inputs["ltx_audio_mode"]` 或
`--ltx-audio-mode` 独立指定，可选 `joint` 和 `frozen`：Ref2VA 省略该参数时采用 `joint`，
其他四种模式采用包内默认值（当前 schema 9 包为 `frozen`）。显式指定会覆盖默认值：

统一接口的两个策略字段均接受相应的字符串值；自动选择时省略对应字段。直接调用底层
`SolH3Runtime` 时，两个参数仍可使用默认值 `None`。

| 策略 | LTX 阶段的音频行为 |
| --- | --- |
| `joint` | 音视频加噪并在三步中一起去噪；是本次 Ref2VA 中段 guide 扩样本的实测设置 |
| `frozen` | H3 音频编码得到的干净 latent 保持不变，音频 sigma 为 0；三步只更新视频；本次中段 guide 扩样本未覆盖该设置 |

固定素材、prompt 和 seed 可对比两种音频策略及 `none` 回退。两种音频策略最终均将 H3
生成的原 PCM 音轨封装进视频；`joint` 更新的 LTX 音频 latent 用于精修时的音视频交互，
不替换最终音轨。两种策略的初始视频噪声抽样顺序一致。

扩样本对照使用四套单人单图 Ref2VA、两种时长及两个 seed，共 16 对，均为 `joint`。
六个固定时刻的单名 AI 助手盲评中，midguide 的外观接近 H3 为 13 较好、0 较差、3 无明确差异；
细节不稳定更好，五官和服饰仍可能被重画。另有一组精灵多参考样例覆盖 5 秒／15 秒，
这些结果不能推广为所有视频、音频参考或动态场景都有效；未做逐帧人工播放或口型验证。
详细方法和结果见本机 `work_dirs/sol_h3_guide_generalization_20261104/REPORT.md`。

统一接口示例使用随仓咖啡馆对白提示词，并显式选择 `frozen`：

```python
from pathlib import Path
from xhmodel_merak.model_inference import inference_model

prompt = Path("examples_merak/aigc/sol_h3/prompts/cafe_fl2va_5s.txt").read_text(encoding="utf-8")
result = inference_model(
    artifact_dir="work_dirs/sol_h3_dual_audio_20261103/bundle_5s", device="cuda:0",
    inputs={"prompt": prompt, "image": "data/sol_h3/cafe_first_hd.png",
            "last_image": "data/sol_h3/cafe_last_15s.png", "seed": 43,
            "ltx_audio_mode": "frozen", "output_path": "/tmp/cafe.mp4"},
)
print(result.stats["ltx_audio_mode"], result.stats["ltx_audio_frozen"])
```

命令行示例使用随仓咖啡馆图片和音频，通过默认 Ref2VA 策略生成画外旁白场景；
女孩不说话，参考音频只用于画外旁白的音色：

```bash
REFS='[{"kind":"image","path":"data/sol_h3/cafe_first_hd.png"},
       {"kind":"audio","path":"data/sol_h3/cafe_reference.wav"}]'
PROMPT=$(cat <<'PROMPT'
subject_definitions:
<Subject 1> is the adult woman at the cafe counter shown in <Picture 1>.
<Audio 1> is a voice-timbre reference for an off-screen narrator (S2), not speech from <Subject 1>.

summary:
[reference generation + audio reference] A single five-second cafe shot with an off-screen narrator. The woman remains silent.

retention_analysis:
<Subject 1>: preserve her facial appearance, hairstyle, accessories and clothing from <Picture 1>.
<Audio 1>: use only its voice timbre for the narrator; do not reuse its original dialogue or audio signal.

detailed_description: [Shot 1] The woman stands quietly at the cafe counter, with the framing and warm lighting of <Picture 1>. She smiles gently and handles a glass without speaking. An off-screen narrator (S2) says once: <d>[Chinese]忙碌的一天，先坐下来喝杯咖啡。</d> No visible person speaks. The picture remains free of subtitles, captions, letters, logos and watermarks.

overall_soundscape: Soft cafe room tone and faint cup sounds accompany the off-screen narration. No other speech.

non_diegetic_music: N/A
PROMPT
)
python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir work_dirs/sol_h3_dual_audio_20261103/bundle_5s \
  --prompt "$PROMPT" --references "$REFS" --seed 42 \
  --output /tmp/cafe-narration.mp4
```

例如对同一输入回退中段 guide，保留默认 `joint` 音频：

```bash
python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir work_dirs/sol_h3_dual_audio_20261103/bundle_5s \
  --prompt "$PROMPT" --references "$REFS" --ltx-guide-policy none --seed 42 \
  --output /tmp/cafe-narration-no-midguide.mp4
```

统一接口中同样可传入该开关；省略两项策略时，Ref2VA 使用 `midguide` 和 `joint`：

```python
from xhmodel_merak.model_inference import inference_model

result = inference_model(
    artifact_dir="work_dirs/sol_h3_dual_audio_20261103/bundle_5s", device="cuda:0",
    inputs={"prompt": "A woman at a cafe smiles gently.",
            "references": [{"kind": "image", "path": "data/sol_h3/cafe_first_hd.png"}],
            "ltx_guide_policy": "none", "seed": 42,
            "output_path": "/tmp/cafe-no-midguide.mp4"},
)
print(result.output_path, result.stats["ltx_guide_policy"], result.stats["ltx_audio_mode"])
```

CLI 的 `--ltx-audio-mode` 与统一接口的 `inputs["ltx_audio_mode"]` 接受相同枚举，输出 stats
记录所选的 `ltx_guide_policy`、`ltx_audio_mode` 和 `ltx_audio_frozen`。
`ltx_guide_policy` 为 `none` 仅表示未使用 Ref2VA 中段 guide，首尾图的 guide 不受影响。
底层图在原输入末尾增加独立的
INT32 `[1]` `audio_mode`：`0=joint`、`1=frozen`；`step` 仍为 0–2。
默认首尾帧静态 guide 图为 14 个输入，target-only 图为 12 个输入。模型内部按该输入选择
相应折叠表，Host 按同一选择决定是否加噪和更新音频。
每次生成开始前确定策略，三个去噪步使用同一个值；直接调用 ONNX 时也必须同时匹配音频输入和 Host 更新规则，不能只改图输入或中途切换。
旧脚本原本采用联合去噪时，保留原音频加噪和 Euler 更新，并传入 INT32 `[0]`；
改用冻结策略时，传入 INT32 `[1]`，音频输入保持干净的编码 latent，Host 不对它加噪或更新。
直接使用上述统一接口／CLI 会自动完成这些配套操作。

上述音频可选 ABI 原本需要重导两个时长的 LTX DiT，不能只改旧包 metadata；这次中段
guide 则复用现有 schema 9 图，无须再导出。已上传的 schema 8 图固定 `frozen`；加载器
可读且通过产物校验的 schema 3/4/5 旧图固定 `joint`。运行时根据静态 guide 槽、输入 RoPE
及控制输入检查图能力；旧包不支持默认中段 guide 时会报错，可显式传入
`ltx_guide_policy="none"` 回退。schema 8 的 Ref2VA 还须显式设置
`ltx_audio_mode="frozen"`，以匹配该图固定的音频策略。直接集成 ONNX 的旧 13 输入脚本
须同步音频控制输入和 Host 规则。

例如使用上述 `$PROMPT` 和 `$REFS` 以原策略运行 schema 8 包：

```bash
python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir work_dirs/sol_h3_external_mask_20261103/bundle_5s \
  --prompt "$PROMPT" --references "$REFS" --ltx-guide-policy none \
  --ltx-audio-mode frozen --seed 42 --output /tmp/cafe-schema8.mp4
```

两种时长的参数用法一致，时长由 `--artifact-dir`／`artifact_dir` 选择，不能用 5 秒包生成 15 秒。
例如 15 秒咖啡馆 I2VA 对白，在仓库根目录运行：

```bash
python -m examples_merak.aigc.sol_h3.generate \
  --artifact-dir work_dirs/sol_h3_dual_audio_20261103/bundle_15s \
  --prompt "$(cat examples_merak/aigc/sol_h3/prompts/cafe_i2va_15s.txt)" \
  --image data/sol_h3/cafe_first_hd.png --ltx-audio-mode frozen --seed 43 \
  --output /tmp/cafe-15s.mp4
```

## 当前 schema 9 模型与验收

本轮模型位于以下本机目录；每个包均支持五种生成模式，只重导本时长的 LTX DiT，
其余组件复用已验证的 schema 8 包：

| 时长 | 模型目录 | ONNX 图数 | LTX 成片 |
| --- | --- | ---: | --- |
| 5 秒 | `work_dirs/sol_h3_dual_audio_20261103/bundle_5s` | 20 | 1344×768，121 帧，24 FPS |
| 15 秒 | `work_dirs/sol_h3_dual_audio_20261103/bundle_15s` | 23 | 1344×768，361 帧，24 FPS |

相对下文已上传的 schema 8 包，LTX 新增 `audio_mode` 输入和两套音频条件表，
每个包的 LTX external data 增加约 1.14 GiB；DiT 主矩阵权重共用，不是两张完整 LTX 图。
非 Ref2VA 模式保留包默认的 `frozen`，Ref2VA 默认使用 `joint`；H3、VAE 和外部 mask 接口沿用原包。
下面的 Artifactory 链接仍对应固定冻结音频的 schema 8 包，不能切换这两个策略。

实际权重的固定 H3 输出回放共六条：5 秒精灵 Ref2VA、5 秒和 15 秒咖啡馆 I2VA，
每条分别运行 `joint`／`frozen`。精灵的两种结果分别与独立的单策略回放 RGB 逐位一致，
`joint` 未出现原冻结结果新增的尾巴；咖啡馆两档的 `frozen` 在编码前 RGB、PCM 上均与
schema 8 基线逐位一致。每组两种策略的最终 H3 PCM 相同，初始视频噪声也相同。

另外完成两条 `inference_model` 全权重完整流程：5 秒精灵 Ref2VA 选 `joint`，
15 秒咖啡馆 I2VA 选 `frozen`，分别输出 121／361 帧，均为 24 FPS。
两条完整流程在三步 LTX 输入、音视频 velocity、编码前 RGB 和 PCM 上，均与对应的固定 H3
回放逐位一致。咖啡馆回放的图片和提示词与上述随库文件一致。

本轮选中 CI 共 1020 项通过，覆盖五模式的小模型构建、搬迁、统一接口推理，以及全部三档
`step`、同一图 session 的 `0→1→0` 策略切换数值对照；仅新增一个参数化的策略导图测试。
全权重导出与生成固定使用 SDK `98997416` 和原对照使用的 hsum，避免混入 SDK 数值变化。
本机 SDK `bd9e21d8` 另通过两项小模型动态策略导图测试，不能据此推断全权重结果也逐位一致。
这些记录均为 CUDA HMONNX simulator 验证，未验证 XH2a 真机性能，也未新增定量口型评分。
本机 `work_dirs/sol_h3_dual_audio_20261103/validation_summary.json` 汇总导出、数值对照和生成记录。

## 已上传的 schema 8 包与 11 月 2 日上传包的区别

已上传的 schema 8 本机模型目录为 `work_dirs/sol_h3_external_mask_20261103/bundle_5s` 和
`work_dirs/sol_h3_external_mask_20261103/bundle_15s`；分别包含 20、23 张 ONNX，均支持五种模式。
11 月 3 日的整包已按 `zip -r -0` 打包并上传 Artifactory：
[5 秒模型](http://10.10.1.53:8081/artifactory/model_zoo2/houmo/minimax_h3/solh3/sol_h3_5s_20261103.zip)、
[15 秒模型](http://10.10.1.53:8081/artifactory/model_zoo2/houmo/minimax_h3/solh3/sol_h3_15s_20261103.zip)。
ZIP CRC 通过，服务器文件大小及 SHA-256 与本地一致；包内只包含 Manifest 登记的模型文件。
对照基线是 Artifactory 上的 `sol_h3_5s_20261102.zip`、`sol_h3_15s_20261102.zip` 整包，
已直接读取 ZIP 内 metadata 和 Manifest 核对：

| 项目 | 11 月 2 日整包 | 11 月 3 日已上传模型 |
|---|---|---|
| runtime schema | 5 | 8，需配套新 runtime／SDK |
| Qwen Vision | 1024、1280 两张图；patch mask 和整数长度 | 共用 1280 一张图；外部 `[1,1,1,1280]` key mask |
| H3 文本 refiner | 输入有效文本长度，在图内生成 mask | base/ref 直接输入外部 mask，容量 1792／3328 |
| FlashAttention | 旧 mask 槽位，短 KV 仍用 FlashAttention | mask 在下标 4；静态 KV <512 使用显式 MatMul 分支 |
| LTX 音频 | 与视频联合去噪 | 干净音频固定，音频时间步为零，三步仅更新视频 |
| 首尾图检查 | 未强制高清尺寸 | I2VA／L2VA／FL2VA 要求宽至少 1344、高至少 768 |
| 图数（5s／15s） | 21／24 | 20／23 |

H3 的 4 步、LTX 的 3 步、生成尺寸、帧数及首尾帧 guide 机制保留。
图接口和 LTX 条件语义都发生过变化，不能只替换 Vision 或只改 metadata。
最近一次 schema 7→8 仅重导了 Vision 和 base/ref 文本 refiner 三个共享组件，
其余 37 个图文件与该轮 schema 7 包一致；此结论不代表与 11 月 2 日 schema 5 整包一致。
原始图像／视频 VAE 的像素输入契约保留，提示词调整本身不需要重导模型。
15 秒包中 H3 音频 decoder、LTX 音频／图像 encoder 及三种视频 decoder 的文件 SHA 虽有变化，
逐图核对仅为 `pkg.torch.onnx.stack_trace` 导出注释差异；计算图及 external data 的 SHA 一致。
本次打包、ZIP CRC、SHA-256 和上传记录位于本机 `work_dirs/sol_h3_release_cleanup_20261103/`。

## 咖啡馆提示词的 token 预算

按当前包自带 tokenizer、`add_special_tokens=False`，对示例命令去掉末尾换行后的文本实测：

| 模式 | 5s prompt tokens | 15s prompt tokens |
|---|---:|---:|
| T2VA | 513 | 552 |
| I2VA | 230 | 257 |
| L2VA | 247 | 275 |
| FL2VA | 280 | 308 |
| Ref2VA | 450 | 477 |

当前最长为 552，距用户提示词上限 1024 还有 472，因此保留现有上限。
1024 只限制用户文本，不包含随后加入的媒体特征和 presentation 标签。
图像／视频的 Qwen token 另计入容量规划；五模式包的 base 文本容量为 1792、ref 为 3328，
公共 Qwen cache 容量为 3328。不要把它们与 1024-token 用户文本预算混为一项。
增强后的随库图像、视频和原音频已通过两档五模式的真实 Host 预处理，视觉时间组均能放入 1280 档。
当时的横屏 YAML 分别为 `sol_h3_all_384p_768p_w4.yaml`（5s）和
`sol_h3_all_384p_768p_15s_w4.yaml`（15s）；单模式、W8 和 2s 的历史 Sol-H3 YAML 已移除，
Native H3 的独立配置归入 `configs_merak/workflows/xh2a/other_models/h3/`。
本轮配置与示例整理的选中 CI 共 1034 项通过，包含 Native H3 full／LoRA 和 Sol-H3 full
的真实小模型构建、搬迁及推理；不代表增强参考素材已完成全权重质量对照。
此次目录和示例整理没有重导 ONNX，上传的是上述已验证 schema 8 包。

## 当前构建选项与新增组件

当前默认构建使用固定五角色 H3 布局，以及一张最大长度 LTX DiT；两者均使用 SDK/编译器的
广播 key mask 接口。需要本变更配套 SDK 和 hmcc 前端补丁；CUDA simulator 验证不等于芯片编译验收。
新图 ABI 需要重新导出，不能用下述历史包的验证记录替代当前构建验收。

Sol-H3 导出不再运行完整 H3/LTX 去噪采样，使用固定种子、有界且尺寸正确的 latent
作为下游组件的导图示例；GPTQ/AutoRound 的真实量化校准保持不变。完整音视频生成
需在发布后独立验收，导出成功不代替生成验证。Native H3/LoRA 保留原采样构建流程，
共享的视觉和文本 refiner 也采用下述外部 mask 接口，Native 包升级为 schema 3。
此优化不追溯到已启动导出的进程，历史大包须按各自源码快照和验收记录判断。

新构建使用 schema 9 和上述运行时可选的 LTX 音频策略。先前 schema 8
的 `ltx_audio_frozen: true` 表示固定冻结音频；它的历史验收结果仅适用于当时的图和策略。
新图不需要下载新 checkpoint，也不改变 VAE encoder 输入；质量应使用相同 H3 latent 独立对照。

schema 8 和 9 使用 `attention_contract: {mask_input_index: 4, min_flash_kv_length: 512}`。
带二值 key mask 的 FlashAttention 使用 `Q, K, V, LUT, FP16 mask` 五输入 ABI；
旧的下标 11 mask 产物不再适配当前 SDK，需要重新导出。
静态 KV 容量小于 512 时，使用 SDK MatMul → MaskedAdd → Softmax → MatMul，
保留原 scale、causal/GQA 和有效长度语义。Sol-H3 参考音频 VAE 的 KV 为 200；
LTX 音频 self-attention、视频查询音频的 cross-attention 在 5 秒、15 秒档分别使用 126、376 的 KV，
这些路径使用显式 MatMul，短 KV 分支不额外清零 K/V。
导出与加载均检查实际 ONNX，拒绝 KV 小于 512 的 FlashAttention 及旧 mask 槽位。

Qwen Vision 和所有 H3 文本 refiner 直接接收 Host 生成的 FP16 `key_mask`，
形状为 `[1,1,1,静态容量]`，有效 key 为 `0`，padding 为 `-65504`。
默认五模式包仅导出一张 1280-patch 视觉图，mask 为 `[1,1,1,1280]`；
图像和视频时间组共用它，Host 补齐输入并裁回有效的 merge 后特征。
视觉图没有 `patch_mask`、`input_length`，不清零 Q/K/V、残差或输出的 padding 行；
mask 直接作为各层 FlashAttention 的第 5 个操作数。
`text_encode_h3_refine_*` 仅输入 `text_features, key_mask`，默认 base/ref 容量分别为 1792/3328，
mask 分别为 `[1,1,1,1792]`、`[1,1,1,3328]`；图内不再从 `valid_text_length` 生成 mask。
Qwen 语言模型的因果 prefill 仍保留其 cache 与整数长度接口。
这次图接口变更需要重导视觉和文本 refiner，旧 schema 6/7 整包不能直接使用新 runtime；
不能只改 metadata 版本。以下 schema 7 的 45 图视频验收记录是本次接口简化前的结果。

2026-11-03 的外部 mask 更新包分别包含 20 张（5 秒）和 23 张（15 秒）图；
共享 Vision、base/ref 文本 refiner 从原 checkpoint 重导，其余组件与前一版逐文件 SHA-256 一致。
实际权重回放中，1008-patch 图片、1176-patch 视频时间组，以及从咖啡馆图片构造的
1260-patch 上界样本，其有效 image features 和三路 DeepStack 均与旧图逐位一致。
base 文本 refiner 的有效长度 7/1024/1792、ref 分支的 7/1024/3328 也与旧图逐位一致。
新包的全部 43 张实际图及五模式视图均通过加载审计。
5 秒、15 秒的 I2VA／Ref2VA 共四条完整 CUDA HMONNX 流程与前一版使用相同输入和 seed；
H3 音视频 latent、编码前 RGB 和 PCM 均逐位一致，输出分别为 121／361 帧。
本轮选中的 CI 为 466 项通过；后续保留 Native／自定义大视觉档位的路由收窄后，
相关容量、单／多档视觉和 Native 回归另有 44 项通过，两次运行的用例有重叠。
本机新包位于 `work_dirs/sol_h3_external_mask_20261103/bundle_5s` 和 `bundle_15s`，
同目录保存图审计、组件数值对照和四条完整流程的比较记录。
以下 schema 7 的十条视频记录仍仅对应旧图；本轮新图没有重跑全部五模式十条视频。

```yaml
export:
  norm_fp16: true
  upscaler_conv3d_mode: slice
  upscaler_conv3d_fastpath: false
  ltx_reference_conditioning:
    enabled: true
    require_native_resolution: true
    encoder_norm_eps: 5.960464477539063e-8
```

`norm_fp16` 只改变 H3/LTX DiT 的 RMSNorm 累加路径；Qwen、文本 refiner 和 VAE 保持宽归约。
`upscaler_conv3d_mode` 仅控制 `h3_upscale_t20`、`h3_upscale_t42`：默认 `slice`，
普通时间卷积由 SDK 转为 Slice→Conv2d→Add；显式设为 `gather` 才打包时间窗口。
`upscaler_conv3d_fastpath` 默认关闭，两个档位的 `1×1×1` 点卷积和纯时间 depthwise
也保留原 Slice→Conv2d→Add；显式设为 `true` 才折叠为 Reshape→Conv2d→Reshape。
它与 `slice/gather` 独立。DW 折叠会改变数值计算过程，浮点数学等价不代表量化后逐位一致。
SDK 必须支持 `conv3d_to_conv2d_fastpath`；否则导出在加载组件前拒绝，防止旧 SDK 静默开启优化。
这是导出配置；已导出的 ONNX 需要重新导出才能改变实现，运行 demo 时切换 YAML 不会改写旧图。
首尾参考一致性默认开启，新增一张共享 W16A16 LTX 图像 encoder：
`[1,3,1,768,1344] → [1,128,1,24,42]`，残差流缩放32。
I2VA/L2VA/FL2VA 分别编码首图、末图、两图；T2VA/Ref2VA 不把参考素材解释为目标首尾帧。
5 秒／15 秒 LTX DiT 分别使用 17136／47376 个视频 token，包含一帧 guide，使用 `first_role:int32[1]` 与
`video_valid_length:int32[1]`。首图固定首1008个目标token；末图追加1008个guide，参与attention，
三步结束后丢弃。没有末图且未启用 Ref2VA 中段 guide 时，有效长度分别为 16128／46368，
dummy guide 保持零且不能影响目标或音频。
目标最后latent仍参与更新，不能用末图覆盖它所对应的八帧。

`enabled:false` 构建不带图像 encoder 的 16128／46368-token 目标档，保留原生首 target marker；
它只关闭LTX约束，H3原有参考条件继续生效。两种构建的ABI由metadata明确区分。
`encoder_norm_eps` 默认是 FP16 最小正值 `2^-24`，对缩放后的 PixelNorm epsilon 取此下限；
该完整 encoder 的37处均使用普通宽累加 RMSNorm，residual scale仍为32。
这是抬高 epsilon 的有损方案，与 `finfo.tiny=2^-14` 不同；原本更大的 epsilon 不会被降低。
显式 `null` 仅用于保留尺度补偿语义的有界 RMS 诊断。修改此值必须重导 encoder 并同步产物清单，不能只改字段。

encoder、DiT、decoder按阶段顺序加载与卸载；运行时不导入作者LTX源码或读取源checkpoint。

## 首尾帧清晰度与无字幕提示词

当前 5 秒、15 秒 W4 五模式 workflow 开启 `require_native_resolution: true`。
I2VA、L2VA、FL2VA 的首尾图片在按原生 EXIF 方向旋转后，宽、高分别至少为
1344、768；提供两张图时分别检查。在加载 Qwen/H3 前拒绝需要放大的图片。
应提供人物五官清晰、压缩损伤少、与输出宽高比接近的图片；把模糊小图 resize 成大图
只能满足像素尺寸检查，不能补回真实细节。Ref2VA 的素材仍使用其独立参考预算。
旧产物未声明该字段时保留原输入规则，避免改变历史包的运行语义。

当前设计保留首帧 latent，末图作为结尾时刻的 guide 参与 attention。
高清原图会重新经过 LTX 图像编码，帮助保留比 H3 384p 草稿更丰富的细节。
`frozen` 策略下，LTX 使用干净且固定的 H3 音频 latent，经音视频交叉注意力精修画面和口型；
也可根据场景选择前述 `joint` 策略。
最终音轨仍是 H3 PCM，未额外串接独立的口型校正网络。

提示词采用英文画面描述，中文台词仅写在 `<d>[Chinese]…</d>` 中；
环境声放进 `overall_soundscape`，不重复台词。下例用于 5 秒 I2VA：

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] A single continuous five-second photographic shot begins with the adult woman and cafe counter shown in <Picture 1>. Retain her facial features, hairstyle, accessories, clothing, hand position, framing and warm lighting. The camera stays still. She looks toward the viewer, smiles gently and makes a small natural movement. Between 0.5 and 4.4 seconds, the woman (S1), with a relaxed, clear Mandarin voice, says once: <d>[Chinese]帅哥，要喝一杯吗，微醺的感觉真好！</d> Her lips follow the spoken words; she then closes her mouth and holds a soft smile. All dialogue is audible speech only. The picture remains free of subtitles, captions, letters, logos, watermarks and graphic overlays throughout the shot.

overall_soundscape: Low cafe room tone with faint cup and glass sounds. No other people speak.

non_diegetic_music: N/A
```

字段、首尾图时间指令和语音标签依据 [H3 官方提示词规范](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)。
Sol-H3 官方[人物说话样例](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/super_acceleration/assets/talking8_fbcache49_5s.json)
也使用“不要字幕、文字”或 `No subtitles, captions`。
本地咖啡馆女孩高清首帧的 5 秒样例使用上述提示词，121 帧 OCR 未检出字幕候选，
独立 ASR 的台词与目标一致（忽略标点）。这是单样本验证；提示词不能硬性保证无字幕，
交付前仍应检查输出，OCR 也可能漏检不可辨认的生成字形。

2026-11-03 的 schema 7 重导验收中，5 秒 T2VA 在同一结构化提示词、seed 43 和 44 下均出现字幕，
两条音轨的台词仍与目标一致。seed 43 的 H3 latent 经原生 FP32 VAE 解码后，
小图预览中已经有字幕，后续 LTX 成片保留了它；这两条样例不能按“无字幕”验收通过。
同轮 15 秒 T2VA、seed 43 的成片也出现字幕；361 帧 OCR 扫描中有 260 帧下部文字候选，
七帧预览中的第 60、120、240 帧可直接看到字幕行。
本次 I2VA、L2VA、FL2VA、Ref2VA 的 5 秒样例抽样未见字幕行，
逐帧 OCR 的背景和饰物候选仍需人工复核。

### T2VA 的正向提示词

T2VA 使用 [5 秒模板](prompts/cafe_t2va_5s.txt) 或 [15 秒模板](prompts/cafe_t2va_15s.txt)。
这两个模板不传参考图，直接从三个核心字段开始；不要附加 I2VA 的首图对齐指令。
当前模板在已测正向场景结构上调整了人物外观和手的位置，以贴合随库咖啡馆女孩；
这次修改后尚未重新生成或验收。T2VA 只使用文本，不保证人物身份与参考图一致。
台词只在 `<d>[Chinese]…</d>` 中出现一次，说话人 `(S1)`、语气、时间和动作放在标签外。
这些格式来自上述 MiniMax 官方规范。
NVIDIA Spark 的 [T2VA 样例](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-Spark/examples/prompts.jsonl)
仍有简短的禁用文字描述，但没有这段中文对白的无字幕验收；其人物说话清单是首图条件样例，
不能直接作为 T2VA 无字幕的证据。

[ComfyUI 的 H3 集成指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide)
建议正面描述希望出现的内容，避免堆叠禁用概念；其 BasicGuider 在 CFG=1 时没有负向分支。
本工程的四步蒸馏 H3 也只计算正向条件，并未开放独立 `negative_prompt`；
不能把禁用句当作独立的负向约束。

模板明确人物、构图、杯子和桌面，描述画面从上到下持续呈现咖啡馆实体场景，
尤其是下方连续的木纹、杯子、手和袖口。声音绑定到画面中的人物，
给定说话区间，并描述结束后闭口微笑到片尾。
正文不再重复字幕、文字、水印等概念，也不用双引号包裹台词。
这里的模板长短是本地实验选择，官方基础规范没有要求固定词数。

上一轮在 schema 8 的同一 5 秒包上，固定四步采样、每个 seed 的初始音视频噪声和原台词，
仅替换提示词。当时完整正向模板描述黑色兔耳发箍和深色服装，与当前外观文字不同。
以下 H3 原生 FP32 VAE 小图及成片结果均对应旧外观文字，不是当前精确模板的验收：

| 提示词 | seed 43 | seed 44 | H3 音频独立 ASR |
|---|---|---|---|
| 旧模板，包含禁用字幕句 | 可见字幕 | 可见字幕 | 两条台词匹配 |
| 仅删除禁用字幕句 | 可见字幕 | 未见字幕 | 两条台词匹配 |
| 当时的完整正向场景模板 | 未见字幕 | 未见字幕 | 两条台词匹配 |

六条 H3 小图共扫描 744 帧。仅删禁用句的 seed 44 有下部 OCR 候选，
复核为咖啡馆标牌或边缘场景文字，不能按候选数直接判为字幕。
当时的正向模板同时改变了构图描述、细节和长度；此对照支持这种正向结构，
但不能证明当前外观修改的效果，也不能单独证明某一句话的效果。
上述旧外观正向模板的完整流程结果如下，两种时长都使用 seed 43、44：

| 时长 | H3 小图 | LTX 成片 | H3／最终音频独立 ASR |
|---|---|---|---|
| 5 秒 | 两条均未见字幕 | 两条均未见字幕，121 帧／条 | 4/4 台词匹配 |
| 15 秒 | 两条均未见字幕 | 两条均未见字幕，361 帧／条 | 4/4 长台词匹配 |

5 秒复用上述对照中的 H3 latent，实际运行 bridge、干净音频条件、三步 LTX 和解码；
15 秒通过公开 `inference_model` 从头运行。四条成片均为 1344×768、24 FPS，
逐帧 OCR 共扫描 964 帧，并复核七帧预览和代表候选。
5 秒 seed 44 的 101 帧候选在左上咖啡馆装饰，15 秒 seed 43 的 361 帧候选在右上墙面画框，
均未见对白字幕叠加；另外两条成片没有 OCR 候选。
八份 ASR 记录中，5 秒的两份 H3 记录来自完全相同的已测音轨，未重复识别。
最终编码前 PCM 与 H3 的保留部分逐位相同，15 秒仅裁去每声道末尾 1067 个采样以对齐画面。
提示词仍属于软约束；生成后应检查 H3 小图与成片，并复核 OCR 候选。
本轮证据位于 `work_dirs/sol_h3_t2va_prompt_20261103/`，其中 `review.html` 汇总对照小图和四条成片。

## 15 秒台词与时间线

MiniMax 原版的 [H3 Prompt Writing](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md)
覆盖 4～15 秒，要求描述覆盖完整目标时长。官方还提供
[H3-Context-IR 接口](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)，
支持 `duration: 15`，返回可交给 H3 的 `content.prompt`；该改写服务没有随权重开源。
本工程的 Qwen 只提取输入特征，不会自动把简短中文要求改写成 Context-IR。

使用 15 秒产物时，将 [咖啡馆 I2VA 示例](prompts/cafe_i2va_15s.txt) 的内容作为 `prompt`。
该示例将原句扩写为 46 个汉字，在 1～12.5 秒自然说一遍，随后闭口微笑，
期间保留咖啡馆环境声。它沿用已验证 5 秒示例的高清参考图和无字幕描述。
时长由所加载产物决定，提示词中的时长描述应与其对应。改写时保留以下规则：

- 画面、动作和声音说明用英文；中文台词只放在 `<d>[Chinese]…</d>` 内，不在其他字段重复。
- 明确说话人 `(S1)`、说话时段，以及说完后直到片尾的动作和环境声。
- 基础四模式使用上述三个字段；Ref2VA 使用官方
  [六字段格式](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)，
  明确 `<Audio N>` 用于音色参考还是直接复用原音轨。
- 改写结果仍须满足当前产物的 1024-token 用户 prompt 上限；不要直接截断台词或参考标签。

另有 [15 秒 Ref2VA 示例](prompts/cafe_ref2va_15s.txt)，对应三张图片、一个带音轨的视频和
一份独立参考音频；`<Audio 1>` 对应视频音轨，`<Audio 2>` 对应独立音频。
二者只参考音色，目标台词只写一次。参考素材变化时应同步调整标签和引用关系。

2026-11-02 的 15 秒短台词 H3 阶段对照固定原参考图、seed 42、四次 DiT 计算和音频解码器，
只替换提示词。T2VA、I2VA、L2VA、FL2VA 的独立 ASR 均识别出完整目标台词且无重复；
I2VA 的 seed 43 也通过。Ref2VA 的全部原参考、仅视觉参考、单份音色参考三组也通过。
此结果对应咖啡馆单场景的八次生成，验证台词忠实度，
不代表所有提示词、人工听感或最终 LTX 画面质量均已通过。

46 字长台词另以高清图验证了 I2VA、Ref2VA 各两个 seed（42、43）的完整 15 秒流程，
输出均为 1344×768、361 帧、24 FPS。四条 H3 原音与四条最终成片的独立 ASR
均与文案一致；最终 PCM 与 H3 PCM 的对应前缀完全相同，仅裁掉约 33 ms 尾部以对齐画面。
推荐咖啡馆样例使用 I2VA、seed 43：其 361 帧 OCR 未检出任何文字候选。
其他三条成片的 OCR 候选经原始帧或候选裁图复核，为背景、发饰或口手纹理，未见字幕叠加。
这些结果仍限于该素材与所测 seed；Ref2VA 会重绘部分饰品细节，不代表身份已被锁定。

## 2026-11-03 编译器 ABI 重导验证

5 秒、15 秒包的所有组件均重新导出，复用现有 AutoRound W4 checkpoint，未复用旧 ONNX。
两包均使用 schema 7；45 个实际图通过 mask 下标 4、FlashAttention 静态 KV ≥512 和组件清单审计。
以下为 CUDA HMONNX simulator、咖啡馆女孩素材、seed 43 的十条主样例结果：

| 时长 | ONNX 图数 | 五模式生成及媒体校验 | 输出帧数 | H3／最终音频 ASR |
|---|---:|---:|---:|---:|
| 5 秒 | 21 | 5/5 | 121 | 10/10 台词匹配 |
| 15 秒 | 24 | 5/5 | 361 | 10/10 台词匹配 |

成片均为 1344×768、24 FPS、32 kHz 双声道。封装前的音轨与本次 H3 PCM 的保留部分逐位一致：
5 秒保留每声道 165600 个采样；15 秒从 482400 裁到 481333 个采样，裁尾 1067 个采样以对齐画面。
独立 ASR 忽略标点，全部台词匹配。另完成 2410 帧 OCR、每条七帧抽样及代表性 OCR 候选复核、SyncNet 评分。
T2VA 两个时长均可见字幕；其他四模式的上述检查未见字幕叠加，15 秒 L2VA 的下部候选位于咖啡机标签。

本机模型目录为 `work_dirs/sol_h3_mask_fa_20261103/bundle_5s` 和
`work_dirs/sol_h3_mask_fa_20261103/bundle_15s`。
同目录的 `review.html` 汇总十条视频，`verification.json` 保存逐视频媒体、语音、PCM、OCR 和口型证据。
`baseline_comparison.json` 记录与旧包同 prompt、seed、素材路径的 15 秒 I2VA／Ref2VA 对照；
跨版本的 H3 latent、PCM 和最终视频均有数值差异。

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

此节记录的旧包不能代替当前版本验收；当前版本的可执行命令见下文“五模式推理示例”。
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

新 Sol-H3 统一包及其模式视图使用 schema 9；schema 5/6/7/8 包属于历史格式。
默认固定五角色的 H3 Dense 图使用 9 个输入：`video_tokens, audio_tokens, text_features, step, cos, sin, role_pack, head_unpack, key_mask`。
每个角色槽位向上对齐 256；`role_pack` 将各角色有效行放到对应前缀，`head_unpack` 在输出投影前还原视频/音频行序，
`key_mask` 屏蔽无效 key，输出裁回原生长度。
LTX 的四组 RoPE（共 8 个 cos/sin 张量）作为共享输入，随包保存在 `hf_config/ltx_rope.npz`。
加上 `video, audio, step, first_role, video_valid_length, audio_mode`，默认首尾参考去噪图共 14 个输入，
避免展开后的 RoPE 常量在每层重复占用显存。
全部部署图使用 `.onnx` 后缀，算子仍为 HMONNX；图数随模式、文本分组及 VAE 档位变化。
加载时校验包内精度策略与组件契约；缺少当前 RMS 策略字段的旧包需要重新导出，不能混用不同文本尺度的组件。

统一包共用 Qwen 语言特征图；1280 以内的视觉容量合并为一个共享档。
五模式默认参考配置使用 1280 档，图片实际的 1008 patches 和视频时间组的 1260 patches
均补齐到 1280，传入对应 key mask；每个视频时间组独立编码后按原顺序拼接。
原生 H3 或自定义配置允许更大的参考图片时，较大的档位继续独立路由，普通图不补齐到这些大档。
视觉图使用非因果注意力，并输出 image features 和三路 DeepStack 特征。
容量向上对齐 256，不改变 processor 的图片尺寸和送入 LLM 的有效 token 数；
静态 KV 小于 512 的小型验证图使用显式 MatMul，其余使用 FlashAttention。
当前五模式配置的 Ref 图片使用四个固定空间档，视频使用一张固定 T17 图；其中
`1×384×672` 图片 encoder 与普通首尾帧共用。
LTX 视频 decoder 按实际时间 tile 长度导出：5 秒为一个档，15 秒为多个档。
其余音频、桥接和 LTX 组件也共用。H3 保留 `base`、`ref` 两条权重分支，各自包含文本 refiner
和 DiT；前四种模式使用 base，Ref2VA 使用 ref。包内 `mode_profiles` 记录每种模式的输入约束、
容量和分支绑定。运行时根据 `image`、`last_image`、`references` 选择模式，每次请求新建 KV cache。

固定尺寸配置导出的参考 VAE encoder 使用原生 GroupNorm 和空间反射边界，每张图仅接收
`pixels`，没有空间 mask。`reference_vae_layout="fixed_v1"` 标识此布局；
`reference_vae_tiles` 按准确的时间和空间尺寸路由，`reference_vae_base` 描述普通图。
原有 `split_v1` 包仍按其记录的 mask 加载，独立 Ref2VA 单模式的原有可变尺寸策略也保留。
LTX decoder 仍按其原有空间 mask 和 tile 拼接策略工作。
视频 T=17 是原生固定 clip 长度，不是图内动态上限。Host 将长视频切成17帧片段；
不足17帧的末段复制末帧补齐，再按原生 `token_drop` 裁后验输出。不会改成任意T图，
也不会把视频尾片当作图片档位执行。

## 本地构建输入

`--model-dir` 使用轻量 source-index 目录，例如 `/models/sol-h3-inventory`，
其中包含清单、Qwen 原生配置和校准输入索引。
避免将实验输出或虚拟环境放入该目录，公开 build 会对整个输入目录做内容指纹检查。
权重和作者源码可以留在各自原位置。下面的路径是示例，须替换为本机位置；
清单文件为 `checkpoint-paths.json`：

```json
{
  "h3_model": "/models/MiniMax-H3",
  "dense_lora": "/models/FastH3/dense-datafree/adapter_model.safetensors",
  "ref2va_lora": "/models/Minimax-h3-Turbo/ref2va.safetensors",
  "transformer": "/absolute/path/ltx-2.5-transformer.safetensors",
  "refiner_lora": "/absolute/path/ltx-2.5-distilled-lora.safetensors",
  "output_video_vae": "/absolute/path/ltx-2.5-conv-video-vae.safetensors",
  "audio_vae": "/absolute/path/ltx-2.5-audio-vae.safetensors",
  "h3_upscaler_checkpoint": "/absolute/path/h3-upscaler.safetensors",
  "adapter_dir": "/absolute/path/h3-ltx-adapter",
  "sana_source": "/sources/Sana",
  "ltx_source": "/sources/LTX-2",
  "upscaler_source": "/sources/upscaler-source.py",
  "refiner_prompt_cache": "/models/fixed-prompt-int8-a100.pt"
}
```

W4 构建和 Qwen 量化 checkpoint 复用都要求 source-index 根目录同时放置 Qwen3-VL
的原生 `config.json`，内容必须与清单 `h3_model` 下的 `text_encoder/config.json` 一致：

```bash
cp /models/MiniMax-H3/text_encoder/config.json /models/sol-h3-inventory/config.json
```

这只是复制配置到构建索引，不移动或修改原始权重。构建只读并校验两份 JSON，不会自动生成
或修正它们；显式 W16 构建不要求此配置镜像。

构建只读本地文件，不下载 Hugging Face 权重。外部源码只在构建时加载，导出的 HMONNX 不依赖这些源码。
默认统一包还需要下文的 `calibration_inputs.json`，以及 H3 目录中的 `transformer_ref`。

构建源码版本：Sana `62da34b14a53d7289e82fa56692f34e96e089bee`，LTX-2
`d151147788a9284cca791edc6ce898007e727fe6`，upscaler companion
`d7c01b9011f2e8439493f6c02c29995a27df276f`。不要将上游 Gemma 的 Transformers 5.14.1 环境覆盖到 Merak
的 Transformers 5.16.1 环境；本构建使用预先生成的固定提示词 cache。上游源码与模型权重的许可证随原下载保留。

## 构建和推理

默认 workflow 将 Qwen3-VL 实际使用的前 50 层导出为一张 `h3_feature_00_50.onnx`，
`text_layers_per_graph=50`。两条 H3 分支各自保留独立的文本 refiner。
两个五模式 profile 均使用 AutoRound W4/group64 checkpoint，Qwen 语言图为 W4A8，视觉图为 W8A8。

从仓库根运行，`CUDA_VISIBLE_DEVICES` 限定单卡，进程内统一使用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.sol_h3.build \
  --model-dir /models/sol-h3-inventory \
  --quant-checkpoint /models/qwen3-vl-autoround-w4g64 \
  --output-dir /outputs/sol-h3-all
```

默认构建 5 秒包；增加 `--profile all-dense-384p-768p-15s` 可构建 15 秒包。
构建前需准备上述权重、checkpoint 和下文的校准媒体索引。

构建竖屏包时分别指定 profile 和输出目录，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.sol_h3.build \
  --model-dir /models/sol-h3-inventory --quant-checkpoint /models/qwen3-vl-autoround-w4g64 \
  --profile all-dense-384p-768p-portrait-5s --output-dir /outputs/sol-h3-portrait-5s
CUDA_VISIBLE_DEVICES=0 python -m examples_merak.aigc.sol_h3.build \
  --model-dir /models/sol-h3-inventory --quant-checkpoint /models/qwen3-vl-autoround-w4g64 \
  --profile all-dense-384p-768p-portrait-15s --output-dir /outputs/sol-h3-portrait-15s
```

竖屏首尾条件图片经 EXIF 方向修正后至少为 768×1344（宽×高）；下文横屏示例的图片不满足此要求。
横屏包不能通过交换输出宽高当作竖屏包使用，应选择对应竖屏 profile 重新导出。
自动产物名不区分时长或方向，构建多个档位时应分别指定 `output_dir`；已有目录不会被覆盖。

### 五模式推理示例

从仓库根目录运行。先将 `ARTIFACT_DIR` 改为当前代码导出的五模式包；
`DURATION=5` 使用 5 秒包，改为 `15` 时需同时改成 15 秒包。示例输入随仓库保存在
[`data/sol_h3`](../../../data/sol_h3/README.md)，首尾图片均满足 1344×768 最小分辨率。
图像条件提示词及有序媒体来自咖啡馆验证样例；T2VA 只使用文本描述，当前外观改动尚未重新验收。

```bash
ARTIFACT_DIR=/outputs/sol-h3-all
DURATION=5
PROMPT_DIR=examples_merak/aigc/sol_h3/prompts
OUT_DIR=/tmp/sol-h3-cafe
mkdir -p "$OUT_DIR"

# T2VA
python -m examples_merak.aigc.sol_h3.generate --artifact-dir "$ARTIFACT_DIR" \
  --prompt "$(cat "$PROMPT_DIR/cafe_t2va_${DURATION}s.txt")" \
  --seed 43 --output "$OUT_DIR/t2va.mp4"

# I2VA
python -m examples_merak.aigc.sol_h3.generate --artifact-dir "$ARTIFACT_DIR" \
  --prompt "$(cat "$PROMPT_DIR/cafe_i2va_${DURATION}s.txt")" \
  --image data/sol_h3/cafe_first_hd.png --seed 43 --output "$OUT_DIR/i2va.mp4"

# L2VA
python -m examples_merak.aigc.sol_h3.generate --artifact-dir "$ARTIFACT_DIR" \
  --prompt "$(cat "$PROMPT_DIR/cafe_l2va_${DURATION}s.txt")" \
  --last-image data/sol_h3/cafe_last_15s.png --seed 43 --output "$OUT_DIR/l2va.mp4"

# FL2VA
python -m examples_merak.aigc.sol_h3.generate --artifact-dir "$ARTIFACT_DIR" \
  --prompt "$(cat "$PROMPT_DIR/cafe_fl2va_${DURATION}s.txt")" \
  --image data/sol_h3/cafe_first_hd.png --last-image data/sol_h3/cafe_last_15s.png \
  --ltx-audio-mode frozen --seed 43 --output "$OUT_DIR/fl2va.mp4"

# Ref2VA: 依次为三张图、一段带音轨的视频和一段独立音频
REFS='[{"kind":"image","path":"data/sol_h3/cafe_first_hd.png"},
       {"kind":"image","path":"data/sol_h3/cafe_ref_middle.png"},
       {"kind":"image","path":"data/sol_h3/cafe_ref_last.png"},
       {"kind":"video","path":"data/sol_h3/cafe_reference.mp4"},
       {"kind":"audio","path":"data/sol_h3/cafe_reference.wav"}]'
python -m examples_merak.aigc.sol_h3.generate --artifact-dir "$ARTIFACT_DIR" \
  --prompt "$(cat "$PROMPT_DIR/cafe_ref2va_${DURATION}s.txt")" \
  --references "$REFS" --seed 43 --output "$OUT_DIR/ref2va.mp4"
```

视频音轨是 `<Audio 1>`，独立音频是 `<Audio 2>`；两者各 5 秒，模板仅要求参考音色。
这些是请求时输入，运行时不会读取构建校准媒体。2026-11-03 的 schema 8 外部 mask 包
重跑了 5 秒和 15 秒的 I2VA／Ref2VA；其余模式和 T2VA 旧外观正向模板的验证范围分别见上文。

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
两份五模式 `*_w4.yaml` 均显式要求 W4，直接 workflow 缺少相应 checkpoint 时会报错，不静默回退 W8。

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

unified API 提供横竖屏、两种时长的五模式共用包：

| 模式 | `build_model(profile=...)` | 推理媒体输入 |
| --- | --- | --- |
| 五模式共包 | `all-dense-384p-768p-5s`（默认） | 按下述媒体输入自动选择 |
| 五模式共包 | `all-dense-384p-768p-15s` | 15 秒输出，按下述媒体输入自动选择 |
| 竖屏五模式共包 | `all-dense-384p-768p-portrait-5s` | 5 秒输出，按下述媒体输入自动选择 |
| 竖屏五模式共包 | `all-dense-384p-768p-portrait-15s` | 15 秒输出，按下述媒体输入自动选择 |

| 推理模式 | 媒体输入 |
| --- | --- |
| T2VA | 无 |
| I2VA | `image` |
| L2VA | `last_image` |
| FL2VA | `image`、`last_image` |
| Ref2VA | 有序 `references: [{kind, path}, ...]` |

默认 YAML 在 `configs_merak/workflows/xh2a/other_models/sol_h3/`。
首尾帧数量由模式固定为 1/1/2，按目标画布预处理，不使用 `references` 容量配置。
五模式共包的 Ref2VA 默认允许三张图、一段最多 5 秒视频和一段最多 5 秒独立音频；
视频自带音轨不占独立音频数量。图片按原始宽高比选择最接近的尺寸，直接缩放到
`384×672`、`448×672`、`672×448`、`512×512` 四档之一；视频直接缩放到 `384×672`。
输入与选中档位的宽高比不一致时会有拉伸，宽高比超出 1:4 至 4:1 的输入会报错。
竖屏包额外提供 `672×384`（高×宽）参考图档，参考视频固定缩放到该竖屏尺寸。
参考媒体最大时长独立于输出时长，各档位均可输入完整 5 秒参考视频。
在当前本地 checkpoint/tokenizer 下，横竖屏包的公共 Qwen KV 均为 3328；5 秒 base/ref 的
五角色 H3 静态长度分别为 12288/23040，15 秒为 30720/41472。替换 processor 后应重新规划。

默认不覆盖 Qwen processor 的像素预算。本地官方 checkpoint 的 image `longest_edge` 为
16777216，video 为 25165824；修改模型来源后以其实际 processor 配置为准。
`image_max_pixels` / `video_max_pixels` 仍可在自定义 YAML 显式覆盖。
当前固定版本的 video processor 把视频预算用于整段采样视频的 `T×H×W`，不是每帧面积。
实际 token 数使用原生 patch、merge、时间采样和取整规则计算。
例如 120 帧、384×672 的参考视频按 2 fps 采样为 10 帧，官方预算下保持画面尺寸，
时间 patch 2、空间 patch 16、merge 2，产生 `5×12×21=1260` 个 Qwen token。

历史独立 Ref2VA 单模式的图像 VAE 曾沿用官方短边 2048，无额外面积压缩，宽高比在 1:4 到 4:1 内，
因此最大画布是 2048×8192（或转置），按 32 对齐。Qwen 接收同一张原生预处理图像，
再按其 processor 规则处理，单图最多 16384 个 Qwen token。
参考视频 VAE 原生策略沿用生成画布的短边和面积约束；本 profile 生成画布仍为 384×672，
没有因恢复参考图片的官方短边而改变生成尺寸。这是旧单模式配置的记录，不是当前五模式默认值。
音频不进入 Qwen 视觉编码器，其 presentation 标签进入文本序列，音频 latent 进入 DiT。

导出自动记录 `runtime_meta_info.json.capacity_plan`：

- `text_tokens`：提示词 token 上限＋所有媒体 Qwen token 上限＋原生标签、时间戳开销。
- `text_capacity`：`text_tokens` 向上对齐到 256。
- `kv_capacity`：同时对齐到 256 和 `prefill_chunk_length` 的公倍数。
- `video_capacity` / `audio_capacity`：生成段 token＋参考媒体 VAE token，再分别向上对齐到 256。
- `dit_capacity`：上述 text、video、audio 三项之和；DiT 输入使用有效行压紧的布局。

历史单模式 prompt 为 1024、使用单模式官方参考预算并按 256 对齐后，使用本地官方 H3 tokenizer/processor 和当时五份单模式 YAML 推算的静态上限如下
（不是每个请求的实际 token 数）：

| 模式 | Qwen 总容量 | KV 容量 | DiT 总容量 |
| --- | ---: | ---: | ---: |
| T2VA | 1,024 | 1,024 | 11,008 |
| I2VA / L2VA | 1,536 | 1,536 | 11,776 |
| FL2VA | 1,792 | 1,792 | 12,288 |
| Ref2VA | 18,944 | 18,944 | 54,784 |

旧 `w4a8_five_modes_20261022` 包仍为 prompt 512；2026-10-26 的统一候选包采用旧的单图大参考预算。

Ref2VA 的未对齐文本上界为 `1024+16384+1350+64=18822`。
旧共包的公共 Qwen KV 因而取 18944。base 分支共用 text/video/audio 槽位
1792/9984/512，ref 分支使用 18944/34560/1280。该单模式表仅供历史对照。
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
但受音频时长限制，导出也会为每个视频预留音轨容量。固定尺寸五模式包直接使用相应档位编码；
原有单模式包按其配置使用原生 tile 拆分及对应的 encoder 图。

### 校准输入与构建

条件模式需要真实媒体，由原生 processor 准备尺寸、token 布局和导出样例。
字段仍名 `export.calibration_inputs`，不替代独立的 GPTQ/AutoRound 权重量化校准。
使用 unified API 的默认 YAML 时，在 model_dir 放置 `calibration_inputs.json`，按模式保存媒体。
例如先将随仓素材复制到 source-index，再使用下面的内容：

```bash
mkdir -p /models/sol-h3-inventory/media
cp data/sol_h3/cafe_* /models/sol-h3-inventory/media/
```

```json
{
  "i2va": {"image": "media/cafe_first_hd.png"},
  "l2va": {"last_image": "media/cafe_last_15s.png"},
  "fl2va": {"image": "media/cafe_first_hd.png", "last_image": "media/cafe_last_15s.png"},
  "ref2va": {"references": [
    {"kind": "image", "path": "media/cafe_first_hd.png"},
    {"kind": "image", "path": "media/cafe_ref_middle.png"},
    {"kind": "image", "path": "media/cafe_ref_last.png"},
    {"kind": "video", "path": "media/cafe_reference.mp4"},
    {"kind": "audio", "path": "media/cafe_reference.wav"}
  ]}
}
```

Ref2VA 开启视频容量时必须提供校准视频，开启音频容量（含视频音轨）时必须提供校准音频。
校准图片不固化为运行时参考图。Ref2VA 源清单还需要 `ref2va_lora`，H3 目录需要
`transformer_ref`；融合复用现有 PEFT 加载器与 LoRA alpha/rank 规则。
该 JSON 中相对路径以 model_dir 为基准；自定义 YAML 中的媒体建议使用绝对路径。
YAML 显式 calibration_inputs 优先。校准清单参与统一构建输入指纹；校准媒体不随产物加载。
当 YAML 的 `mode: all` 时，`export.calibration_inputs` 也使用上述按模式组织的字典；
构建从 FL2VA 和 Ref2VA 校准样本分别准备 base/ref 分支，共享编码图只导出一次。

```python
from pathlib import Path

from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

build_model(model_id="sol-h3", model_dir="/models/sol-h3-inventory",
            profile="all-dense-384p-768p-5s", output_dir="/outputs/sol-h3-all", device="cuda:0")
prompt = Path("examples_merak/aigc/sol_h3/prompts/cafe_fl2va_5s.txt").read_text(encoding="utf-8")
inference_model(artifact_dir="/outputs/sol-h3-all", device="cuda:0",
                inputs={"prompt": prompt, "image": "data/sol_h3/cafe_first_hd.png",
                        "last_image": "data/sol_h3/cafe_last_15s.png", "seed": 43,
                        "ltx_audio_mode": "frozen",
                        "output_path": "/tmp/sol-h3-fl2va.mp4"})
```

CLI 的五模式命令见上文；自定义配置可使用 `build --workflow ...`。
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
删除源权重后生成 121 帧带音频视频。条件模式按请求选择，构建读取校准媒体清单。
条件 DiT 使用十输入 ABI，固定参考前缀、仅更新生成行。完整权重五模式的独立验收见上节；
小模型和完整权重的 GPU simulator 验证均不代表 XH2a 真机性能。
