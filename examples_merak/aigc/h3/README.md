# 原生 H3 与可切换 LoRA

模型 ID 为 `minimax-h3`。实现复用 `sol_h3/` 的 Qwen3-VL、参考条件编码、
H3 图和 Host 调度，直接使用 H3 视频/音频 VAE 输出，不经过 LTX。
`sol-h3` 仍是原来的四步 H3 加三步 LTX；旧 `minimax-h3-fl2va` 接口不变。

统一 API 默认且仅提供 `all-native-768p-5s`：768×1344、124 帧、24 FPS，
一个包支持 T2VA、I2VA、L2VA、FL2VA、Ref2VA。标准 H3 与 LoRA 使用同一套构建配置，
运行时按请求选择模式，并可切换包内已导出的适配器。
其他尺寸和参考预算保留为 workflow YAML，使用下文的工作流入口导出。
标准 H3 默认实际执行 50 次 DiT，视频/音频 shift 为 12/3。
`native_sampling.steps` 表示实际 DiT 调用数；传给原生 scheduler 的 sigma 点数是 steps+1。
Qwen 语言部分默认 W4A8，视觉部分 W8A8，沿用官方 processor 预算及 256 对齐。

当前运行 metadata 使用 schema 3。共用的 Qwen Vision 仅接收一个外部 FP16
`key_mask`（`[1,1,1,patch_capacity]`），取代原来的 `patch_mask` 和 `input_length`。
标准与 LoRA 的全部文本 refiner 均输入 `text_features, key_mask`，mask 末维为对应分支
的文本容量；有效 key 为 `0`，padding 为 `-65504`，Host 负责生成 mask 和裁取有效视觉特征。
schema 2 旧包需重导这两个编码接口。1280 以内的视觉档位合并，较大原生参考预算保留独立档位。

## 验证状态（2026-10-25）

七个 384×672 静态档位已完成完整权重构建、包搬迁和标准/LoRA 推理验收。

- 小型真实模型验证包括统一 API 构建、包搬迁和删除源权重后的推理：full 包覆盖五模式，
  LoRA 包覆盖参考分支及 base→LoRA→base 切换；回切 base 输出一致。
- 旧 all 大预算的完整权重构建曾在参考分支 DiT FFN 遇到 SDK GEMM 索引溢出：
  `111360 × 28672 = 3192913920` 个输出元素超过有符号 INT32 寻址范围。
  这是旧短边 2048 加 5 秒参考视频预算的历史失败；新静态档位已降低容量，
  上述七个新静态档位的完整权重验证均未触发该问题。
- 七个 384×672 静态档位已完成完整权重构建和包搬迁。T2VA、I2VA、L2VA、FL2VA、
  Ref2VA image384、Ref2VA image768、Ref2VA video2s 均已通过
  标准 50 步→turbo8→标准 50 步验证。
- 完整权重验证比较编码前的视频像素与 PCM 的字节摘要、形状和 dtype，
  已通过档位回切标准模型后完全一致。MP4 编解码会引入变化，不作为逐字节回切判据。
  各轮输出为 124 帧、384×672、24 FPS，音频为有限值双声道。
- 完整权重验证使用 strace 检查搬迁后的推理未访问原始模型、Qwen 量化 checkpoint、
  LoRA 源目录或原包路径；共享源权重没有删除。实际删除源文件的验证仅针对小模型测试夹具。
  验收记录位于实验目录 `native_h3_static_modes_20261024/acceptance_matrix.json`，
  单档记录位于 `<profile>/validation_raw/acceptance.json`。
  768p 档位与 all 档的完整权重端到端验证、音频听感及口型同步评价不在上述通过范围内。

Sol-H3 完整包的验证结果不能替代这里的原生 H3/LoRA 验证。下述命令描述
已实现的接口。新档位按实际模式和媒体类型分配容量，避开旧 2048 短边加 5 秒参考视频的
统一大预算；完整权重验收范围为上述七个 384×672 档位。

## 构建

`model_dir` 可以是完整原生 H3 目录，包含 transformer、transformer_ref、text_encoder、
vae、audio_vae、tokenizer 和 processor；也可以是带 `checkpoint-paths.json` 的目录：

```json
{"h3_model": "/path/to/MiniMax-H3"}
```

W4 构建和 Qwen 量化 checkpoint 复用推荐使用上述轻量 source-index，例如 `/inputs/h3`。
该索引目录还必须包含 Qwen3-VL 原生配置，与 `h3_model/text_encoder/config.json` 的 JSON
内容一致：

```bash
cp /path/to/MiniMax-H3/text_encoder/config.json /inputs/h3/config.json
```

只复制配置到索引目录，保留原始模型和权重。构建只读并校验配置，不会自动写入或修正源目录；
显式 W16 构建不要求这个配置镜像。

校准输入由 `model_dir/calibration_inputs.json` 按模式登记。默认统一包需要四种条件模式的
完整校准媒体；Ref2VA 预算为一张短边 768 的参考图、一个 2 秒视频和一个 2 秒音频。
单模式 YAML 仅需要所选模式的媒体；文生 YAML 不需要参考媒体。视频档构建的校准视频
需要带真实音轨，以校准对应参考音频 encoder；发布包推理时可输入无音轨视频。
已有 Qwen W4 checkpoint 可以复用，不重新做权重量化：

```bash
python -m examples_merak.aigc.h3.build \
  --model-dir /inputs/h3 \
  --quant-checkpoint /inputs/qwen3vl_w4 \
  --output-dir /outputs/h3
```

### 按 workflow YAML 导出其他档位

以下是 YAML 的 `export.profile`，不是统一 API 的可选 profile。
文件位于 `configs_merak/workflows/xh2a/other_models/h3/h3_native*.yaml`；
`{384,768}` 分别对应输出 H×W 为 384×672、768×1344，均输出 124 帧。

| YAML 中的 profile | 支持的输入 |
|---|---|
| `t2va-native-{384,768}p-5s` | 文本，无参考媒体 |
| `i2va-native-{384,768}p-5s` | 一张首帧图 |
| `l2va-native-{384,768}p-5s` | 一张尾帧图 |
| `fl2va-native-{384,768}p-5s` | 首帧、尾帧各一张图 |
| `ref2va-native-{384,768}p-image384-5s` | 一张参考图，VAE 短边 384；无参考视频/独立音频 |
| `ref2va-native-{384,768}p-image768-5s` | 一张参考图，VAE 短边 768；无参考视频/独立音频 |
| `ref2va-native-{384,768}p-video2s-5s` | 一段最长 2 秒的参考视频，可带自身音轨；无独立图像/音频 |
| `all-native-768p-5s` | 统一 API 默认五模式包；Ref2VA 默认一张短边 768 图、一个 2 秒视频、一个 2 秒音频 |

参考图的短边与输出尺寸独立，例如 `ref2va-native-768p-image384-5s` 用短边 384 的
参考图生成 768×1344 视频。短边档位仍保留原生纵横比处理，并非强制正方形。
`image384` 缩小的是输入短边及对应 DiT/Qwen token 预算；masked VAE encoder 画布仍为
768×768，以保留原生 768 tile、横竖方向覆盖、GroupNorm 统计与拼接语义。
把 VAE tile 直接改成 384 会改变数值语义，本轮不做该变化。
视频档的独立音频 `max_count=0` 不禁用视频自身音轨，容量会为该音轨预留 2 秒。
提示词默认 1024 token；Qwen 视觉与 H3 VAE 的容量分别按原生 processor 和实际媒体预算推导，
静态容量取 256 的倍数，单模式不继承其他模式的最大预算。

例如导出 384p 的 I2VA，使用已有 `AutoWorkflow` 接口并复用匹配的 Qwen checkpoint：

```python
from pathlib import Path

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_other_model.workflows.result import QuantResult
from xhmodel_merak.xh_other_model.models.sol_h3.native_runtime import NativeH3Runtime

workflow = AutoWorkflow.from_config(
    model_dir="/inputs/h3",
    config_path="configs_merak/workflows/xh2a/other_models/h3/h3_native_i2va_384p_w4.yaml",
)
quant = QuantResult(raw_model_dir="/inputs/h3", quanted_model_dir="/inputs/qwen3vl_w4")
result = workflow.export(quant, output_dir="/outputs/h3_i2va", device="cuda:0")

runtime = NativeH3Runtime(result.work_dir, device="cuda:0")
try:
    prompt = Path("examples_merak/aigc/sol_h3/prompts/cafe_i2va_5s.txt").read_text(encoding="utf-8")
    runtime.generate(prompt, image="data/sol_h3/cafe_first_hd.png", seed=43, output_path="/outputs/i2va.mp4")
finally:
    runtime.close()
```

需要重新量化时，将 `quant` 改为
`workflow.quant(output_dir="/outputs/qwen_quant", device="cuda:0")`，校准配置来自所选 YAML。
需要 LoRA 时，在 `workflow.export` 增加
`config_overrides={"export.model.lora.path": ["/inputs/turbo8"]}`。
工作流图包位于 `result.work_dir`，即 `/outputs/h3_i2va/h3_export`，含
`runtime_meta_info.json`、`export_meta_info.json` 和所选模式的静态图；它不经过统一 API
的 Manifest 发布层，按上例用 `NativeH3Runtime` 加载。统一 API 发布的默认包仍使用下方推理命令。

LoRA 使用独立适配器目录，`h3_adapter.json` 显式声明名称、分支和采样设置：

```json
{
  "name": "turbo8",
  "branches": {
    "base": "fl2v_8step.safetensors",
    "ref": "ref2v_8step.safetensors"
  },
  "sampling": {"steps": 8, "video_shift": 6.0, "audio_shift": 3.0}
}
```

两份权重使用对应的官方 Diffusers 格式 768p 8step checkpoint；不要把 FL2V 权重用于 Ref2VA。
把权重复制或硬链接到目录内。构建时增加 `--lora /inputs/turbo8`，可重复传多个目录。
每个目录名称必须唯一；仅提供一个分支时，其他分支的模式不能选择这个适配器。

原生 8 步权重支持的目标为 DiT 和文本 refiner 的 attention/FFN Linear。
A/B 保持独立，基权重不融合；AdaLN、额外替换权重及未知目标会明确拒绝。
每个 adapter 都带自己的文本图、DiT 图、采样表和资源说明，共享 Qwen、参考编码器与 VAE。
`export_meta_info.json` 的 `lora` 字段按适配器名称分组。

## 推理与切换

```bash
python -m examples_merak.aigc.h3.generate \
  --artifact-dir /outputs/h3 --lora-adapter turbo8 \
  --prompt "$(cat examples_merak/aigc/sol_h3/prompts/cafe_i2va_5s.txt)" \
  --image data/sol_h3/cafe_first_hd.png --seed 43 \
  --output /outputs/turbo8.mp4
```

这两条命令从仓库根目录运行，复用 [Sol-H3 咖啡馆素材](../../../data/sol_h3/README.md)；
Sol-H3 的生成验收不等于原生 H3／LoRA 对该提示词的质量验收。
不传 `--lora-adapter` 即标准 H3。`--image`、`--last-image`、`--references` 分别选择
首帧、尾帧/首尾帧、Ref2VA，参数与 Sol-H3 示例一致。
统一 API 使用 `inference_model(..., inputs={..., "lora_adapter": "turbo8"})`。
运行时可在同一 `NativeH3Runtime` 上依次调用 `generate`，每次请求选择不同适配器，
省略名称回到 base。切换会清空前一视图的 session/KV 状态并加载选定静态图，
与 Qwen3.5 的包内适配器视图设计一致；不是同一 SDK session 内热更新权重。

推理只使用发布包与请求媒体，构建源权重和 LoRA 源目录可移走。
当前 CI 只构建 full、LoRA 两个公开小包，复用包验证五模式和适配器切换；
不为每个模式或另一个量化位宽重复导出。CPU 保留独立调度、LoRA B×A、视频分块和关键拒绝边界；
视频 decoder 另保留一张真实图的原生数值对照。完整权重质量须独立验收。
