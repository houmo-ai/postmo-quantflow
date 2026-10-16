# MiniCPM-o-4.5 Merak

本目录提供 MiniCPM-o-4.5 的 Merak workflow、Hugging Face demo、HMONNX demo 和流式 demo。
模型实现位于 `xhmodel_merak/xh_llm/models/minicpm_o_4_5/`，配置位于
`configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/`。

## 正式配置

仓库保留两个正式配置：

| 配置 | LLM | GPTQ | TTS |
| --- | --- | --- | --- |
| `minicpm_o_4_5_xh2a_w8a8_gptq.yaml`（默认） | `w8a8_sefp` | 8 bit，group size 64 | `w8a8_sefp` |
| `minicpm_o_4_5_xh2a_w4a8_gptq.yaml` | `w4a8h0_ssfp` | 4 bit，group size 64 | `w8a8_sefp` |

两者使用同一份 80 条混合校准集。GPTQ 只量化 LLM 主干；各组件的实际导出精度以各自的
`quant_type` 为唯一来源：Vision、Audio、Flow、HiFT 和 Speaker 为 `w8a16_sefp`，TTS 为
`w8a8_sefp`。

## 环境

传统 demo 使用 `transformers==4.57.1`；统一接口使用 recipe 声明的 `transformers==5.16.1`。
真实导出还需要仓库、MiniCPM-o/Token2Wav 依赖、可用 CUDA 设备，以及 XH2a 导出和 HMONNX 运行环境。

在已有公共依赖的 CI 环境中，从仓库根目录分别准备依赖：

```bash
python benchmark_test/ci_test_transformer5.16.0/prepare_dependencies.py \
  --model-id minicpm-o-4.5 --phase inference \
  --packages pillow librosa moviepy resampy hyperpyyaml torchaudio ninja --install
python -m pip install --no-deps minicpmo-utils==1.0.6
```

不要安装 `[tts]`/`[all]`，也不要将 minicpmo-utils 放入上面普通安装的 `--packages`。
使用支持公共 Pillow 11.3.0 的 MoviePy 2.2.1，以及公共 librosa 0.11.0；
音频重采样显式使用 `kaiser_best`（resampy 0.4.3）。
其余依赖仍需由 CI 按 recipe 准备。这只避免安装器降级公共依赖，
不消除 minicpmo-utils 对 Pillow/librosa/MoviePy 的旧版本约束；`pip check` 仍会报告冲突，
依赖档案仍为 `blocked_distribution`，不是可直接干净安装的锁文件。
后续可选项：切换新版 librosa 默认的 `soxr_hq` 重采样，待整体精度验证后再决定。

## 校准数据

正式配置使用：

```text
xh2modelzoo://data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl
```

该数据集参考 Qwen-VL 的校准来源，由 CMMMU、COCO、DocVQA、MMMU 各 18 条和
Wikitext 8 条组成，共 80 条；固定 seed 为 1024，记录按来源轮询交织。构造脚本保留在
`build_vl_calibration_jsonl.py`，可从仓库根目录重新生成：

YAML 中的 `xh2modelzoo://` 前缀表示仓库内资源，不依赖调用命令时的当前目录；若代码安装在仓库
之外，可设置 `XH2MODELZOO_ROOT` 指向仓库根目录。

```bash
python examples_merak/llm/minicpm_o_4_5/build_vl_calibration_jsonl.py \
  --source CMMMU=data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json \
  --source COCO=data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json \
  --source DocVQA=data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json \
  --source MMMU=data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json \
  --text-source Wikitext=data/calib_data/wikitext-2-raw-v1.jsonl \
  --samples-per-source 18 \
  --text-samples-per-source 8 \
  --sampling-mode random \
  --seed 1024 \
  --output data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl
```

脚本同时写入同名 `.meta.json`，记录来源计数和生成文件的 SHA256。GPTQ 入口还支持显式配置
预计算 `inputs_embeds` 校准文件，或将其与 JSONL 文本混合；正式 YAML 不启用这些实验模式。

## 导出

默认导出 W8A8 LLM + GPTQ G64 + W8A8 TTS：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --quant-output-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_quant \
  --export-output-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export \
  --device cuda:0 \
  --overwrite
```

上例显式展示了 `--quant-output-dir` / `--export-output-dir` 可以传入；不传时默认目录按
config stem 生成，结果同样是：
`work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_quant` 与
`work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export`。

后续导出/Golden 示例均省略这两个参数，使用默认目录。

导出 W4A8 LLM + GPTQ G64 + W8A8 TTS：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w4a8_gptq.yaml \
  --device cuda:0 \
  --overwrite
```

该配置的默认目录为 `work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_quant` 与
`work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_export`。

传统 demo 仍需原始模型目录中的 processor、tokenizer 和 remote code；Token2Wav 权重已在导出图中，
不再加载原始 `flow.pt` / `hift.pt`。
`export.mode` 决定导出的运行模式：

- `offline`：非流式 `chat`；
- `session`：`streaming_prefill` / `streaming_generate`；
- `stream`：Duplex。

默认导出 `offline`，可通过 `--export-mode session` 或 `--export-mode stream` 选择其他模式。

## HMONNX Demo

统一接口使用 `minicpm-o-4.5`，通过 `profile` 选择 `offline`、`session` 或 `stream`，LLM 为 W4A8、prefill 256、
context 32k。构建会打包 Host 配置、processor、tokenizer 和 embedding，推理不再需要原始权重目录：

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

artifact = build_model(
    model_id="minicpm-o-4.5", model_dir="<model_dir>",
    quant_checkpoint="<gptq_llm_dir>", output_dir="work_dirs/minicpm_o_4_5_unified",
    profile="offline", device="cuda:0",
)
result = inference_model(
    artifact_dir=artifact.output_dir,
    inputs={"prompt": "请简短描述视频内容。", "video": "<video.mp4>", "max_new_tokens": 128},
    device="cuda:0",
)
print(result.output_text)
```

`offline` 运行整段多模态问答；`session` 分块输入后生成回答；`stream` 按全双工协议逐块听/说。
统一入口每次处理一段完整素材，分块过程中复用同一 runtime，返回完整结果及分块记录，不跨调用保留会话。
例如，构建时选择 `profile="stream"`，推理时传入音频和参考音频：

```python
result = inference_model(
    artifact_dir=artifact.output_dir,
    inputs={
        "prompt": "请描述听到的内容。", "audio": "<audio.wav>",
        "generate_audio": True, "reference_audio": "<reference.wav>",
    },
    device="cuda:0",
)
```

`golden=["llm", "tts", "token2wav_flow_frontend", "token2wav_flow_decoder", "token2wav_hift"]`
可在推理时采集指定组件的真实 Golden；所选组件需要在该次输入中实际执行。

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_hmonnx_demo.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --work-dir work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_export \
  --video /path/to/video.mp4 \
  --exec-device cuda:0
```

需要语音回复时增加 `--generate-audio` 和 `--ref-audio /path/to/reference.wav`。输出目录包含
文本和 metadata；启用语音回复后还会写入 WAV 和 speech token。

## 流式 Demo

流式 demo 与导出模式的对应关系：

- `session`：`session_audio_text`、`session_audio_reply`；
- `stream`：`duplex_audio_text`、`duplex_audio_reply`、`duplex_omni_reply`。

示例：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_streaming_demo.py \
  --case session_audio_reply \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --work-dir work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_export_session \
  --media /path/to/input.wav \
  --exec-device cuda:0 \
  --output-dir work_dirs/minicpm_o_4_5_streaming_demo
```

`minicpm_o_4_5_hmonnx_demo.py` 使用 `offline` 产物。

## Golden

`--dump-golden` 默认使用真实样本记录运行图，并用合成输入补齐未执行的参考图。
`offline` 使用模型 assets 中的视频，`session` 使用同一视频的音频，`stream` 使用
默认双工音频样本；可分别用 `--golden-video` 或 `--golden-audio` 覆盖。
仅需逐图合成 Golden 时使用 `--golden-mode synthetic`。

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --device cuda:0 \
  --dump-golden
```
