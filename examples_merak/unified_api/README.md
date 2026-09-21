# 统一接口示例

先按[仓库安装说明](../../README.md#安装)准备 Python 环境，以下命令都在仓库根目录运行。
查模型不加载权重，也不需要 GPU；构建和推理需要对应模型的 SDK、依赖和本地权重。

开发新模型时看 [recipe.py](recipe.py)，接入要求见 [开发约定](../../xhmodel_merak/model_integrations/README.md)。
运行 `python -m examples_merak.unified_api.recipe` 可以看完整注册结果：它在独立 catalog 中
注册现有 Ling recipe，打印参数、组件和输入输出说明，不构建模型、不修改正式模型列表。

## 查模型、看参数

```bash
python -m examples_merak.unified_api.catalog
python -m examples_merak.unified_api.catalog qwen3.5-4b
python -m examples_merak.unified_api.catalog qwen3.5-4b --full
```

分别调用 `list_models()`、`describe_model()` 和 `describe_model(full=True)`，打印 JSON。
第一个返回模型条目列表；第二个只列可改的参数、默认值和约束；第三个还包含固定配置和推理输入格式。
`describe_model()` 返回的是说明，不是可以直接传给 `build_model(**...)` 的参数字典。

## 构建产物

以 Qwen3.5-4B 为例，把路径换成自己的：

```bash
python -m examples_merak.unified_api.build \
  --model-id qwen3.5-4b \
  --model-dir /path/to/Qwen3.5-4B \
  --output-dir ./work_dirs/qwen35_4b_api \
  --device cuda:0
```

脚本直接调用 `build_model()`，默认按模型配置量化并导出。`output_dir` 是最终目录，不再套一层。
不传 `--output-dir`（Python 中省略或传 `None`），就在当前目录按发布名和日期建目录，
路径见返回的 `output_dir`。已有目录不覆盖，包括自动命名重名的情况。
已有匹配的量化权重时，加 `--quant-checkpoint /path/to/quantized`，仍保留原始 `--model-dir`。
来源会从权重元数据识别，不重跑量化。需要调整组件图精度时，加
`--deployment-precision '{"visual":"w8a16h1_sefp"}'`；可改哪些组件先看 `describe_model`。

Qwen 可加 `--low-memory` 走分层导出；`--no-low-memory` 明确关闭。不传时使用变体默认值，
Ling 默认开启。Gemma 和 Qwen LoRA 暂不支持开启；这不改变推理方式。
没有指定的组件沿用当前 YAML，已量化的权重不会因此重做。
需要改上下文时，加 `--context-length 4096 --prefill-chunk-length 256`；可用范围先查 `catalog`。

返回 `BuildResult`，脚本将它转成 JSON。主要看 `output_dir`、`manifest_file`、`components` 和 `loadable`。
这些脚本只展示常用参数；精度、LoRA 等设置可以直接在 `build_model()` 调用处补充，不需要改 YAML。

## 用产物推理

推理只传产物目录，不用再指定模型名或某个 ONNX 文件：

```bash
python -m examples_merak.unified_api.generate \
  --artifact-dir ./work_dirs/qwen35_4b_api \
  --prompt '用一句话介绍你自己' \
  --max-new-tokens 64
```

图片问答以及生成 golden：

```bash
python -m examples_merak.unified_api.generate \
  --artifact-dir ./work_dirs/qwen35_4b_api \
  --prompt '图里有什么？' \
  --image /path/to/image.jpg \
  --golden visual prefill decode
```

脚本调用 `inference_model()`，JSON 中的 `output_text` 是回答，`golden_artifacts` 是本次生成的 golden 路径。
不传 `--golden` 就只推理；重复运行会清理上次未纳入 manifest 的 golden 输出。
动作等模型的输入格式看 `catalog --full` 返回的 `input_schema`。

## Kokoro 中文语音

```bash
python -m examples_merak.unified_api.build \
  --model-id kokoro-82m-v1.1-zh \
  --model-dir ./work_dirs/model_assets/kokoro \
  --output-dir ./work_dirs/kokoro_api --device cuda:0

python -m examples_merak.unified_api.synthesize \
  --artifact-dir ./work_dirs/kokoro_api \
  --text '千里之行，始于足下。' --output-wav ./work_dirs/kokoro.wav
```

原始资源要求见 [Kokoro 说明](../audio/kokoro/README.md#环境与原始资源)。构建导出全部四档；
推理返回 24 kHz 波形，脚本保存为单声道 WAV。当前只支持中文、数字和中文标点，
最多 256 token（含首尾标记）；超长请自己切句，不会截断或自动分段。

包内有 `hf_config`、voice.npy 和 16 个组件目录（三张 NPU 图、一张 CPU 相位图，各四档），
外加统一 manifest 和 `effective_workflow.yaml`。图的原始发布名称和 external_data 不改，
集成组件路径读 `export_meta_info.json`；运行契约读 Manifest 指向的 `runtime_meta_info.json`。旧 Kokoro 包要重新构建。
`build_model` 不生成 Golden；全档串联 Golden 仍使用 Kokoro workflow 的独立入口。

## MOSS-Transcribe-Diarize 会议转写

音频必须是本地 16 kHz 单声道 16-bit PCM WAV/FLAC。30 分钟会用 64k 图，并显式传 `prefill_chunk_length=4096`：

```bash
python -m examples_merak.unified_api.catalog moss-transcribe-diarize-0.9b --full

python -m examples_merak.unified_api.build \
  --model-id moss-transcribe-diarize-0.9b \
  --model-dir data/models/MOSS-Transcribe-Diarize \
  --output-dir ./work_dirs/moss_transcribe_diarize_64k_api \
  --device cuda:0 \
  --context-length 65536 \
  --prefill-chunk-length 4096

python -m examples_merak.unified_api.transcribe \
  --artifact-dir ./work_dirs/moss_transcribe_diarize_64k_api \
  --audio work_dirs/moss_llm_migrate_clip30s.wav \
  --device cuda:0
```

`transcribe.py` 只调用 `inference_model()`。workflow 直出没有统一 manifest，逐步打开三张图并打耗时用 [HMONNX infer 示例](../asr/moss_transcribe_diarize/README.md)。

## 产物长什么样

Qwen3.5 完整包的主要文件如下，具体文件清单以 `model_build_manifest.json` 为准：

```text
qwen35_4b_api/
├── model_build_manifest.json
├── export_meta_info.json
├── effective_workflow.yaml
├── golden_meta_info.json
├── hf_config/
├── quant_embedding.pt
├── visual_gears.json
├── visual_m96/
├── visual_m196/
├── visual_m384/
├── visual_m704/
├── visual_m1536/
├── prefill/
└── decode/
```

各组件目录里是图和相关权重。ViT 五档平铺在根目录，`visual_gears.json` 记录每档图的位置。
集成商读取 `export_meta_info.json` 的 `{path, type}`，路径相对发布根目录；图指向文件，
tokenizer 指向目录，LoRA 位于 `lora.<checkpoint>` 下。完整规则见
[产物布局约定](../../docs/architecture/artifact-layout.md)。
搬运时复制整个产物目录，不要只拿一个 ONNX 文件。旧 workflow 直接导出的目录没有统一 manifest，不能直接交给这个推理接口。

构建交付时不包含临时子图、逐层导出工作文件、日志、评测报告和摘要缓存。
推理可能在目录里新增缓存或临时文件，不影响再次加载；已交付的模型文件仍会校验完整性。
MTP/DFLASH 等图只在选择对应变体时出现，按 `mtp_*` / `dflash_*` 平铺，不再增加一层目录。
