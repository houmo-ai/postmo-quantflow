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

## Gemma4：复用量化基座并增加 MTP

`gemma4-e2b`、`gemma4-e4b`、`gemma4-12b-unified`、`gemma4-31b` 和
`gemma4-26b-a4b` 使用同一接口。选择 MTP 时必须提供对应尺寸、架构的 assistant：

E2B/E4B/31B/26B-A4B 的 Transformers 范围为 `>=5.5.0,<5.17.0`；
12B Unified 单独要求 `>=5.13.0,<5.17.0`。普通版与 MTP 版遵循各自 recipe
的同一版本门禁，构建/推理前检查；版本范围不代表每个补丁版本都跑过完整模型评测。

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

result = build_model(
    model_id="gemma4-12b-unified",
    model_dir="/path/to/gemma-4-12B-it",
    quant_checkpoint="/path/to/gemma4_12b_quant",
    sources={"mtp": "/path/to/gemma-4-12B-it-assistant"},
    speculative={"mode": "mtp", "num_draft_tokens": 4},
    context_length=8192,
    low_memory=False,
    output_dir="./work_dirs/gemma4_12b_mtp",
    device="cuda:0",
)
answer = inference_model(
    artifact_dir=result.output_dir,
    inputs={"prompt": "用一句话介绍你自己", "max_new_tokens": 64},
    device="cuda:0",
)
print(answer.output_text, answer.stats)
```

`quant_checkpoint` 复用主模型已量化权重，跳过主干搜索与校准。省略时执行 recipe 的
主模型量化流程。普通版与 MTP 版要对齐，须显式使用同一个 checkpoint。
assistant 主干沿用 W8、独立 Head 沿用现有 W4 导出配置，执行导出 PTQ；
这不表示 assistant 经过 AutoRound/GPTQ 校准。E2B/E4B 的 ordered head 还包含
候选词选择，不能将整个候选表都视为已打包 W4。

Gemma4 assistant 使用主模型的输入 embedding 和只读 KV，但拥有自己的输出 Head，
不继承 Qwen 的共享 Head 规则，也不改变主模型 Head 精度。导出包包含 target
prefill/verify、assistant decode 和独立执行所需资源；没有 assistant prefill。
推理不再需要原 assistant 路径。当前 unified MTP 验证入口为文本、greedy、无 Golden，
返回草稿数、接收数及接收率；MTP 图包含多模态主干不等于该验证入口已支持多模态输入。

## Qwen：固定基座后复用 MTP 和 LoRA

Qwen3.5/3.6/3.8 recipe v3 的新量化默认校准 Head W5。普通版与 MTP/DFlash
使用同一主模型精度；MTP、DFlash、DFlash2 继承完整 head 载荷。
旧 draft head 字段与主模型相同则兼容，冲突时报错。Qwen3-Next 保持原有量化
recipe 默认值，采用相同的 head 继承规则，不提供 LoRA 组合。
DFlash2 沿用 `speculative={"mode": "dflash"}` 和 `sources.dflash`，由 draft
checkpoint 的架构声明识别，不增加一个 head 或精度选择参数。
例如 Qwen3.8 的 AutoRound 主干先通过 W5 head workflow 补校准，之后统一接口
读取新 checkpoint 中的 W5 元数据并让 target/draft 共用 head：

```python
build_model(
    model_id="qwen3.8-27b",
    model_dir="/models/Qwen3.8-27B",
    quant_checkpoint="/models/autoround_quant_head_w5",
    sources={"dflash": "/models/Qwen3.8-27B-DFlash2"},
    speculative={"mode": "dflash"},
    context_length=8192,
    output_dir="/exports/qwen3_8_dflash2_w5_8k",
    device="cuda:0",
)
```

`context_length=262144` 选择 256K FlashAttention workflow；两档的 RoPE 表
长度均由 `max_pe_length=262144` 控制。

先生成并保留量化基座（不生成 Golden）：

```python
from xhmodel_merak.model_build import build_model

base = build_model(
    model_id="qwen3.6-35b-a3b",
    model_dir="/models/original",
    speculative={"mode": "mtp", "num_draft_tokens": 4},
    quant_output_dir="/models/quantized",
    output_dir="/exports/base_mtp",
    low_memory=False,
    device="cuda:0",
)
```

MTP 非共享专家高阶校准 W4，其余 W8；Dense MTP 则 decoder block 的 Linear
高阶校准 W4，融合 FC W8。Head 与主模型共享，算法跟随主干生产端。
35B-A3B 需要 block 混合精度时，在首次调用增加 `precision={"weight": "wmix"}`；
搜索配置由 recipe 固定，不向客户暴露 KL、head bit 或 MTP bit。

第二阶段从返回的 **同一个 `base.quant_artifact`** 导出各个 LoRA：

```python
build_model(
    model_id="qwen3.6-35b-a3b",
    model_dir="/models/original",
    quant_checkpoint=base.quant_artifact,
    speculative={"mode": "mtp"},
    lora={"paths": ["/adapters/adapter_a", "/adapters/adapter_b"]},
    output_dir="/exports/lora_mtp",
    low_memory=False,
    device="cuda:0",
)
```

LoRA 增量分支 W8A16。LoRA + MTP 要求已有完整量化 checkpoint，不重新搜索，
也不自动补 head/MTP 校准；载荷缺失或冲突直接报错。adapter 修改 head、embedding
或 MTP 自身时不支持该组合。量化产物及 LoRA 内容指纹写入 manifest。
各 adapter 子目录保留独立运行时配置和共享 draft 组件。

从同一 checkpoint 导出普通版时省略 `speculative`；省略 `precision` 时自动读取
checkpoint 中的混合精度标记。复用旧 checkpoint 保留其已记录的 head 位宽，
不会因为 recipe 默认值变为 W5 而改写旧权重。默认临时量化目录会被清理；首次
需要后续复用必须传 `quant_output_dir`。相同主模型权重是关闭/开启 MTP 确定性
解码一致性的前提，实际 token 一致性与接收率仍需在部署 runtime 上验收。
