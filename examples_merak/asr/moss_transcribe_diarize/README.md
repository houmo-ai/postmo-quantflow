# MOSS-Transcribe-Diarize

导出并跑 MOSS-Transcribe-Diarize 0.9B 的 encode / prefill / decode 图。命令都在仓库根目录执行。

| 脚本 | 用途 |
|---|---|
| `moss_transcribe_diarize_workflow.py` | 导出。默认 2k YAML；加 `--context-length` 走统一 `build_model` |
| `moss_transcribe_diarize_hmonnx_infer.py` | 看产物 metadata，或在 XH2a 上按生产路径转写并打耗时 |
| `compare_pytorch.py` | 30s clip：官方 HF greedy vs 本产物，打前缀一致率 |

对外入口是 `model_id=moss-transcribe-diarize-0.9b` 的 [统一接口](../../unified_api/README.md)（`catalog` / `build` / `transcribe`）。本目录是 workflow 直出和逐步 `session.run` 演示。

## 音频

本地 **16 kHz 单声道 16-bit PCM** WAV 或 FLAC。encode 组件固定 30s Whisper 窗，多窗在 Host 拼进同一份 KV，不按会议切段。CPU 跑不了这些 XH2a 图。

## 导出

默认 YAML（2k、FA 关、prefill 512）：

```text
configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/moss_transcribe_diarize.yaml
```

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py \
  --model-dir data/models/MOSS-Transcribe-Diarize \
  --device cuda:0 \
  --output-dir work_dirs/moss_transcribe_diarize_2k \
  --overwrite
```

Golden 只接受不超过 30s 的单窗，且只能走默认 YAML 路径（不能和 `--context-length` 一起用）：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py \
  --model-dir data/models/MOSS-Transcribe-Diarize \
  --golden-audio work_dirs/moss_llm_migrate_clip30s.wav \
  --dump-golden --overwrite
```

30 分钟推荐 64k FA 图。`build_model` **不会**读长档 YAML 里的 4096，必须显式传 `prefill_chunk_length`：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py \
  --model-dir data/models/MOSS-Transcribe-Diarize \
  --output-dir work_dirs/moss_transcribe_diarize_64k \
  --device cuda:0 \
  --context-length 65536 \
  --prefill-chunk-length 4096 \
  --overwrite
```

同一 `model_id` 分四档，不要把默认 YAML 改成 64k/128k：

| `context_length` | prefill | FA | 用途 |
|---|---|---|---|
| ≤4096（默认 2048） | 512 | 关 | 单窗 30s |
| 8192 / 16384 | **必须 4096** | 开 | 约 5–10 分钟 greedy 验证 |
| 65536 | **必须 4096** | 开 | 30 分钟推荐 |
| 131072 | **必须 4096** | 开 | 更长 cache 的 greedy；load 时同样把 FA 图转成 PageAttention |

prompt 和生成预算必须同时放进本图 cache，溢出直接失败。省略 `max_new_tokens` 时按约 10 token/s 估（下限 512、上限 65536）。30 分钟默认约 1.8 万 token，16k 图会失败。旧产物缺契约就重新导出，不手改 metadata。

Decoder 用 FP32 RoPE 表 + 整数下标，默认表长 256k（`max_pe_length=262144`），与 16k/64k/128k 档相同。`rope_mode` 或几何对不上会被拒绝。

## HMONNX 推理

加速在三张已 lowering 的图上。Host 仍做 log-mel、音/文融合和 greedy `lm_head`。

只看产物：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_hmonnx_infer.py \
  --artifact-dir work_dirs/moss_transcribe_diarize_2k
```

2k（FA 关，保持 CacheTensor）：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_hmonnx_infer.py \
  --artifact-dir work_dirs/moss_transcribe_diarize_2k \
  --audio work_dirs/moss_llm_migrate_clip30s.wav \
  --device cuda:0 \
  --disable-page-attention
```

64k greedy 在加载期把 FA+inplace 转成 PageAttention，decode 走 CUDA Graph：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_hmonnx_infer.py \
  --artifact-dir work_dirs/moss_transcribe_diarize_64k \
  --audio work_dirs/moss_alimeeting_clip10min.wav \
  --device cuda:0 \
  --enable-page-attention
```

`--artifact-dir` 可以是 workflow 的 `output_dir`，也可以是 `build_model` 包。省略 PA 开关时，runtime 按产物 `model_config.enable_page_attention` / FA 决定是否转换。

## 对照官方 PyTorch

30s smoke，不是 AliMeeting 全量：

```bash
PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/compare_pytorch.py \
  --audio work_dirs/moss_llm_migrate_clip30s.wav \
  --artifact-dir work_dirs/moss_transcribe_diarize_64k \
  --model-dir data/models/MOSS-Transcribe-Diarize \
  --device cuda:0 \
  --enable-page-attention
```

## 统一接口

`build_model` 产物带 `model_build_manifest.json`，才能交给 `inference_model`。workflow 直出没有这份 manifest。

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
