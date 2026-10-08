# Qwen3-ASR Merak Workflow 示例

本目录提供 Qwen3-ASR 迁移到 `xhmodel_merak/xh_other_model` 后的导出和 HMONNX 推理示例。所有命令默认从仓库根目录执行。

## 环境安装

环境名称和 Python 版本可按实际工程要求调整。示例安装步骤如下：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install qwen_asr
pip install onnx-ir==0.1.14 onnx==1.16.2 onnxscript==0.5.7
```

Qwen3-ASR 依赖 `qwen_asr`，该包会带入对应的 `transformers` 版本约束。若运行环境已有统一的 xhquant/HMONNX 运行时，请以该环境的依赖版本为准。

## 配置

默认 workflow YAML：

```text
configs_merak/workflows/xh2a/other_models/qwen3_asr/0_6b/qwen3_asr.yaml
```

Qwen3-ASR 月辉支持 0.6B 和 1.7B，对应导出配置：

```text
configs_merak/workflows/yuehui/other_models/qwen3_asr/0_6b/qwen3_asr_0_6b.yaml
configs_merak/workflows/yuehui/other_models/qwen3_asr/1_7b/qwen3_asr_1_7b.yaml
```

`--model-dir` 必须显式传入。默认输出目录是相对当前工作目录的字符串路径：

- 量化目录：`work_dirs/qwen3_asr_quant`
- 导出目录：`work_dirs/qwen3_asr_export`

YAML 默认导出两个组件：

- `encoder`
- `prefill_decode`

workflow 会在导出前把实际使用的 YAML dump 到导出目录，文件名使用 workflow 配置文件名。`AutoWorkflow` 会通过 YAML 中的 `export.model.type` 解析模型，并绑定到 `Qwen3ASRWorkflow`。

## 导出

Qwen3-ASR-1.7B 月辉导出：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir /data01/nfs_shared/ASR_TTS/Qwen3-ASR-1.7B \
  --config-path configs_merak/workflows/yuehui/other_models/qwen3_asr/1_7b/qwen3_asr_1_7b.yaml \
  --export-output-dir work_dirs/qwen3_asr_1_7b_yuehui_export \
  --device cuda:0 \
  --overwrite
```

0.6B 使用同一工作流，替换模型和配置路径：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir /data01/nfs_shared/ASR_TTS/Qwen3-ASR-0.6B \
  --config-path configs_merak/workflows/yuehui/other_models/qwen3_asr/0_6b/qwen3_asr_0_6b.yaml \
  --export-output-dir work_dirs/qwen3_asr_0_6b_yuehui_export \
  --device cuda:0 \
  --overwrite
```

XH2a 默认配置导出：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --device cuda:0 \
  --overwrite
```

覆盖音频长度和 prefix token 预算：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --max-audio-length 6000 \
  --prefix-token-budget 512 \
  --export-output-dir work_dirs/qwen3_asr_export_60s \
  --device cuda:0 \
  --overwrite
```

Qwen3-ASR 当前的 prefill/decode 量化在导出阶段执行。workflow 的 `quant()` 阶段会记录跳过状态并返回 `QuantResult`，随后 `export()` 按 YAML 配置导出 encoder 和 prefill/decode。

## HMONNX 推理

导出完成后，把导出目录传给 HMONNX demo。以下示例使用默认导出目录：

```text
work_dirs/qwen3_asr_export
```

分段离线推理：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo.py \
  --work-dir work_dirs/qwen3_asr_export \
  --audio <audio_file> \
  --device cuda:0 \
  --max-audio-length 1500
```

累计音频加文本 prefix 推理：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo_chunk_prefix.py \
  --work-dir work_dirs/qwen3_asr_export \
  --audio <audio_file> \
  --device cuda:0 \
  --max-audio-length 6000 \
  --chunk-seconds 20
```

`hmonnx_demo.py` 每个音频片段独立推理，只需要 `max-audio-length` 覆盖单个片段。

`hmonnx_demo_chunk_prefix.py` 会基于累计音频重复推理，并在后续片段注入已识别文本作为 prefix。使用该模式时，导出时需要设置更大的 `max-audio-length` 和足够的 `prefix-token-budget`。

## TCIM/HMM 推理

HMM 与 HMONNX 不在同一目录时，先生成独立的 runtime metadata；该命令不创建软链接，也不修改
原始 `export_meta_info.json`：

```bash
PYTHONPATH=$PWD python -m xhmodel_merak.exec_backends.link_hmm \
  --hmm_path <compiled_hmm_dir> \
  --hmonnx_path work_dirs/qwen3_asr_export \
  --source-meta export_meta_info.json
```

然后仅通过 backend 环境变量和 `--config` 切换运行路径：

```bash
XHMODEL_INFERENCE_BACKEND=tcim PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo.py \
  --config work_dirs/qwen3_asr_export/hmm_meta_info.json \
  --audio <audio_file>
```

未显式传 `--cache-len` 时，demo 使用 `hmm_meta_info.json` 中由 HMM 输入 ABI 更新后的 cache
维度。TCIM runtime 要求 metadata 中模型路径明确指向 `.hmm`/`.hmms`，不会在运行时猜测或替换
HMONNX 路径。

## Golden

使用真实音频生成可复现 golden 时，在 `hmonnx_demo.py` 中加入 `--golden`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo.py \
  --config work_dirs/qwen3_asr_export/export_meta_info.json \
  --audio <audio_file> \
  --golden
```

组件图的实际输入、输出和 activation 分别写入发布根目录下的
`encode/step_*`、`prefill/step_*`、`decode/step_*`（旧布局则写入 metadata 中模型所在的组件目录）。
本次请求的输入来源写入同级 `golden_input/`：原始音频复制为 `input_audio.<原扩展名>`，并保存
`prompt.txt`、保留 special token 的 `tokenizer_output.txt`、padding 前的二维 `token_ids.npy` 和
`input_meta.json`。每次运行会先重建 `golden_input/`；多 chunk 推理时 tokenizer 文件对应最后一次
prefill，`input_meta.json` 记录实际 prefill 次数，各组件的 `step_N` 对应第 N 次图执行。Host 侧 embedding 是 prefill 图输入，由
`prefill/step_*` 捕获，不在 `golden_input/` 重复保存。

为了至少产生一次 decode golden，`--max-new-tokens` 必须大于 0。TCIM 当前不支持 HMONNX node-level golden，因此
`XHMODEL_INFERENCE_BACKEND=tcim` 与 `--golden` 同时使用会在推理前明确失败。

workflow 的 `--dump-golden` 仍用于导出阶段的随机输入 smoke baseline，与上述真实音频 demo golden
用途不同：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```
