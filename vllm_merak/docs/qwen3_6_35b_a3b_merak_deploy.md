# Qwen3.6-35B-A3B Merak/HMONNX 单卡/多卡部署与性能验收

本文给出 Qwen3.6-35B-A3B 导出产物的可复用 vLLM 部署流程。Phase 1 支持恰好 1、2 或 3 张唯一 GPU，并固定 `MAX_NUM_SEQS=1`；实际大小 cache backing 允许在资源充足的 80GB 卡上单卡部署一个 35B 服务。`MAX_MODEL_LEN=262144` 时默认容量 planner 规划 4304 个全局 block。旧的 `CPU_KV=48` GiB 结果仅作为显式 virtual budget 的历史证据，不再是推荐默认值；2k 配置只用于低成本 smoke。

目标模型目录：

```text
/data01/home/yujy/work/v4/xh2modelzoo/work_dirs/qwen3_6_35b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260710
```

> 多卡由 HMONNX V2 `--auto-offload` 完成。服务始终只有一个 vLLM worker，启动命令必须保留 `--use-v2 --auto-offload --cuda-graph`，**不要添加 `--tensor-parallel-size`**。当前固定地址 PageAttention metadata staging 仅支持 `MAX_NUM_SEQS=1`。

## 1. 当前静态证据

2026-07-10 已核对导出目录，静态事实如下：

| 检查项 | 当前证据 |
|---|---|
| loader 兼容类型 | `hf_config/config.json` 的 `architectures` 为 `Qwen3_5MoeForConditionalGeneration`，`model_type` 为 `qwen3_5_moe` |
| Merak metadata | `golden_meta_info.json` 的 `meta.class_name` 为 `Qwen3_5Moe_ModelMeta` |
| 最大上下文 | `golden_meta_info.json` 的 `max_context_tokens` 和 `context_max_length` 均为 `262144` |
| HMONNX 图 | `prefill/`、`decode/`、`visual/` 及对应 external data 均存在 |
| 图像能力判定 | `golden_meta_info.json` 根级 `visual_config.hmonnx` 指向存在的 visual HMONNX；服务自动注册多模态 loader |
| Merak 入口配置 | 当前目录已补齐 `merak_config.json`；更换或重建导出目录时仍需检查 |

这些静态事实本身不证明服务成功加载或性能达标；已完成的运行证据、未达标项和未测试项都在第 7 节单独披露。

## 2. 环境与前置检查

```bash
conda activate vllm
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak
export MODEL_DIR=/data01/home/yujy/work/v4/xh2modelzoo/work_dirs/qwen3_6_35b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260710

export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_6_35b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260722

export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_122b_w4a8_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_122b_a10b_w4a8_256_256k_mpe256k_448x448_20260714

export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260723

export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_9b_flashattention_16_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a16_256_256k_mpe256k_448x448_20260725
```

启动脚本要求模型目录包含 Merak 配置。当前 `20260710` 目录已经补齐；若更换或重建导出目录，可用仓库模板补齐，`-n` 会避免覆盖已有配置：

```bash
cp data/merak_config.json "$MODEL_DIR/merak_config.json"
```

检查必需文件：

```bash
test -f "$MODEL_DIR/merak_config.json"
test -f "$MODEL_DIR/golden_meta_info.json"
test -f "$MODEL_DIR/hf_config/config.json"
test -d "$MODEL_DIR/prefill"
test -d "$MODEL_DIR/decode"
```

检查插件环境和空闲 GPU：

```bash
python - <<'PY'
import vllm, vllm_merak, xhmodel_merak, xhquant
print("vllm", vllm.__version__, "| merak dependencies ok")
PY
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
```

## 3. 1–3 卡 2k smoke（仅预检）

以下示例用一张空闲的 80GB 卡；也可以把 `0` 换成两个或三个唯一 GPU 编号。2k smoke 默认使用容量 planner，同时使用 `MAX_MODEL_LEN=2048`、`MAX_NUM_SEQS=1` 和端口 `18036`：

```bash
CUDA_VISIBLE_DEVICES=0 \
SERVED_MODEL_NAME=qwen3_6_35b_a3b_w4a8 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen36_35b_w4a8_1gpu_256k_server.log

CUDA_VISIBLE_DEVICES=6 \
SERVED_MODEL_NAME=qwen3_6_35b_a3b_w4a8 \
PORT=18308 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen36_35b_w4a8_1gpu_256k_server_18308.log


VLLM_LOGGING_LEVEL=INFO \
AUDIT_LOG_MODE=both \
AUDIT_LOG_DIR=logs/audit_qwen36 \
CUDA_VISIBLE_DEVICES=7 \
SERVED_MODEL_NAME=qwen3_5_9b_w4a8 \
PORT=18408 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen35_9b_w4a8_1gpu_256k_server_18408.log

CUDA_VISIBLE_DEVICES=0 \
SERVED_MODEL_NAME=qwen3_6_35b_a3b_w4a16 \
PORT=18307 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen36_35b_w4a16_1gpu_256k_server.log


CUDA_VISIBLE_DEVICES=7 \
SERVED_MODEL_NAME=qwen3_6_35b_a3b_w4a16 \
PORT=18309 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen36_35b_w4a16_1gpu_256k_server_18309.log

CUDA_VISIBLE_DEVICES=1,2,4 \
SERVED_MODEL_NAME=qwen3_5_122b_a10b_w4a8 \
PORT=18405 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen35_122b_w4a8_1gpu_256k_server.log
```

脚本会在启动 Python 前检查 `merak_config.json` 和 GPU 列表。`CUDA_VISIBLE_DEVICES` 只接受恰好 1、2 或 3 个唯一非负整数；空项、重复卡、前导零、非整数和 4 卡都会直接报错。脚本支持以下环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
|---|---:|---|
| `MODEL_DIR` | 本文目标目录 | HMONNX 导出目录 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 恰好 1、2 或 3 个唯一非负整数 GPU 编号 |
| `PORT` | `18036` | OpenAI API 端口 |
| `MAX_MODEL_LEN` | `2048` | smoke 或长上下文上限 |
| `MAX_NUM_SEQS` | `1` | 固定为 1；脚本拒绝任何其他值 |
| `AUDIT_LOG_MODE` | `off` | `off`、`aggregate`、`sse` 或 `both`；定位请求/回复问题时推荐 `both` |
| `AUDIT_LOG_DIR` | `logs/audit` | 聚合 JSONL 和原始 ASGI/SSE 分片日志目录 |
| `VLLM_CPU_KVCACHE_SPACE` | 未设置 | 可选显式 virtual scheduler budget，单位 GiB；不是物理 backing |
| `CPU_KV` | 未设置 | 上述标准变量的兼容别名；两者同时设置且不同会报错 |

未设置两个容量变量时，脚本使用确定性 planner。2K smoke 按实际几何规划小容量；256K 配置默认需要 4099 个 block，并加 5% headroom 得到 4304。启动日志会同时披露 virtual scheduler budget 与逐设备实际 physical cache backing。

### 请求与回复审计日志

需要排查多轮对话、reasoning、工具调用或流式分片时，在原启动命令前增加：

```bash
umask 077
VLLM_LOGGING_LEVEL=INFO \
AUDIT_LOG_MODE=both \
AUDIT_LOG_DIR=logs/audit_qwen36 \
CUDA_VISIBLE_DEVICES=0 \
SERVED_MODEL_NAME=qwen3_6_35b_a3b_w4a8 \
PORT=18306 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee tmp/qwen36_35b_w4a8_server.log
```

审计日志独立于 vLLM logger，不需要把 `VLLM_LOGGING_LEVEL` 调到 `DEBUG`。
`aggregate` 每个请求记录一条重组后的请求/回复；`sse` 逐条保存原始 ASGI/SSE
body 分片；`both` 同时生成两类文件：

```text
logs/audit_qwen36/audit-aggregate-YYYYMMDD-p<PID>.jsonl
logs/audit_qwen36/audit-sse-YYYYMMDD-p<PID>.jsonl
```

SSE 记录同时包含便于阅读的 `body_text` 和用于逐字节还原的 `body_base64`。
查看日志：

```bash
python -m vllm_merak.audit_log_viewer \
  logs/audit_qwen36/audit-sse-YYYYMMDD-p<PID>.jsonl | less
```

日志包含完整请求正文和模型回复，可能带有敏感数据，并且 `both` 模式增长较快。
问题定位结束后应恢复 `AUDIT_LOG_MODE=off`，生产环境还应配置日志轮转和清理策略。

### 串行执行边界

当前 HMONNX CUDA Graph 会在每个 runtime model、prefill/decode stage 和 consumer device 上复用固定地址的 PageAttention metadata staging buffer。该设计要求同一个 runtime model 实例的 `set_context -> forward` 严格串行，因此启动脚本强制 `MAX_NUM_SEQS` **必须精确等于 `1`**；设置 `2` 或更大值会在 Python 启动前直接失败，`0`、`01` 或非数字同样不会被接受。

客户端可以同时发起多个 HTTP 请求，但这些请求只能由单一 EngineCore 排队串行执行，不能同时进入同一 model 实例的 forward。提升 `MAX_NUM_SEQS` 前，必须先实现并验证 request-local 的 graph/context（或每个并发请求独立的 runtime/graph 实例），不能继续共享当前可变 staging buffer。

另一个终端执行健康检查和最小生成：

```bash
conda activate vllm
curl -fsS http://127.0.0.1:18036/health
curl -fsS http://127.0.0.1:18036/v1/models | python -m json.tool
curl -fsS http://127.0.0.1:18036/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3_6_35b_a3b_merak",
    "prompt": "用一句话介绍北京。",
    "max_tokens": 64,
    "temperature": 0
  }' | tee /tmp/qwen36_35b_2gpu_smoke.json
```

## 4. 按实际物理容量选择 1、2 或 3 卡

实际大小 cache backing 不再按 virtual page padding 稠密分配，因此资源充足的 80GB 卡可以运行一个单卡 35B 服务。优先尝试满足部署目标的最少卡数；只有启动日志明确出现 OOM/容量不足时，才依次改用两张或三张空闲卡。停止旧服务后只改可见卡列表，其余参数不变：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MODEL_DIR="$MODEL_DIR" \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee /tmp/qwen36_35b_3gpu_2k_server.log
```

无论使用 1、2 还是 3 卡，都不要添加 `--tensor-parallel-size`。HMONNX 日志中的 device mapping 才是实际放置证据；仅看到多张卡可见不等价于每张卡都被使用。

## 5. 正式 streaming benchmark

`tools/bench_openai_stream.py` 接受 `--base-url`、`--model`、`--tokenizer` 或 `--model-dir`、目标 prompt token 数、输出 token 数、warmup 和重复次数。每个 warmup/测量轮次输出一行 JSON，最后输出一行 `type=summary` 的 median/p95；逐轮和 summary 都披露 target、本地实际值及服务 usage 的 prompt tokens。

工具执行严格的完整性校验：本地 prompt 必须精确命中 target；每轮服务 `prompt_tokens` 必须同时等于本地值和 target；SSE 必须包含 usage 与 `[DONE]`，且不能包含 error event。请求固定 `ignore_eos=true`，所以 `finish_reason` 必须为 `length`，`completion_tokens` 必须等于 `max_tokens`，否则该轮直接失败，不进入 summary。decode 吞吐严格按服务返回的 `completion_tokens - 1` 计算，计时区间为首个文本 token 到最后一个文本 token，不把最终 usage/`[DONE]` 的网络尾延迟算入。

2k 基准：

```bash
conda activate vllm
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak
python tools/bench_openai_stream.py \
  --base-url http://127.0.0.1:18036/v1 \
  --model qwen3_6_35b_a3b_merak \
  --model-dir "$MODEL_DIR" \
  --target-prompt-tokens 1800 \
  --max-tokens 128 \
  --warmup 2 \
  --repeat 5 \
  | tee /tmp/qwen36_35b_2gpu_2k_bench.jsonl
```

也可以绕过 `MODEL_DIR/hf_config` 自动推导，显式指定 tokenizer：

```bash
python tools/bench_openai_stream.py \
  --tokenizer "$MODEL_DIR/hf_config" \
  --target-prompt-tokens 1800
```

要做受控的“35B-A3B decode 比 9B 快”模型比较，必须使用相同 GPU 数量/型号、目标 prompt tokens、`max_tokens`、warmup/repeat 和服务热态，分别保存 9B 与 35B-A3B JSONL，并比较 summary 中 `decode_tok_s.median`。若按各自实际部署拓扑测试，也可以比较相同请求协议下的服务吞吐，但必须同时披露卡数和卡号，不能把差异仅归因于模型。旧文档中的单轮数据不能替代正式重复测量。

## 6. 256k 验证

2k smoke 通过后，再停止 smoke 服务并以 256k 配置重启。不要继承旧的 8/48 GiB 环境覆盖；默认 planner 会为一个 256K 请求规划 4304 个全局 block。启动前确认主机和目标设备资源充足：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_DIR="$MODEL_DIR" \
MAX_MODEL_LEN=262144 \
MAX_NUM_SEQS=1 \
bash examples/vllm_server_start_qwen36_35b.sh \
  2>&1 | tee /tmp/qwen36_35b_1gpu_256k_server.log
```

如单卡出现明确 OOM，再把 `CUDA_VISIBLE_DEVICES` 改为两张卡，必要时再改为三张卡；始终保持单 worker、无 tensor parallel。长上下文基准为 prompt 和输出预留总长度空间：

```bash
python tools/bench_openai_stream.py \
  --base-url http://127.0.0.1:18036/v1 \
  --model qwen3_6_35b_a3b_merak \
  --model-dir "$MODEL_DIR" \
  --target-prompt-tokens 256000 \
  --max-tokens 64 \
  --warmup 1 \
  --repeat 3 \
  | tee /tmp/qwen36_35b_256k_bench.jsonl
```

## 7. 证据与验收门槛

### 日志必须证明

```bash
grep -E \
  'Merak launch config|ENABLE_HMINFERENCE_V2|Auto-offload device mapping|CudaGraphAutoOffLoadGraphModuleInterpreter|CUDA Graph capture completed|Application startup complete|out of memory|OOM' \
  /tmp/qwen36_35b_*_server.log
```

其中 `Merak launch config` 与容量 planner 行必须随每份 benchmark 结果保存，至少记录 `CUDA_VISIBLE_DEVICES`、`card_count`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`VIRTUAL_KV`、global block 数、逐设备 physical backing 和 `PORT`。受控模型对比要求资源一致；实际部署拓扑对比则必须明确披露资源差异。本次历史结果属于后者：35B 使用双卡 `0,1`，9B 使用单卡 `6`。

验收门槛：

1. `/health` 返回 HTTP 200，`/v1/models` 包含 `qwen3_6_35b_a3b_merak`。
2. server log 明确出现 HMONNX V2、auto-offload device mapping、HMONNX CUDA Graph interpreter/capture；没有未恢复的 OOM 或 traceback。
3. smoke completion 有非空正文，且服务 usage 的 `completion_tokens > 1`。
4. 同时发起 4 个 HTTP completion 时，EngineCore 以 `MAX_NUM_SEQS=1` 排队串行处理；四个响应均正确、prompt/输出互不串扰且服务无异常。该项只验证排队边界，不计入性能数据。
5. 最少卡数优先：单卡通过则不扩卡；若失败，保留明确 OOM 证据后依次用两卡、三卡通过同一组门槛。
6. 2k benchmark 至少 2 轮 warmup + 5 轮 measured，JSONL 完整，summary 含实际 prompt tokens、TTFT 与 decode median/p95；每轮 `finish_reason=length`、`completion_tokens=max_tokens`。
7. 35B-A3B 的 `decode_tok_s.median` 高于相同请求协议的 9B 实际部署对照；否则部署可用但服务吞吐目标未达标。由于两者卡数不同，即使高于对照也不能声称这是同资源模型性能优势。
8. 256k 服务能完成精确 256000 prompt tokens + 64 completion tokens 的请求，usage、本地计数和 target 三者严格相等，过程中无 OOM/崩溃。

### 2026-07-10 历史实测结果（新 planner / exact backing 之前）

以下数据保留用于回归对照，来自旧的显式 `CPU_KV=48` virtual budget 和统一页稠密 backing，不能当作当前默认容量或物理内存公式。历史服务使用物理 GPU `0,1`、端口 `18035`、`MAX_MODEL_LEN=262144`、`MAX_NUM_SEQS=1` 和 `CPU_KV=48` GiB。`/health` 返回 HTTP 200，`/v1/models` 披露 `max_model_len=262144`，vLLM 分配的 KV 容量为 295847 tokens。服务稳定态显存约为 GPU0 66989 MiB、GPU1 43251 MiB。

HMONNX 日志分别确认 prefill/decode 的 stage 0 (`cuda:0`) 与 stage 1 (`cuda:1`) 完成 CUDA Graph capture 和 replay，未出现 OOM、traceback 或未恢复错误。功能验收 7/7 通过（普通 completion、固定答案、required tool call、四个串行城市请求）；四个并发 HTTP 请求 4/4 正确，并按 `MAX_NUM_SEQS=1` 排队完成，无串扰。10 分钟稳定性检查共 30 次请求，30/30 HTTP 200 且语义检查通过，结束时 `/health` 仍为 200。

旧统一页稠密 backing 下的单卡 256k 对照曾在 KV 分配阶段 OOM：模型已占用约 78.25 GiB、仅余约 1020.94 MiB，而下一次分配需要 4.80 GiB。该结果解释了当时为何保留两卡生产拓扑，但不能否定当前实际大小 cache backing 的单卡能力；当前单卡仍需按本节门槛重新验收。

| 场景 | visible GPUs/card count | max len / max seqs / CPU KV GiB | prompt → completion | TTFT | decode | 结论 |
|---|---|---|---|---|---|---|
| 35B-A3B 短 prompt 热态 | `0,1` / 2 | `2048 / 1 / 8` | `16 → 256` | 单轮探针，不作正式统计 | `10.5056 tok/s` | 高于 9B 单卡实际部署的 `10.1199 tok/s`；拓扑不同 |
| 35B-A3B 正式 1800（修复前基线） | `0,1` / 2 | `262144 / 1 / 48` | `1800 → 256`，2 warmup + 5 measured | median `19.0733 s` | median `7.3265 tok/s` | 低于 9B；定位到 serial request 间 decode block-table bucket 收缩导致重复 graph capture |
| 35B-A3B 正式 1800（修复后最终） | `0,1` / 2 | `262144 / 1 / 48` | `1800 → 256`，2 warmup + 5 measured，均为 `length` | median `2.624489 s`，p95 `2.625844 s` | median `11.260519 tok/s`，p95 `11.265927 tok/s` | 相同请求协议下为 9B 的 `1.3414x`；35B 是双卡，不能称同资源模型优势 |
| Qwen3.5-9B 正式 1800 | GPU6 / 1 | `262144 / 1 / 48` | `1800 → 256`，2 warmup + 5 measured | median `14.0163 s` | median `8.3945 tok/s` | 单卡实际部署对照；请求 shape/warmup/repeat 相同，卡数不同 |
| 正式 8000 / 16000 | 8000 中途停止；16000 未执行 | 同上 | `8000/16000 → 256` | 无有效统计 | 无有效统计 | 用户接受当前结果后停止继续 benchmark/调优，未形成完整 measured 结果或 summary |
| 精确 256000 prompt | `0,1` / 2 | `262144 / 1 / 48` | `256000 → 64` | 未测试 | 未测试 | 只验证了 256k 服务配置、KV 容量和短请求稳定性，不能声称端到端 256000-token 验收通过 |

修复后正式 1800 结果保存在 `tmp/qwen36_35b_dual_256k_tuned_20260710_163650/formal_tuned_35b_1800.jsonl`，包含 5 个 measured run 和完整 summary；修复前 35B 与 9B 对照分别保存在 `tmp/qwen36_35b_dual_256k_20260710_155733/formal_35b_1800.jsonl` 与 `formal_9b_1800.jsonl`。最终结果证明“两卡 256k 配置可部署并稳定服务串行请求”，且在**相同请求协议、不同实际部署拓扑**下 35B 正式 1800 decode 吞吐高于 9B；它不证明同资源模型性能优势，也不证明精确 256000-token 请求已完成。8000 中途停止且没有有效 summary，16000 与精确 256000 prompt 未执行，均保留为明确的后续验证风险。

## 8. 图像输入与跨 prefill chunk 验证

Qwen3.6-35B-A3B 的 HF 源架构仍沿用 Qwen3.5 multimodal contract。vllm-merak 根据导出 metadata 选择实际 loader：

- 根级 `visual_config.hmonnx` 存在且文件有效：加载 `Qwen3_5MerakForCausalLM`，支持图像。
- 根级 `visual_config` 缺失：加载 `Qwen3_5MerakTextForCausalLM`，不构造 visual runtime，继续支持纯文本服务。
- metadata 声明了 visual HMONNX 但文件不存在：启动失败，不静默退化为纯文本。

模型层不设置固定图片数量上限，同一请求可以按消息顺序输入多张图片；实际可用数量受上下文长度、视觉编码开销和显存约束。vLLM 当前部署默认每种模态最多 999 项，生产环境建议按资源增加例如 `--limit-mm-per-prompt '{"image": 8}'` 的服务级上限。视频仍不支持。visual HMONNX 必须位于 GPU，代码会在启动期校验 `visual.device`；日志验收应同时看到 visual ONNX 进入 HMONNX V2，以及 auto-offload 使用 CUDA：

```text
.../visual/...onnx through HMONNXInferenceV2
Auto-offload device mapping completed. Devices used: {'cuda:0'}
```

可用下面的 OpenAI 请求验证当前 `18307` 服务；示例把文本放在图片前，顺序无需调整：

```bash
IMAGE=/path/to/image.jpg python - <<'PY'
import base64, mimetypes, os
from openai import OpenAI

path = os.environ["IMAGE"]
mime = mimetypes.guess_type(path)[0] or "image/jpeg"
data = base64.b64encode(open(path, "rb").read()).decode()
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18307/v1")
response = client.chat.completions.create(
    model="qwen3_6_35b_a3b_w4a16",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "直接用一句话描述图片。"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
    }],
    max_tokens=256,
    temperature=0,
)
print(response.choices[0].message.content)
PY
```

图片 placeholder 不受“必须位于第一个 prefill chunk”限制。2026-07-14 的 35B W4A16 单卡实测中，长文本在图片之前，服务返回 HTTP 200；请求共 1668 个 prompt tokens，image token 位于 1449–1635，187 个 image tokens 实际跨 HMONNX prefill chunk 5 和 6，模型正确描述了图片内容。实现按当前 chunk 中的 placeholder 数量切分 image embeddings，并把 vLLM 生成的 request-local 三轴 M-RoPE positions 路由给对应 HMONNX chunk，因此不依赖首 chunk 的可变 rope 状态。

多图沿用同一条通用路径：visual encoder 按消息中的图片顺序产出 embedding tuple，合并后保持 placeholder 顺序；prefill router 只按每个 chunk 实际包含的 image token 数连续消费 embedding rows，因此单张图跨 chunk、多张图分布在不同 chunk，以及前一张结束与后一张开始落在同一 chunk 都不需要特殊分支。

移除模型级单图限制并重启服务后，2026-07-14 又完成 35B W4A16 单卡双图实测：一个请求依次传入白底红色正方形和白底蓝色圆形，服务返回 HTTP 200、`finish_reason=stop`，正文按顺序正确描述两张图片；usage 为 416 prompt tokens、222 completion tokens，请求后 `/health` 仍为 HTTP 200，`/v1/models` 仍只暴露原模型 ID `qwen3_6_35b_a3b_w4a16`。日志保存在 `tmp/qwen36_35b_w4a16_multi_image_20260714.log`。

Cherry Studio 对自定义 OpenAI 模型主要依赖客户端模型能力设置，而标准 `/v1/models` 没有统一的视觉能力字段。当前服务继续只暴露 `qwen3_6_35b_a3b_w4a16`，在 Cherry Studio 中手动为该模型勾选“图像/视觉识别”即可；无需服务端别名。
