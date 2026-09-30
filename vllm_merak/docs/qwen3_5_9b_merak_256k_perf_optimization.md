# Qwen3.5 9B Merak 256k 性能优化记录

本文记录为了让 `docs/qwen3_5_9b_merak_deploy.md` 中的 256k 上下文部署跑通并尽量接近 2k token/s，已经做过的改动、验证结果、哪些应保留、哪些已经清理，以及后续真正逼近目标还需要什么。

## 当前结论

- 256k 服务已经能稳定启动，`/v1/models` 返回 `max_model_len=262144`。
- CUDA Graph 不能关：关闭后 8k TTFT 从约 14.7s 退化到约 31.4s，decode 从约 13 tok/s 退化到约 4.3 tok/s。
- 本轮代码优化主要改善 **decode 长上下文路径** 和 **首轮 CUDA Graph capture 开销**；长 prefill 的根瓶颈仍然是当前导出的 prefill ONNX 静态 chunk 只有 256 token。
- 实测 64k TTFT 仍约 147.9s，说明当前代码侧优化还没有达到 2k token/s 目标。要继续逼近目标，优先方向应是重新导出更大 `prefill_chunk_length`（例如 512 / 1024 / 2048）或对 256-token prefill graph 做更深融合。

## 三个仓库改动盘点

### 1. `vllm-merak`

已提交：`95724b4 Reduce decode block-table overhead for long-context Merak`

必须保留：

- `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py`
  - 对 decode block table 做 power-of-two bucket，减少长上下文 decode 每步传入 HMONNX 的 block table 宽度。
  - bucket 变化时清 CUDA Graph，避免旧形状 graph 被错误复用。
- `tests/test_merak_model_runner_v1.py`
  - 覆盖 decode bucket 和 CUDA Graph clear 行为。

不再保留的尝试：

- prefill block table bucket 化曾经用 `VLLM_MERAK_PAGE_ATTENTION_BUCKET_PREFILL=1` 试验，第二个请求触发 CUDA illegal memory access，已回退，不应恢复。

### 2. `xhquanttool`

当前仍是未提交改动；提交前需按 `xhquant-commit` 规范提供 JIRA 号。

必须保留：

- `xhquant/backend/hmops/fused_ops/flash_attention/xh2/paged_sefp_flash_attention_xh2.py`
  - 新增真正的 paged SEFP Triton attention kernel，decode 不再先 gather 成连续 KV。
  - 长 prefill 在非 CUDA stream capture 时把 paged KV 纯重排成 contiguous SEFP layout，然后复用 contiguous quantized attention kernel；这是当前 8k/16k prefill 可接受的主要路径。
- `xhquant/backend/hmops/fused_ops/flash_attention/xh2/_flash_attention_xh2_triton.py`
  - `HMOPS_XH2_NUM_SEGMENTS_3D` 默认从 16 提到 64，用于长 decode split-KV。
  - contiguous quantized prefill 恢复 full-visible fast path：因本模型长 prefill 是 causal、无 sliding window、无 mm prefix，且 `max_seq_q >= 128 && max_seq_kv >= 8192` 时可用。
- `xhquant/xhonnxruntime/hmonnx_inference_v2.py`
  - 配置里的 auto-offload layer node 名称和当前 FX graph 不一致时，不再直接启动失败，而是 warning 后回退到 `infer_auto_device_map()`。
  - auto-offload CUDA Graph warmup 默认降为 `HMOPS_CUDA_GRAPH_WARMUP_RUNS=1`、`HMOPS_CUDA_GRAPH_GRAPH_WARMUP_RUNS=2`，降低 cold TTFT；仍可用环境变量调回更保守值。
- `tests/testing/xh2a/ops/test_flash_attention_qmodel.py`
  - 增加 paged decode split-KV、long prefill contiguous fast path 的数值回归测试。

已清理/不保留：

- `xhquant/xhonnxruntime/cuda_graph_execution_strategy.py` profiling：该文件不是当前 auto-offload V2 256k 热路径，已回退。
- `xhquant/xhonnxruntime/parsers/page_attention.py` profiling：只用于定位 `write_k/write_v/attn` 是否是瓶颈，证据表明不是主瓶颈，已回退。
- `hmonnx_inference_v2.py` 中的 HMONNX module profiler / CUDA Graph profiler：只用于诊断，默认关闭但会增加维护面，已清理，仅保留 warmup 优化。
- prefill paged split-KV 备用分支：默认会被 contiguous fast path 截获，当前没有收益证据，已删除，避免保留无证据备用路径。
- 基于 `inspect.getsource()` 的静态结构测试：过于脆弱，已删除；保留数值回归测试。
- `.gitignore` 的脏改动与本任务无关，不纳入本轮整理。

### 3. `xh2modelzoo`

当前工作区无新改动。

已依赖的既有导出/提交：

- 模型目录：`hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260707`
- 当前 prefill ONNX 输入形状仍是 `[1, 256, 4096]`，`golden_meta_info.json` 中 `prefill_chunk_length=256`。

## 关键验证数据

在 256k 启动配置、CUDA Graph 开启、单并发下：

| prompt tokens | TTFT | decode |
|---:|---:|---:|
| 1808 cold | 8.29s | 11.98 tok/s |
| 1808 hot | 3.41s | 11.81 tok/s |
| 8k | 14.70s | 13.22 tok/s |
| 16k | 29.38s | 12.87 tok/s |
| 64k | 147.86s | 10.86 tok/s |

关闭 CUDA Graph 的对照结果：

| prompt tokens | TTFT | decode |
|---:|---:|---:|
| 1808 hot | 8.04s | 4.34 tok/s |
| 8k | 31.43s | 4.30 tok/s |
| 16k | 61.82s | 4.37 tok/s |

CUDA Graph profile 诊断结论：

- 当前热 prefill 主要被一个 256-token prefill graph replay 卡住，单次 replay 约 400ms。
- 64k 需要约 250 个 256-token chunk，TTFT 线性放大，因此代码侧小修无法把 64k/256k prefill 推到 2k token/s。

QTL-411 合并后的 256k 服务已重新验证 1808-token 热请求（`max_tokens=64`）：TTFT 为 3.41s、decode 为约 11.81 tok/s；服务日志同时确认 HMONNX 使用 `CudaGraphAutoOffLoadGraphModuleInterpreter`。该值是一次 64-token 健康检查，后续性能回归应使用多轮统计。

## 复现实验命令

启动 256k：

```bash
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak
conda activate vllm
export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260707
CUDA_VISIBLE_DEVICES=6 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_CPU_KVCACHE_SPACE=48 \
python examples/vllm_chat_server.py \
  --model "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 18006 \
  --served-model-name qwen3_5_9b_merak \
  --trust-remote-code \
  --config-format merak_llm \
  --load-format merak_llm \
  --enforce-eager \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --use-v2 \
  --auto-offload \
  --cuda-graph \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

验证：

```bash
curl -sS -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:18006/health
python tmp/qwen35_merak_logs/bench_openai_stream.py --target-prompt-tokens 8000 --max-tokens 8
python tmp/qwen35_merak_logs/bench_openai_stream.py --target-prompt-tokens 16000 --max-tokens 8
python tmp/qwen35_merak_logs/bench_openai_stream.py --target-prompt-tokens 64000 --max-tokens 8
```

## 后续如果继续冲 2k token/s

1. 优先重新导出更大 `prefill_chunk_length` 的 HMONNX 模型。当前 256-token chunk 是最硬的线性瓶颈。
2. 如果不能重导出，则需要针对当前 prefill graph 的 5900+ ONNX node 做更大范围 fusion / kernel replacement；这已经不是 vLLM 调度层或 block table 小优化能解决的问题。
3. 保持 CUDA Graph 开启，且不要恢复 prefill block-table bucket 实验，除非先修复第二请求 CUDA illegal memory access。
