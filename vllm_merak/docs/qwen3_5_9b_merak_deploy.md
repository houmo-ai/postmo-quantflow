# Qwen3.5 9B Merak/hmonnx 模型部署教程

本文记录如何在 `vllm` conda 环境中，用 GPU 6 部署以下 Merak/hmonnx 导出模型，并测试普通对话、Phase 1 串行排队边界和工具调用（tool call）。

```bash
/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260707
```

> 该模型是 hybrid 结构（full-attention + linear-attention）。当前稀疏 Linear checkpoint store 属于 Phase 1：固定 `MAX_NUM_SEQS=1`、关闭 prefix cache，不支持 service-level preemption，也不宣称并发 forward。多个 HTTP 请求可以由 EngineCore 排队串行执行，但不能把排队成功解释成 `max_num_seqs>1` 已受支持。

## 1. 进入环境

```bash
conda activate vllm
cd /data01/home/yujy/work/xh2modelzoo/vllm_merak
export MODEL_DIR=/data01/home/yujy/work/xh2modelzoo/work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260707
```

## 2. 检查并补充 Merak 配置

导出目录需要有 `merak_config.json`，没有就从仓库模板复制：

```bash
cp -n data/merak_config.json "$MODEL_DIR/merak_config.json"
```

检查关键文件齐全：

```bash
ls "$MODEL_DIR"/merak_config.json "$MODEL_DIR"/golden_meta_info.json \
   "$MODEL_DIR"/hf_config/config.json
ls -d "$MODEL_DIR"/prefill "$MODEL_DIR"/decode "$MODEL_DIR"/visual_m*
```

图像输入能力由导出 metadata 自动判定，不依赖模型名称猜测。下面的检查应输出一个存在的 visual HMONNX 文件；若根级 `visual_config` 缺失，服务会按纯文本模型加载，若声明的 `hmonnx` 文件不存在则启动直接失败：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["MODEL_DIR"])
meta = json.loads((root / "golden_meta_info.json").read_text())
visual = meta.get("visual_config")
assert isinstance(visual, dict), "export is text-only: visual_config is absent"
path = Path(visual["hmonnx"])
path = path if path.is_absolute() else root / path
assert path.is_file(), path
print(path)
PY
```

> 注意：`merak_config.json` 里的 `enable_page_attention` 即使是 `false` 也没关系。hybrid 模型（本模型）会在加载时自动强制开启 paged full-attention KV cache——这是 Full Attention 请求状态隔离的前提，配置里的 false 会被忽略；它不改变 Phase 1 的单请求 forward 限制。

## 3. 安装 vllm-merak 插件

```bash
uv pip install -v -e . --no-deps
```

`vllm_merak` 是 editable 安装，改源码直接生效。验证可导入：

```bash
python - <<'PY'
import vllm, vllm_merak, xhmodel_merak, xhquant
print("vllm", vllm.__version__, "| vllm_merak / xhmodel_merak / xhquant ok")
PY
```

## 4. 启动 OpenAI API Server

推荐通过仓库启动脚本使用 GPU 6、监听 `18006`，模型别名 `qwen3_5_9b_merak`：

```bash
CUDA_VISIBLE_DEVICES=6 \
PORT=18006 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
bash examples/vllm_server_start.sh "$MODEL_DIR"
```

脚本在未设置容量覆盖时使用确定性 planner：2K smoke 会按实际几何规划小容量；`MAX_MODEL_LEN=262144` 时默认规划 4304 个全局 block（4099 最小需求加 5% headroom）。`VLLM_CPU_KVCACHE_SPACE` 仍可作为单位为 GiB 的**显式 virtual scheduler budget**；它不是实际物理 cache backing，也不再是推荐默认值。兼容别名 `CPU_KV` 会映射到标准变量；若两者同时设置且数值不同，脚本会拒绝启动。

参数说明（Merak 专属 + 关键项）：

| 参数 | 作用 |
|------|------|
| `--config-format merak_llm` / `--load-format merak_llm` | 用 Merak 的配置/权重加载路径（必需） |
| `--use-v2` | 启用 HMONNX V2 运行时 |
| `--auto-offload` | HMONNX 自动把权重放到 GPU（cuda:0 = 物理 GPU 6） |
| `--cuda-graph` | 启用 HMONNX CUDA Graph（需与 `--use-v2` 同用） |
| `--reasoning-parser qwen3` | 让 vLLM 按 Qwen3/Qwen3.5 的 `<think>` 协议拆分 thinking，避免推理文本混入普通 `content`；不要用它替代 `--tool-call-parser` |
| `--max-num-seqs 1` | Phase 1 硬限制；启动脚本拒绝其他值，多个 HTTP 请求只能排队串行执行 |
| `--enforce-eager` | 显式保持 vLLM 外层 eager；当前 Merak 的 PageAttention metadata 不支持 vLLM CUDA Graph，但这不会关闭 HMONNX CUDA Graph |
| `--enable-auto-tool-choice` | **工具调用必需**：允许模型自动决定是否调用工具 |
| `--tool-call-parser qwen3_xml` | **工具调用必需**：Qwen3.5 的 tool call 解析器（`qwen3_coder` 亦可，两者实测等价） |
| `--generation-config vllm` | 使用 vLLM 默认生成配置，避免模型仓库 generation_config 覆盖服务侧采样/长度行为；Cherry Studio 链路按 AgentWorld-style 保留该参数 |

> `--auto-offload` 和 `--cuda-graph` 一起用时，**必须**带 `--use-v2`，否则启动脚本会直接报错退出。
>
> 当前 Merak 被注册为 vLLM CPU backend，且 PageAttention metadata 声明不支持 vLLM CUDA Graph；因此需传 `--enforce-eager`，明确保持 vLLM 外层 eager。这不会关闭 HMONNX 的 CUDA Graph：`--cuda-graph` 仍会让 HMONNX V2 使用 `CudaGraphAutoOffLoadGraphModuleInterpreter`。不要把 vLLM 的 `cudagraph_mode: NONE` 误判成 HMONNX CUDA Graph 未生效。
>
> 上面的 `--max-model-len 2048` 是部署 smoke / 工具调用 / 串行排队验证配置，目的是快速确认服务链路和 HMONNX decode CUDA Graph 可用；它**不等价于 256k 长上下文性能验证**。256k 验证见第 9 节。

启动成功后日志出现：

```text
INFO ... Starting vLLM server on http://127.0.0.1:18006
INFO:     Application startup complete.
```

确认 HMONNX 走 GPU（`cuda:0` 对应物理 GPU 6，因为设了 `CUDA_VISIBLE_DEVICES=6`）：

```text
Auto-offload device mapping completed. Devices used: {'cuda:0'}
```

确认 HMONNX V2 + auto-offload + CUDA Graph 生效：

```text
Environment variable ENABLE_HMINFERENCE_V2 is enabled
Auto-offload CUDA Graph uses a single stage on cuda:0.
using CudaGraphAutoOffLoadGraphModuleInterpreter in HMONNXInferenceV2.
CUDA Graph capture completed
```

PageAttention prefill chunk 走 eager 是预期行为；只要后续 decode 出现 `CUDA Graph capture completed`，如下日志不是错误：

```text
CUDA Graph capture skipped for current input: RuntimeError: PageAttention CUDA graph capture is disabled for prefill chunks.
CUDA Graph capture skipped for current input: RuntimeError: PageAttention CUDA graph capture is disabled for padded prefill chunks.
```

如果看到下面这种永久禁用日志，则说明 PageAttention decode 没有被 CUDA Graph 加速，需要先修复 xhquant/xh2modelzoo/vllm-merak 的 PageAttention CUDA Graph 兼容补丁：

```text
CUDA Graph capture disabled: ... cudaErrorStreamCaptureUnsupported ...
CUDA Graph capture disabled: ... cudaErrorStreamCaptureInvalidated ...
```

模型较大（8GB+ initializer），加载约 2-4 分钟，请耐心等 `Application startup complete`。

### 4.1 完整请求与回复审计日志

服务自带的访问日志只记录 URL 和状态码。需要定位多轮对话、reasoning、工具调用或
流式分片问题时，可以启用 Merak 的独立审计中间件：

```bash
umask 077
AUDIT_LOG_MODE=both \
AUDIT_LOG_DIR=logs/audit \
CUDA_VISIBLE_DEVICES=6 \
PORT=18006 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
bash examples/vllm_server_start.sh "$MODEL_DIR"
```

审计日志不依赖 vLLM 的 logger，因此不需要设置 `VLLM_LOGGING_LEVEL=DEBUG`。
保持默认 `INFO` 即可，避免 EngineCore 每个 decode step 输出大量 DEBUG 信息。

`AUDIT_LOG_MODE` 支持：

| 值 | 行为 |
|---|---|
| `off` | 默认值，不创建审计文件 |
| `aggregate` | 每个请求完成后写一条 JSONL，保存原始请求和重组后的结构化回复 |
| `sse` | 保存请求开始、响应头、每个原始 ASGI/SSE body 分片和请求结束事件 |
| `both` | 同时保存聚合日志和原始分片日志，适合问题定位 |

日志按日期和 API Server 进程分文件：

```text
logs/audit/audit-aggregate-YYYYMMDD-p<PID>.jsonl
logs/audit/audit-sse-YYYYMMDD-p<PID>.jsonl
```

聚合日志包含原始请求体的 Base64、解析后的完整 JSON、重组后的 `content`、
`reasoning`、`tool_calls`、`finish_reason`、`usage`、状态码和耗时。SSE 日志把每个
原始响应 body 分片以 Base64 保存，可以逐字节还原服务实际发送的流式数据。
新生成的 SSE 记录还包含可直接阅读的 `body_text`；Base64 字段继续保留用于无损校验。
旧日志可以直接用查看器解码：

```bash
python -m vllm_merak.audit_log_viewer \
  logs/audit/audit-sse-20260716-p12345.jsonl | less
```

中间件只审计 `POST /v1/chat/completions`，不保存 Authorization/Cookie 等请求头，
不截断请求或回复，也不会缓存或重写 StreamingResponse。`both` 模式的日志增长较快，
问题定位结束后应恢复 `AUDIT_LOG_MODE=off`，并使用 `logrotate` 做轮转和压缩。

## 5. 健康检查

```bash
curl http://127.0.0.1:18006/health          # 期望 HTTP 200
curl http://127.0.0.1:18006/v1/models        # 期望看到 qwen3_5_9b_merak
```

## 6. 普通对话（不带工具）

```bash
curl http://127.0.0.1:18006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5_9b_merak",
    "messages": [{"role":"user","content":"你好呀，你有哪些能力。"}],
    "max_tokens": 32768,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

Python 方式：

```bash
python - <<'PY'
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18006/v1")
r = c.chat.completions.create(
    model="qwen3_5_9b_merak",
    messages=[{"role":"user","content":"你好，用一句话介绍北京。"}],
    max_tokens=64,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print("finish_reason:", r.choices[0].finish_reason)
print(r.choices[0].message.content)
PY
```

### 6.1 think / reasoning 配置

本模型是 reasoning 模型，默认 chat template 可能开启 think，正文会带 `Thinking Process:` / `<think>` 推理段。多轮对话时，如果客户端把这些推理段原样作为上一轮 assistant 内容回传，下一轮 prompt 会被污染，表现为开头出现 `Writing PDF thinking capsule`、重复上一轮“大模型/LLM”主题等内容。

Cherry Studio 等客户端通常不是直接从普通 `content` 里猜 Qwen3 的推理文本，而是优先读取 OpenAI 兼容响应里的 reasoning 字段（不同客户端/服务版本可能命名为 `reasoning` 或 `reasoning_content`）。服务侧应只做两件事：

1. 带 `--reasoning-parser qwen3`，让 vLLM 按 Qwen3 的 `<think>` / `</think>` 协议拆分 thinking；
2. 带 `--generation-config vllm`，保持与 AgentWorld 可用启动脚本一致，避免模型仓库 generation_config 改写服务侧生成参数。

不要在 `examples/vllm_chat_server.py` 里加响应重写 / alias 中间件。实测 Starlette `BaseHTTPMiddleware` 版本会让 vLLM API Server 卡在请求入口，纯 ASGI 重写虽然能通过 curl，但对 Cherry Studio 的“无正文输出”问题没有帮助；最终恢复正常的是 AgentWorld-style 启动参数（尤其 `--generation-config vllm`）加干净的 vLLM OpenAI 流式响应链路。

如果 `finish_reason=length` 且只有 reasoning、`content` 为空，说明 `max_tokens` 在 thinking 阶段耗尽了；需要提高 max tokens，或在确认客户端支持的前提下请求侧显式关闭 thinking。

不要在 Cherry Studio 这类客户端链路上把服务级默认值强制设成 `enable_thinking=false`：实测可能出现 prompt 正常消耗但 completion 只有 1 token、界面无正文输出。若客户端确认支持非 thinking 模式，可在 **请求侧** 显式关闭：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

如果需要显式开启 think，则改成：

```json
"chat_template_kwargs": {"enable_thinking": true}
```

Cherry Studio 联调时可以用下面两个现象快速定位是否走错链路：

- 异常链路：vLLM request log 里 prompt token 约 1500、`max_tokens` 很大但 output 只有 1，UI 看起来“没有返回”。
- 正常链路：同一条“你好呀，你有什么能力”请求约 15 个 prompt token、数百个 output token，并能在 Cherry Studio 正文区显示。

Python OpenAI 客户端通过 `extra_body` 传 vLLM 扩展参数：

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

注意：

- `stop: ["</think>"]` 只是遇到 think 结束标记时截断输出，**不等价于关闭 think**。
- 如果使用 xh2modelzoo 原生脚本 `examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py`，命令里带 `--think` 就是显式开启 think；想关闭就不要传 `--think`。
- `finish_reason=length` 且 `completion_tokens` 正好等于 `max_tokens`，说明被 `max_tokens` 截断；若仍开启 think，短 `max_tokens` 可能只截到思考过程，建议关闭 think 或适当调大 `max_tokens`。

### 6.2 图像输入

当前 Qwen3.5 Merak 服务支持 OpenAI Chat Completions 的 `image_url` 输入。约束如下：

- 模型层不设置固定图片数量上限，支持同一请求按消息顺序输入多张图片。实际可用数量仍受上下文长度、视觉编码开销和显存约束；vLLM 当前部署默认每种模态最多 999 项，生产环境建议按资源使用例如 `--limit-mm-per-prompt '{"image": 8}'` 设置服务级上限。
- 暂不支持视频输入。
- visual HMONNX 和语言 HMONNX 都必须运行在 GPU；即使视觉图能在 CPU 创建，服务也会拒绝启动，避免静默走 CPU。
- 图片占位 token **不要求位于第一个 prefill chunk**。长文本可以放在图片前面，图片 span 也可以跨多个 HMONNX prefill chunk；运行时会逐 chunk 路由对应 image embeddings，并使用 vLLM 已计算的 request-local M-RoPE positions。
- 是否支持图片由 `golden_meta_info.json` 根级 `visual_config.hmonnx` 及文件存在性决定；没有视觉导出的 Qwen3.5 产物仍可作为纯文本服务启动。

使用本地图片时，推荐转成 data URL，避免开放服务端任意本地路径：

```bash
IMAGE=/path/to/image.jpg python - <<'PY'
import base64, mimetypes, os
from openai import OpenAI

path = os.environ["IMAGE"]
mime = mimetypes.guess_type(path)[0] or "image/jpeg"
data = base64.b64encode(open(path, "rb").read()).decode()
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18006/v1")
response = client.chat.completions.create(
    model="qwen3_5_9b_merak",
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

`content` 数组中可以先写很长的 text，再写 `image_url`；不要为了适配 runtime 把图片强制移到 prompt 开头。实现会在启动期校验 `visual.device` 必须是 `cuda:*`，日志验收应同时看到 visual ONNX 被 HMONNX V2 加载，以及 auto-offload 使用 CUDA，例如：

```text
.../visual/...onnx through HMONNXInferenceV2
Auto-offload device mapping completed. Devices used: {'cuda:0'}
```

Cherry Studio 对自定义 OpenAI 模型不会通过试发图片自动探测能力，而且标准 `/v1/models` 响应没有统一的图像能力字段。保留服务原模型 ID，在 Cherry Studio 的模型设置里手动勾选“图像/视觉识别”即可，不需要增加模型别名。

2026-07-14 使用上述 `20260707` W4A8 导出在物理 GPU 0、端口 `18006` 完成实机 smoke：`/health` 为 HTTP 200，日志确认 visual ONNX 进入 HMONNX V2 且 auto-offload 使用 `cuda:0`，96×96 白底红色方块的 data URL 请求返回 HTTP 200，模型识别出“红色正方形”。该次请求因 `max_tokens=160` 在 reasoning 阶段达到 `finish_reason=length`；这也说明图片识别成功不等价于正文一定非空，正文为空时仍应按 6.1 节检查 thinking token 预算。

移除模型级单图限制后，同日又完成双图实机验证：一个请求依次传入白底红色正方形和白底蓝色圆形，服务返回 HTTP 200、`finish_reason=stop`，正文按顺序回答“第一张图片是一个红色的正方形”“第二张图片是一个蓝色的圆形”；usage 为 416 prompt tokens、242 completion tokens，请求后 `/health` 仍为 HTTP 200。日志保存在 `tmp/qwen35_9b_multi_image_verify_20260714.log`。

## 7. Phase 1 串行排队测试

保持 `MAX_NUM_SEQS=1`。可以让多个客户端同时提交 HTTP 请求，以验证 EngineCore 排队后每个请求均正确完成；这不是并发 forward，也不是提高 `--max-num-seqs` 的授权。

用 asyncio 同时发 4 个不同城市的请求：

```bash
python - <<'PY'
import asyncio, aiohttp, time
URL="http://127.0.0.1:18006/v1/chat/completions"
CITY={"北京":"Beijing","上海":"Shanghai","广州":"Guangzhou","深圳":"Shenzhen"}
async def ask(s, city):
    body={"model":"qwen3_5_9b_merak",
          "messages":[{"role":"user","content":f"用一句话介绍{city}。"}],
          "max_tokens":40,"temperature":0,
          "chat_template_kwargs":{"enable_thinking":False}}
    async with s.post(URL, json=body) as r: d=await r.json()
    return city, d["choices"][0]["message"]["content"]
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        t0=time.time()
        res=await asyncio.gather(*[ask(s,c) for c in CITY])
        print(f"4-way concurrent done in {round(time.time()-t0,1)}s")
        cross=0
        for city,out in res:
            leaked=[CITY[o] for o in CITY if o!=city and CITY[o] in out]
            if leaked: cross+=1
            print(f"[{city}] {'CROSSTALK'+str(leaked) if leaked else 'clean'}")
        print("VERDICT:", "PASS no crosstalk" if cross==0 else f"FAIL {cross}")
asyncio.run(main())
PY
```

历史实测输出（4 个 HTTP 请求排队完成，无串扰）：

```text
4-way concurrent done in 8.6s
[北京] clean
[上海] clean
[广州] clean
[深圳] clean
VERDICT: PASS no crosstalk
```

> Phase 1 不允许把 `--max-num-seqs` 调到 4 或 8。并发 forward、prefix cache 和 service-level preemption 需要后续阶段单独实现和验证。

## 8. 工具调用（tool call）

### 8.1 启动侧需要增加什么

工具调用**必须**在启动命令里加这两个参数（第 4 节命令已包含）：

```bash
--enable-auto-tool-choice        # 允许模型自动决定调用工具
--tool-call-parser qwen3_xml     # Qwen3.5 的 tool call 解析器（qwen3_coder 亦可）
```

缺了这两个，即使请求里带了 `tools`，模型也不会返回结构化的 `tool_calls`。

### 8.2 请求工具调用

验证工具通道建议用 `tool_choice="required"`（强制返回 tool call，最稳定）：

```bash
python - <<'PY'
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18006/v1")
tools=[{"type":"function","function":{
    "name":"get_weather","description":"查询指定城市的实时天气",
    "parameters":{"type":"object",
        "properties":{"city":{"type":"string","description":"城市名"}},
        "required":["city"]}}}]
r = c.chat.completions.create(
    model="qwen3_5_9b_merak",
    messages=[{"role":"user","content":"请查询上海今天的天气。"}],
    tools=tools, tool_choice="required", max_tokens=64, temperature=0)
ch=r.choices[0]
print("finish_reason:", ch.finish_reason)
print("tool name:", ch.message.tool_calls[0].function.name)
print("tool args:", ch.message.tool_calls[0].function.arguments)
PY
```

实测输出：

```text
tool name: get_weather
tool args: {"city": "上海"}
```

### 8.3 工具结果回填

拿到 tool 结果后，作为 `role:"tool"` 消息回填，让模型基于结果作答：

```bash
python - <<'PY'
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18006/v1")
msgs=[
 {"role":"system","content":"你会根据工具结果回答用户。"},
 {"role":"user","content":"上海今天的天气怎么样？"},
 {"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function",
   "function":{"name":"get_weather","arguments":'{"city":"上海"}'}}]},
 {"role":"tool","tool_call_id":"call_1","name":"get_weather",
  "content":'{"city":"上海","weather":"晴","temperature":"30℃"}'},
 {"role":"user","content":"请根据工具结果直接回答。"},
]
r = c.chat.completions.create(model="qwen3_5_9b_merak", messages=msgs,
    max_tokens=64, temperature=0)
print(r.choices[0].message.content)
PY
```

实测输出会引用工具返回的天气信息（晴 / 30℃）作答。

### 8.4 重要：用 `tool_choice="required"`，不要用 `"auto"`

实测发现（`qwen3_coder` 和 `qwen3_xml` 两个 parser 表现一致）：

| tool_choice | 结果 |
|---|---|
| `"required"` | ✅ 正常返回 `get_weather` + `{"city":"上海"}` |
| `"auto"` | ❌ 输出退化成 `?!?!?!?!...` 乱码，不返回 tool_calls |

带 `tools` 且用 `tool_choice="auto"` 时，tool schema 注入会让这个量化模型陷入 repetition collapse（连普通问题都乱码）。**这是模型层行为，与 parser 选择无关。**

结论：调用工具时**显式指定** `tool_choice="required"`，或指定具体函数：

```python
tool_choice={"type":"function","function":{"name":"get_weather"}}
```

不带 tools 的普通对话（第 6 节）不受影响，正常。

## 9. 256k 长上下文验证

第 4 节的 `--max-model-len 2048` 只用于快速 smoke。要验证 256k 长上下文，必须单独启动 256k 配置，并继续保持 Phase 1 单请求限制。默认 planner 会给一个 256K 请求规划 4304 个全局 block：

```bash
CUDA_VISIBLE_DEVICES=6 \
PORT=18006 \
MAX_MODEL_LEN=262144 \
MAX_NUM_SEQS=1 \
bash examples/vllm_server_start.sh "$MODEL_DIR"
```

验收要点：

1. `/health` 返回 HTTP 200，`/v1/models` 中 `max_model_len` 是 `262144`。
2. 日志确认 HMONNX V2/auto-offload/CUDA Graph 生效，尤其是 `CUDA Graph capture completed`。
3. PageAttention prefill skip 可以出现，但不能出现 `CUDA Graph capture disabled: ...` 永久禁用。
4. 用同一长 prompt 分别测试 `--cuda-graph` 开/关，比较 decode 阶段 token/s；256k 慢通常要看 decode 重放是否进入 CUDA Graph，而不是只看 prefill。

性能相关默认值（来自当前 xhquanttool 优化）：

- `HMOPS_CUDA_GRAPH_WARMUP_RUNS=1`、`HMOPS_CUDA_GRAPH_GRAPH_WARMUP_RUNS=2`：降低 auto-offload CUDA Graph cold capture 开销；需要保守验证时可手动调大。
- `HMOPS_XH2_NUM_SEGMENTS_3D=64`：长 decode split-KV 分段数默认值。
- 长 prefill 会在非 CUDA stream capture 时把 paged SEFP KV 纯重排成 contiguous SEFP layout，并复用 contiguous quantized attention 的 full-visible fast path。

当前 256k 性能状态：

| prompt tokens | TTFT | decode |
|---:|---:|---:|
| 1808 cold | 8.29s | 11.98 tok/s |
| 1808 hot | 3.41s | 11.81 tok/s |
| 8k | 14.70s | 13.22 tok/s |
| 16k | 29.38s | 12.87 tok/s |
| 64k | 147.86s | 10.86 tok/s |

> 这些结果说明服务链路和 decode 优化已生效，但还没有达到 2k token/s 的长 prefill 目标。当前模型导出的 prefill ONNX 静态 chunk 是 256 token，长 prompt 会按 chunk 数线性放大；继续冲 2k 应优先重新导出更大的 `prefill_chunk_length`，详见 `docs/qwen3_5_9b_merak_256k_perf_optimization.md`。
>
> 1808-token 数据已于 QTL-411 合并后的 256k 服务重新验证：使用 `bench_openai_stream.py --target-prompt-tokens 1808 --max-tokens 64`，连续同形状请求的热 TTFT 为 3.41s、decode 为约 11.81 tok/s。该小样本 decode 数值用于健康检查，不应替代多轮统计性能基准。

历史边界（旧的统一页稠密 backing，保留作对照，不是当前推荐配置）：

- `VLLM_CPU_KVCACHE_SPACE=16` 会启动失败：vLLM 报 262144 至少需要约 34.02GiB KV cache，16GiB 只能估算到约 123136 长度。
- `VLLM_CPU_KVCACHE_SPACE=48` 可启动，日志显示 `GPU KV cache size: 369,777 tokens`、`Maximum concurrency for 262,144 tokens per request: 1.41x`，`/v1/models` 返回 `max_model_len=262144`。

> 当前实现应优先使用 planner 默认值并核对日志中的 `global_blocks=4304`、virtual budget、逐设备 physical backing 和 padding saved。只有需要复现历史容量或进行受控诊断时，才显式设置例如 `VLLM_CPU_KVCACHE_SPACE=48`；该值是 virtual override，并不表示会物理分配 48 GiB。任何低于 4099 block 的 256K 计划都会在 cache allocation 前失败。

## 10. 停止服务

```bash
# 找主进程
ps -fu "$USER" | grep vllm_chat_server.py | grep -v grep
kill <PID>
# 若有 EngineCore 子进程残留
ps -fu "$USER" | grep 'VLLM::EngineCore' | grep -v grep
```

## 11. 常见问题

- **端口**：本教程用 `18006`，换端口改 `--port` 和客户端 URL 即可。
- **模型目录必须用 `--model`**：vLLM 0.24 里若只作 positional 参数可能误用默认模型。
- **`cuda:0` = 物理 GPU 6**：因为设了 `CUDA_VISIBLE_DEVICES=6`，进程内只看得到一张卡，编号 `cuda:0`。
- **`hsum extension` warning**：fallback 到 Python 实现，不影响服务启动和请求。
- **Phase 1 并发边界**：`--max-num-seqs` 必须为 1；多个 HTTP 请求只能排队。工具调用参数不改变这一限制。
- **工具调用必须用 `tool_choice="required"`**：`"auto"` 会让模型输出 `?!?!?` 乱码（见 8.4）。
- **Cherry Studio 显示不支持图片**：在该自定义模型的设置中手动启用“图像/视觉识别”；这不代表服务端视觉链路不可用。
- **图片必须放在最前面吗**：不需要。图片 placeholder 可以晚于首个 prefill chunk，也可以跨 chunk；保持符合业务语义的消息顺序即可。
- **图片请求被拒绝**：检查是否超过 `--limit-mm-per-prompt` 的部署级 image 上限、`golden_meta_info.json` 根级 `visual_config.hmonnx` 是否指向真实文件，以及 visual runtime 的设备是否为 `cuda:*`。
