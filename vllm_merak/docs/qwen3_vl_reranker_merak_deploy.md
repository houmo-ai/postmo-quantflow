# Qwen3-VL-Reranker-2B 的 vLLM Merak 部署

本入口加载 ModelZoo 导出的视觉档位和 Prefill HMONNX，使用 vLLM 的 pooling / classify
执行相关性评分。它不加载原始 HF 语言权重，也不再次转换已导出的 yes-minus-no head。

## 支持范围

| 项目 | 状态 |
| --- | --- |
| 原始浮点权重来源的 W8 发布包 | verified |
| AutoRound W4G64 权重、W8 基础配置发布包 | verified；沿用图内实际 SSFP W4 权重 |
| 文本、图片、图文、Query 带图、多图 | verified |
| 视频、Query 带视频、多视频、图文视频混合 | verified；合成四帧视频的 W8 HMONNX 接入 smoke，未运行数据集评测 |
| 动态视觉、主特征和三路 DeepStack | verified；复用已有五档视觉 runtime |
| 32K、256-token HMONNX 分块、FlashAttention | verified |
| vLLM 分块调度、最终单 token、重复/换序请求 | verified |
| 离线 `LLM.score`、在线 `/rerank` | verified |
| 并发提交请求 | verified；单个引擎按 `max_num_seqs=1` 排队执行 |
| PageAttention、prefix caching、投机解码、TP/PP、多设备 offload | unsupported |
| 芯片真机性能 | blocked；本轮运行环境为 GPU HMONNX 后端 |

vLLM 的 `score_type="cross-encoder"` 对应 classify 任务。LAST pooler 选择最终有效 token，
FP32 classifier head 对导出图第 0 路 logit 做 sigmoid。图中其余 63 路仅用于硬件对齐，
不参与 softmax，也不作为分类标签。

## 环境

采用仓库 `runtime_profiles.toml` 的独立服务环境：Python 3.12、vLLM 0.23.x、
Torch 2.11.0+cu129、Transformers 5.13.0、ONNX 1.22.0、匹配的 xhquanttool 0.9.2。
插件安装及 ABI 检查见 [主 README](../README.md)。量化/导出仍使用原来的独立环境。

Reranker 媒体处理使用 ModelZoo 的 `qwen-vl-utils==0.0.14`。MP4 解码需其支持的
Decord 或 TorchCodec；本机验收使用 `decord==0.6.0`。服务环境中与 Torch 2.11
配套的 Torchvision 已移除旧 `read_video` API，因此不能仅依赖 qwen-vl-utils 的 Torchvision fallback：

```bash
python -m pip install --no-deps decord==0.6.0
```

从 ModelZoo 根目录运行以下命令，`python` / `vllm` 应来自服务环境。

## 准备发布包

```bash
PYTHONPATH="$PWD" python vllm_merak/examples/vllm_merak_reranker.py \
  --model work_dirs/qwen3_vl_reranker_2b_w8a8 --prepare-only
```

命令校验评分 ABI，并在发布包根目录补充 `merak_config.json`：

```json
{
  "architectures": ["Qwen3VLRerankerMerakForSequenceClassification"],
  "runner_type": "pooling",
  "num_labels": 1,
  "enable_page_attention": false,
  "xh_model": {
    "model_type": "hmonnx",
    "meta_info": "golden_meta_info.json"
  }
}
```

源 `hf_config/config.json` 的 architecture 保持 `Qwen3VLForConditionalGeneration`。
插件从 Reranker metadata 识别专用 loader，普通 Qwen3-VL 和 Embedding 不会被误识别为 Reranker。
已有服务配置与评分约定冲突时拒绝覆盖。

若旧发布包的 external_data 与 Golden 使用了硬链接，ONNX 1.22 会拒绝读取它们。
增加 `--materialize-external-data`，仅对有多个硬链接的外部权重做独立文件复制：

```bash
PYTHONPATH="$PWD" python vllm_merak/examples/vllm_merak_reranker.py \
  --model work_dirs/qwen3_vl_reranker_2b_w4a8 \
  --prepare-only --materialize-external-data
```

该操作逐文件核对 SHA-256 后原子替换，不改变图、权重字节或已登记的 Manifest 摘要。
Golden 保留原文件；物化后的文件增加磁盘占用。初版 AutoRound 包已完成此步骤；
其本机目录后来已删除，上述路径用于说明准备方法。

## 离线评分

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python vllm_merak/examples/vllm_merak_reranker.py \
  --model work_dirs/qwen3_vl_reranker_2b_w8a8
```

使用 AutoRound 包时替换 `--model` 即可。自定义输入使用 `--input-json`，格式与 ModelZoo 一致：

```json
{
  "query": {"text": "Find the matching scene."},
  "documents": [
    {"image": "/absolute/path/image.jpg", "text": "A beach scene."},
    {"text": "A train in the snow."}
  ]
}
```

Query 同样支持 `image`；图片字段支持路径列表。示例将本地图片编码成数据 URL，保持图片先于
文本的顺序。可传非空 `instruction`。输出包含按输入顺序排列的 `scores` 和降序 `ranking`。

## 视频输入

视频沿用原生 vLLM 的标准 `content` / `video_url` 输入。示例同时接受原有 ModelZoo
文本/图片格式和标准 content 数组；视频不需要新增模板，Query 和候选都可包含视频：

```json
{
  "query": "Find a moving red block.",
  "documents": [
    {"content": [{"type": "video_url", "video_url": {"url": "file:///absolute/path/clip.mp4"}}]}
  ]
}
```

上述结构可直接作为 `/rerank` 的业务字段，HTTP 请求再添加 `model`。
离线示例用 `--input-json` 读取它；本地 file:// 媒体还需设置
`--allowed-local-media-path /absolute/path`。服务端同样使用 vLLM 的
`--allowed-local-media-path`，只允许该目录下的媒体。也可使用服务端可访问的 HTTP URL
或视频 data URL。`--limit-mm-per-prompt '{"image":8,"video":2}'` 是本文的默认服务配置。

评分入口的解码、抽帧与 resize 复用官方 `qwen_vl_utils`；完成后通过原生 vLLM
Qwen3-VL processor 生成 patch、grid 和时间戳，关闭 HF 的二次 resize / 抽帧。
`RerankerMultiModalProcessor` 为展开后的视频保留官方外层视觉边界。M-RoPE 和语言侧
Prefill 图继续复用现有实现。
视觉编码复用 WeMM 的 `video_frame_grids`，将每个 temporal-patch 单元送入已有视觉档位，
保留主特征和三路 DeepStack，再按原视频顺序合并。没有修改上游 vLLM、WeMM 或 Qwen3.5 文件，
也没有为此次接入重导出图或重做量化。

若输入已抽帧，须保留原帧索引、FPS 和 `do_sample_frames` metadata，避免重复抽帧。
该支持属于 vLLM Merak 评分入口，ModelZoo 原生 `inference_model` 的视频输入未在本次扩展。

## 与官方预处理对齐

本轮以浮点评测会话使用的
[官方 revision 393e2978 的 Reranker wrapper](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_reranker.py)
为对照，沿用发布包中的源 HF processor/tokenizer。权重目录自带的旧 `scripts/` 版本
默认值不同，不能混合作为同一评测协议。

| 项目 | Merak 评分入口的默认处理 |
| --- | --- |
| 提示词与分隔符 | 官方 system 文本、默认 instruction、`<Instruct>: `、`<Query>:`、`\n<Document>:`；不添加额外空格或换行 |
| 同一 Query / Document 的内容顺序 | 视频 → 图片 → 文本；每类内部保留输入顺序 |
| 图片预算 | `min_pixels=4×32²`、`max_pixels=1800×32²`；由 qwen-vl-utils resize 并处理透明图片白底合成 |
| MP4 / 编码视频 | 官方 `fetch_video`，默认 `fps=1`、`max_frames=64`，保留原视频帧索引和 FPS |
| 已抽帧列表 | 超过 64 帧时按官方整数 linspace 采样，使用 `total_pixels=7680×32²`；帧列表默认 sample FPS 为 2 |
| patch / merge | `patch_size=16`、`temporal_patch_size=2`、`spatial_merge_size=2` |
| HF 后处理 | `do_resize=False`、`do_sample_frames=False`，避免重复处理 |
| 视频边界 | 保留最外层 start/end，内部各时间单元继续保留时间戳及其 start/end |
| 输出锚点 | 最后五个模板 token 完整保留，LAST 取最后有效位置；图内 yes-minus-no logit 后做 FP32 sigmoid |

原生 vLLM 会用时间单元序列替换整个 `<|vision_start|><|video_pad|><|vision_end|>`，
而 HF processor 保留外层 start/end。原生路径少两个 token，上一轮接口 smoke 没有检查出
这一点。本轮仅在 Reranker processor 的 prompt replacement 中补回外层边界，HF 文本路径、
token 路径及视觉 placeholder mask 均保持一致；不要求评测客户端再手工增加一层边界。
浮点评测会话的 `prepare_pair` 为原生 vLLM 手工补过一层外部边界。切换到 Merak 时，
该客户端补丁须按后端关闭，否则会重复增加边界；官方正常模板和官方媒体 metadata 继续保留。

可通过引擎级 `--mm-processor-kwargs` 或 HTTP 的 `mm_processor_kwargs` 显式覆盖
`min_pixels`、`max_pixels`、`total_pixels`、`fps` 和 `max_frames`；浮点对照须采用相同值。
离线 `LLM.score` 可通过 `PoolingParams.extra_kwargs["mm_processor_kwargs"]` 传递。
这些参数作用于 qwen-vl-utils；不支持让 HF 再次 resize / 抽帧，也不支持视频 token pruning。
本地文件目录和 HTTP 域名限制仍由 vLLM MediaConnector 检查。

现有视觉导出档位的上限仍为每张图片 / 每个视频时间单元 1536 个合并后视觉 token，
小于官方图片预算上限 1800。Merak 不再按 1536 覆盖官方图片预算；官方处理结果超出档位时
在 CPU 预处理阶段明确报错。要覆盖这部分大图，需补充更大的视觉导出档位。本次未重导出图。
所有视觉 token、时间戳、文本和模板合计仍受本次指定的 32K 总容量约束。

## 超长输入与评分末尾

原生 Reranker 和 vLLM 的离线 `LLM.score` / 在线 `/rerank` 共用保护式截断规则：
保留所有特殊 token，按原顺序保留预算内最靠前的普通 token，完整接回最后五个模板 token。
末尾必须为 `<|im_end|>\n<|im_start|>assistant\n`，本机 tokenizer 对应
`[151645, 198, 151644, 77091, 198]`；运行时通过 tokenizer 编码校验，不硬编码 ID。
缺失或错位时明确报错，避免 LAST pooler 读取错误位置。

32768 是包含模板和视觉 token 的总容量，正文最多占 32763，末尾五个 token 另占五个位置。
官方脚本的 `truncate_tokens_optimized` 接收正文预算，截断后还会拼接五个尾 token；
本实现采用相同保留策略，并在总容量内预留这五个位置。普通 token 无需截断时保持原序列。
特殊 token 和视觉 token 本身已超过容量时拒绝请求，不删除视觉占位符来凑长度。
视频的 placeholder 区间还包含时间戳文本，因此整段保留并计入容量，避免普通 token
截断删除时间戳后破坏视频位置与特征的对应关系。

vLLM 在图片占位符展开后截断，再同步更新 `mm_placeholders` 偏移，保留视觉特征及对应顺序。
部分 HF tokenizer 版本及 vLLM 的缓存视图中，`all_special_ids` 没有完整包含视觉 token，因此同时读取
`added_tokens_decoder` 的 special 标记。请求中的 `truncate_prompt_tokens` 可指定更小的总预算，
仍使用上述保留规则；不支持在评分末尾之后添加 prompt padding。

这一约束参考了 WeMM 的 pooling anchor 修复：
[processor](../../xhmodel_merak/xh_llm/models/wemm_embedding/processor.py) 会补齐、去重并移动
`<embedding>` 到最后有效位置；[回归用例](../../benchmark_test/ci_test_transformer5.16.0/test_wemm_embedding_contract.py)
覆盖缺失、重复、turn suffix 和左右 padding。WeMM 的专用 token 不能照搬到 Reranker，
Reranker 必须保留自己的 assistant 模板末尾。

## 在线服务

先执行准备步骤，再启动：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=merak \
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH="$PWD" \
vllm serve work_dirs/qwen3_vl_reranker_2b_w8a8 \
  --config-format merak_llm --load-format merak_llm --runner pooling \
  --served-model-name qwen3-vl-reranker-2b \
  --max-model-len 32768 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --limit-mm-per-prompt '{"image":8,"video":2}' \
  --chat-template vllm_merak/examples/qwen3_vl_reranker.jinja \
  --enforce-eager --trust-remote-code --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-reranker-2b","query":"What is the capital of France?","documents":["Paris is the capital of France.","Whales live in the ocean."]}'
```

多模态请求使用 vLLM 的 `content` 数组和 `image_url` 格式。服务当前限制一个活动请求，
多个 HTTP 请求可以并发提交，由调度器排队。默认关闭 prefix caching，KV 由 HMONNX 独占，
新请求清空缓存，续跑时校验 request ID 和 past length，避免候选间串状态。

## 调度及复用边界

视觉预处理和 M-RoPE 复用已有 Qwen3.5 Merak helper；DeepStack 合并复用上游 Qwen3-VL
的多尺度特征拆分。视觉编码仍调用 ModelZoo Reranker runtime，五档之间共享 GPU 权重。
Qwen3.5 模型代码和上游 vLLM 源码均未修改。

`MerakPoolingScheduler` 通过 vLLM 的 scheduler 扩展点接入，仅为 Reranker 选择。
vLLM 0.23 的生成调度器在续跑 Prefill 时为生成 token 预留一个位置；pooling 没有生成 token，
因此调度期间允许消费最后一个输入位置。输入校验、KV manager 和导出图的容量仍是 32768，
不会因此接受 32769-token 输入。

## 本机验证证据

工作区 `work_dirs/` 中保存以下记录，不将权重或日志提交 Git：

- 官方输入对齐：`reranker_official_alignment_red.log` 为新增回归实现前的失败记录；
  `reranker_official_ci.log` 为最终影响范围回归，188 passed，其中本模型 33 项。
  初轮定向记录 `reranker_official_alignment_contracts.log` 及最终 CI 覆盖真实 HF/vLLM processor 的文本和 token
  两条视频展开路径、外层边界、patch tensor、M-RoPE、MP4 抽帧和帧列表、RGB/RGBA/灰度图片。
  `reranker_official_real_video_inputs.json` / `.log` 在 CPU 上复核另一会话同一条 MSR-VTT
  预检样本：原生路径 2798 tokens，修正后 2800 tokens，与官方保存的 IDs 完全一致；
  patch、grid、M-RoPE 位置和 delta 也完全一致，没有加载浮点权重。
  `reranker_official_smoke.json` / `.log` 为 W8 HMONNX 的八组合成输入验收；其中七组
  未截断输入与官方 CPU 预处理逐 token 一致，另验证 512-token 视频保护式截断与重复请求。
  `reranker_official_http.json` / `.log` 验证默认视频、MP4、中文 instruction 与显式
  FPS/max_frames、奇数帧视频四组 HTTP 请求；均返回 200，默认分数与离线入口一致，
  指令/采样覆盖后的 token 数与官方 CPU 处理结果一致。
  `reranker_official_demo.log` 验证公开示例读取本地 MP4，分数与上述 W8 smoke 一致。
  `reranker_official_capacity_red.log` 记录超档位图像/视频在前端尚未拒绝的失败用例，
  最终 CI 验证其在 CPU 预处理阶段拒绝，避免进入 HMONNX 执行。
  旧 `reranker_video_*` 为修正官方外层边界和默认预处理之前的接口证据，不能作为官方输入对齐依据。
- 视频接入：`reranker_video_red.log` 记录实现前 3 项失败；
  `reranker_video_contracts.log` 为 20 passed，包含真实小 Qwen3-VL 视频视觉网络的时间单元
  拆分数值对照、三路 DeepStack、图像/视频顺序、时间戳保护和非视觉 token 合并检查。
  `reranker_video_smoke.json` / `.log` 记录五组 W8 HMONNX 短视频接入和重复请求隔离；
  `reranker_video_demo.log` 为公开示例读取本地合成 MP4 的结果；
  `reranker_video_http.json` / `.log` 为视频帧序列及 MP4 的 HTTP 入口验收。
  `reranker_video_ci.log` 为影响范围回归：216 passed、3 failed、2 deselected。
  三项失败均为已有 A16 YAML 的 `lm_head` 精度与旧测试固定 W8A8 断言不一致；
  这些 YAML 属于另一会话的改动，本次保留原样。两项未选用例涉及 AutoRound 和完整发布包
  构建/推理，不属于此次复用现有图的视频接入验证。
  本轮没有启动浮点模型或 MMEB/MMTEB/JinaVDR/ViDoRe 数据集评测，不将 smoke 当作精度结果。
- 截断修复：`reranker_truncation_contracts.log` 为 16 passed，新增保留末尾、特殊 token
  溢出、真实 cached tokenizer 的 special 元数据、视觉偏移和 native mask/type 同步测试。
  `reranker_truncation_wemm_anchor.log` 为 4 passed，复核所参考的 WeMM anchor 回归。
- `reranker_truncation_qwen3_vl_reranker_2b_w8a8.json`：W8 的短输入、恰好 32K、超长文本、
  超长 Query 后的图片、显式 513-token 双图共五组。截断后 input IDs 和五个尾 token 均一致，
  vLLM 与同一 HMONNX 原生路径分数误差为 0。官方保留算法的对照使用完整 special 元数据，
  并将正文预算设为总容量减五；不声称复现官方旧脚本在 HF 5 下漏保留视觉 token 的行为。
- `reranker_truncation_http.json`：三个长输入请求并发提交，全部 HTTP 200；有效长度分别为
  32768、32768、513，分数与原生路径误差为 0；临时服务验证后关闭。
- `reranker_truncation_export_ci_final.log`：207 passed、4 failed、23 skipped、1 deselected。
  本模型真实 tiny 构建、搬移、Golden 和推理已通过；fixture 原来的 WhitespaceSplit 会丢失
  末尾换行，现已添加换行/assistant 词表及保留换行的分词。剩余失败来自工作区原有三份
  A16 YAML 的 lm_head 精度与断言不符，以及环境缺少 Pi0.5 用例所需的 lerobot。
  AutoRound 用例因 datasets 4.8.5 不满足 3.6.0 门禁而未复跑；初次失败记录在
  `reranker_truncation_export_ci.log`。服务环境 CI 为 212 passed、5 failed，其中两个构建
  用例不满足导出依赖，另三个为上述 A16 配置。W4 原包已删除，本轮截断修复未复测 W4。
- 以下 `reranker_merak_*` 为初版接入证据，W4 结果对应当时尚未删除的发布包。
- `reranker_merak_qwen3_vl_reranker_2b_w8a8.json`、
  `reranker_merak_qwen3_vl_reranker_2b_autoround4_w8a8.json`：每种精度 8 组、11 对样本，
  含 Query 图片、多图、竖图、513-token 和 32768-token 输入。
  与直接 HMONNX 相比，所有有效 input IDs 相同，分数最大绝对误差为 0；重复、换序请求也一致。
- `reranker_merak_gears_qwen3_vl_reranker_2b_w8a8.json`、
  `reranker_merak_gears_qwen3_vl_reranker_2b_w4a8.json`：补充 196、704、1536 档位，
  两种精度全部通过，input IDs 相同、分数最大误差为 0。结合前面的 96、384 档，覆盖全部五档。
- `reranker_merak_http.json`、`reranker_merak_w4_http.json`：每种精度并发提交 4 个文本/图片请求，
  全部 HTTP 200，按候选索引还原的分数与直接 HMONNX 完全一致。
- `reranker_merak_demo_final.log`：本文公开离线示例执行通过。
- `reranker_merak_red.log`：新注册、DeepStack 和 pooler 测试实现前失败。
- `reranker_merak_contracts_final.log`：7 passed，包含真实 LAST pooler、DeepStack 合并、
  metadata 路由、独占 cache、实际调度器的满容量边界及 ONNX 外部权重物化。
- `reranker_merak_ci_final.log`：162 passed、1 failed、22 errors；未通过项均属于 GLM5，
  服务环境 Transformers 5.13 缺少 `transformers.models.glm5_next`。
  `reranker_merak_glm_baseline.log` 在适配前插件快照中复现同一缺失模块。
- `reranker_merak_adjacent.log`：145 passed、1 failed；失败为已有 PageAttention
  `_get_context` SDK 接口差异，`reranker_merak_adjacent_baseline.log` 在适配前快照复现。
- 新增 Python 文件通过 Ruff check/format。共享文件既有的 31 项 Ruff 诊断保持原样，
  `reranker_merak_shared_ruff_baseline.json` 记录无新增诊断。

上述对照验证插件调度和输出语义；它不改变已有量化模型相对官方浮点模型的精度结论，
也不替代检索业务指标验收。

本轮 `ocr` 工具不可用，已人工审查注册、图 ABI、cache 所有权和评分路径。
