# WeMM Embedding 的 vLLM Merak 部署

## 原生浮点 WeMM

原生浮点模型通过插件包内的 `wemm` general plugin 加载，不需要修改上游 vLLM
源码，也不需要改写官方 checkpoint 的 `config.json`：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/vllm_wemm_embedding.py \
  --model /path/to/official-wemm \
  --text "Represent this input for retrieval."
```

运行脚本会在导入 vLLM 前设置 `VLLM_PLUGINS=wemm`。该插件只在显式选择时将官方
配置中的 `Qwen3_5ForConditionalGeneration` 映射到 `WeMMEmbedding`，因此不会覆盖
普通 Qwen3.5 模型。图像使用 `--image`，视频使用 `--video` 和 `--video-frames`。
`--max-model-len` 必须容纳动态视觉 token；示例默认值为 4096。

2026-09-29 使用官方 WeMM-2B 对文本、动态图像和 4 帧视频完成真实 GPU smoke，
均返回 2048 维 FP32 归一化向量。与官方 Transformers 对同一输入的 embedding
余弦相似度分别为 `0.999795`、`0.999391` 和 `0.997787`。

## HMONNX 产物要求

WeMM HMONNX 导出目录需要包含 `golden_meta_info.json`、`merak_config.json`、
语言模型图、量化 embedding 和视觉图。新的 xh2modelzoo 导出流程会写入：

```json
{
  "architectures": ["WeMMEmbeddingMerakForEmbedding"],
  "runner_type": "pooling",
  "config_format": "merak_llm",
  "load_format": "merak_llm"
}
```

vLLM Merak 会读取 `golden_meta_info.json` 中的
`model_config.model_type=WeMMEmbedding`，据此选择 WeMM 专用 loader。`runner_type`
由 xh2modelzoo 导出 workflow 显式写入，vLLM Merak 配置解析器不再为 WeMM
补写该字段。旧产物如果缺少该字段，需要重新导出，或者手动补上
`"runner_type": "pooling"`。

运行时还会校验以下 embedding 合约：

- `embedding_output=hidden_states`
- `embedding_pooling=last_token`，且最后一个有效 token 必须是 `<embedding>`
- `embedding_normalize=true`
- `embedding_dimension` 是正整数

## 离线推理

先让当前 shell 能找到 vLLM 环境中的 `ninja`，再执行示例：

```bash
export PATH=/data01/home/jing.huang/miniconda3/envs/vllm/bin:$PATH

python examples/vllm_merak_embedding.py \
  --model /path/to/wemm_hmonnx_export \
  --text "A red car is parked beside a tree."
```

图像 embedding：

```bash
python examples/vllm_merak_embedding.py \
  --model /path/to/wemm_hmonnx_export \
  --image /path/to/image.jpg \
  --text "Represent this image for retrieval."
```

视频 embedding：

```bash
python examples/vllm_merak_embedding.py \
  --model /path/to/wemm_hmonnx_export \
  --video /path/to/video.mp4 \
  --video-frames 4 \
  --text "Represent this video for retrieval."
```

示例使用 `runner="auto"`。vLLM 从产物中的 `runner_type="pooling"` 选择 pooling
runner，调用 `LLM.embed()` 返回最后一个 token 的 L2 归一化 FP32 向量。自定义
预分词路径必须保留最后的 `<embedding>` token；WeMM 必须池化该 token 的 hidden
state，同时不要在 embedding 的 chat template 中添加 generation prompt。

## 图像与视频处理

图像保留原始宽高比进入 Qwen3.5 视觉预处理，并根据实际 patch token 数选择能够
容纳输入的最小视觉档位。当前五档产物支持 96、196、384、704、1536 个合并后
视觉 token。

视频沿用 Qwen3 VL 的时间戳 prompt 格式。每个时间单元会展开成独立的视觉 span，
WeMM loader 再根据 `video_grid_thw` 把这些 span 归并回原视频，并逐时间单元调用
动态视觉图。传入 vLLM 的原始视频数据必须附带 metadata；示例中的
`VideoMediaIO` 会同时生成帧数组和 metadata。

图像或视频 token 加上文本 token 后必须小于 `--max-model-len`。增加
`--video-frames` 会提高视频覆盖度，也会增加视觉 token 数和推理时间。

## 调度与缓存

WeMM 导出必须在 `merak_config.json` 中设置 `enable_page_attention: true`。
WeMM loader 继承 Qwen3.5 的 `paged` hybrid cache 策略：全注意力层使用
PageAttention key/value cache，线性注意力层使用 convolution cache 和 recurrent
state。vLLM 负责请求输入、视觉预处理、cache 调度、pooling 输出和离线 API；
HMONNX runtime 执行量化语言图与视觉图。

当前 Merak 平台仍要求包含线性注意力状态的 hybrid 模型设置 `max_num_seqs=1`，
因此示例保留该参数。这个限制来自通用 hybrid linear-state 调度约束，不再由 WeMM
专属的连续 KV cache 策略引入。

Merak 在 vLLM 中注册为 CPU 类型平台，但 HMONNX 的实际输入和 cache backing 位于
CUDA。当前环境的 CPU 版 vLLM 不提供 Triton CUDA slot-mapping kernel；如果按
tensor 是否位于 CUDA 来选择上游实现，会调用不可用的 Triton 函数。Merak runner
使用等价的 PyTorch 运算生成 slot mapping，使这种 CPU 调度、CUDA HMONNX 执行的
组合能够工作。

普通运行时在 HMONNX session 构造时固定到 Merak runner 的主 GPU，避免多张 GPU
可见时 HMONNX 自动放置造成输入和索引跨设备。启用 HMONNX auto offload 时继续
使用 auto offload 自己的多设备映射。

## 已验证配置

开发验证使用 2B W4A8、2K context、五档动态视觉图产物，分别执行文本、图像和
4 帧视频。三类请求均得到 2048 维有限 FP32 向量，L2 范数误差小于 `2e-7`。
