# MiniMax-H3 Merak Workflow

该目录提供 MiniMax-H3 的 Merak HMONNX 导出入口和局部图推理 demo。默认配置先导出 `vae_encoder`，用于验证 HMONNX 图接入完整 pipeline；未导出的组件继续使用 FP 权重。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/minimaxh3/minimax_h3_workflow.py \
  --model-dir <model_dir> \
  --device cuda:0 \
  --output-dir work_dirs/minimax_h3_w8a8ssfp \
  --overwrite
```

选择多个组件时，组件名必须来自 `vae_encoder`、`vae_decoder`、`audio_vae_encoder`、`audio_vae_decoder`、`transformer`、`vision_encoder` 和 `text_encoder`：

```bash
PYTHONPATH=$PWD python examples_merak/aigc/minimaxh3/minimax_h3_workflow.py \
  --model-dir <model_dir> \
  --components vae_encoder vae_decoder \
  --device cuda:0 \
  --output-dir work_dirs/minimax_h3_vae \
  --overwrite
```

导出结果包含 `export_meta_info.json`。增加 `--dump-golden` 可以为导出的每个组件生成 HMONNX golden。

## HMONNX 推理

`vae_encoder`、`transformer`、`vae_decoder`、`audio_vae_encoder` 和 `audio_vae_decoder` 可独立替换；
`text_encoder` 与 `vision_encoder` 会绑定一起导出/安装。其他未导出的组件保持 FP：

导出 `text_encoder`/`vision_encoder` 时，会额外写出 host 侧轻量权重：

- `text_encoder/text_embeddings.safetensors`（`embed_tokens.weight`）
- `vision_encoder/vision_geometry.safetensors`（`pos_embed.weight`）

`hmonnx_demo.py` 检测到这些文件后，会跳过加载完整原始 `text_encoder`，改为构建仅含 embedding/geometry 的轻量 host，再安装 HMONNX TE/VE。TE ONNX ABI 仍从 `hidden_states` 开始，不变。

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/minimaxh3/hmonnx_demo.py \
  --model-dir <model_dir> \
  --export-dir work_dirs/minimax_h3_w8a8ssfp \
  --workflow fl2va \
  --prompt "a clay fox walking through a forest" \
  --image <reference_image> \
  --output work_dirs/minimax_h3_demo.mp4 \
  --device cuda:0
```

workflow 和 demo 默认都会尝试加载并融合仓库中的 MiniMax-H3 FL2V LoRA：

```text
data/models/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors
```

如需指定其他 LoRA，可以传入相同的参数：

```bash
--lora-path /path/to/minimax_h3_lora.safetensors \
--lora-alpha 8 \
--lora-scale 1.0
```

若本地没有 LoRA，默认会跳过并继续；需要强制存在时加 `--require-lora`，或用 `--no-lora` 显式关闭。

导出 `transformer` 时 LoRA 会在 HMONNX 导出前融合；运行 demo 时同一份 LoRA 会融合到未导出的 FP transformer，保证两条路径使用一致的 transformer 权重。

## 分辨率预设

`wrapper_demo.py` / `minimax_h3_workflow.py` 支持 `--resolution`：

| preset | 像素 | transformer packing (frames=124, text=512) |
| --- | --- | --- |
| `demo` | 608x352 | Nv=7942, Na=414, S=8868 |
| `480p` | 832x480 | Nv=14820, Na=414, S=15746 |
| `720p` | 1344x768 | Nv=38304, Na=414, S=39230 |
| `720p_16x9` | 1280x704 | Nv=33440, Na=414, S=34366 |

导出时 `--resolution` 只覆盖 `input_shapes.transformer`；VAE tile / text / audio 示例形状保持不变。

`transformer` 导出图不再接收 `timestep` / `token_tags` / `position_ids`。
demo 安装 HMONNX transformer 后，会在 host 侧用保留的 `time_proj` / `time_embedder` / `rope`
预计算 `temb` / `adaln_indices` / `rotary_cos` / `rotary_sin`，并要求 packing 提供
additive `encoder_attention_mask`（给 `token_refiner`）、host 展开后的
`packed_attention_mask`（给主 blocks），以及 `pack_gather_indices`
（`Concat([text,video,audio])` 后的 Gather 逆置换，避免图内 `ScatterElements`）；
安装时会立刻释放 FP transformer 主体，不再保留 runtime fallback。

验证单个或多个 HMONNX 部分时，使用 `--components`；也可用 `--components all`
启用导出目录中全部可用组件：

```bash
PYTHONPATH=$PWD python examples_merak/aigc/minimaxh3/hmonnx_demo.py \
  --model-dir <model_dir> \
  --export-dir work_dirs/minimax_h3_vae_encoder \
  --components vae_encoder \
  --workflow fl2va \
  --prompt "a clay fox walking through a forest" \
  --image <reference_image> \
  --output work_dirs/minimax_h3_vae_encoder.mp4 \
  --device cuda:0
```

验证 transformer 时请同时带上 text 路径，或只写 `transformer`（demo 会自动补齐）：

```bash
PYTHONPATH=$PWD python examples_merak/aigc/minimaxh3/hmonnx_demo.py \
  --model-dir <model_dir> \
  --export-dir work_dirs/minimax_h3_w8a8ssfp \
  --components text_encoder transformer \
  --workflow fl2va \
  --prompt "a clay fox walking through a forest" \
  --image <reference_image> \
  --output work_dirs/minimax_h3_transformer.mp4 \
  --device cuda:0
```

将 `--components vae_encoder` 替换为任一支持的组件名，或一次传入多个组件名；
若指定 `text_encoder` 或 `vision_encoder` 任一，demo 会同时启用两者。
若指定 `transformer`，demo 还会自动启用 `text_encoder`/`vision_encoder`，
以保证 fixed-length text embeds 与 `encoder_attention_mask` 可用。
不传该参数，或传入 `--components all` 时，demo 会启用导出目录中所有可用的 MiniMax-H3 HMONNX 组件。

`vision_encoder` 使用固定最大 patch 容量（默认 `vision_max_patches=4352`）：
processor 产出的真实 patch 不足时会右 pad 到该长度，并用 additive attention mask
屏蔽 pad token；不同分辨率图片只要 `prod(image_grid_thw) <= 4352` 都可走同一张图。
demo 侧无需手动 pad，`install_hmonnx_components` 会在 host 侧构造
`pixel_values / position_embeddings / rotary_cos / rotary_sin / attention_mask`，
并在返回 pipeline 前裁掉 pad 后的无效 image token。

`vae_encoder` 图只包含单个 tile 的 `quant_conv(encoder(tile))`。时间分块、空间 tiling、tile 拼接、token drop、posterior sampling 和 latent normalization 均保留在图外，因此该 demo 的输入输出契约与 FP pipeline 一致。