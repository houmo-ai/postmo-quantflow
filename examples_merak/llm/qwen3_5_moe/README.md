# Qwen3.5 MoE

Dense/MoE 已共用 [Qwen3.5 系列 workflow](../qwen3_5/readme.md)。
本目录保留兼容脚本，已有调用不用改；新任务使用
`examples_merak/llm/qwen3_5/qwen3_5_workflow.py`。

## 配置与量化

配置在 `configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/`：

- Qwen3.6-35B-A3B：`35b_a3b/qwen3_6_35b_a3b_full.yaml`；head W4/W5 分别选
  `qwen3_6_35b_a3b_full_head_w4.yaml` / `qwen3_6_35b_a3b_full_head_w5.yaml`，见下文。
- Qwen3.5-122B-A10B：普通版用 `122b_a10b/qwen3_5_122b_a10b_full.yaml`；
  MTP 版用同目录的 `qwen3_5_122b_a10b_full_mtp.yaml`，沿用主模型 W5 head 权重，默认生成 3 个 draft token。
- MTP/DFlash 选对应后缀；GPTQ 选 `*_gptq.yaml`。
- `ornith-1.0-35b_full.yaml` 是独立命名配置，不要用它替代 Qwen3.6 的发布名。

AutoRound 默认 `int_sym_xh`、n256/i400、bs8/ga1、G64。
Attention、shared expert 和 routed expert 的位宽看 YAML 的 `quant.moe`；
W8、FA16 等对照配置不因默认 dtype 变化而改变。
MTP 直接复用 Base 的量化目录，导出时加 `--export-from-quanted-model`。

## 自动 block 混合精度

Qwen3.5/Qwen3.6 MoE 可以在 `quant` 下添加 `auto_mixed_precision: true`。
推荐使用 KL 0.03 + Head W5 的完整配置
[qwen3_6_35b_a3b_full_head_w5_auto_mixed.yaml](../../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w5_auto_mixed.yaml)。
同一份配置、一次 workflow 会依次完成：

1. 32 条文本 + 64 条 RealWorldQA 短答案做轻量 RTN 搜索，按 block 确定 attention 和共享专家的 W4/W5/W8；非共享专家固定 W4。
2. 恢复浮点权重，按分配表用 AutoRound 校准主干（Pile 256 条、长度 2048、400 次迭代、batch size 8、seed 42）。
3. 回放已量化主干，使用其输出激活校准浮点 Head 为 W5，保存完整 checkpoint 并继续导出。

`quant.head.bits: 5` 和导出 `lm_head: w5a8h1_sefp` 已在同一 YAML 中配置，
无需中间 Head W4 模型或手工执行 head 补量化。`--quant-output-dir` 是包含主干、
Head W5 和分配表的最终目录，临时主干在流程结束后清理。

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.6-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w5_auto_mixed.yaml \
  --quant-output-dir work_dirs/Qwen3.6-35B-A3B_auto_mixed_head_w5 \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_auto_mixed_head_w5_export
```

Head W4 配置仍可选
[qwen3_6_35b_a3b_full_head_w4_auto_mixed.yaml](../../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4_auto_mixed.yaml)。
未配置或设为 `false` 时沿用原量化流程。需要调节搜索时改为字典：

```yaml
quant:
  method: autoround              # 也支持 gptq
  head: {bits: 5}                # export 的 lm_head 同步使用 W5
  auto_mixed_precision:
    candidates: [4, 5, 8]
    initial_bits: 5
    max_kl: 0.03                 # RTN 搜索 KL，nats / 有效预测 token
    max_ppl_increase: 0.02       # 搜索抽样 PPL 相对增幅，0.02 = 2%
    nsamples: 32
    seqlen: 512
    positions_per_sample: 64
    max_trials: 24
    seed: 42
    image_dataset: xh2modelzoo://data/calib_data/realworldqa_search_64.jsonl
    image_nsamples: 64
    image_seqlen: 2048
    image_max_pixels: 393216
  moe:
    expert_bits: 4              # 非共享专家固定配置，不参加搜索
    expert_down_bits: 4
```

每层有两个独立搜索组：attention 主投影与 shared expert 的 gate/up/down。
Full attention 的 q/k/v/o 同位宽；linear attention 的 qkv/z/out 同位宽。
router、shared_expert_gate、linear attention 的 a/b 小投影、卷积、norm、视觉及 head
不参加搜索，继续使用各自配置。35B-A3B 有 40 层、80 个搜索组；层数由模型读取。
W5 使用本地 8 bit 槽位、5 bit 数值范围，逻辑权重开销按 5 bit 计算。

生产端先以原生浮点模型为参考采集文本输出和投影输入，用对称分组 RTN 生成候选。
输入重建误差只用于排序；从全 W5 开始，每次实际运行整模，检查累计方案的输出 KL
与抽样 NLL/PPL。超阈值则按排序分批回退 W8（批量由层数和试验预算计算，
确保必要时能评估全 W8）。找到可行方案后，分批撤销不必要的 W8 回退，
仅对拒绝的批次二分细查，最后继续尝试逐组下降到 W5/W4。已评估的相同方案复用指标。
这是有预算的贪心搜索，不保证全局最优；预算用完仍无可行方案会报错，不继续量化。
选定分配后恢复全部浮点权重，再调用所选 AutoRound/GPTQ 完成最终校准和保存。

KL 使用相同输入历史上的 `KL(P_float || Q_candidate)`，全词表、温度 1、自然对数，
按有效预测 token 平均；排除 padding 和没有下一 token 的位置。
`positions_per_sample` 均匀抽取位置，报告的 PPL 是这些位置上的相对指标，不是全语料 PPL。
配置 `image_dataset` 后，文本与图像回答的 KL/PPL **分别**检查同一组阈值，必须都通过。
图像样本格式为 `{"image":"images/a.png","text":"问题","answer":"参考答案"}`，
图片路径相对 JSONL，也可使用本地绝对路径。也支持将 `image` 替换为 `image_base64`，
值为原图片文件字节的 Base64（不带 data URI 前缀）；两者只能提供一个。
可选 `image_sha256` 校验原始文件字节。默认 64 条 JSONL 已内嵌全部图片，拉取仓库即可使用，
不依赖旁边的图片目录或下载流程；需配套支持 `image_base64` 的 GPTQModel。
实际运行原生 processor、视觉编码和 M-RoPE；
缓存不变的视觉输入后回放完整语言模型。只评分回答 token，不评分图像占位符与问题。
支持 OCR、描述及问答，不要求选择题；可选 `options` 提供单 token 答案标签，
用于额外记录正确数和选项一致率，问题中须自行包含选项文本。
未设置 `image_dataset` 时仍可只搜索文本；显式提供后缺图片或格式错误会报错。
`image_seqlen` 限制完整图文序列长度，超长报错，不截断图像 token。
示例的 64 张 RealWorldQA 图片来自之前的候选筛选池，属于搜索数据，不能作为独立验收。
图片准备说明见 [校准数据](../../../../data/calib_data/README.md)。

报告的 `metrics.modalities` 保存两种模态各自的结果；顶层 KL/PPL 取较差的模态。
文本已失败的候选会跳过图像回放并记录 `skipped_modalities`；接受的候选必须两者齐全。
搜索前额外测一次全 W8 对照，不占 `max_trials` 候选预算。
默认阈值是筛选起点，需要按业务验证调整；它们**不承诺最终 checkpoint 的任务精度**。
搜索时未搜索的模块保持浮点；最终专家/head 的量化误差、A8、图片、长上下文和 decode
仍须单独验证。图像回答评分参与搜索，不代表全量多模态任务验收通过。
最终 AutoRound/GPTQ 校准仍由其独立的 `quant.calibration` 数据配置控制。

搜索默认复用量化数据源，也可设置 `auto_mixed_precision.dataset` 指向独立 JSONL
或 HF 文本数据集，`text_key` 默认 `text`；`dataset` 和 `image_dataset` 都支持 ModelZoo 资源 URI。
建议独立保留最终验收集，避免用搜索样本报告任务精度。搜索当前把原生模型放到单卡，
可用 `auto_mixed_precision.device: cuda:0` 指定；35B-A3B 建议使用 80 GB GPU。
这与最终校准的 `runtime.device_map` / `low_gpu_mem_usage` 分开。
首版不支持搜索已经打包的 checkpoint、rotation、磁盘 offload 或只量化部分层。

成功的 checkpoint 内包含 `auto_mixed_precision.json`（样本摘要、阈值、每次试验与分配）
和 `config.json -> quantization_config.meta.auto_mixed_precision`（静态逻辑位宽）。
head 补校准会保留它们。导出自动读取并校验实际 packed 张量形状与位宽配置，把结果
设置到每个投影的 Merak 量化配置；显式逐节点设置与产物冲突时直接报错。
全局 W8 和 `lm_head` 配置不需要手工展开成 40 层。

已有自动搜索产物可照常用 `--export-from-quanted-model`；不会再次搜索。
开启此字段却传入没有分配表的旧产物会报错，应先从浮点模型运行新 recipe。
失败或成功搜索的中间报告写到量化输出目录旁的
`<output_dir>.auto_mixed_precision_search.json`，便于检查失败原因。

## Qwen3.5 / Qwen3.6 视频推理

图片和视频共用已有 visual gear 图，无需重新导出或新增 `video_visual` 组件。
视频使用 checkpoint 原生 video processor 采样、生成时间戳和占位 token；
模型 runtime 逐时间组编码，并把视频特征送入 LLM。每组空间 token 数不能超过最大 gear。
`full` 和普通 `lora` 支持视频；MTP / DFlash 投机推理仍只接受文本。

```python
from xhmodel_merak.model_build import build_model
from xhmodel_merak.model_inference import inference_model

built = build_model(model_id="qwen3.6-35b-a3b", model_dir="/path/to/model",
                    output_dir="/path/to/new_artifact", device="cuda:0")
result = inference_model(artifact_dir=built.output_dir, device="cuda:0", inputs={
    "prompt": "描述这段视频。", "video": "/path/to/video.mp4", "max_new_tokens": 64,
})
print(result.output_text)
```

已有统一发布目录可直接运行：

```bash
python examples_merak/llm/qwen3_5_moe/qwen3_6_video.py \
  --artifact-dir /path/to/artifact --video /path/to/video.mp4 --prompt '描述视频。'
```

`video` 也允许 RGB 帧目录：按文件名排序，所有帧尺寸相同，时间轴按 2 FPS 解释。
帧目录保留所有输入帧；末组不足 `temporal_patch_size` 时由原生 processor 补齐末帧。
视频文件使用原生采样，并实际调用 TorchCodec 解码。PyTorch 2.8 对应
TorchCodec 0.6 / 0.7（推荐 `pip install torchcodec==0.7.0`），还需要 FFmpeg 共享库，
兼容关系见 [官方版本表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)。
文本、图片和帧目录不需要安装视频文件解码器。损坏文件或不兼容解码环境直接报错。
底层独立 visual 发布的运行元数据仍为 `visual_meta_info.json`；完整模型请使用统一发布目录。

## Qwen3.6-35B-A3B head W4 / W5

现有 Qwen3.5/Qwen3.6 YAML 和 W8 head 路径保持不变。只有导出 head 小于 8 bit
时要求校准；W8 不要求 GPTQModel 产物包含量化 head，继续使用导出侧的 RTN SEFP。

提供三份可直接使用的完整配置：

| 配置 | quant.head.bits | export head |
| --- | --- | --- |
| [qwen3_6_35b_a3b_full.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml) | 未配置 | `w8a8h1_sefp` |
| [qwen3_6_35b_a3b_full_head_w4.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml) | 4 | `w4a8h1_sefp` |
| [qwen3_6_35b_a3b_full_head_w5.yaml](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w5.yaml) | 5 | `w5a8h1_sefp` |

两份新配置基于 `full.yaml`，只增加 `quant.head.bits` 并调整 export 的 lm_head 位宽。
主体保持 AutoRound W4、attention/shared expert W8，使用 `int_sym_xh`、G64、
n256/s2048/i400；视觉和其他导出参数均沿用原配置。W5 head 使用本地 5 bit 数值、8 bit 槽位的 GPTQ 布局。
quant/export 的 head 位宽必须一致；低位宽导出缺少 `quant.head` 会提前报错。
同一份配置兼容原生 HF 整模量化和已有量化产物补 head，不需要 `scope`。

原生浮点 HF 输入照常运行完整 recipe。GPTQModel 内部先处理主体，再校准 head，
最终只保留指定输出目录。head 默认跟随主体算法，AutoRound 跟随 AutoRound、GPTQ
跟随 GPTQ；校准样本和迭代数沿用 quant 设置。遇到 tied weights，会先复制 head，
保持 embedding 数值不变，并在新 checkpoint 中将根配置/text_config 的
`tie_word_embeddings` 设为 false，避免重载时重新绑定。

已有 GPTQModel/AutoRound 产物沿用 `--export-from-quanted-model`。该参数跳过主体
量化，但保留 YAML 中的 head 要求和校准设置：

- 已量化且符合目标：直接复用传入目录，不读校准数据、不前向主体、不复制模型。
- head 仍为浮点：warning 提示补校准算法和输出路径，校准后保存同级完整 checkpoint，
  目录后缀为 `_head_w4` 或 `_head_w5`，本次导出自动消费新目录。
- 已打包但位宽/分组/格式不符：报错，避免默默将低位宽导出变成二次量化。

先执行 `conda activate xhquant_55`，从仓库根运行现有 CLI。`PYTHONPATH=.` 确保使用当前
工作树，避免 conda 环境的 editable 安装指向另一份仓库：

从原生 HF 模型开始量化并导出 W4 head：

```bash
PYTHONPATH=. python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.6-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml \
  --quant-output-dir work_dirs/Qwen3.6-35B-A3B_autoround_head_w4_n256_iter400 \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_head_w4_export
```

已有 GPTQModel/AutoRound 产物跳过主体量化并导出 W4 head：

```bash
PYTHONPATH=. python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/Qwen3.6-35B-A3B_autoround_int_sym_xh_n256_iter400 \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_head_w4_export
```

W5 使用相同命令，将配置改为 `qwen3_6_35b_a3b_full_head_w5.yaml`，输出目录的
`head_w4` 相应改为 `head_w5`。

若需要补校准，新模型路径为
`work_dirs/Qwen3.6-35B-A3B_autoround_int_sym_xh_n256_iter400_head_w4`。
后续将它直接作为 `--model-dir`，配合同一份 W4 head YAML 和跳过量化参数。
同名输出已存在时不会覆盖，请直接传入该新模型复用，或先整理已有目录。

补校准默认按产物生产信息选择算法；AutoRound 常写入 `quant_method=gptq`，应优先
看 provider/provenance，不能据此误判成 GPTQ 算法。补校准只替换 head 张量及相应
配置，保留主体、视觉/MTP、tokenizer 和其他辅助文件，并记录
`lm_head_quantization.json`。head 输入来自已量化主体的真实路由及 final norm。
当前消费契约为 G64、对称、无 activation ordering、int32 打包字、不启用 rotation。

## 122B 分层导出

122B 需要启用 placeholder 分层导出，降低峰值内存：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/Qwen3.5-122B-A10B_autoround_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/Qwen3.5-122B-A10B_export \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --quick-test
```

多卡并行导出可设 `XH2MODELZOO_EXPORT_WORKERS`，默认 1；可见 GPU 和机器内存
都要够用。多卡 golden 另外传 `--golden-device-map cuda:0 cuda:1`，
设备编号相对于 `CUDA_VISIBLE_DEVICES`。

122B MTP 的 `num_draft_tokens: 3` 对应主图验证长度 4；主模型和 MTP draft
从同一个补全 W5 head 与 W4 MTP 专家的 checkpoint 导出。已有主体量化目录若仍是
浮点 head、浮点 MTP，先按下文的补量化流程产生新目录；unified 的
`quant_checkpoint` 只接受完整产物，不会在复用时修改输入目录。
122B MTP 校准把主模型权重留在主机内存，逐 block 使用 `--device` 指定的 GPU
汇集校准样本的 hidden；主干回放结束后才把 MTP 层移到 GPU 做专家量化。

维护 placeholder wrapper 时，attention、GDR、MoE 子图返回值只能是 tensor
或扁平 tensor 容器，不能带 `None`，否则会破坏主图与子图的端口对应关系。

## 旧入口

本目录的生成、PPL、legacy meta 转换脚本继续保留。新产物直接使用
`golden_meta_info.json`，一般不需要转换 meta；调用方式见
[统一验证说明](../qwen3_5/readme.md#验证产物)。


## 共享 head 与 MTP 标准入口

主模型 `export.model.quant_scheme.nodes.lm_head` 决定主模型及 draft 的 head 位宽，
并与 `quant.head.bits` 校验一致。MTP/DFlash/DFlash2 不再填写
`spec_draft_head_weight_bits`；旧字段显式不同会报错。规则适用于 Qwen3.5 dense/MoE
及同架构的 Qwen3.6/3.8；Qwen3-Next 的 MTP 也使用同一规则。
既有 W8 标准配置仍继承 W8；使用 W5 时选择下面完整入口。

| Workflow 配置（35b_a3b 目录） | 主体 | 共享 head | MTP 主干 |
| --- | --- | --- | --- |
| `qwen3_6_35b_a3b_full_head_w5_mtp.yaml` | 非共享专家 W4，其余 W8 | AutoRound W5 | 非共享专家 AutoRound W4，其余 W8 |
| `qwen3_6_35b_a3b_full_head_w5_auto_mixed_mtp.yaml` | 非共享专家 W4，attention/shared expert 按 block 搜索 W4/W5/W8 | AutoRound W5 | 非共享专家 AutoRound W4，其余 W8 |

一次 workflow 依次完成 bit 分配（混合入口）、主体校准、在量化主体上校准 Head W5、
MTP 专家校准（启用 `quant.mtp` 时），再导出主图与 MTP 图。MTP head 直接复用
主 head 的量化载荷，不单独校准或 RTN。

```bash
python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w5_auto_mixed_mtp.yaml \
  --quant-output-dir work_dirs/Qwen3.6-35B-A3B_mix_head_w5_mtp_w4 \
  --export-output-dir work_dirs/Qwen3.6-35B-A3B_mix_head_w5_mtp_w4_export
```

DFlash/DFlash2 保留各自 assistant 配置和导出 ABI，仅 head 继承目标模型。
各 HMONNX 图分别携带继承的 head 权重；运行时 session 独立加载，不要求底层存储共享。

## 文本 AutoRound checkpoint 的消费校验

独立 `Qwen3_5MoeForCausalLM` 的 GPTQ-v2 导入识别完整的
`qweight/qzeros/scales/g_idx`，包括 untied packed `lm_head`；已有 packed head
优先于额外提供的原始 head。四张量残缺时报格式错误。

当前 SSFP 消费接口同时使用整数码和浮点权重推导 group scale。若 checkpoint 同时
保存浮点副本，两份必须一致；scale 和整数码还必须能通过该接口精确保留，否则导入
提前拒绝。负 scale 仅在符号吸收不越过 W4/W8 范围时接受，禁止静默饱和。
这是一项已有量化产物的消费约束，不在导出过程中重新校准或修复 producer 权重。
调用方显式传入的 `trust_remote_code=False` 同时约束配置和模型加载。
