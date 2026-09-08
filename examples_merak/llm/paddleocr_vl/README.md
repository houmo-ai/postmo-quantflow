# PaddleOCR-VL 1.6：Merak 工作流与问题复测

本目录提供 PaddleOCR-VL 的 Merak workflow、校准样本准备及 HF/HMONNX 案例复测入口。
已修复 R1 + AutoRound 的磁盘 offload 保存问题；视觉导出已改为与 Qwen3.5 相同的 token-capacity gear 路由。
Jira 认证及附件读取已解决，固定尺寸 10 项、原生尺寸 7 项 HF/HMONNX 对照均已完成。
原生 HMONNX 银行主表的选定单元格为 37/37，但标签之外仍发现账号数字错误，菜单和手写也仍有误识。
实际结果及复现证据见
[实验报告](../../../work_dirs/paddleocr_vl/REPORT.md)。

## 模型与执行规格

本次使用官方 `PaddlePaddle/PaddleOCR-VL-1.6`，固定下载 revision：

```text
c5630abae1d940eafe0697512a0325494b02ab42
```

下载目录为 `/data01/datasets/PaddleOCR-VL-1.6`。
[下载记录](../../../work_dirs/paddleocr_vl/download_manifest.json) 保存来源、revision、文件清单及
`proxy_disabled=true`。模型下载前必须在下载进程所在 shell 执行 `UP` 关闭代理。
后续对比应复用固定 revision，避免将上游模型变更混入量化或部署差异。

模型加载使用已安装的 Transformers 原生实现。仓库内
[HF loader](../../../xhmodel_merak/xh_llm/models/paddleocr_vl/hf.py) 保留原始配置的
`tie_word_embeddings=false`，避免旧版 flat config 经 Transformers 5.16 解析时意外绑定
两份独立的 embedding/lm_head 权重；不要绕过该 loader 建立对比基线。

默认部署配置是
[paddleocr_vl_1_6.yaml](../../../configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6.yaml)：

| 项目 | 完整规格 |
| --- | --- |
| 目标 | `chip_arch: XH2a` |
| 模型 dtype | `float16`；HF 对照可明确选择 `bfloat16` |
| 图像尺寸 | 896 × 896 |
| ViT patch / spatial merge | 14 / 2 |
| 图像 patch grid | `[1, 64, 64]`，4096 个 patch，合并后 1024 个 image token |
| context / max position length | 16384 / 16384 |
| prefill chunk | 256 |
| 文本与视觉图量化配置 | `w8a8_sefp` |

小图、小 tensor 和短生成用于定位或 CPU 回归，不能替代此规格的全量验收。
`quant: null` 只跳过独立权重量化阶段，`export.model.quant_scheme` 和视觉量化配置仍然生效。

## 环境与单卡执行

所有命令从仓库根运行，模型、校准集、checkpoint、日志及验证产物均放入本仓库。
本机当前可使用 `xh2` 环境，并通过 `PYTHONPATH` 读取现有较新版本的 xhquant SDK：

```bash
export PADDLEOCR_PYTHON=/data01/home/xuzk/anaconda3/envs/xh2/bin/python
export PYTHONPATH="$PWD:/data01/home/xuzk/workspace/xh2/xhquanttool${PYTHONPATH:+:$PYTHONPATH}"
export CC=/data01/home/xuzk/anaconda3/envs/xh2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data01/home/xuzk/anaconda3/envs/xh2/bin/x86_64-conda-linux-gnu-c++
export HF_HOME="$PWD/work_dirs/paddleocr_vl/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$PWD/work_dirs/paddleocr_vl/cache/torch_extensions"
export TRITON_CACHE_DIR="$PWD/work_dirs/paddleocr_vl/cache/triton"
export XDG_CACHE_HOME="$PWD/work_dirs/paddleocr_vl/cache/xdg"
export TMPDIR="$PWD/work_dirs/paddleocr_vl/tmp"
mkdir -p "$TMPDIR"
unset TORCH_COMPILE_DISABLE
export TORCHDYNAMO_DISABLE=1
```

上述外部 SDK 路径只作为已有依赖读取，不修改其代码。环境路径是本机配置，其他机器应替换
为其已验证环境。若需要改用 `xhquant` 环境，记录 Python、Transformers、GPTQModel 和 SDK
实际来源，不将环境差异隐藏在实验结果中。

本机默认 `c++` 命令缺失，但 xh2 环境中已有 conda GCC 11.2；上述 `CC` / `CXX`
显式选择它，供 SDK 的 CUDA 扩展构建使用。已有工具链通过 CUDA 12.4 编译检查，见
[编译器探针](../../../work_dirs/paddleocr_vl/diagnostics/compiler_probe/compiler_probe_report.json)。
实验使用 `TORCHDYNAMO_DISABLE=1` 跳过普通 `torch.compile` 路径。
不要同时设置 `TORCH_COMPILE_DISABLE=1`：在当前 Torch 2.8 中，它还会关闭
Merak 严格导出所需的 Dynamo 图捕获，使完整量化图在 CPU 上实算，最终无法导出。
四种开关组合的严格导出对照见
[诊断记录](../../../work_dirs/paddleocr_vl/diagnostics/export_dynamo_strict_probe.json)。

每次启动前检查 GPU 当前剩余显存和运行任务，将 `PADDLEOCR_GPU` 设为一张足够空闲的物理卡编号：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
export CUDA_VISIBLE_DEVICES="$PADDLEOCR_GPU"
```

`PADDLEOCR_GPU` 必须是单个编号。设置后命令中的 `--device cuda:0` 指这张可见卡。
每次实验只占一张卡；同一张卡不叠加导出、量化或生成任务。等待对应实验终端自然退出，
再读取结果并调度下一步，不频繁轮询或因观察超时重复启动任务。

## Merak workflow

[paddleocr_vl_workflow.py](paddleocr_vl_workflow.py) 调用
`AutoWorkflow.from_config()` → `quant()` → `export()` → `dump_golden()`。
保持 [Merak 架构](../../../docs/architecture/merak.md) 中 workflow、模型语义、图与 runtime
的职责划分；量化接入边界见 [quantization-boundary.md](../../../docs/architecture/quantization-boundary.md)。

PaddleOCR-VL 也注册为统一接口模型 `paddleocr-vl-1.6`。校准集是该 recipe 的必需资源，
构建时通过 `resources` 显式传入，不能把本机实验路径写进公开配置：

```python
from xhmodel_merak.model_build import build_model

result = build_model(
    model_id="paddleocr-vl-1.6",
    model_dir="/path/to/PaddleOCR-VL-1.6",
    output_dir="work_dirs/paddleocr_vl/api_artifact",
    device="cuda:0",
    resources={"calibration": "/path/to/calibration.jsonl"},
)
```

产物可直接交给统一推理入口；图片为必需输入：

```bash
python -m examples_merak.unified_api.generate \
  --artifact-dir work_dirs/paddleocr_vl/api_artifact \
  --prompt 'OCR:' --image /path/to/document.png \
  --max-new-tokens 1024 --device cuda:0
```

下列命令描述当前 CLI；执行成功和精度结论必须由真实产物及复测补充：

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/paddleocr_vl_workflow.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --config configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6.yaml \
  --output-dir work_dirs/paddleocr_vl/experiments/base_896_16k_256 \
  --device cuda:0 \
  --golden-image work_dirs/paddleocr_vl/cases/bank_statement/input.png \
  --golden-text 'OCR:'
```

默认 YAML 的 `quant: null` 使用原始权重进入导出。
GPTQModel/AutoRound 使用新增的
[paddleocr_vl_1_6_r1_autoround_w8.yaml](../../../configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6_r1_autoround_w8.yaml)，
保留完整 `export` 规格，并启用下列 `quant` 参数：

```yaml
quant:
  algorithm: gptqmodel
  method: autoround
  calibration_jsonl: null  # 通过 CLI 指定本次校准数据
  nsamples: 32
  bits: 8
  group_size: 64
  batch_size: 1
  iters: 200
  rotation: hadamard
  quantize_vision: true
  vision_rotation: hadamard
  offload_to_disk: false
```

用 `--calibration-jsonl` 指定本仓库的实际校准集：

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/paddleocr_vl_workflow.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --config configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6_r1_autoround_w8.yaml \
  --calibration-jsonl work_dirs/paddleocr_vl/calibration/synthetic_ledgers32_otsl_v2/calibration.jsonl \
  --output-dir work_dirs/paddleocr_vl/experiments/r1_autoround_w8_896_16k_256 \
  --device cuda:0 \
  --golden-image work_dirs/paddleocr_vl/cases/bank_statement/input.png \
  --golden-text 'OCR:'
```

`--calibration-jsonl` 要求配置中已启用 `quant`，不能用于默认 `quant: null` 配置。
上述命令展示当前参数接口，不构成已量化成功的记录。`rotation: null` 可用于无旋转对照。
当前 [recipe](../../../recipes/paddleocr_vl.py) 在两个干净模型实例上分别执行 AutoRound，
再把语言 decoder 和 Vision encoder 的 GPTQModel 权重合并为一个可重载 checkpoint。
语言侧 `rotation: hadamard` 是离线 R1 hidden-basis 变换，包含 projector 输出到语言隐藏空间
的连接；Vision 侧 `vision_rotation: hadamard` 是逐 attention head 的 V/O 离线等价变换，
不改 Q/K 与二维 RoPE。Vision encoder 的 `q/k/v/out_proj` 和 `fc1/fc2` 使用 W8G64；patch
Conv2d 与跨模态 projector 保持浮点。由于 Vision 的可学习 LayerNorm 和 projector 边界，
这里不宣称启用了贯穿整个 Vision residual stream 的全局 R1，也不启用在线 R2/R4。
部署图自身的 SEFP 量化仍属于后续导出阶段。

当前 R1 路径要求 `offload_to_disk: false`。已复现磁盘 offload 保存时恢复原始 projector
和末层 RMSNorm、同时保留旋转后 embedding/lm_head 的错误，造成 hidden basis 不一致。
该组合必须在量化前拒绝；诊断见
[保存路径检查](../../../work_dirs/paddleocr_vl/diagnostics/rotation_save_path_diagnosis.json)。

量化 checkpoint 必须完成保存/重载、部署消费与生成检查后，才能作为可用产物交付。

## 校准 JSONL

[load_calibration_inputs](../../../xhmodel_merak/xh_llm/models/paddleocr_vl/calibration.py)
读取每行一个 JSON 对象，必需字段为 `image`、`prompt`、非空 `answer`：

```json
{"image":"images/page_001.png","prompt":"OCR:","answer":"Document number: 12345. Total: 82.50."}
{"image":"images/table_001.png","prompt":"Table Recognition:","answer":"<fcel>Date<fcel>Amount<nl><fcel>2026-09-08<fcel>82.50<nl>"}
```

相对图片路径按 JSONL 所在目录解析。loader 用与复测相同的固定尺寸预处理和原生 processor
构造图文输入，并将 `answer` 与 EOS 接入校准序列；attention mask 和多模态 token 标记随之扩展。
这里的原始图文 JSONL 由 workflow 的 `calibration_jsonl` 消费，不能与 recipe 直接接受的
已处理 tensor/model-input 字典格式混用。

可用 [prepare_calibration.py](prepare_calibration.py) 生成独立的英文数字账本：

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/prepare_calibration.py \
  --output-dir work_dirs/paddleocr_vl/calibration/synthetic_ledgers32_otsl_v2 \
  --samples 32 --seed 42
```

输出包括 896 × 896 图片、每页 16 行交易的 OTSL 表格答案、`calibration.jsonl` 和 `provenance.json`。
`Table Recognition:` 的原生输出使用 `<fcel>`、`<ecel>`、`<nl>` 等标记；历史服务输出的 HTML
经过了表格转换，不能把 HTML 直接视为本次原生模型的回答格式。合成页均为无合并、无空单元格的
17 × 5 单元格表格，使用 `<fcel>` 与 `<nl>` 编码；旧的 HTML 校准草稿不用于本次量化。
格式依据为 [PaddleX 原生 OTSL 转换实现](https://github.com/PaddlePaddle/PaddleX/blob/ffb64904d23708863ff5b8da312a5cbd52a7f462/paddlex/inference/pipelines/paddleocr_vl/uilts.py#L707-L970)。
每次使用新目录，脚本不会覆盖已有目录。
这些样本为合成英文印刷表格，使用已知数字标签，不包含留出的三个问题原图。
它们不代表中文手写、多语言、复杂合并表格、真实扫描噪声或银行水印分布；不能仅凭该校准集
上的结果宣称用户案例精度得到改善。问题案例应持续作为留出集，分别检查各类误差。

## 原始问题与复测数据

已读取 [飞书问题文档](https://houmo.feishu.cn/wiki/BQphwReXpi78F5kQAFgcxy6mnAc)
revision 163，下载 6 张图和 `对比.rar`，解压得到 58 个文件。
本地 [cases manifest](../../../work_dirs/paddleocr_vl/cases/manifest.json) 保存原图 SHA-256、
9 组历史输出、3 个整图入口及 22 个可恢复 layout 区域。

| 飞书案例 | 本地输入（仓库相对路径） | 历史现象 |
| --- | --- | --- |
| 手写举报信 | `work_dirs/paddleocr_vl/cases/handwriting/input.jpg` | 字词、日期和金额误识别；整图后摩输出出现连续 795 个 `7` |
| 银行流水 | `work_dirs/paddleocr_vl/cases/bank_statement/input.png` | 数字、日期和列归属错误，尾行缺失；layout 后摩输出出现连续 1925 个 `0` |
| 每周菜单 | `work_dirs/paddleocr_vl/cases/weekly_menu/input.jpeg` | 日期/餐次分组、配料重量关联与底行缺失；layout 后摩 HTML 额外输出 139 个空行 |

历史 GPU 输出本身也有错误，不能直接当完整 ground truth。
[visual_assertions.json](../../../work_dirs/paddleocr_vl/cases/visual_assertions.json) 仅包含人工看原图
核查的部分短语、银行单元格与菜单断言。图片自动 alt 描述没有用于标注；手写对比截图中的
绿色表示“双方都错”。银行原图包含看似不合理的合成数值，应按印刷字符识别，不按会计逻辑改写。

历史后摩结果的 `engine` 为 `xh2-hmm-quantized`、`model` 为
`paddleocr-vl-1.6-0.9b-xh2-quantized`，可在
[银行历史结果](../../../work_dirs/paddleocr_vl/cases/bank_statement/houmo_layout.json) 的同名字段核对。
已下载归档的 39 个文本文件没有给出具体 w8a8/SEFP 方案、checkpoint revision/hash、SDK 版本或
Git 分支/commit；5 份 `api_response.json` 的 `log` 字段均为空。
因此当前 `w8a8_sefp` 是本次明确选择的配置，不能宣称与历史量化方案完全相同。
当前官方模型下载 revision 也不能用来补写 2026-09-07 历史运行的 checkpoint 版本。

[QTL-510](http://jira.houmo.ai/browse/QTL-510) 和
[CMS-787](http://jira.houmo.ai/browse/CMS-787) 已通过认证读取，正文、全部评论和附件保存于
`work_dirs/paddleocr_vl/sources/jira/authenticated_20260908T134542Z/`。
CMS-787 新增两张手写原图，已加入 `work_dirs/paddleocr_vl/cases/jira_manifest_v2.json`；
QTL-510 要求验证动态转静态的影响，验收指标仍待确认。旧 401 记录保留为历史证据。

## HF 与 HMONNX 使用同一输入

[evaluate_cases.py](evaluate_cases.py) 默认跑全部三个整图案例。
它调用公共 [prepare_inputs](../../../xhmodel_merak/xh_llm/models/paddleocr_vl/processing.py)，
将图像转 RGB、固定 resize 后交给原生 processor，并关闭 processor 的再次动态 resize。
不会自动旋转或任意纠正原图。HF 与 HMONNX 必须使用相同的图像像素、prompt、尺寸、
dtype 和生成预算进行配对比较。

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/evaluate_cases.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --manifest work_dirs/paddleocr_vl/cases/manifest.json \
  --backend hf --dtype bfloat16 --device cuda:0 \
  --max-new-tokens 4096 \
  --output-dir work_dirs/paddleocr_vl/evaluation/hf_bfloat16_full
```

这里显式使用 4096 输出 token，避免把默认整图 1024 上限导致的截断误当模型精度。
HMONNX 导出完成后用实际 metadata 路径复测；`PADDLEOCR_META` 应指向本仓库内该次导出的 metadata：

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/evaluate_cases.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --manifest work_dirs/paddleocr_vl/cases/manifest.json \
  --backend hmonnx --meta-file "$PADDLEOCR_META" \
  --dtype float16 --device cuda:0 --max-new-tokens 4096 \
  --output-dir work_dirs/paddleocr_vl/evaluation/hmonnx_float16_full
```

HF 默认 `bfloat16`，HMONNX 默认 `float16`，后者显式指定 `bfloat16` 会报错：
当前部署 KV-cache ABI 固定为 FP16，不能将 BF16 标签写入实际 FP16 的实验记录。
部署配置是 FP16 时，还须建立相同 FP16 的 HF 对照；上面的 BF16 基线用于单独观察 dtype 因素，
不能将不同 dtype 的输出差异全部归因于部署量化。

HF 的 `--hf-cache` / `--no-hf-cache` 可分别显式打开或关闭生成 KV cache。
省略该参数时保持原行为，遵循 checkpoint 的 generation config；本次固定版本的
`generation_config.json` 中 `use_cache=false`，因此默认 HF 长生成可能较慢。
例如，另开一个目录运行缓存开启的 FP16 对照：

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/evaluate_cases.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --backend hf --dtype float16 --hf-cache --device cuda:0 \
  --max-new-tokens 4096 \
  --output-dir work_dirs/paddleocr_vl/evaluation/hf_float16_cache_full
```

HMONNX 拒绝这两个 HF-only 参数，由 runtime 管理自己的 KV-cache scope，
加载时使用 `device_map=[device]` 限定运行设备。每例 `result.json` 记录实际选择的
`use_cache` 及 `use_cache_source`，区分 checkpoint 默认值、CLI 覆盖和部署固定 cache ABI。
切换 cache 策略必须留作独立实验因素，不覆盖已经开始或结束的基线记录。

HF 入口也接受本 recipe 生成的 GPTQ checkpoint：把 `--model-dir` 指向保存目录，入口通过
Merak 的权重消费接口还原量化权重到原生浮点算子，再执行 HF generate。该对照包含权重量化
误差，尚未包含部署图的激活量化，可用于区分两个阶段的影响。加载后通过
`set_attn_implementation("eager")` 与原始 HF 对照保持相同 attention 后端。

`--case handwriting`、`--case bank_statement`、`--case weekly_menu` 可重复指定。
`--include-regions` 在整图之外增加 manifest 标记的主要问题区域：两张表格及手写后三段。
默认预算来自 manifest：OCR 1024、table 4096；`--max-new-tokens` 显式覆盖，并检查
`prompt_tokens + max_new_tokens <= 16384`。所有生成均为 `do_sample=False`。

`--native-resize` 使用模型原生 processor 的尺寸选择和像素预算。它保留原图或区域的
宽高关系，不套用 manifest 的固定尺寸、stretch 或 letterbox。HF 和 HMONNX 可使用
相同参数；HMONNX 必须先导出对应 patch grid，未覆盖的 grid 会明确报错。

```bash
"$PADDLEOCR_PYTHON" examples_merak/llm/paddleocr_vl/evaluate_cases.py \
  --model-dir /data01/datasets/PaddleOCR-VL-1.6 \
  --manifest work_dirs/paddleocr_vl/cases/native_focus_manifest_v1.json \
  --include-regions --native-resize --backend hf --dtype float16 --hf-cache \
  --max-new-tokens 4096 --device cuda:0 \
  --output-dir work_dirs/paddleocr_vl/evaluation/native_hf_new_run
```

原生模式的 `prepared.png` 保存 **processor 输入的原图/恢复裁剪**；实际缩放、归一化和
patch 展开结果由 `input_tensors` 的形状、dtype 和 SHA-256 记录。字段
`preprocessing.prepared_pixels_stage=source_crop_before_native_resize` 标明该区别。
尺寸对照同时改变有效像素预算，不能单独归因为宽高比。

[token gears 配置](../../../configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6_r1_autoround_w8_token_gears.yaml)
使用 `[96, 196, 384, 704, 1024, 1280]` 个合并后视觉 token 的容量档位。HF processor 仍按原图和
`smart_resize` 生成动态 `image_grid_thw`；host 将 row-major patch 重排为 2×2 merge-major，选择能容纳
有效 token 的最小档位，补零后同时传入位置插值 IDs/weights、二维 RoPE IDs 和 additive key mask。
HMONNX 只切掉尾部 padding 输出，不改变有效图像区域或 HF 的缩放结果。

| image token gear | patch 容量（merge=2） | 路由规则 |
|---:|---:|---|
| 96 | 384 | 有效 token ≤ 96 |
| 196 | 784 | 96 < 有效 token ≤ 196 |
| 384 | 1536 | 196 < 有效 token ≤ 384 |
| 704 | 2816 | 384 < 有效 token ≤ 704 |
| 1024 | 4096 | 704 < 有效 token ≤ 1024 |
| 1280 | 5120 | 1024 < 有效 token ≤ 1280 |

该上限与 PaddleOCR-VL processor 的 `max_pixels=1003520` 对齐：`1003520 / (14×14×2×2) = 1280`。
非方形 2×6 patch grid 放入 16-patch gear 的最小真实模型回归中，有效输出相对 HF 原生视觉塔的
最大绝对误差为 `7.45e-09`。原五个精确分辨率 HMONNX 仍是历史精度报告的证据；它们不具备新
ABI，需要按 token gears 配置重新导出后才能用于任意受支持的动态 grid。

回放保留以下区别和假设：

- 历史整图服务记录为 896 × 896 `stretch`、无独立 layout；它未保存 literal prompt 或生成限额。
  manifest 的 `OCR:`、1024 输出 token 和 16K context 是已声明的回放默认值。
- 历史 layout 区域使用 896 × 896 `letterbox`，每区 1024 image token、1037 prompt token、16K context；
  OCR/table 的输出限额分别为 1024/4096。两张历史 table 都达到 `max_new_tokens`。
- RAR 未保存 crop 图片，但保留 crop_box、polygon mask、padding、合并白底及有效内容尺寸。
  复测按这些信息恢复区域；历史插值和居中细节未入档，当前明确采用 PIL bicubic 与居中白底。
  `OCR:`/`Table Recognition:` 的原始字面量同样未入档，不把任务推导的 prompt 当成历史事实。
- 银行 table 的有效内容只有 896 × 454；手写窄条可能有超过 90% 白边。
  应将 layout/resize 与量化、dtype 因素分别记录，不能用不同预处理的输出推断量化损失。

每例保存 `prepared.png`、原图/预处理像素/tensor SHA-256、输入和输出 token IDs、
`raw_output.txt`、`output.txt`、生成长度、EOS/到上限标记、耗时与 `result.json`；
整体记录在 `evaluate.log`、`arguments.json`、`summary.json`。
输出目录必须为空，以保留每次实验证据。

诊断的 `table_format` 区分 `plain_text`、`html` 和原生 `otsl`。已有 HTML 行数、闭合标签数、
`tables` 等字段继续保留；OTSL 在 `otsl.rows` 中保存原始格子及 `<fcel>/<ecel>/<lcel>/<ucel>/<xcel>`
标记，以 `<nl>` 分行并保留原始行宽，不自动补齐、截短或修复。合并延续格组成的整行不算空行。
无合并且行宽一致的银行表格可按表头核验选定行列；包含合并、畸形 OTSL 或不同行宽时，
`bank_cells.status=not_evaluable`，不生成伪位置分数。`closed_table_count` 仍只计 HTML 闭合标签，
不能据此判断 OTSL 是否包含表格。

重复诊断同时记录连续相同字符和 `phrase_repetition`（长度 2–64 字符的精确连续重复）。
例如菜单的 `菜名 配料 重量克 ` 循环可有 509 次，而最长相同字符仅为 1；重复表头本身也可能合理，
这些指标仅报告原始输出证据，不提前停止生成或清洗结果。诊断还保存部分 anchor 命中/缺失。
anchor 命中只是部分信息出现，不能证明完整 OCR 正确；区域输出使用页级标签时，缺失 anchor
可能在区域外。菜单仍需核对 colspan/rowspan、餐次和底行；有效 HTML 或 `table_html_valid=true`
不能证明没有重复、缺字或列错位。

## 产物阶段与实测边界

下表和后续 `r1_autoround_w8_*` 路径记录的是上一轮**仅语言 decoder 权重量化**的基线产物；
它们没有包含本次新增的 Vision encoder QuaRot + AutoRound 权重，不能作为新双阶段 recipe 的
完成证据。新实现已通过最小真实模型的两阶段量化、合并和 GPTQModel 重载验证；全量 32 样本、
200 iterations checkpoint 及其 HMONNX/Jira 对照需要生成新的独立产物记录。

| 阶段 | 应留存证据 | 当前边界 |
| --- | --- | --- |
| 固定模型与输入 | 下载 revision、原图和 SHA-256、历史输出及回放参数 | 官方模型、飞书和两张 Jira 工单均已取得；五张原图的固定尺寸 HF/HMONNX 复测已完成 |
| 独立权重量化 | GPTQModel checkpoint、recipe provenance、保存/重载与生成结果 | 语言侧基线完成 32 样本/200 iterations/R1/8-bit；Vision 双阶段实现已完成最小真实模型验证，全量产物待新的运行记录 |
| 图导出 | 完整 16K/256 的 aligned 转换、实际 HMONNX 图和 metadata | 历史五档精确分辨率 visual、共享 prefill/decode 已导出；token-gear ABI 已通过最小真实模型 TorchFX/图量化，完整 checkpoint 产物待重导 |
| golden | 实际图输入/输出及对照记录 | 五份视觉、五份 prefill 和一份 decode 输出均符合规格且有限；四档原生输入与 HF tensor 证据一致 |
| 问题复测 | HF/HMONNX 配对生成、数值与结构判据、结束原因、原始全文 | 固定尺寸 10 项及原生尺寸 7 项 HMONNX 均完成，无运行失败；原生尺寸 7 项均 EOS，精度问题仍未全部解决 |

固定 896 HMONNX 银行主表生成 38×9 OTSL、无空尾行，选定单元格为 17/37，与两组 HF 总数相同，
但相对修正 W8 HF 有 4 项改善和 4 项退化，不能宣称精度不变。菜单整图的循环由表头变为
内容行，最长精确短语重复 76 次；菜单主表 750 token 后 EOS、无空尾行，15/15 短语出现，
仍有合并和不齐行宽，短语命中不等于行列、重量或餐次全部正确。
HMONNX 与 HF 的全部输入控制记录匹配；每例 revision 未独立记录，比较器明确保留该证据缺口。
部署阶段还引入图数值路径、视觉及文本权重再次量化和激活量化，不能将差异只归因于激活量化。

原生尺寸 HMONNX 银行主表保持 38×9，选定单元格提升到 37/37，与两组原生 HF 相同；
但另行核对发现最后一条交易的账号仍有一位数字错误，见 [银行复核](../../../work_dirs/paddleocr_vl/diagnostics/native_hmonnx_bank_semantic_review_v1.md)。
菜单整图本次 670 token 后 EOS，仍缺内容且列归属有误；菜单主表为 10×19 原始槽位，
源图和两组原生 HF 支持 11 行，实际缺失“臊子面”并压缩、错位晚餐内容，见 [菜单复核](../../../work_dirs/paddleocr_vl/diagnostics/native_hmonnx_menu_semantic_review_v1.md)。
手写新增第二页日期恢复为 `2023年11月23日`，但 `220万元` 被识别为 `210万无`，见 [手写复核](../../../work_dirs/paddleocr_vl/diagnostics/native_hmonnx_handwriting_semantic_review_v1.md)。
不能把 EOS、行宽齐全或稀疏字串命中当作完整精度通过。

当前阶段记录：

- [原始 HF cache/dtype 对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_hf_cache_dtype_v1/comparison.md)
- [首个 R1 + AutoRound HF 对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_hf_r1_w8_v1/comparison.md)
- [修正 checkpoint 的 HF 对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_hf_r1_w8_no_offload_v2/comparison.md)
- [原始 HF、修正 W8 HF 与 HMONNX 最终比较](../../../work_dirs/paddleocr_vl/evaluation/comparison_final_hf_hmonnx_v3/comparison.md)
- [Jira 新增页面三组对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_jira_new_threeway_v1/comparison.md)
- [原模型固定/原生尺寸对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_original_fixed_native_v1/comparison.md)
- [原生尺寸原模型/修正 W8 对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_native_hf_w8_v1/comparison.md)
- [原生尺寸 HF/W8/HMONNX 三组对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_native_threeway_v1/comparison.md)
- [固定/原生尺寸 HMONNX 对照](../../../work_dirs/paddleocr_vl/evaluation/comparison_hmonnx_fixed_native_v1/comparison.md)
- [7 项原生 HMONNX 保存证据审计](../../../work_dirs/paddleocr_vl/diagnostics/native_hmonnx_evaluation_audit_v1.json)
- [8 项 HMONNX 运行与对照审计](../../../work_dirs/paddleocr_vl/diagnostics/final_hmonnx_evaluation_audit.json)
- [修正 checkpoint 的旋转检查](../../../work_dirs/paddleocr_vl/diagnostics/full_v2_rotation_audit.json)
- [修正 checkpoint 的导出与 golden 完成记录](../../../work_dirs/paddleocr_vl/runs/r1_autoround_w8_no_offload_export_v3/completion.json)
- [最终 HMONNX 与 golden 产物检查](../../../work_dirs/paddleocr_vl/diagnostics/final_export_golden_audit.md)

修正 checkpoint 为 `work_dirs/paddleocr_vl/runs/r1_autoround_w8_no_offload_896_16k_256_v2/checkpoint`，
其 HMONNX 与 golden 位于 `work_dirs/paddleocr_vl/runs/r1_autoround_w8_no_offload_export_v3/`。
历史五档精确分辨率 HMONNX 与 golden 位于 `work_dirs/paddleocr_vl/runs/r1_autoround_w8_native_profiles_export_v3/`，
加载入口见该目录的 `completion.json`；完整检查见 [五档产物审计](../../../work_dirs/paddleocr_vl/diagnostics/full_native_profile_bundle_audit_v3.md)。
早期 `r1_autoround_w8_896_16k_256_v1` checkpoint 和 `r1_autoround_w8_export_v2` 导出目录
仅保留为错误保存状态的对照证据。
