# Qwen3-VL-Reranker-2B

将 Query 和 Document 的文本、图片或图文组合输入同一 Qwen3-VL 主干，输出每个候选的相关性分数及降序索引。
实现复用 Qwen3-VL 的语言包装、VL-Embedding 的 Prefill-only 导出，以及 Qwen3.5 已有的视觉 token 档位实现。

本次适配、Merak 视频与超长截断支持、模型卡、核心 CI 和正式报告集中在本目录及当前仓库。
评测脚本、工具专用测试和精确结果保存在[工作目录支持包](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/README.md)。
量化配置选择依据见下文；[五精度评分头 A8/A16 对照](evaluation/HEAD_A8_A16_713_REPORT.md)
和[三套局部 A16 筛选报告](evaluation/ACTIVATION_A16_713_COMPARISON_20261028.md)
均使用原 713 条查询、9 个任务的完整冻结候选集，不是官方全量 MMEB 总分。

[Model card](model_cards/m50/qwen3_vl_reranker_2b.yaml) 以两节点 A16 为主工作流，
仅记录两节点 A16＋head W8A16 在原 713 条上的图像、文档、视频宏平均，
附 BF16 和百分点差值，不填逐任务或九任务总均值。

另提供 [W4A16 model card](model_cards/m50/qwen3_vl_reranker_2b_w4a16.yaml)，
关联全 196 个语言 Linear 激活 A16、head W8A16 的配置，单独记录同一原 713 条数据的三个模态宏平均。

## 支持范围

| 项目 | 范围 |
| --- | --- |
| 模型 | `Qwen/Qwen3-VL-Reranker-2B`，28 层语言模型、2048 hidden size |
| 目标与执行 | XH2a HMONNX；通过 CUDA 上的 HMONNXInferenceV2 数值模拟器验证，未声明芯片真机验证 |
| 部署精度 | 维护两节点 W4A8/A16 和全语言 W4A16 两份 YAML；原生 checkpoint 默认导出 W8A8；评分头均为 W8A16，视觉 W8A8，KV cache 为 FP16 |
| 输入 | Query 和 Document 支持文本、单图、多图、图文组合；Merak 另支持视频及显式帧/时间戳输入；按实际图像 token 数选择 96/196/384/704/1536 档位 |
| 序列 | 默认 context 32768、Prefill chunk 256；语言 Attention 导出为 FlashAttention；图内更新 KV cache，Host 按全局位置切分输入 |
| 输出 | `scores` 按输入候选顺序返回；`ranking` 是稳定的降序候选索引 |
| 精度状态 | 已完成五种语言精度在评分头 A8/A16 下的原 713 条对照，以及三套局部 A16 完整验证；量化误差与检索指标分别记录，不宣称无损 |
| 当前边界 | 底层每次执行 batch=1，候选顺序处理；保护特殊 token、视频时间戳跨度及末尾五个模板 token 后截断；仅保留项已超容量时明确报错 |

`precision` 描述输入 checkpoint，默认 W16；`deployment_precision` 选择导出的基础 QuantScheme，默认 W8A8。
已有整数权重可覆盖对应节点的实际权重格式，位宽以节点属性为准。
`quant: null` 表示不运行独立 GPTQ/AutoRound，图导出阶段仍执行 SDK 的权重量化。
当前 YAML 的 `context_max_length` 均为 32768，与官方发布的 32K 规格一致。`max_pe_length=262144` 是位置编码表上限，不代表官方测试了 256K 上下文，也不扩大本图的 KV cache 容量。

## 环境与构建

使用当前仓库 CI 的 Transformers 5.16.1、PyTorch 2.8.0 和匹配的 xhquant SDK。
精确依赖由 recipe 的 `dependencies` 声明；构建包的 `environment.json` 和 `environments/` 保存同一份声明。
安装来源及干净环境验证状态遵循该文件，不将本机可运行解释为公共 wheel 已发布。

以下命令在仓库根目录执行，`PYTHONPATH` 必须指向本工作区：

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/qwen3_vl_reranker_workflow.py \
  --model-dir /data01/datasets/Qwen3-VL-Reranker-2B \
  --output-dir work_dirs/qwen3_vl_reranker_2b_w8a8 \
  --device cuda:0
```

输出目录必须不存在。可用 `--context-length` 和 `--prefill-length` 选择实际导出容量。
统一 Python 接口：

```python
from xhmodel_merak.model_build import build_model, describe_model

print(describe_model("qwen3-vl-reranker-2b", full=True))
result = build_model(
    model_id="qwen3-vl-reranker-2b",
    model_dir="/data01/datasets/Qwen3-VL-Reranker-2B",
    output_dir="work_dirs/reranker_release",
    device="cuda:0",
)
```

发布目录包含平铺的 `visual_m96/`、`visual_m196/`、`visual_m384/`、`visual_m704/`、`visual_m1536/`、`prefill/`、`hf_config/`、token embedding 的 `.pt/.bin`、
`golden_meta_info.json`、有效 workflow、依赖声明、`export_meta_info.json` 与构建 Manifest。
没有 Decode 图。图名使用真实部署精度，context 和 chunk 的真实值从运行 metadata 读取。
`export_meta_info.json` 是 `{path, type}` 集成清单，运行契约由 Manifest 指向的 metadata 解释。
加载完整目录即可，不需要源 checkpoint。

## AutoRound 四位权重量化

W4 主配置为 `configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml`。
`quant` 配置沿用 Qwen3.5 full 的 AutoRound dense / llm-only 接口：W4G64、对称 `int_sym_xh`、不旋转、
256 条 Pile 校准文本、长度 2048、400 次优化、batch 8。校准数据使用仓库资源 URI，独立校准长度不影响 32K 部署上下文。
AutoRound 阶段保留视觉塔和原始 tied embedding / 完整语言 head 的浮点值；导出时由原始 head 计算 yes-minus-no，评分 head 为 W8A16，视觉塔默认 W8A8。
AutoRound 先优化舍入并输出 GPTQ 格式 checkpoint；随后 export 消费该载荷生成 XH2a W4 图。
这是 AutoRound 算法，GPTQ 是保存格式名称。不会把它当作单纯的导出阶段 RTN W4。

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/qwen3_vl_reranker_workflow.py \
  --model-dir /data01/datasets/Qwen3-VL-Reranker-2B \
  --weight-bits 4 --quant-output-dir work_dirs/reranker_autoround_weights \
  --output-dir work_dirs/reranker_autoround_release --device cuda:0
```

统一 Python 接口指定 `precision={"weight": "w4", "activation": "a8"}`，recipe 自动选择两节点 A16 配置；
指定 `precision={"weight": "w4", "activation": "a16"}` 则选择全语言 W4A16。
默认 W16/a8 输入跳过独立量化、移除两个语言节点例外，导出语言 W8A8，评分头保留 W8A16。
后续复用已有权重时加 `--quant-checkpoint <AutoRound目录>`，
公开 API 对应 `quant_checkpoint`，并保持 `precision` 为 W4；无需重复校准。
也可通过 `AutoWorkflow.from_config(model_dir=..., config_path=<W4 YAML>)` 分别调用 `quant()`、`export()` 和 `dump_golden()`。
`export.model.hf_model: null` 是输入路径占位符，会由 workflow 用原生或已量化目录填充；`export` 本身不是 null。

当前两份 YAML 均省略 `max_size_t`、`patch_size`、`temporal_patch_size`，复用 Qwen3.5 已有的
`XHQwen3_5_VisualConfig` 默认值 2、16、2；省略这些字段不改变图像动态档位。

注意当前 SDK 对已有整数权重的处理：GPTQ 格式检查点的加载器不仅恢复浮点 `weight`，还保留
`quant_weight`。SDK 发现这个整数载荷后，会把对应 Linear 的权重设置为 SSFP，并依据整数范围确定
实际位宽。因此复用本次 AutoRound W4 检查点时，即使设置
`deployment_precision={"prefill": "w8a8h1_sefp"}`，196 个语言 Linear 仍然导出为 SSFP W4；
基础 scheme 的名字不足以证明这些节点实际采用 W8。没有整数载荷的原始浮点权重则按 W8 SEFP 导出。
这也解释了 Qwen3.5 full YAML 同时配置 AutoRound `bits: 4` 和基础 W8 scheme 的行为。
当前 W4 YAML 的 `quant.bits: 4`、`group_size: 64` 和 AutoRound 方法保持明确，
两节点配置的 `export.model.quant_scheme.quant_type` 使用 `w8a8h1_sefp`，全语言 A16 配置使用 `w8a16h1_sefp`。
直接量化和复用检查点两条路径都接受这个基础格式，仍验证实际检查点为 AutoRound W4G64；
W4 的 chunk=256 约束也依据检查点执行，不会因为基础格式写成 W8 而跳过。
此前实际转成 W8 的独立对照从同一检查点的反量化浮点值导出，仅在临时副本中移除整数载荷，
保存前后逐参数校验 SHA-256；不修改原始检查点、模型实现或 SDK。三种情况的结果见 PIPELINE。

## 量化配置选择与局部 A16

当前仅保留两节点 A16 和全语言 W4A16 两份 YAML，评分头均采用 `w8a16h1_sefp`。
五种语言精度的历史对照配置及完整结果保存在评测归档，正式报告继续随代码保留。
评分头是 `(W_yes - W_no)` 合并后的相关性评分层；
五精度 head A8/A16 配对实验表明提高其激活精度能减少检索损失，单独将 head 权重再升 W16
没有带来同等幅度的收益，因此保留 W8 权重、采用 A16 激活。完整任务和模态对照见
[评分头报告](evaluation/HEAD_A8_A16_713_REPORT.md)，历史 head A8 配置与对应分数保存在评测归档中。

语言主干的 A8 激活仍有明显影响。保持同一 W4G64 权重、head W8A16、视觉和 Attention 配置，
扫描 28 层后进一步比较敏感层内的 21 个 Linear，最终用原 713 条完整候选集验证三套方案。
局部 A16 优先使用
[`qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml`](../../../configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml)：

```yaml
model_language_model_layers_26_mlp_up_proj:
  act_scheme:
    bits: 16
model_language_model_layers_27_mlp_gate_proj:
  act_scheme:
    bits: 16
```

层号从 0 开始。只有 `model.language_model.layers.26.mlp.up_proj` 和
`model.language_model.layers.27.mlp.gate_proj` 的激活升至 A16，其余 194 个语言 Linear 仍为 A8，
全部语言权重仍为 W4。评分头 W8A16、视觉 W8A8 五档和 FlashAttention Q8/K8/V8/P8 不变。
该文件保留实际正式验证的 `combo_nodes02` 有效配置，仅删除当前 FlashAttention 路径未命中的
根级 `ops.MatMul` 覆盖；其余解析字段完全相同。复用已有 AutoRound W4G64 checkpoint 即可，
不需要再次校准。可通过 `AutoWorkflow.from_config(..., config_path=...)` 显式选择；
公开 `build_model` 的 W4/a8 请求也使用这两个覆盖项，默认原生 W16/a8 请求仍导出基础 W8A8。

| 原 713 条方案 | 语言 Linear A16 数 | 图像宏平均 | 文档宏平均 | 视频宏平均 | 九任务等权宏平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | — | 75.67 | 80.19 | 56.00 | 70.62 |
| 全语言 W4A16、head W8A16 | 196 | 74.33 | 80.28 | 56.00 | 70.20 |
| L26、L27 的 7 个 Linear | 14 | 75.00 | 79.26 | 53.33 | 69.20 |
| L5、L26、L27 的 7 个 Linear | 21 | 74.67 | 79.55 | 54.00 | 69.41 |
| L26 up_proj、L27 gate_proj | 2 | 73.67 | 79.11 | 56.67 | 69.82 |

两节点方案在本次三套局部 A16 中综合得分最高，升位节点最少，因此作为局部 A16 的优先配置；
优势主要来自视频，图像和文档仍是整层方案更好。不将本次子集上的优先选择解释为所有任务最优，
也不将更少 A16 节点直接等同于已经测得的吞吐提升。诊断使用的间隔误差只用于定位，
最终选择依据完整候选集的检索指标。完整 14/21/2 节点名称、四位小数成绩、配置和证据见
[筛选报告 Markdown](evaluation/ACTIVATION_A16_713_COMPARISON_20261028.md)
与 [Word](evaluation/ACTIVATION_A16_713_COMPARISON_20261028.docx)。

### 全语言 A16 参照配置

同目录的 `qwen3_vl_reranker_2b_xh2a_w4a16.yaml` 使用 AutoRound W4G64，
语言 Linear 的基础 scheme 为 `w8a16h1_sefp`，复用整数载荷后实际为 SSFP W4＋A16。
它与两节点方案共享同一权重、校准预算、32K 上下文、chunk=256 和五档动态视觉。
评分 head 保持 W8A16、视觉塔保持 W8A8，FlashAttention 保持 Q8/K8/V8/P8。
使用 `AutoWorkflow.from_config` 显式选择，或通过统一接口指定 W4/a16。

当前语言 Prefill 图没有独立 MatMul，已移除不命中的根级 `ops.MatMul` 覆盖。
FlashAttention 的位宽由 `flash_attention` 单独控制；视觉图中的 MatMul 由独立的
`visual_config.quant_scheme` 控制，不继承语言图的这项覆盖。以后若关闭 FlashAttention，
应重新明确普通 MatMul 两个输入的精度，不能沿用本次无命中的判断。
`quant`、`export` 和 Golden 仍分别调用。历史 W8A16/W16A16 对照使用原始浮点 checkpoint，
对应配置已归档；SEFP W16/A16 是 SDK 的共享指数格式，不代表全模型原生 FP16。

早期三层小模型验证已执行导出、HMONNX 图属性检查、图文重复评分和 Golden，
检查到各有 21 个语言 Linear 使用对应的 W4/W8/W16 与 A16；当时评分 head 为 W8A8。
小模型验证将上下文设为 256、视觉档位设为 `[4,8]`；W4 校准使用 2 条样本、1 次优化。
配置及保护校验回归 44 项、CI policy/recipe 门禁 133 项通过。
证据为 `work_dirs/reranker_a16_config_smoke.json`、`reranker_a16_config_unit.log` 和
`reranker_a16_policy.log`；临时导出产物已清理。之后已完成完整 2B 五精度、head A8/A16 的导出与评测，
以及上述三套局部 A16 的正式验证，以链接的最终报告为准。

模型权重、导出包、Golden、冻结候选、分数和缓存均放在 `work_dirs`，不提交到 Git。
两个临时评测工作区的完整运行资料集中保存在
`/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/`，
具体目录映射和迁移核验见[评测入口](evaluation/README.md)。早期适配记录见 PIPELINE。

AutoRound 通常比直接 RTN 更能保留低比特精度，实际收益取决于模型、校准样本与优化预算。
2B Reranker 对分数变化敏感，应分别检查绝对分数、排序与业务检索指标；不能仅凭压缩比声明精度。
本次实际验证预算与结果见 PIPELINE，不将缩小预算的 smoke 运行写成默认 400 次的质量评估。

## 统一推理

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/hmonnx_demo.py \
  --artifact-dir work_dirs/qwen3_vl_reranker_2b_w8a8 --device cuda:0
```

默认示例比较 Paris 和无关候选。传 `--input-json pairs.json` 可指定：

```json
{
  "query": {"text": "A woman and a dog playing on the beach."},
  "documents": [
    {"image": "data/images/qwen2_vl_demo.jpeg"},
    {"text": "A train crossing a snowy mountain."}
  ],
  "instruction": "Given a search query, retrieve relevant candidates that answer the query."
}
```

Query 也可以携带 `image`；`image` 可以为本地图片路径列表。底层 runtime 另支持 PIL 图片。
统一接口校验路径并返回结构化结果，模板、分块、数值和状态逻辑均由模型包承担。

## 评分与精度对照

官方实现取最终 RMSNorm 之后、最后一个有效 token 的 hidden state，并计算：

```text
logit = (W_yes - W_no) @ hidden_state
score = sigmoid(logit)
```

本地 checkpoint 的 `yes/no` ID 是 9693/2152，实际加载时从 tokenizer 读取，不写死到模型逻辑。
checkpoint 使用 tied embeddings，构造评分权重时读取已正确绑定的 `lm_head.weight`。
评分 head 补齐至 64 个输出以适配现有图算子，只有第 0 项有效，其余权重为零。
它不输出 embedding，也不做 L2 normalize。

超长输入采用官方的保护式截断：保留特殊 token，按顺序保留预算内的普通 token，
完整保留末尾五个 assistant 模板 token。32768 是总容量，正文预算为 32763；
视觉 token 也计入总容量，若仅特殊/视觉 token 就已超限则报错。原生和 vLLM Merak
共用该规则，详见 [超长输入与评分末尾](../../../vllm_merak/docs/qwen3_vl_reranker_merak_deploy.md#超长输入与评分末尾)。

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/compare_native_hmonnx.py \
  --model-dir /data01/datasets/Qwen3-VL-Reranker-2B \
  --metadata work_dirs/qwen3_vl_reranker_2b_w8a8/golden_meta_info.json \
  --image data/images/qwen2_vl_demo.jpeg \
  --output work_dirs/reranker_comparison.json --device cuda:0
```

该程序依次执行 checkpoint 自带官方脚本的 FP16 模型、本仓库浮点 wrapper 和真实 HMONNX，
检查各 pair 的 token ID、attention mask、图片 patch/grid、多模态 token 类型哈希、分数差及候选排序。
官方旧脚本在 Transformers 5 下新增的多模态 token 类型
仅做保值的 Tensor 转换。官方与目标路径使用相同的图片像素上限，图片保留比例；除原六组外，增加五个档位，共 11 组、22 个 pair。这是代表样本后端对照，不是完整检索 benchmark。
默认浮点最大分数差阈值为 0.02，量化最大分数差阈值为 0.1，同时要求各组排序一致。

## Golden

构建不会自动生成 Golden。对完整发布包分别执行：

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/dump_golden.py \
  --metadata work_dirs/qwen3_vl_reranker_2b_w8a8/golden_meta_info.json --device cuda:0
```

```bash
PYTHONPATH="$PWD" python examples_merak/llm/qwen3_vl_reranker/dump_golden.py \
  --metadata work_dirs/qwen3_vl_reranker_2b_w8a8/golden_meta_info.json \
  --image data/images/qwen2_vl_demo.jpeg --device cuda:0
```

合成命令执行五个视觉图的满档位基线，再走完整 rerank 链路；真实图片命令按正常路由采集实际选中的视觉图及 Prefill 图，数据位于各组件的
`golden_synthetic/step_N` 与 `golden_real/step_N`。多 chunk 继续使用 Prefill，包含仅剩一个 token 的尾块。
同名 Golden 目录在本次生成前清理，避免混入旧样本。Golden 属于显式验证产物，不加入模型构建清单。
每个 step 中的图和外部权重使用普通硬链接；当前发布目录禁止软链接，因此生成后仍可通过统一入口加载。

## CI 与证据

新增测试归属当前 CI：`test_qwen3_vl_reranker.py` 覆盖 head、模板、多图、chunk 边界、FlashAttention、真实小语言主干和动态视觉与官方视觉网络的数值对照；
`test_unified_model_qwen3_vl_reranker_2b.py` 使用真实小模型执行构建、搬移、删除源目录、加载及重复/换序推理。
独立的 `test_qwen3_vl_reranker_autoround.py` 执行真实小模型的 AutoRound W4 量化、packed checkpoint 检查、导出、搬移、源权重删除、Golden 和推理。不 mock 被测神经网络或 runtime。

2026-10-28 整理验证：评测工具与 Merak 的 78 项回归、模型/配置及 CI policy/recipe 的 179 项回归通过；
真实小模型的统一构建和 AutoRound 两项端到端测试均通过。额外检查实际 Prefill 图的评分头均为 W8A16，
AutoRound 图包含 21 个 W4G64 语言 Linear，FlashAttention 与 Golden 齐全，原生 GPU 扩展无 Python 回退。
命令、JUnit、环境和图审计保存在 `work_dirs/consolidation_20261028/validation/`；
此轮验证提交整合与配置，没有重新执行 2B 的 713 条评测。

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

具体执行结果、适用范围、源码调用链及工具链版本见 [PIPELINE.md](PIPELINE.md)。
官方资料：[模型与权重](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)、
[评分参考实现](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py)。
