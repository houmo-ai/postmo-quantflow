# Qwen3.6-35B-A3B：MoE 剪枝与 SAFE 精度记录

整理日期：2026-10-20。全量结果来自 2026-09-21 至 2026-10-10 的已有实验，本次没有重新推理。

本卡记录 `moe_prune_v2` 研究版 vLLM 实测结果，供本仓库 SAFE 适配与后续复测参考。
这些分数尚未验证于 ModelZoo 导出模型、HMONNX、SEFP 量化或芯片运行时，不能直接标为这些后端的精度。

## 模型与方法

| 项目 | 记录 |
| --- | --- |
| 模型 | Qwen3.6-35B-A3B，`Qwen3_5MoeForConditionalGeneration` |
| 路由结构 | 40 层，每层 256 个路由专家，原始 Top-K = 8 |
| 模型权重精度 | 研究版推理使用 BF16；不训练或修改专家权重 |
| 研究后端 | vLLM 0.28.0 + `moe_prune_vllm_acep` 插件 |
| 普通 SAFE | ACEP（AMP/AIMER + Top-P）提名候选，ExFold 有向投影表决定安全折叠；实际折叠数量不限奇偶 |
| 偶数 SAFE | 同一流程增加 `nearest_mass` 策略，每个 decode token、每层的实际折叠数为 0/2/4/6 |
| 公共约束 | prefill 保持原始 Top-8；仅 decode 折叠；保护原始 router gate Top-1；系数为正、投影损失有限且 ≤ 1.0；无安全目标则保留 |
| gate 补偿 | 将源 gate × 投影系数累加到保留目标，随后恢复原 Top-K gate 总和 |
| 校准协议 | WikiText-2 raw train，连续 128 × 2048 = 262,144 tokens |

`coverage=0.85` 是重要性累计质量目标，不是固定剪枝率。`nearest_mass` 在奇数可折叠数时比较少折一个和多折一个后的保留质量距离；向上方案须让全部折叠源仍有安全目标，平局向下。因此它不是始终少折一个，最终保留质量也不保证严格 ≥ coverage。

## 全量精度对照

表格内容为 **数据集得分（实测 decode 槽位折叠率）**。Dense 为本项目复现结果，不采用模型卡参考分数。
不同列存在采样设置、输出上限、硬件和 TP 差异，具体协议见下节；尤其不能把偶数 SAFE 与历史 Dense 的差值直接归因为剪枝效果。

| 数据集（Think） | vLLM Dense，历史基线 | 普通 SAFE，coverage=0.85 | 普通 SAFE，coverage=0.875 | 偶数 SAFE，nearest_mass，coverage=0.85 |
| --- | --- | --- | --- | --- |
| RealWorldQA（关，765 条） | 76.86%（0%） | 76.21%（17.61%） | 74.90%（14.55%） | 77.39%（25.18%） |
| RealWorldQA（开，765 条） | 78.82%（0%） | 77.65%（15.16%） | 无全量回执 | 81.31%（25.13%） |
| CC-OCR-V2（关，7,093 条 / 16 子集） | 76.92%（0%） | 76.52%（全量折叠率未闭合，见注） | 无全量回执 | 76.26%（25.12%，TP4） |
| CC-OCR-V2（开，7,093 条 / 16 子集） | 待核对完整分数回执 | 无已核验全量回执 | 无全量回执 | 无已核验全量回执 |

- RealWorldQA 使用 accuracy。历史 Think-on 的 78.82% / 77.65% 为取最后一个大写 `ANSWER` 的严格重评分，分别正确 603 / 594 题；旧 EvalScope 报告误取思考中的首个 `ANSWER`，原始报告分数不作为本卡正式值。
- CC-OCR-V2 使用 EvalScope 官方 `normalized_score`，不是答对题数比例。关 Think 的 macro 分数：Dense 73.54%，普通 SAFE 73.10%，偶数 SAFE TP4 72.58%。
- 普通 SAFE 的 CC-OCR 76.52% 是完整 7,093 条评分；其中 2,390 条复用有效预测、4,703 条重新生成。回执中的 **16.0435%** 折叠率仅属于后一次 4,703 请求服务，不能当作全 7,093 条折叠率。
- 最新偶数 SAFE 的 CC-OCR 主表保留 TP4 的 76.26%。另一次独立 TP2 全量为 76.22%、macro 72.55%、折叠率 25.1203%；两次均各自完整 7,093 条，不合并预测、不平均分数，也不据此声称 TP4 稳定优于 TP2。
- 本次只汇总可核验的既有分数。缺格不填中途快照、子集分数、第三方参考值，也不据旧的运行状态推断任务目前仍在运行。

## 最新偶数 SAFE 的精度与实际计数

以下三项均为 `nearest_mass` 全量验收结果。[S1]、[S2]、[S3] 给出原始来源。

| 数据集（Think） | 全量结果 | decode 折叠槽位 / 原始槽位 | 折叠率 | 评测硬件 |
| --- | --- | --- | --- | --- |
| RealWorldQA（关） | 592 / 765 = 77.385621% | 14,547,312 / 57,768,640 | 25.1820% | 4 × RTX PRO 6000 Blackwell 96GB，TP4 |
| RealWorldQA（开） | 622 / 765 = 81.307190% | 253,215,182 / 1,007,599,680 | 25.1305% | 4 × A100-SXM4-80GB，TP4 |
| CC-OCR-V2（关） | 7,093 条全部完成；normalized_score = 0.7626 | 201,955,880 / 804,019,840 | 25.1183% | 4 × A100-SXM4-80GB，TP4 |

实测折叠率的定义为：`正式数据集 decode_folded_slots / decode_total_slots`。
计数已扣除启动探针；同一运行各 TP rank 的镜像统计一致，只取一份，不将 rank 计数重复相加。
分母为参与该方法的路由专家 Top-8 槽位，不包括共享专家；prefill 剪枝槽位与奇数折叠行均为 0。
这是实际采集的路由计数，尚不是物理专家 FLOPs、执行时间或延迟下降的测量。

验收同时检查了样本数量、索引闭包、API 错误及 rank 一致性；三项均无 API 错误。
RealWorldQA Think-on 有 1 条 reasoning-only 输出达到 32,768 上限，该条保留在 765 条全量分母中。
CC-OCR TP4 的 16 个子集、7,093 条索引及预测完整性通过，空最终文本为 0。

## 评测协议

### 最新偶数 SAFE，2026-10-09 / 10

评测为 EvalScope 1.12 Native；BF16、eager、chunked prefill 开启、prefix cache 关闭，使用 `qwen3` reasoning parser。
三项均 batch_size=64、max_num_seqs=64、max_num_batched_tokens=65,536，generation seed=0。

| 参数 | RealWorldQA Think-off | RealWorldQA Think-on | CC-OCR-V2 Think-off |
| --- | --- | --- | --- |
| max_model_len（输入 + 输出总上下文） | 32,768 | 65,536 | 32,768 |
| max_tokens（输出上限） | 4,096 | 32,768 | 4,096 |
| temperature | 0 | 1 | 0 |
| 采样 top_p | 1.0 | 0.95 | 1.0 |
| 采样 top_k | 20 | 20 | 20 |
| presence_penalty | 0 | 1.5 | 0 |
| repetition_penalty / min_p | 1.0 / 0.0 | 1.0 / 0.0 | 1.0 / 0.0 |

这里的采样 `top_p` / `top_k` 是生成参数，与 MoE 的 coverage / 原始路由 Top-K 不同。
三份配置文件分别为：

- `safe_even_nearest_mass_tp4_rwqa_thinkoff_server29_20261009_config.json`
- `safe_even_nearest_mass_tp4_rwqa_thinkon_server28_20261009_config.json`
- `safe_even_nearest_mass_tp4_ccocr_thinkoff_full_server28_20261010_config.json`

配置存放于下文证据根目录。冻结的研究插件 `exfold.py` SHA256 为：

```text
af16cc73bd30cbf491793e8cdee1a35032502da64e8ed240096a1614c38abae6
```

### 历史普通 SAFE 与 Dense，2026-09

- RealWorldQA Think-off：EvalScope 1.4.1 Native，8 × A100 / TP8，temperature=0、max_tokens=512、eval batch=8。普通 SAFE 两个 coverage 点的正确数分别为 583 / 765、573 / 765；同批 Dense 为 588 / 765。
- RealWorldQA Think-on：EvalScope 1.4.1 Native，8 × A100 / TP8，context=65,536、output=32,768、batch=8、temperature=0、top_p=0.95、top_k=20；未设置 presence penalty 和 reasoning parser。Dense / SAFE 分别有 61 / 66 条输出达到上限，正式精度保留全部样本；不能以排除触顶后的子集精度代替全量结果。
- CC-OCR Think-off：EvalScope 1.12 Native，8 × A100 / TP8，context=32,768、output=4,096、batch/seq=64、batched tokens=131,072、temperature=0、seed=0。Dense 与普通 SAFE 为已有协议对照；SAFE 的缓存修复与折叠计数范围见上文。

历史和最新结果的生成协议不同；当前没有在最新偶数 SAFE 三项设置下全部重跑、逐项匹配的 Dense / 普通 SAFE 对照，因此不把主表跨列差异解释为纯算法增益。

## 与本仓库部署实现的关系

研究入口为 `moe_prune_v2/vllm_acep_plugin/src/moe_prune_vllm_acep/exfold.py::apply_decode_ace_exfold`。
研究代码已归档于 [moe_prune_v2 提交 502d28f](https://github.com/xzktx003/moe_prune_v2/commit/502d28fe4c1fecad68299c50d97e277a1e7172ac)。

ModelZoo 适配说明见 [Qwen3.5/3.6 SAFE decode-only](../docs/qwen3_5_safe_decode.md)。
导出准备会把 AMP/AIMER 预融合为逐专家 scalar，并把 scalar、投影系数和损失表打包后交给 MoE 算子。
研究版使用 FP32 双分支评分且开方内有 eps；ModelZoo v4 默认使用 FP16 融合常量和在线路由，并调整了防零处理。
边界处的排序、保留集合与 gate 数值可能变化，不能把本卡研究版分数直接复制给适配后的算子。

| 后端 / 产物 | 本卡已有证据 |
| --- | --- |
| 研究版 vLLM BF16 + SAFE / 偶数 SAFE | 上述全量实测精度及路由计数 |
| ModelZoo SAFE v4 导出 / xhquanttool MoE | 尚缺同 checkpoint、同数据集、同采样参数的全量精度回执 |
| HMONNX、SEFP 量化模型或芯片运行时 | 尚缺本卡所述 SAFE 配置的全量精度与物理加速回执 |
| PPL、实际专家 FLOPs、TTFT / TPOT、同后端延迟改善 | 本卡未登记对应的已核验测量 |

旧 `floor` 偶数策略的 77.12% / 9.6171% 已撤回归档，不计入 `nearest_mass`。
102 条触顶困难子集、CC-OCR 分片和启动失败记录也不进入全量主表。
普通 SAFE 的 coverage 从 0.85 提高到 0.875 后，RealWorldQA 精度反而降低，已有结果不支持精度随折叠率单调变化的结论。

## 证据索引

以下均是本机保存的 JSON 回执；相对链接跨到研究工程，离开本工作区时需另行带上回执。
完整原始 prediction、review、report 路径及其哈希由对应回执记录。本次核对回执、配置和冻结代码，没有重新读取远端全部原始预测。

证据根目录：`workspace/ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/`。

| 编号 | 回执 |
| --- | --- |
| S1 | [偶数 SAFE 全量 RealWorldQA 开 / 关 Think](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_safe_even_nearest_mass_dual_host_tp4_20261009.json) |
| S2 | [偶数 SAFE CC-OCR 关 Think TP4 全量](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_safe_even_nearest_mass_tp4_ccocr_thinkoff_full_server28_20261010.json) |
| S3 | [偶数 SAFE CC-OCR 关 Think TP2 独立全量](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_safe_even_nearest_mass_tp2_ccocr_thinkoff_full_server28_20261010.json) |
| S4 | [普通 SAFE RealWorldQA 关 Think，两种 coverage](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_realworldqa_safe_decode_ace_exfold_full_20260921.json) |
| S5 | [普通 SAFE RealWorldQA 开 Think](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_realworldqa_safe_thinkon_c085_tp8_max32k_20260922.json) |
| S6 | [Dense RealWorldQA 开 Think](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_realworldqa_dense_thinkon_tp8_max32k_20260923.json) |
| S7 | [普通 SAFE CC-OCR 关 Think，含缓存及计数范围说明](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_ccocr_safe_c085_tp8_retry2_32k_20260922.json) |
| S8 | [Dense CC-OCR 关 Think](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_ccocr_dense_tp8_20260922.json) |
| S9 | [Dense RealWorldQA 关 Think及 ACEP 历史对照](../../../ai_agent/paper_wrighting/papers/moe_dynamic_topk/experiments/results/qwen36_35b_realworldqa_topp_ace_085_tp8_20260921.json) |

本次核对时的回执 SHA256：

```text
S1 b9def8252b08cdc2a2109109bcca5bd33db858215b459ea6cf5298a2dba82bc5
S2 62988c43cf4788cf46c2ef09fd0d00391bb8de689db5d24bec8f3a3232303141
S3 63b0d3227e6eee6a3180df860ce04388695f80abbb691f0c298e038901766cb8
S4 bf70649beb9ee8f839b553c39402a9aaec7d39241ec1b59ff0001033c80841b9
S5 bc657716f9409d59e11e26ce12a29f6cb342db079f506b5e1ac92f891e38ff22
S6 71bcbcfb6ae7e9131f89b0279d1c04c7736c265122ce72510de23f173d932aa9
S7 aec236612aaf58e86aa36428b8e36f884eb5cd92ccc9df5d8569ece3ae2d6fe9
S8 f0654483551e2f1547b8601696e220d9ff1893b5aba49812466d7bd7d3439057
S9 15cc1236b23bab17ec928fed38b8804477cfd582ee5f000003e9a363c7487a1f
```
