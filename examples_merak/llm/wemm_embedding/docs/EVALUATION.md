# WeMM-Embedding 评测配置与结果索引

整理日期：2026-10-16。2B、4B、9B 的自测浮点与 W4A8 均已完成 MMEB-V3 非音频全量评测。
本文件记录实际运行配置、主要代码入口、最终报告位置和清理后的留档方式。

## 评测范围与评分

每份结果保留 182 个执行任务：179 个官方非音频任务和 3 个 VisDoc 诊断任务。
官方汇总任务为 Image 36、Video 18、VisDoc 24、Text 53、Tool 35、GUI 8、Memory 4、MCMR 1。
MCMR 的原始结果位于 image 模态，但单独计分，不进入 Image 或 MMEB-v2。

| 项目 | 规则 |
|---|---|
| Text、VisDoc 主指标 | `ndcg_linear@5` × 100 |
| Image、Video、Tool、GUI、Memory、MCMR 主指标 | `hit@1` × 100 |
| 分类汇总 | 该分类任务分数的算术平均，不按样本量加权 |
| MMEB-v2 | 36 个 Image + 18 个 Video + 24 个 VisDoc，共 78 项 |
| Agent | 35 个 Tool + 8 个 GUI + 4 个 Memory，共 47 项 |
| Supported-Only | 179 个官方非音频任务分数之和 ÷ 179 |
| V3-All | 179 个非音频任务分数之和 ÷ 190；11 个音频任务按 WeMM 官方约定记零 |
| 量化差值 | W4A8 − 自测浮点，单位为百分点 |

WeMM 不支持音频，报告中的 V3-All 包含 11 个零分音频任务，不表示执行过音频推理。
以下诊断任务保留逐任务结果，但不进入 VisDoc、MMEB-v2、Supported-Only 或 V3-All：

- `ViDoRe_biomedical_lectures_v2`
- `ViDoRe_economics_reports_v2`
- `ViDoRe_esg_reports_v2`

2026-10-16 修正了通用聚合器原先排错诊断任务名称的问题。旧聚合器误排除
`ViDoSeek-doc`、`MMLongBench-doc`、`MMLongBench-page`，导致部分旧 `summary.json`
与按官方任务映射生成的最终报告不一致。本次从未修改的原始分数重新汇总，并验证
4B/9B 官方参考文件中的 179 个非音频任务集合；三个尺寸使用同一正确集合。
新目录内的 `REPORT.md`、`summary.json` 和对比报告采用统一口径。

## 最终报告位置

可随仓库保存的 [MMEB-V3 精简最终报告](MMEB_V3_REPORT.md) 包含 2B、4B、9B 各自的浮点自测与 W4A8 HMONNX 分类对比。模型卡位于 `../model_cards/m50/`，`precision.comparisons` 的 evidence 指向该报告对应尺寸的章节。

以下为本机完整运行归档，包含原始逐任务结果，不随 Git 拉取。

本机结果根目录：

```text
/data01/home/jing.huang/xh2modelzoo/work_dirs/wemm_embedding/mmeb_v3_runs
```

| 模型 | 自测浮点 vs 官方浮点 | W4A8 vs 官方浮点 | W4A8 vs 自测浮点 |
|---|---|---|---|
| 2B | [float/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/2b/float/aggregate/REPORT.md) | [w4a8/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/2b/w4a8/aggregate/REPORT.md) | [comparison/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/2b/comparison/REPORT.md) |
| 4B | [float/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/4b/float/aggregate/REPORT.md) | [w4a8/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/4b/w4a8/aggregate/REPORT.md) | [comparison/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/4b/comparison/REPORT.md) |
| 9B | [float/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/9b/float/aggregate/REPORT.md) | [w4a8/aggregate/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/9b/w4a8/aggregate/REPORT.md) | [comparison/REPORT.md](../../../../work_dirs/wemm_embedding/mmeb_v3_runs/9b/comparison/REPORT.md) |

4B/9B 官方成绩使用根目录 `references/WeMM-Embedding-{4B,9B}.json` 中的逐任务分数，
2B 官方汇总参考使用聚合器内保存的官方公布值，精度为一位小数。量化精度变化以自测浮点为基准。

| 模型 | 自测浮点 V3-All | W4A8 V3-All | 差值 | 自测浮点 Supported-Only | W4A8 Supported-Only | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 55.75 | 55.25 | -0.50 | 59.18 | 58.64 | -0.53 |
| 4B | 57.93 | 57.88 | -0.05 | 61.49 | 61.44 | -0.05 |
| 9B | 59.33 | 58.80 | -0.53 | 62.98 | 62.41 | -0.57 |

每个 `float/` 或 `w4a8/` 目录包含：

```text
evaluation_config.json         实际运行参数摘要，包括并发、批大小、显存设置、分片和执行批次
run_manifest.json              归档任务清单，以及原始文件的 SHA256 校验值
aggregate/REPORT.md            当前后端与官方浮点的分类对比
aggregate/summary.json         完整机器可读汇总、逐任务分数、协议检查结果
aggregate/per_task.csv         182 项分数、指标、是否进入汇总、官方值和有效文件路径
results/<modality>/<dataset>/
    score.json                 原始完整指标文件，字节内容未修改
    protocol.json              原始运行协议，字节内容未修改
    config.yaml                原始任务配置，字节内容未修改
    timing.csv                 原始计时记录（原运行提供时保留）
evidence/                      原始 manifest、运行状态、补跑历史、启动命令和迁移校验记录
```

每个 `comparison/` 目录包含 `REPORT.md`、`comparison.json`、`per_task_delta.csv`。
逐任务对照包含 182 项，分类汇总排除三个诊断项。完整清理记录在根目录 `cleanup_manifest.json`。

## 实际评测配置

以下值来自留存的 `protocol.json` 和原始 manifest，而非当前 Python 环境版本。

| 配置 | 实际值 |
|---|---|
| 官方评测代码版本 | `5f539a2f5196b08bbcf29ac71fc90a95ccad7f11` |
| MMEB-V3 数据版本 | `4a5560b2b64384204b6fea8a82ea986eba51f5aa` |
| MMEB-v2 media 版本 | `e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1` |
| 浮点后端 | 原生 vLLM + WeMM 专用模型及 processor 注册 |
| 量化后端 | vLLM Merak + HMONNX 导出图；未回退到 HF 浮点模型 |
| vLLM | `0.23.1.dev0+g0fc695fc6.d20260921.precompiled` |
| Transformers | `5.13.0` |
| PyTorch | `2.11.0+cu129` |
| 运行环境 | `/data01/home/jing.huang/miniconda3/envs/vllm/bin/python` |
| 最大上下文 | `262144` processed tokens，包含模板、特殊 token 和视觉 token |
| 浮点 `max_num_seqs` | `16`，表示引擎最多同时调度的序列数 |
| W4A8 `max_num_seqs` | `1` |
| 评测器 batch size | Text/GUI 为 `1`，其他模态为 `8` |
| DataLoader workers | `4` |
| Prefix caching | 关闭 |
| 输出 | `<embedding>` 锚点的最后一层 hidden state，FP32 L2 归一化 |
| 超长文本 | 按官方协议截断，保留末尾 `<embedding>` 锚点，并留 1 token 安全余量 |

`max_num_seqs=16` 与最大上下文长度无关。独立多引擎与 `tensor_parallel_size=4` 也不同：
本次每个引擎独立执行模型，GPU 之间分配数据集或 embedding 分片，不将一个模型按四卡张量并行切分。

| 运行 | 主要 GPU / 引擎安排 | 显存利用率参数 | 备注 |
|---|---|---:|---|
| 2B 浮点 | GPU 0–3，每卡 1 引擎 | 0.60 | 数据集分组，每组最多 8 个数据集 |
| 2B W4A8 | GPU 0–3；历史阶段由每卡 1 引擎调整为每卡 2 引擎 | 0.60 | 最终合并 138 项既有有效结果、36 项加速补跑、8 项修复后 GUI 重跑 |
| 4B 浮点 | GPU 4–7，每卡 2 引擎 | 0.45 | 独立数据集动态领取；部分任务分片；失败项补跑 |
| 4B W4A8 | GPU 0–3，每卡 2 引擎 | 0.60 | 独立数据集动态领取；大任务 4 分片；失败项补跑 |
| 9B 浮点 | GPU 4–7，每卡 2 引擎 | 0.58 | 每引擎 KV cache 9 GiB；大任务 4 分片；失败项补跑 |
| 9B W4A8 | GPU 0–3，每卡 2 引擎 | 0.60 | 独立数据集动态领取；大任务 4 分片；失败项补跑 |

补跑阶段可能改为每卡一个引擎；准确的 GPU、任务耗时、启动参数及重试结果保存在各目录的 `evidence/`。
2B W4A8 是经过验证的分阶段合并结果，不能称为所有任务使用同一个代码哈希的一次运行。
`evidence/original_merge_provenance.json` 保留原始分阶段策略、各批次哈希和逐任务归属；
三个批次的原始 manifest/status 也分别保存，未为获得统一外观而改写旧哈希。

### 图像与视频

| 配置 | 实际值 |
|---|---|
| 图像 | 保持宽高比，动态分辨率；不固定为 448×448 |
| 图像最小 / 最大像素 | `65536` / `1572864` |
| 每输入最多图像数 | `16` |
| 量化视觉图 | patches 输入，合并后 token 档位 `[96, 196, 384, 704, 1536]` |
| 视觉图选择 | 最小可容纳档位；补齐静态图，保留有效 patch mask；超容量报错 |
| 视频评测帧数 | 官方数据集 YAML 的 `num_frames=64`，不足时按数据集规则处理 |
| 视频 `frame_size` 参数 | `224`；不是强制所有最终 processor 帧均为 224×224 |
| 视频 processor 像素范围 | 每帧 `4096`–`262144`，总预算 `8388608` |
| 视频采样 | 官方帧序列及均匀采样顺序，消息采样 `fps=1`、最多 64 帧 |
| WeMM 视频网格 | `[T,H,W]` 按官方模型语义展开为 T 个 `[1,H,W]` 帧组 |

### 模型与导出配置

浮点模型位于 `/data01/datasets/WeMM-Embedding-{2B,4B,9B}`。
W4A8 产物位于 `work_dirs/wemm_embedding/<size>_256k_context/wemm_embedding_export/` 下的实际导出子目录；
各份 `evaluation_config.json` 记录完整模型路径。对应 recipe 为：

```text
configs_merak/workflows/xh2a/llm_models/wemm_embedding/2b/wemm_embedding_2b_256k_context_xh2a_w4a8.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/4b/wemm_embedding_4b_256k_context_xh2a_w4a8.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/9b/wemm_embedding_9b_256k_context_xh2a_w4a8.yaml
```

语言分支 AutoRound：W4、group size 64、`int_sym_xh`、256 个校准样本、400 次迭代、seed 42。
256K 是导出上下文配置，不表示重新用 256K 的校准序列量化权重。语言图 W4A8，视觉图 W8A8。
`prefill_chunk_length=256`、`linear_chunk_size=64`；FlashAttention 开启，Q/K/V/S/P 均为 8 bit。
已评测产物的 `merak_config.json` 中 `enable_page_attention=true`、`runner_type=pooling`。
vLLM 的外层调度预算与 HMONNX 单次 256 token prefill 图长度是两层不同的分块。

## 保留的主要代码

### 本次脚本整理与兼容性

- `mmeb_v3_eval.py` 增加按连续 query/candidate 范围编码的分片模式，保存 FP32 embedding、样本 ID、范围和维度清单；通过临时文件和原子替换发布完整分片。分片阶段不计算局部检索分数。
- `run_mmeb_v3_vllm_merak_full.py` 增加独立数据集作业、动态领取和指定大任务分片；核对协议、源码/配置哈希及完整分片范围，按原始顺序合并 embedding，再调用官方 evaluator 对完整候选池统一评分。失败任务记录在运行状态中，其他作业继续执行；有效结果和有效分片可在相同条件下续跑复用。已完成的归档拒绝作为新推理输出目录。
- 聚合器排除三项 ViDoRe 诊断任务，保留 MMLongBench 和 ViDoSeek 正式文档任务。此修正影响旧汇总数值，当前报告使用修正后的任务集合；已有原始 `score.json` 未改写。
- 模型卡路径统一为 `model_cards/m50/wemm_embedding_{2b,4b,9b}.yaml`，旧的直接文件路径已移除。显式传入旧路径的调用方需要更新路径；默认按 M50 扫描可发现新卡片。
- 本次新增的成绩使用 `float/hmonnx/delta`，只读取旧 `quant` 字段的消费者需增加 `hmonnx` 读取支持。Schema 继续接受旧 `quant`，也允许迁移期间两字段并存；这不是互斥字段约束。当前 WeMM 卡片仅写 `hmonnx`，避免重复数值来源。未引入 HM-Eval 的条件分组发布实现。
- 插件包装测试同步更新评测模块路径，并调整长行格式；Schema 校验器也做了局部格式整理。

| 入口 | 用途 |
|---|---|
| [run_mmeb_v3_vllm_merak_full.py](../mmeb/run_mmeb_v3_vllm_merak_full.py) | 主评测调度：按数据集动态领取、大任务分片、协议校验、续跑、全池合并评分 |
| [mmeb_v3_eval.py](../mmeb/mmeb_v3_eval.py) | 调用固定官方 evaluator、数据 parser 与 ranking metrics；支持 HF 对齐及目标后端 |
| [mmeb_v3_vllm_merak.py](../mmeb/mmeb_v3_vllm_merak.py) | 原生 vLLM / Merak embedding 后端适配 |
| [aggregate_mmeb_v3.py](../mmeb/aggregate_mmeb_v3.py) | 检查协议并按正确 179 项口径生成汇总、逐任务表及官方对比 |
| [compare_mmeb_v3_aggregates.py](../mmeb/compare_mmeb_v3_aggregates.py) | 逐任务核对输入配置和前后处理，生成 W4A8 vs 自测浮点报告 |
| [run_accuracy_suite.sh](../flickr/run_accuracy_suite.sh) | 可选 Flickr30K 完整图文检索与 cosine 校验；复用同目录 Flickr30K helper |
| [compare_native_hmonnx.py](../checks/compare_native_hmonnx.py) | 可选小样本官方浮点 / HMONNX 输出对齐 |
| [evaluate_wemm_embedding.py](../checks/evaluate_wemm_embedding.py) | 可选小样本文本检索冒烟测试，不作为 MMEB-V3 正式得分 |
| [audit_dynamic_resolution.py](../checks/audit_dynamic_resolution.py) | 可选真实图像动态尺寸与档位审计 |

独立 watcher、阶段交接控制器、临时修复副本、旧两项图像任务脚本和固定串行队列已清理。
调度、分片和续跑逻辑集中在主调度器中；新评测需要失败补跑时可对相同有效输出重新运行主调度器。
运行前仍需验证当前环境和代码的官方输入输出对齐，不能把旧环境记录当作新环境验证结论。

## 重新运行与重建报告

先激活 vLLM 环境并确认插件从当前 ModelZoo 的 `vllm_merak/` 导入。
下面是新一轮 4B W4A8 的启动示例，输出使用新目录，不覆盖已归档结果。

先在终端创建持久会话：

```bash
tmux new-session -s wemm_4b_new
```

进入该会话后运行：

```bash
conda activate vllm
cd /data01/home/jing.huang/xh2modelzoo
export PYTHONPATH="$PWD/vllm_merak:$PWD:$PWD/work_dirs/wemm_embedding/mmeb_v3_official/mmeb_v3_eval${PYTHONPATH:+:$PYTHONPATH}"
python -u examples_merak/llm/wemm_embedding/mmeb/run_mmeb_v3_vllm_merak_full.py \
  --repo "$PWD" --backend vllm_merak --vllm-merak "$PWD/vllm_merak" \
  --official "$PWD/work_dirs/wemm_embedding/mmeb_v3_official/mmeb_v3_eval" \
  --data "$PWD/work_dirs/wemm_embedding/mmeb_v3_data" \
  --model "$PWD/work_dirs/wemm_embedding/4b_256k_context/wemm_embedding_export/hmquant_xh2_wemm_embedding_4b_w4a8_256_256k_mpe256k_visualm96_196_384_704_1536_20260922" \
  --source-model /data01/datasets/WeMM-Embedding-4B \
  --output "$PWD/work_dirs/wemm_embedding/mmeb_v3_runs/new_4b_w4a8" \
  --python /data01/home/jing.huang/miniconda3/envs/vllm/bin/python \
  --gpus 0,0,1,1,2,2,3,3 --job-mode dataset \
  --batch-size 8 --dataloader-workers 4 --vllm-max-num-seqs 1 \
  --gpu-memory-utilization 0.60 \
  --shard-dataset video:NExTQA --shard-dataset video:MVBench \
  --shard-dataset image:MCMR --shard-dataset text:text/LongEmbed/narrativeqa \
  --shard-dataset text:text/BRIGHT/leetcode --shard-dataset gui:GAE-Mind2Web_q2t \
  --shards-per-dataset 4 --retries 1 \
  > work_dirs/wemm_embedding/mmeb_v3_runs/new_4b_w4a8.log 2>&1
```

按 `Ctrl+B`、再按 `D` 脱离会话；使用 `tmux attach -t wemm_4b_new` 返回。

其他大任务可按 `evaluation_config.json` 的 `shard_datasets` 追加 `--shard-dataset`。
`--gpus 0,0,1,1,2,2,3,3` 表示四卡八独立引擎；需要单引擎补跑时改为 `0,1,2,3`。
浮点评测改为 `--backend vllm --model /data01/datasets/WeMM-Embedding-4B --vllm-max-num-seqs 16`，
显存参数按模型和实测容量设置。9B 浮点历史运行还配置了每引擎 9 GiB KV cache；
其实际启动参数记录在 `9b/float/evidence/` 中，不能只更换模型路径就假定显存足够。

tmux 会话与 SSH/VS Code 窗口独立。退出服务器不会结束会话，服务器重启会结束进程；
这种情况下在同一个新运行输出目录重启主调度器，只有协议和哈希一致的有效任务或分片会被复用。
分片只并行生成 embedding；全部分片合并后仍对完整 candidate 池进行官方评分，不平均局部分数。

重建已归档的 4B 报告不需要 GPU：

```bash
python examples_merak/llm/wemm_embedding/mmeb/aggregate_mmeb_v3.py \
  work_dirs/wemm_embedding/mmeb_v3_runs/4b/float \
  --title 'WeMM-Embedding-4B float MMEB-V3 256K' \
  --official-scores work_dirs/wemm_embedding/mmeb_v3_runs/references/WeMM-Embedding-4B.json
python examples_merak/llm/wemm_embedding/mmeb/aggregate_mmeb_v3.py \
  work_dirs/wemm_embedding/mmeb_v3_runs/4b/w4a8 \
  --title 'WeMM-Embedding-4B W4A8 MMEB-V3 256K' \
  --official-scores work_dirs/wemm_embedding/mmeb_v3_runs/references/WeMM-Embedding-4B.json
python examples_merak/llm/wemm_embedding/mmeb/compare_mmeb_v3_aggregates.py \
  work_dirs/wemm_embedding/mmeb_v3_runs/4b/float \
  work_dirs/wemm_embedding/mmeb_v3_runs/4b/w4a8 \
  --title 'WeMM-Embedding-4B MMEB-V3：W4A8 vs 自测浮点' \
  --output work_dirs/wemm_embedding/mmeb_v3_runs/4b/comparison
```

9B 替换尺寸和官方参考文件；2B 省略 `--official-scores`，使用保存的官方公布汇总值。
归档模式校验保留文件的 SHA256 和原始运行 manifest，不拿今天修改后的源码哈希否定历史结果。
新推理运行继续校验当前 adapter、模型和插件哈希，不放宽续跑门槛。

## 清理与来源记录

结果目录按尺寸统一为 `{2b,4b,9b}/{float,w4a8,comparison}`，旧运行目录名与新位置的对应关系
记录在根目录 `cleanup_manifest.json`。

删除内容包括已被完整结果覆盖的快速测试、错误/中止运行、embedding/query/candidate 缓存、
分片中间产物、逐样本预测与输入明细、临时日志、PID/lock、监控器及无关的整仓源码副本。
保留官方数据和模型权重/导出图；它们不在此次结果目录清理范围内。

每份结果保留原始分数、协议、任务 YAML、启动命令、执行及补跑状态。
`evidence/file_receipts.json` 记录原路径、新路径和内容哈希。
原始协议和 manifest 内的历史绝对路径作为运行证据原样保留；当前有效文件路径使用新
`run_manifest.json`、`aggregate/per_task.csv`，无需恢复旧目录。
相关冻结评测/插件/模型代码压缩保存在 `sources/*.tar.gz`，文件明细和哈希见 `sources/manifest.json`；
不保留整仓副本及可执行监控器。`validation/` 保留小体积的前后处理对齐与代码恢复一致性证据，
不参与正式汇总。

清理后能重新校验分数并重建报告；不能从已删除 embedding 或预测明细重新排名、重算指标，
也不能恢复此前推理缓存。需要改变评分指标或复查逐样本排名时，应在新输出目录重新运行推理。
主调度器会拒绝将已完成的分数归档目录当作推理输出，避免覆盖留档结果。
