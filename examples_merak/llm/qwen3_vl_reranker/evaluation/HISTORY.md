# 评测批次与完成状态

整理：2026-10-28。此处合并原分模型启动、自动接续、评分头重导出和扩样记录，
只保留最终状态及证据入口；旧文中的“进行中”不再作为当前状态。
正式 713 查询成绩见[五模型 head A8/A16](HEAD_A8_A16_713_REPORT.md)和
[三套局部 A16](ACTIVATION_A16_713_COMPARISON_20261028.md)，诊断结论见
[精度诊断与配置选择](ACCURACY_DIAGNOSIS_FINAL.md)。

## 批次时间线

| 日期 | 批次 | 最终状态与区别 |
| --- | --- | --- |
| 10月23–24日 | 原评分头 W8A8 的精简集 | BF16 及六种量化配置均完成原 713 查询；六种包括五种语言精度和 W4A8＋视觉 W8A16。分模型启动和接续记录已被最终结果取代。 |
| 10月24–25日 | head W8A16 五精度复评 | 五轮于10月25日01:29全部完成，每轮 713 查询、66,809 候选对；W8A8 使用 corrected r2，先前额外改变视觉精度的包未评测。 |
| 10月26日 | head W16A16 试验 | 五套导出完成；正式评测只完成 W4A8、W8A8 各 713 查询，W4A16 完成 408 条后按用户要求停止，W8A16/W16A16 未开始。不得标记五模型评测完成。 |
| 10月26–27日 | 同九任务扩至 3414 查询 | BF16、W8A8＋head W8A16、W4A8＋head W8A16 全部完成；W4A16 完成 2784/3414 后按用户要求停止。它与原 713 查询成绩分开报告。 |
| 10月27–28日 | 语言激活定位与三方案验证 | 135 查询用于定位，完成 28 层、21 个层内 Linear 及组合对照；两整层、三整层、两节点三套方案均完成原 713 查询，10月28日02:32生成最终报告。未再运行 3414 条验证。 |

每个模型、评分策略和查询集合保留独立协议。相同输入可复用原始媒体、召回、标签及兼容视觉缓存，
不同模型或评分头之间不复用重排分数。停止状态保留已完成查询与候选批次，不把局部分数冒充整轮指标。

## 原 713 查询与扩样的边界

原集为九任务 713 查询、66,809 个候选对。扩样保持任务、原查询和原候选身份，
增至 3414 查询、314,431 个候选对；新增 2701 查询和 247,622 候选对。
仅同模型、同协议的原 713 结果可继承到扩样，新增查询需要实际推理。
扩样仍不是官方完整 MMEB-v2，不能与原 713 直接混合平均。

三套局部 A16 的验证又回到原 713 条：135 条定位集、135 条筛选备用集、443 条未参与本轮选择的保留集。
保留集不是新增数据；历史基线可见，因此也不是新的外部盲测。

## 最终状态和原始证据

下列运行资料均已进入长期归档，查看结果不需要重新启动旧 watchdog 或 supervisor。
原 plan/SQLite 中的旧绝对路径按归档 `PATH_MAP.json` 定位，不能修改 SHA 绑定文件来强行恢复旧协议。

| 批次 | 归档入口 |
| --- | --- |
| 原评分头六精度最终汇总 | [MMEB_QUICK_EVALUATION_REPORT.md](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/reports/MMEB_QUICK_EVALUATION_REPORT.md) |
| head W8A16 完成与复算 | [最终汇总](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/headw8a16_20261024/FINAL_SUMMARY_20261026.md)、[完成审计](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/headw8a16_20261024/completion_audit_20261026.json) |
| head W16A16 停止记录 | [stop_requested_20261026.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/headw16a16_20261026/stop_requested_20261026.json) |
| 3414 条 BF16/W8A8 对比 | [REPORT.md](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/expanded_mmeb_20261026/REPORT.md) |
| 扩样 W4A8 完成、W4A16 停止 | [W4A8 quant_summary.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/expanded_mmeb_20261026/w4a8_headw8a16/quant_summary.json)、[successors/pipeline.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/expanded_mmeb_20261026/successors/pipeline.json) |
| 三套激活方案完成记录 | [followup_nodes02/completion.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/activation_sensitivity_20261027/followup_nodes02/completion.json) |

## 整理前文档快照

详细预检、设备分配、启动命令和过程吞吐保存在以下逐文件快照；原日期、数值及旧状态未改写。
这些文件用于追溯，不作为仍有评测进程运行的说明。

[完整快照清单及逐文件 SHA](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/snapshot_manifest.json)
还记录了整理前的缓存说明、重复索引、迁移清单和三套原始摘要。清单中的 `snapshot` 为完整目录，
旧文内的历史主仓路径可按该目录与相对文件名定位；快照本身保留原字节。

| 原文档 | 保留内容 |
| --- | --- |
| [EVALUATION_REPORT_20261023.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/EVALUATION_REPORT_20261023.md>) | 早期已完成任务、原计划浮点附录和缓存/吞吐记录 |
| [W4A16_EVALUATION.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/W4A16_EVALUATION.md>) | 原 head W8A8 的 W4A16 预检与启动 |
| [W8A16_EVALUATION.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/W8A16_EVALUATION.md>) | 原 head W8A8 的 W8A16 预检与启动 |
| [SUCCESSOR_EVALUATIONS.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/SUCCESSOR_EVALUATIONS.md>) | 两组 GPU 上的 W16A16、视觉 W8A16 接续调度 |
| [HEAD_W8A16_EVALUATIONS.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/HEAD_W8A16_EVALUATIONS.md>) | head W8A16 五模型准入、r2 修正和完成性检查 |
| [HEAD_W16A16_EVALUATIONS.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/HEAD_W16A16_EVALUATIONS.md>) | 五套重导出、Golden 和中止时保留范围 |
| [EXPANDED_EVALUATION.md](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/EXPANDED_EVALUATION.md>) | 扩样清单、增量复用与接续规则 |
