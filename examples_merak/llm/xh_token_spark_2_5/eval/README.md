# Spark-X2.5 vLLM 服务评测

从 `ifly_spark/examples_merak/llm/xh_token_spark_2_5/` 迁移的 HTTP 评测客户端。
通过已有服务的 `/v1/models`、`/v1/chat/completions` 和 `/tokenize` 运行六项评测；
权重、tokenizer 与 chat template 由服务端加载。

本目录包含：

| 文件 | 用途 |
| --- | --- |
| `vllm_spark_tokens_eval.py` | 流式推理、预算检查、断点续跑与结果汇总 |
| `eval_common.py` | 从原 XH2 入口提取的数据读取、提示词、基础评分和报告函数 |
| `mini_eval_vllm_128k.yaml` | 默认 128K 套件，IFEval 生成预算为 130000 |
| `mini_eval.yaml` | 原始套件，保留原预算，IFEval 生成预算为 1024000 |
| `download_mini_eval_datasets.py` | 下载固定版本数据并规范为 JSONL |
| `score_spark_tokens_ifeval.py` | 调用外部 Google IFEval 评分器 |
| `requirements.txt` / `requirements-ifeval.txt` | 客户端、数据下载及可选评分依赖 |

公共模块保留原函数行为，排除了模型加载和 XH2 推理代码。迁移后的默认数据、输出路径
指向 **xh2modelzoo 仓库根目录**。默认套件改为 `mini_eval_vllm_128k.yaml`；两份 YAML
内容均保持原样。原仓库代码、数据、预测结果和服务运行环境保留在原位置。

## 准备环境和数据

以下命令在 xh2modelzoo 仓库根目录执行。客户端只需 Python 3.10+ 与 PyYAML；
数据下载另需 huggingface-hub，BFCL 下载还需系统 `git`。

```bash
python -m pip install -r examples_merak/llm/xh_token_spark_2_5/eval/requirements.txt
python examples_merak/llm/xh_token_spark_2_5/eval/download_mini_eval_datasets.py --dry-run
python examples_merak/llm/xh_token_spark_2_5/eval/download_mini_eval_datasets.py
```

数据默认写到 `data/evaluation/spark_x2_5/`。GPQA 源数据需要已获授权的 Hugging Face
账号，可通过 `HF_TOKEN` 提供凭据。下载脚本保留原数据源和版本，并写出
`download_manifest.json`。也可复用已准备的数据，通过 `--data-dir /path/to/spark_x2_5`
指定目录，或多次传入 `--dataset NAME=PATH` 指定单项文件。已有预测对应的数据版本
应以原运行的 `run_config.json` 和下载清单为准。

## 运行评测

先检查服务别名、数据和预算：

```bash
python examples_merak/llm/xh_token_spark_2_5/eval/vllm_spark_tokens_eval.py \
    --base-url http://127.0.0.1:18006/v1 \
    --model spark2_5 \
    --suite examples_merak/llm/xh_token_spark_2_5/eval/mini_eval_vllm_128k.yaml \
    --data-dir data/evaluation/spark_x2_5 \
    --thinking --do-sample --temperature 1.0 --top-p 0.95 --top-k -1 \
    --seed 1024 \
    --output-dir work_dirs/eval_spark2_5_1_7b_vllm_128k \
    --dry-run
```

将 `--dry-run` 替换为 `--resume` 开始或继续评测；首次冒烟可增加
`--benchmarks IFEval --limit 2` 并使用独立输出目录。省略 `--benchmarks` 运行全部六项。
`--model` 是服务当前加载模型的别名，单次不会自动切换 YAML 中列出的模型或精度。
服务应提供 131072 的总上下文容量；输入与生成预算之和仍会逐题检查。

128K 套件的生成预算为：

| Benchmark | max_new_tokens |
| --- | ---: |
| IFEval | 130000 |
| GPQA-Diamond | 8192 |
| AIME-2026 | 16384 |
| AA-LCR | 4096 |
| BFCL-V4 | 4096 |
| SciCode | 16384 |

`--max-new-tokens` 可统一覆盖各项预算。保留原入口的参数规则：显式添加
`--do-sample` 启用采样；省略时按默认贪心生成。`--thinking` / `--no-thinking` 可覆盖
YAML 模式。接口鉴权使用 `VLLM_API_KEY`，密钥不写入报告。

`--dry-run` 检查整体计划，不生成回答；实际评测再逐题调用 `/tokenize` 检查输入加输出
预算。超限不会截断输入或缩小输出预算。`finish_reason=length` 保存文本并记为失败。
输出包括 `run_config.json`、`results.json`、`benchmark.md` 和各项 `predictions.jsonl`，
记录请求、最终回答、thinking、工具调用、token 用量与失败原因。

`--resume` 跳过成功题目并重试失败题目；配置或数据变化时拒绝复用结果。
迁移后数据文件的绝对路径变化也会触发该检查。继续原来的运行时，应保留原
`--data-dir` / `--dataset` 与 `--output-dir`，不要改写原运行配置绕过检查。
服务原地替换权重后应使用新输出目录。失败样本或评分器异常导致非零退出码。

## IFEval 官方评分

评分入口只读取预测文件。Google 的 `instruction_following_eval/` 包由外部提供，
本目录未复制该第三方源码；可复用原工作区已有评分器。
原评测使用 google-research commit `e6890f85757dd84e27ca6df2dd30651dafad28e0`。
评分环境需安装依赖和 NLTK 分句资源：

```bash
python -m pip install -r examples_merak/llm/xh_token_spark_2_5/eval/requirements-ifeval.txt
python -m nltk.downloader -d work_dirs/nltk_data punkt_tab
NLTK_DATA="$PWD/work_dirs/nltk_data" python \
    examples_merak/llm/xh_token_spark_2_5/eval/score_spark_tokens_ifeval.py \
    --predictions work_dirs/eval_spark2_5_1_7b_vllm_128k/ifeval/predictions.jsonl \
    --dataset data/evaluation/spark_x2_5/ifeval.jsonl \
    --scorer-dir /path/to/ifeval_scorer
```

`--scorer-dir` 指向包含 `instruction_following_eval/` 的父目录。
也可在生成命令中添加 `--ifeval-scorer-dir /path/to/ifeval_scorer` 自动评分。
自动评分同样需要设置 `NLTK_DATA` 和安装评分依赖。

评分输出在预测目录的 `official_scores/`，包括四项 prompt/instruction × strict/loose
指标、逐题结果与输入 SHA-256。重复 ID 取最后一条，默认失败/截断回答置空计分。
默认要求题目完整覆盖；`--allow-partial` 显式评分部分题目；
`--include-failed-responses` 改为检查保存的失败/截断文本。

GPQA 使用选项准确率，AIME 使用归一化精确匹配。AA-LCR 的精确匹配为诊断分数；
BFCL-V4 与 SciCode 保存单次响应，完整官方工具/代码执行仍需各自 harness。
外部百分制结果可通过 `--resume --score NAME=PERCENT` 合入汇总。

## 验证

回归位于 `benchmark_test/ci_test_transformer5.16.0/test_spark25_vllm_eval.py`，
覆盖流式响应、工具参数、预算检查、失败重试、续跑配置、默认路径及 IFEval 对齐。
