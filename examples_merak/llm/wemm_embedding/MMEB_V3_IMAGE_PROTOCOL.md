# WeMM-Embedding MMEB-v3 图像评测协议

## 固定版本与样本

- 评测代码：官方 WeMM 仓库 `mmeb_v3_eval`，commit `5f539a2f5196b08bbcf29ac71fc90a95ccad7f11`。使用它的数据解析、查询/候选生成和 `RankingMetrics` 计分。
- 图像元数据：`VLM2Vec/MMEB-V3`，revision `4a5560b2b64384204b6fea8a82ea986eba51f5aa`，`image-query/`。
- 已用元数据 SHA256：Visual7W parquet `9c78a0789f6c3ea2ed354a35dcaf76306721304f1e07fafb100168e3b702879e`；ChartQA parquet `d256b11d0f5bb4cbf66d790da4b35833e50e4a601736904cf38f193a0c0e8df1`。
- 图像媒体：`TIGER-Lab/MMEB-V2`，revision `e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1`，`image-tasks/mmeb_v1.tar.gz`。
- 媒体归档 SHA256：`feeaf4a8319cba587b982ec976fba225d13e134a3217e1cad5c251e64299346a`，与 Hugging Face LFS 对象哈希一致。Visual7W 和 ChartQA 的 1,000 条查询图片路径均已逐一检查，缺失数为 0。
- 本地完整归档已解压到 `work_dirs/wemm_embedding/mmeb_v3_data/image-tasks/MMEB/`，并记录 `.done` 标记；解压后再次检查两项任务的图片路径，缺失数仍为 0。
- 评测任务：官方 `image.yaml` 原样摘出的 `Visual7W` 和 `ChartQA`，见 [配置一](mmeb_v3_image_first.yaml)、[配置二](mmeb_v3_image_chartqa.yaml)。每项使用完整 test 集 1,000 条，不截断查询或候选。输出保留官方逐任务分数及查询、候选向量。
- 当前脚本覆盖上述两项图像任务；完整 `image.yaml` 还包含其他图像任务，其中 MCMR 需另行下载 MMEB-v3 的 `image_tasks/MCMR.tar.gz`。
- 输入图像：HF 与 HMONNX 都不设置 `resized_height` / `resized_width`，保持原图宽高比，并使用相同的 `min_pixels=65536`、`max_pixels=1572864`。最大像素数对应 HMONNX 的 m1536 档位容量。HMONNX 按实际 patch token 数选择最小可容纳档位。
- 两端均使用官方 WeMM 指令文本、原始样本顺序、`pooling=last`、L2 归一化和相同候选池。HF 与 HMONNX 输出目录隔离；比较脚本检查查询标签与候选 ID 的顺序一致。

## 接入方式

本仓库的 [评测入口](mmeb_v3_eval.py) 从官方 `eval.py` 启动评测。HMONNX 模式替换模型加载和图文编码；数据集加载、候选构造、相似度排序与分数计算继续执行官方代码。运行时增加了传入完整 `messages` 的入口，用于保留官方指令中的图像占位符位置。

数据准备命令（完整图像媒体归档；当前任务的所有引用图片均已校验存在）：

```bash
REPO=/path/to/xh2modelzoo
DATA=$REPO/work_dirs/wemm_embedding/mmeb_v3_data
hf download VLM2Vec/MMEB-V3 --repo-type dataset \
  --revision 4a5560b2b64384204b6fea8a82ea986eba51f5aa \
  --include 'image-query/**' --local-dir "$DATA"
hf download TIGER-Lab/MMEB-V2 image-tasks/mmeb_v1.tar.gz --repo-type dataset \
  --revision e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1 \
  --local-dir "$DATA/.sources/MMEB-V2-e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1"
mkdir -p "$DATA/image-tasks"
tar -xzf "$DATA/.sources/MMEB-V2-e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1/image-tasks/mmeb_v1.tar.gz" \
  -C "$DATA/image-tasks"
```

运行环境需包含本仓库、官方评测代码、PyTorch、Transformers、`qwen-vl-utils`、`datasets` 和 HMONNX 运行依赖。官方 README 固定 Transformers 5.2.0；每次实际运行版本写入 `protocol.json`。MMEB-v3 官方包导入时会加载视频解析模块，因此其视频依赖即使在图像任务也可能需要；评测入口在缺少 `imageio`/`decord` 时只对图像任务提供导入占位模块。

```bash
cd /path/to/xh2modelzoo
conda activate xhquant_55
bash examples_merak/llm/wemm_embedding/run_mmeb_v3_suite.sh
```

脚本按 2B、4B、9B 分别运行 HF 和 HMONNX，并保存官方分数、逐查询预测、向量和 `protocol.json`。比较脚本逐项核对两端的动态图像协议、查询、候选集合与顺序；完全一致时才计算精度差异。
可用 `WEMM_SIZES` 和 `MMEB_TASKS` 选择规格与任务。脚本对每个规格及任务加锁并复用已完成的对比结果；设置 `MMEB_FORCE=1` 可重新评测。

数据级动态分辨率审计：

```bash
python examples_merak/llm/wemm_embedding/audit_dynamic_resolution.py \
  --flickr-dir work_dirs/wemm_embedding/datasets/flickr30k \
  --mmeb-data-root work_dirs/wemm_embedding/mmeb_v3_data \
  --output work_dirs/wemm_embedding/DYNAMIC_IMAGE_PREPROCESSING_AUDIT.json
```

该文件记录每个数据集的原始/预处理尺寸种类、视觉 token 范围、自然产生的 448×448 样本数和 m96/m196/m384/m704/m1536 命中数。
