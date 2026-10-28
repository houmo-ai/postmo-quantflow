# Qwen3-VL-Reranker-2B 链路与验证证据

本文件保留 2026-10-22 初期适配、对照和清理的历史证据；以下“当前”均指当时阶段，
不代表 2026-10-28 的最终配置、保留模型数量或质量结论。
后续 Merak 视频支持、评分头 W8A16、完整五精度与局部 A16 的正式结果以
[模型 README](README.md)、[五精度 head A8/A16 报告](evaluation/HEAD_A8_A16_713_REPORT.md)
和[三套局部 A16 报告](evaluation/ACTIVATION_A16_713_COMPARISON_20261028.md)为准。
两个临时评测工作区的运行资料已集中到 `xh2modelzoo/work_dirs` 的长期归档，见
[评测资料映射](evaluation/README.md#证据与归档)。

工作区为 `/data01/home/jing.huang/xh2modelzoo-qwen3vl-reranker`，分支为
`feat/qwen3-vl-reranker-2b`，独立克隆基线 `0e52a89f`。初版接入提交为 `63e09623`，
本轮增加动态图像、32K、FlashAttention 和 AutoRound W4。
原有 `/data01/home/jing.huang/xh2modelzoo` 未参与修改。验证日期为 2026-10-22。

## 当前保留产物与清理

两份 YAML 的 `export.model.quant_scheme.quant_type` 均采用 `w8a8h1_sefp`。
AutoRound 工作流接受此基础格式；实际 W4G64 检查点校验和 chunk=256 约束依据检查点执行。
原先仅允许显式 `w4a8h0_ssfp` 的量化校验已调整，显式 W4 覆盖仍可使用。

| 当前保留 | 目录 | 实际内容 |
| --- | --- | --- |
| AutoRound W4 权重、W8 基础格式导出包 | `work_dirs/qwen3_vl_reranker_2b_autoround4_w8a8/` | 196 个 SSFP W4 语言 Linear，评分 head/视觉为 W8 |
| 不运行 AutoRound 的 W8 导出包 | `work_dirs/qwen3_vl_reranker_2b_w8a8/` | 197 个 SEFP W8 语言/评分 Linear，视觉为 W8 |
| AutoRound 源检查点 | `work_dirs/qwen3_vl_reranker_2b_autoround_smoke/` | W4G64，供重载和后续导出；32 样本、长度 512、32 次优化，不是第三份导出包 |

已按用户要求删除六个过时目录：`qwen3_vl_reranker_2b_autoround_dequant_for_w8`、
`qwen3_vl_reranker_2b_autoround_reencoded_w8a8`、`qwen3_vl_reranker_2b_autoround_w4a8`、
`qwen3_vl_reranker_2b_dynamic_w4a8_rtn_baseline`、`qwen3_vl_reranker_2b_dynamic_w8a8_initial`、
`qwen3_vl_reranker_2b_release`，均位于本工作区 `work_dirs/`。
后文指向这些目录的说明属于历史实验记录，已删除产物不作为当前可加载模型；比较 JSON、日志和
逐参数哈希仍保留。考虑硬链接的其他存活引用后，按文件块统计预计回收约 278.10 GiB。
删除前已核验两份保留包全部发布文件的 SHA-256，原始数据集/模型与其他模型的 CI 缓存未参与清理。

当前配置回归为 **171 passed**，包含真实小模型的 AutoRound → W8 基础格式 → 实际 SSFP W4 →
搬移删源、Golden 和推理；另覆盖 W8 基础格式下实际 W4 检查点拒绝 chunk=512。
证据为 `work_dirs/autoround_w8_base_red.log`、`autoround_w8_base_ci.log`。
清理清单、删除目录的原 Manifest 和保留包校验结果见 `work_dirs/reranker_artifact_cleanup.json`。
清理后两份保留包都通过公开 `inference_model` 的文本和图片输入，文本分数与清理前逐项相同；
结果及日志为 `work_dirs/retained_reranker_validation.json`、`retained_reranker_validation.log`。

## 范围与状态

| 能力 | 状态 | 实际证据与边界 |
| --- | --- | --- |
| 文本、单图、多图、图文混合 | verified | 真实 2B，11 组、22 个 pair 的官方处理输入哈希对照和实际推理 |
| 图像动态尺寸 | verified | 保持比例；96/196/384/704/1536 最小适配档位；图内屏蔽 padding，裁回有效主特征与三个 DeepStack |
| 32K cache 与分块 | verified | W8、AutoRound W4 的初版都实际执行 32768 token / 128 个 Prefill chunk；当前 processor 对超长输入执行保留特殊 token 和末尾五个模板 token 的截断，总容量仍为 32768 |
| FlashAttention 导出 | verified | 两种精度的 Prefill 图各有 28 个真实 FlashAttention 节点；加载验证节点数及 Q/K/V/P 位宽 |
| AutoRound W4G64 权重 | verified | 真实小模型端到端；真实 2B 有 196 组 packed language linears，独立保存、重载、消费 |
| AutoRound W4 HMONNX | verified | 196 个 W4 Linear + 1 个 W8 评分 Linear，所有 weight group=64；视觉塔为 W8 |
| Golden | verified | 每个发布包五个视觉图的合成基线及真实 pair；另采集真实图片实际路由图与 Prefill |
| 发布包搬移及删除源权重后加载 | verified | 小模型 W8/W4 的统一 API 测试；真实包由 metadata 的相对路径加载 |
| 绝对分数精度验收 | blocked | 现有 0.1 阈值未通过，数值见下节；不宣称量化质量已完成 |
| 默认 256 样本、2048 长度、400 次 AutoRound 的质量 | blocked | YAML 已提供；真实 2B 本轮为缩小预算 smoke，未运行默认预算的完整质量评估 |
| 视频、GPU batch>1、XH2a 真机、完整 MMEB/MMTEB | unsupported | 本轮只验证 CUDA HMONNXInferenceV2 和代表输入，未扩大能力声明 |

官方模型卡发布的是 32K 上下文；`max_position_embeddings/max_pe_length=262144` 是位置范围，
不能作为官方测试 256K 的证据。两份 YAML 都固定 `context_max_length: 32768`、chunk 256，
位置表沿用上游 262144。模型卡也不意味着所有官方 benchmark 样本都采用 32K 截断。

## 数值结果与限制

同一批 22 个 pair 包含文本、长文本、文本对图片、图片对文本、两侧混合、多图，以及五个视觉档位。
官方 FP16 与目标浮点的 token ID、mask、patch/grid 和多模态 token 类型哈希全部一致，
最大分数绝对差 **0.0021793842**，通过浮点 0.02 门槛。

| 对照（同一批样本） | 最大分数绝对差 | 排序 |
| --- | --- | --- |
| HMONNX W8A8 vs 原始目标浮点 | 0.1226310730 | 11/11 组一致 |
| 直接 RTN W4 vs 原始目标浮点，仅用于对照 | 0.2511559129 | 11/11 组一致 |
| AutoRound W4 smoke 的 HMONNX vs 原始目标浮点 | 0.2924855053 | 11/11 组一致 |
| AutoRound W4 反量化浮点 vs 原始目标浮点 | 0.2864899933 | 单独记录 checkpoint 阶段的漂移 |
| AutoRound W4 HMONNX vs 同一 checkpoint 的反量化浮点 | 0.1167398095 | 用于区分权重量化与部署阶段误差 |

这轮 AutoRound smoke 用的是 **32 条 Pile、长度 512、32 次、batch 4**，耗时约 89 秒，
不是 YAML 的正式预算。其分数漂移在反量化浮点阶段已经明显，不能把它写成 AutoRound 优于 RTN 的证据。
AutoRound 的优化目标与数据、预算有关，Reranker 还需要针对相关性任务做正式校准及检索指标验收。
上述简单正负候选的排序一致不替代完整 benchmark，也不代表近邻候选都能保持排序。

FlashAttention 的 Q/K/V/P 8/16 位组合做了单独对照，没有稳定通过所有输入的 0.1 门槛，
因此最终 YAML 保持参考配置的 Q8/K8/V8/P8，不调宽验收阈值。
初版固定 448、2K、普通 Attention 的六组结果属于 `63e09623`，不能挪用为本轮动态配置的精度证据。

## W8 基础配置与实际 W8 的对照

两份 Reranker YAML 已省略视觉配置中重复的 `max_size_t: 2`、`patch_size: 16`、
`temporal_patch_size: 2`，这些值由直接复用的 `XHQwen3_5_VisualConfig` 提供。
本次使用上节同一份 AutoRound W4G64 smoke 检查点，保持 32K、chunk 256、Flash Q8/K8/V8/P8、
视觉档位和 W8 评分 head 一致，先仅将公开接口的 Prefill 基础 scheme 改为 `w8a8h1_sefp`。

| 项目 | 显式 W4 基础配置 | W8 基础配置，直接复用同一检查点 |
| --- | --- | --- |
| 实际语言 Linear | 196 个 SSFP W4/H0/G64 | 196 个 SSFP W4/H0/G64 |
| 实际评分 Linear | 1 个 SEFP W8/H1/G64 | 1 个 SEFP W8/H1/G64 |
| FlashAttention | 28 个 | 28 个 |
| Prefill 外部权重数据 | 1,587,912,704 字节 | 1,587,912,704 字节 |
| 六个组件的外部载荷 SHA-256 | 参考 | 全部相同 |
| 六个组件的计算图 | 参考 | 去除文件路径、doc string 和 metadata 后全部相同 |
| 22 个 pair 的分数 | 参考 | 逐项完全相同，最大差 0 |
| 相对原始目标浮点的最大分数差 | 0.2924855053 | 0.2924855053 |

两份 ONNX 文件并非逐字节相同：文件名、external_data location 和附带的量化配置等元数据不同；
不能把这些差异写成模型计算变化。`inspection_autoround_w8_vs_w4.json` 同时记录原始文件 SHA-256、
保留所有算子/属性/张量定义后的计算图 SHA-256，以及外部数据 SHA-256。

原因是 `gptqmodel_torch_qlinear_converter` 在恢复浮点 `weight` 时仍保留整数 `quant_weight`。
SDK `QLinearBase._setup` 发现该载荷后将权重切换为 SSFP/H0，`weight_static_quant` 根据整数范围
确定实际位宽，本检查点为 W4。普通 `quant.bits: 4` 没有在 YAML 解析层改写 W8，但实际量化节点
仍会采用检查点的 W4 整数载荷。因此仅根据基础 `quant_type` 判定最终权重位宽是不完整的。
当前 HMONNX 使用 INT8 张量存储这批 W4 权重、FLOAT16 存储 scale；图内逻辑 W4 不等于文件按
两个四位值打包成一个字节，不能单凭磁盘大小判定量化位宽。

为比较实际 W8，将同一检查点反量化成临时浮点副本，仅移除 196 个整数载荷和量化声明，
对全部 625 个参数张量执行保存前后 SHA-256 检查，确认数值完全一致；再从这些浮点值导出 SEFP W8。
这一步是独立部署格式实验，不修改 AutoRound 原始产物、公共模型实现或 SDK。
复用检查点的 W8 基础配置包为 `work_dirs/qwen3_vl_reranker_2b_autoround4_w8a8/`；
真正重新编码为 W8 的历史对照包为 `work_dirs/qwen3_vl_reranker_2b_autoround_reencoded_w8a8/`，已清理。
临时浮点副本 `work_dirs/qwen3_vl_reranker_2b_autoround_dequant_for_w8/` 已清理，逐参数证据
`work_dirs/autoround_dequant_for_w8_provenance.json` 保留。

真正 W8 的 Prefill 图确认有 197 个 SEFP W8/H1/G64 Linear（含评分 head）和 28 个 FlashAttention；
五个视觉组件的计算图和外部权重仍与上面的 W4 包相同。

| 同一批 22 个 pair | 保留 AutoRound 整数载荷的 W4 | 从相同反量化数值重新编码的 W8 |
| --- | --- | --- |
| 相对原始目标浮点：最大绝对差 | 0.2924855053 | 0.2614289820 |
| 相对原始目标浮点：平均绝对差 | 0.1647089882 | 0.1428470923 |
| 相对同一检查点反量化浮点：最大绝对差 | 0.1167398095 | 0.1194910407 |
| 相对同一检查点反量化浮点：平均绝对差 | 0.0397788110 | 0.0590518117 |
| 与原始目标浮点的候选排序 | 11/11 组一致 | 11/11 组一致 |
| Prefill 外部权重数据 | 1,587,912,704 字节 | 1,587,912,704 字节 |

两种实际格式的最大分数差为 0.0879960209。W8 在这些样本上更接近原始模型的分数，但相对同一
检查点浮点基线的部署误差没有改善，不能据此宣称 W8 更能保留 AutoRound 的结果。
两种格式都没有通过既有的 0.1 最大分数差门槛。此对照使用上述缩小预算的 smoke 检查点，
不代表正式 400 次 AutoRound 的质量，也不是完整检索 benchmark。
HMONNX 的 INT8 容器和 FLOAT16 scale 存储使两版外部数据字节数相同，逻辑 W4/W8 的算子差异
应从节点属性读取，不能把当前文件大小当作芯片实际带宽或编译后压缩比。

在同一张物理 RTX 6000D GPU 2 上顺序启动独立进程，CPU 线程数固定为 4，
每种输入先预热一次，再计时三次取中位数，包含 Host 处理、视觉和 Prefill 的完整运行路径。
两种格式的全部有效输入张量哈希、32K token ID 哈希均相同；重复运行分数逐项一致。

| CUDA HMONNXInferenceV2 耗时 | 实际 W4 | 实际 W8 |
| --- | --- | --- |
| 11 组各自中位耗时之和，共 22 个 pair | 33.1998 秒 | 35.8189 秒 |
| 32768 token / 128 chunk，单对输入 | 21.4220 秒 | 19.8605 秒 |
| 32K 的三次计时 | 21.3397 / 21.4220 / 21.4689 秒 | 19.8865 / 19.7869 / 19.8605 秒 |
| 32769 token | 拒绝 | 拒绝 |

32K 文本输入中 W8 比 W4 约快 7.3%，代表组的中位耗时合计则约慢 7.9%；
结果与输入及模拟器算子路径有关，不推断为 XH2a 芯片速度或统一的格式优劣。
首层 Q projection 的实际 INT8 载荷也已检查：W4 的数值范围为 [-8, 7]，W8 为 [-128, 127]，
区别不仅是节点上的位宽标记。

本轮改动的 CI 选择仍为五个目标：模型定向用例 **34 passed**，policy/recipe 和公开 W8/AutoRound W4
端到端用例 **135 passed**，合计 **169 passed**。这些结果对应本轮简化 YAML 后的配置。

| 本轮对照证据 | 路径，相对本工作区 |
| --- | --- |
| 直接复用 W4 检查点、基础 W8 导出 | `work_dirs/build_autoround_w8a8.py`、`build_autoround_w8a8.log` |
| 同一权重值重新编码为实际 W8 | `work_dirs/build_autoround_reencoded_w8a8.py`、`build_autoround_reencoded_w8a8.log` |
| 三种图的节点属性、计算图、载荷与大小 | `work_dirs/inspection_autoround_w8_vs_w4.json` |
| 代表权重 INT8 载荷范围 | `work_dirs/deployment_weight_payload_ranges.json` |
| 基础 W8 仍使用 W4 时的分数 | `work_dirs/comparison_autoround_w8_requested.json` |
| 实际 W4/W8 的输入哈希、分数、计时和 32K | `work_dirs/comparison_autoround_w4_recheck.json`、`comparison_autoround_actual_w8.json` |
| 简化配置后的 CI | `work_dirs/simplified_reranker_configs_tests.log`、`simplified_reranker_configs_ci.log` |
| Golden | `work_dirs/golden_autoround_w8_requested.json`、`golden_autoround_reencoded_w8.json` |
| Golden step 图链接检查 | `work_dirs/golden_deployment_inventory.json` |
| 两份新包的搬移后公开 demo | `work_dirs/deployment_relocation.json` 与 `.log` |
| 汇总结果 | `work_dirs/autoround_deployment_comparison.json` |

两份新包均通过生成 Golden 后的搬移和公开 demo 加载。W8 基础配置、实际 W4 的包重新采集了
全部五个视觉满档位基线及 Prefill，并采集原始真实图片的视觉和七个 Prefill chunk。
实际 W8 的五个视觉图在确认计算图和外部载荷 SHA-256 均相同后，以普通硬链接复用这些视觉基线，
为各 step 链接该包自己的图与权重；W8 Prefill、红色图片和 768×512 真实图片路径重新执行采集。
每个 step 都包含实际 `.npy`，图与本包对应图的 inode 一致；没有软链接，也没有复用 W4 Prefill 的输出。

## 入口、职责和复用

```text
ModelRecipe / build_model
  → W16 输入：W8 YAML、quant 显式跳过
  → W4 输入请求：config resolver 读取 W4 YAML
      → Qwen3.5 public quant adapter
      → GPTQModel qwen35_autoround recipe 的通用 dense / llm-only 路径
      → AutoRound 校准、优化、GPTQ 格式 checkpoint
  → 原生或量化 checkpoint 加载 / 现有 dequant converter
  → Qwen3-VL language wrapper + yes-minus-no head
  → 复用 Qwen3.5 patch-token ViT + Reranker DeepStack + Prefill HMONNX
  → 公共打包、metadata、manifest、export_meta_info.json
  → 官方 processor / 图像最小档位路由 / 全局 M-RoPE
  → 视觉主特征和三个 DeepStack / 分块 Prefill / FP32 sigmoid
```

ModelZoo 只映射参数和消费量化产物，不实现 AutoRound 求解或 GPTQ 打包。
当前 GPTQModel 的公开 dense recipe 已通过 AutoModelForImageTextToText/AutoProcessor 支持该模型，
本轮未修改外部 GPTQModel 或 SDK checkout。

`quant`、`export`、`dump_golden` 保持独立。AutoRound 工作流要求真实 AutoRound W4G64 checkpoint，
并要求 chunk=256；基础格式为 W8 或显式 W4 都执行这些检查，不能跳过 quant 后静默导出原始权重。
`quant: null` 仅属于默认 W8 路径。`export.model.hf_model: null` 是 workflow 输入路径占位符。

直接导入或继承 Qwen3.5 已有的配置、位置几何、token 图表达、档位导出及共享权重 runtime。
Qwen3.5 目录与适配前基线保持一致，不增加 `xhmodel_merak/xh_llm/visual_token_*.py` 公共模块。
Reranker 的 `visual.py` 局部扩展视觉 forward，保留三个 DeepStack 输出；复用原有算子改写及
RoPE 初始化。Qwen3-VL 类型绑定在包装期间生效，退出后恢复注册表，避免改变其他模型的包装行为。
视觉独立加载只读取 `model.visual.*`，导出其他档位时无需反复加载和反量化完整 2B 语言模型。

语言部分沿用 `qwen3_vl/_llm_model_impl.py`、VL-Embedding 的末 token 选择和 Prefill-only 导出。
新增 `flash_attention` 映射到 SDK FlashAttention，显式传递过去长度与本块有效长度，不只是 HF 后端开关。
`binary_score_head` 读取已正确绑定的原始 head，创建独立 64×H 投影，第 0 行为 W_yes−W_no，其余为零，
不改 tied embedding；最终只计算 score_logits[0,0,0] 的 sigmoid。

旧版 M-RoPE 的显式 default 配置与 Transformers 5 的 rope_parameters 做逐字段等价检查，
保留 theta、mrope_section、mrope_interleaved。改变任一值仍拒绝量化 checkpoint。
旧视觉配置类型 `qwen3_vl` 到 `qwen3_vl_vision` 的等价别名限定在此 recipe，不泛化类型豁免。

## 图、状态与产物契约

| 组件/张量 | ABI |
| --- | --- |
| Visual 输入 | pixel_values FP16 `[1,4m,1536]`，来自官方归一化 patch；另有位置索引、插值权重、二维 RoPE IDs、padding mask |
| Visual 输出 | image_embeds 和三路 DeepStack，各 `[1,m,2048]`；Host 裁成实际有效 token 数 |
| Visual padding mask | additive key bias `[1,1,1,4m]`，零前缀加负 padding，通过 MaskedAdd + Softmax |
| Prefill embedding / DeepStack | 每路 FP16 `[1,256,2048]`；全局媒体位置生成后同步切块 |
| position / past / current | 三路位置 INT32 `[256]`；过去长度和本块有效长度 INT32 `[1]` |
| KV cache | 28 对 FP16 `[1,8,32768,128]`；图内 KVcache 原地更新，不跨候选共享状态 |
| score_logits | FP16 `[1,1,64]`，没有 Decode 或 embedding 输出 |

图像预处理、几何、路由、token embedding、全局 M-RoPE 和 chunk 切分属于 Host；
Transformer、视觉 Attention/merger、DeepStack、最终 norm、投影与 cache 更新属于图。
候选逐个处理并重置 cache；单 token 尾块仍运行 Prefill。

发布目录平铺 `visual_m96/` 至 `visual_m1536/`、`prefill/`、`hf_config/`、token embedding、
`visual_gears.json`、运行 metadata、有效 workflow、依赖、manifest 与 export_meta_info 清单。
完整目录可搬移；不依赖原始 checkpoint。视觉图由 MultiHMONNXLoader 共享 GPU 权重。
Golden 的图和外部权重用普通硬链接，遵循当前发布包禁止软链接的检查。

## 本机验证索引

命令和公开用法见 [README.md](README.md)。目录相对本工作区，权重与 Golden 不入 Git。

| 检查 | 证据路径 |
| --- | --- |
| W8 构建、图检查 | `work_dirs/build_dynamic_w8a8_final.log`、`inspection_dynamic_w8a8.json` |
| W8 最终重建与 Golden 复用 | `work_dirs/w8_package_golden_reuse.json`；六个组件所有图及权重的 SHA-256 与原捕获一致，仅重建发布契约 |
| W8 发布包 | `work_dirs/qwen3_vl_reranker_2b_w8a8/` |
| 官方、浮点、W8 对照 | `work_dirs/comparison_dynamic_w8a8.json` 与 `.log` |
| RTN W4 对照 | `work_dirs/comparison_dynamic_w4_rtn.json`，作为 ablation，不是交付的 W4 方案 |
| AutoRound 2B 权重与执行记录 | `work_dirs/qwen3_vl_reranker_2b_autoround_smoke/`、`quant_autoround_smoke.log` |
| AutoRound W4 构建、图检查 | `work_dirs/build_autoround_w4a8.log`、`inspection_autoround_w4.json` |
| AutoRound 当前发布包 | `work_dirs/qwen3_vl_reranker_2b_autoround4_w8a8/`；原显式 W4 包已清理 |
| AutoRound W4 浮点与 HMONNX 对照 | `work_dirs/comparison_autoround_w4.json`、`comparison_autoround_float.json` |
| 32K 边界 | `work_dirs/context_32k.json`、`context_32k_w4.json`；分别约 24 秒执行 128 个 chunk |
| W8 Golden | `work_dirs/golden_dynamic_w8a8_synthetic.log`、`golden_dynamic_w8a8_real.log` |
| AutoRound W4 Golden | `work_dirs/golden_autoround_w4_synthetic.log`、`golden_autoround_w4_real.log` |
| TDD | `dynamic_red.log`：两份 YAML 契约 RED；`w4_guard_red.log`：禁止未校准 W4 导出 RED；实现后均通过 |
| 定向契约 | `contracts_final.log`：100 passed，含真实视觉几何、Flash chunk/reset、M-RoPE、布局 |
| 真实 2B 搬移 CLI | `work_dirs/relocation_dynamic_final.log`；两种精度的最终包在生成 Golden 后搬移并通过公开 demo |
| 真实统一 W8/W4 | `integration_contracts_final.log`：4 passed（两条契约、两条端到端）；W4 验证 packed 权重、W4 nodes、Flash、搬移删源、Golden、重复推理 |
| 扩大 CI | `ci_dynamic.log` 为最初 47 个目标；公共 RoPE 等价检查新增影响的 20 个目标为 `ci_delta.log` |
| 无本轮改动的基线 | `baseline_dynamic_failures.log`：DeepSeek 两例、Gemma12B full、Pi0.5 四个失败均复现 |

本轮将视觉复用方式收敛为直接继承 Qwen3.5 原实现。相对适配前 `0e52a89f`，
整个 `models/qwen3_5/` 无差异，五个 `xh_llm/visual_token_*.py` 公共文件已删除；
同时恢复因抽取公共模块而扩大的 29 份既有 impact 声明。三份 Reranker impact 声明
覆盖直接使用的 Qwen3.5 文件，并保留删除旧公共路径时的回归选择。

- `work_dirs/reranker_local_vision_ci.log`：190 passed、1 failed；Reranker 的真实 W8/W4
  构建、搬移、推理及 Golden 均通过。失败项为 Pi0.5 fixture 导入 LeRobot 时的
  `TypeError: non-default argument 'backbone_cfg' follows default argument`，
  与 `baseline_dynamic_failures.log` 中已复现的环境问题一致。
- `work_dirs/reranker_local_vision_qwen35_regression.log`：51 passed，覆盖原视觉加载、
  metadata 校验及配置注册。
- `work_dirs/reranker_local_vision_runtime.json`：两份现存 2B 包的文本、图文评分与
  重构前逐值相同；每份包的五个视觉档位均实际执行四路输出，且不导入已删除模块。
- `work_dirs/reranker_local_vision_demo.log`：公开 HMONNX demo 执行通过。
- `work_dirs/reranker_changed_files.csv`：更新后的净改动清单，共 34 个文件。

环境为 PyTorch 2.8.0+cu128、Transformers 5.16.1、xhquanttool 0.9.2、GPTQModel 5.8.0、
本地维护的 AutoRound 0.12.0，RTX 6000D。`work_dirs/venv_ci` 补充了 datasets 3.6.0 等 CI 依赖，
未更改共享 Conda 环境。精确部署依赖由 recipe 和发布包 environment 文件记录。
默认量化预算的精度、干净环境安装和芯片真机验证均未执行，不因本机 smoke 可运行就标记通过。

扩大 CI 的依赖/后端阻塞包含：GPTQModel 缺少 DeepSeek dspark 模块、Gemma 校准 CUDA illegal access、
Ling 缺 fla-core、部分模型 worker 依赖不满足、Qwen speculative 路径缺 qwen35_mtp API，
以及 Pi0.5 fixture 的 LeRobot dataclass 接口问题。
已复现的基线类别与其余环境阻塞分别报告，不跳过用例、不修改无关模型或放宽依赖门禁。
首次 47 目标运行是 **1097 passed、20 failed、16 errors**；其中本次遗漏的动态视觉规划入口和
矩阵计数两项已修复。最终 `integration_contracts_final.log` 为 **4 passed**，覆盖这两项契约及
分离后的 W8/W4 两条真实统一接口用例。其余失败/错误为上述依赖/API、CUDA 和 fixture 阻塞。
追加 20 目标为 **111 passed、15 failed**；扩展 CI 不宣称全绿。

`ocr` 在环境中不可用，采用源码和 diff 人工复核；检查图 ABI、直接复用边界、Ruff 与 diff --check。

当前改动 Python 文件通过 Ruff check/format，暂存与未暂存 diff --check 均通过。
