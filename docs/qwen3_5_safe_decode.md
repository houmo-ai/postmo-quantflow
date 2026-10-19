# Qwen3.5/3.6 MoE：SAFE decode-only 适配

## 原剪枝与 SAFE 的区别

原 `dynamic_prune` 使用逐层 Method1/AMP 标量乘 router gate，归一化后按
Top-P coverage 选择专家，并保护 router Top-1。它没有同时使用 AIMER，也没有
源专家到替代专家的投影系数、误差约束或偶数折叠策略，不能称为当前 SAFE。
保留原配置的 `algorithm: acep` 默认值，不静默改变历史实验。

`algorithm: safe` 沿用研究实现 `apply_decode_ace_exfold` 的折叠流程；
当前 v4 沿用融合先验公式和固定折叠控制参数，改用 FP16 存储与在线路由计算；
不与历史 FP32 或 v1 加性平滑版本逐位等价：

1. 导出前按层、按专家预计算 `scalar = sqrt(AMP * AIMER)`。
   推理时取原 router 的 Top-K，只在这些专家内计算
   `q = normalize(gate * scalar)`，不再在线执行双分支及开方。
2. 按 q 排序，以“累加当前专家之前的质量 < coverage”为保留条件。
   跨越阈值的专家保留，router Top-1 始终保护。
3. 对候选删除专家 s，在初始保留集合中找投影损失最小的目标 t。
   仅当系数 `c[s,t] > 0`、损失有限且不超过固定上限 `1.0` 才折叠；
   不满足条件的源专家回退为实际计算。
4. 固定使用 `nearest_mass`：遇到奇数折叠数，比较恢复 q 最大的折叠专家与
   多折一个 q 最小的、非 router Top-1 的初始保留专家两种方案。
   向上取偶数时必须对所有折叠源重新找仍保留的目标，并全部通过误差门槛。
   在合法的上/下方案中，选择保留 q 总质量离 coverage 更近者，平局向下。
   这不是固定剪两个，也不是不看贡献的数量四舍五入。
5. 将 `gate[s] * c[s,t]` 加到目标 gate，源 gate 置零，再归一化恢复原 Top-K
   gate 总质量。专家权重本身不修改，不训练。

融合依据：若 `a = gate * AMP / A`、`b = gate * AIMER / B`，则非负 gate 下
`sqrt(a*b) = gate * sqrt(AMP*AIMER) / sqrt(A*B)`。最后归一化会消去公共分母
`sqrt(A*B)`，所以每个专家只需一个预计算 scalar，而不是整个模型共用一个数。
打包时先将两份先验转为 FP16，计算 `sqrt(AMP) * sqrt(AIMER)`，结果仍为 FP16，
避免先计算乘积导致不必要的上溢/下溢。先验转换后为零或非有限数时明确报错。

v1 开方内的 `+1e-8` 无法精确合入静态先验，自 v2 起移除；所有除法防零保护
保留。该变化可能影响极小权重和阈值边界处的选择，历史评测精度不能直接标成 v4 结果。
v3 固定 `max_loss=1.0`、`parity=2`、`renormalize=True`，仅删除对应控制列和可选分支，
不改变 v2 默认设置下的折叠、补偿和归一化语义。

v4 的 Top-K、重要性乘法、归一化、累计和、偶数方案距离、gate 补偿均使用 FP16
浮点张量，不再显式提升为 FP32。专家 ID/掩码仍为整数/布尔类型；底层归约内核的
内部累加实现由 PyTorch/硬件决定，不能据此宣称每条硬件指令均为 FP16。
`s_scalar_eps: 1e-8` 配置仍兼容，但在线有效分母下限为
`max(eps, 6.103515625e-5)`（FP16 最小正规数），避免 `1e-8` 转为零。
这可能影响极小质量和 coverage 边界，尚不能声称大模型精度不变。

只有 decode 图绑定 SAFE 数据；prefill 图明确移除数据和阈值。不能按输入 token
数量猜测阶段：单 token 的 prefill 仍须 dense。共享专家不参与上述路由专家折叠。

## ModelZoo 与 SDK 职责边界

离线模型准备属于 ModelZoo 的前端图准备/导出工作流：
`safe_calibration.py` 在 HF 包装前根据真实权重和校准文本生成 AMP/AIMER 与 ExFold 表；
`safe_prune.py::pack_safe_data` 校验原始 AMP/AIMER、预计算融合 scalar、处理 ExFold
禁用配对与对角线，再生成 FP16 `[E,2*E+1]` 数据。`prepare_safe_scalars` 负责逐层调用、
checkpoint/文件指纹校验和缓存，随后将数据绑定到 decode 图。

xhquanttool 只消费已经准备好的 `s_scalar`：校验算子输入格式、直接使用 FP16 常量，
在浮点/量化执行及 HMONNX 重载时执行 SAFE 路由。SDK 不实现或导入 ModelZoo 的
打包函数，没有反向依赖或兼容转发包装。原有一维 ACEP 和“不安全则不折叠”的回退不变。

此前 v3 的职责迁移未改变算法；本次 v4 改变存储/计算精度，旧缓存不可直接使用。
融合/打包测试归属 ModelZoo；SDK 测试直接构造预处理后的输入，不依赖 ModelZoo。

## 使用方式

必须同时使用本次更新的 ModelZoo 和配套 xhquanttool。沿用原动态剪枝 workflow，
两份 MoE 剪枝 YAML 默认 `algorithm: safe`、`auto_s_scalar: true`，覆盖率 `0.85`。
没有显式指定 packed cache 或 AMP/AIMER、ExFold 文件时，导出器会在 HF 模型包装前
自动计算先验、执行文本校准，再生成二维 `s_scalar`；不需要提前准备研究仓库的 `.pt` 文件。

独立 CLI 默认选择 text-only YAML 和 `work_dirs/qwen3_5_safe_export`：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_dynamic_prune_workflow.py \
  --model-dir /path/to/checkpoint \
  --export-output-dir work_dirs/qwen3_5_safe_export
```

自动准备分为两步：

1. 从当前消费的真实专家权重计算 exact SLaNC AMP 和 inverse-normalized AIMER。
   AMP 使用 RMSNorm 的有效系数 `1 + HF weight.float()`，逐专家计算以控制临时内存。
2. 默认读取 `Salesforce/wikitext` 的 `wikitext-2-raw-v1` 训练集，全部行按 `"\n".join`
   拼接，不筛掉短行、不打乱、不套对话模板；取前连续 `128 × 2048` 个 token，
   切为 128 个 2048-token 窗口，`batch_size: 1`。在原生 HF 路由上收集同 token
   共同路由专家的未加 gate 输出，只累积 pair 统计，不保存全量专家输出。
   有向标量回归按源输出范数加权，`ridge=1e-3`，系数裁剪到 `[-4,4]`；
   未观察的配对使用系数 0、损失 `1e30`，推理时禁止折叠。

产物位于导出工作目录：`safe_calibration/priors.pt`、`safe_calibration/exfold.pt`，
以及原有 `moe_safe_s_scalar_v4.pt`。自动校准缓存的 identity 绑定 checkpoint、
实际 token SHA256、校准配置和模型 dtype；不匹配时明确报错，
可用 `overwrite_s_scalar: true` 显式重建。显式指定的文件优先于自动校准。

也可在 workflow 的 `export.model.dynamic_prune` 中配置：

```yaml
mode: decode
algorithm: safe
threshold: 0.85
auto_s_scalar: true
calibration:
  dataset: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train
  text_path: null
  nsamples: 128
  sequence_length: 2048
  batch_size: 1
s_scalar_eps: 1.0e-8
```

CLI 可用 `--calibration-text-path` 提供本地原始训练文本，用
`--calibration-nsamples`、`--calibration-sequence-length`、`--calibration-batch-size`
覆盖上述配置。本地文本也须足够提供所需的连续 token；若要复现默认协议，应保存
完整 WikiText 训练行以相同换行方式拼接的文本。改变样本数、序列长度或数据来源后，
应记录实际配置，不能将该结果标为标准 `128 × 2048` WikiText 校准。

已有产物可通过 `--ace-priors-path /path/to/priors.pt` 与
`--exfold-artifact-path /path/to/exfold.pt` 同时指定，或只传
`--s-scalar-path /path/to/moe_safe_s_scalar_v4.pt`。两种输入方式互斥。
`--no-auto-s-scalar` 关闭自动生成，此时必须提供上述手动文件。

`--context-max-length 32768` 可改变导出上下文；它不改变默认校准序列长度，
也不是 Think 评测输出上限。首次可加 `--only-first-block` 做有限规模的图检查，
不能把它当全模型评测。当前部署不支持非偶数 SAFE、floor 策略或其他损失上限；
旧 CLI 参数只接受 `--even-fold-policy nearest_mass --max-projection-loss 1.0`，可直接省略。

两份输入均为可信来源的 PyTorch `.pt` 文件，使用 `weights_only=True` 加载：

- `priors.pt`：`amp` 和 `aimer` 字典，键为从零开始的 body layer index，值为 `[E]`
  正有限浮点向量。必须覆盖实际导出的每一层，两表层号一致。
- `exfold.pt`：`coeff`、`loss` 为 `[L,E,E]`，第一专家轴是源、第二轴是目标；
  `expert_norms` 为 `[L,E]` 正有限数。损失必须有限且非负；`>=1e30` 为禁用配对。
  打包时按降精度前的原始损失屏蔽所有 `>1.0` 的配对，系数置零、损失置为
  FP16 最大有限数 `65504`，防止 `1.0001` 舍入成 `1.0` 后误判为安全。
  原始数据读取和这一步安全校验使用 FP32，不属于在线计算；最终包不含 FP32 列。
  对角线在打包时禁止折叠到自身。有效系数转 FP16 后非有限则报错。

**不能拿原一维 AMP/Method1 文件冒充 SAFE，也不会自动用全 1 先验补缺。**
自动校准使用当前模型权重和训练文本，不更新权重。手动复用外部产物时，使用者仍需保证
它们与 checkpoint、专家顺序和标定方法一致；文件 shape/哈希不能证明语义对应。

默认生成 `moe_safe_s_scalar_v4.pt`（`method: safe_exfold_v4`）：
包含 checkpoint 指纹、输入文件 SHA256、固定策略元数据和
逐层 packed 数据。更换文件或参数时拒绝复用不匹配缓存；只有显式配置
`overwrite_s_scalar: true` 才重建。缓存可通过 `--s-scalar-path` 加载，此时不要再给两份原始文件，
默认 SAFE YAML 无需再传 `--prune-algorithm safe`；自定义 YAML 应显式选择 SAFE。旧 YAML 若显式写了策略/损失上限，只接受上述固定值。
旧默认文件 `moe_safe_s_scalar.pt`、`moe_safe_s_scalar_v2.pt` 和 `moe_safe_s_scalar_v3.pt` 不被覆盖；
v1/v2/v3 缓存不能直接加载，也不能只对旧表调用 `.half()`，必须从两份原始
产物重新打包。原始 AMP/AIMER 和 ExFold 文件格式不变。

## 算子 ABI：不增加输入槽

沿用 XH2a `MoeBlock` 的 `s_scalar`（ONNX 第 15 号、从零计数）：

| 数据 | 含义 |
| --- | --- |
| `[E]` | 原 ACEP 标量路径，语义不变 |
| FP16 `[E,2*E+1]` | SAFE v4 数据包，算子内部拆包 |

每行 e 的列依次为融合 scalar、`coeff[e,:]`、`loss[e,:]`。
三个固定控制值不再写入数据包；缓存中的 settings 仅用于记录和校验固定策略，
不是运行时控制输入，也不会被导出为 ONNX 常量表的额外列。
coverage 仍为已有 `prune_threshold` 属性。LoRA 从第 16 号输入开始的位置不变。
已移除强制保留 FP32 的 `SafeDataBuffersMixin`，数据包是普通 FP16 buffer。
ONNX 两个加载入口保持 SAFE 数据类型，不再强转 FP32；旧一维 ACEP 加载行为不变。
不要对 SAFE 模型调用 `.float()`/`.bfloat16()` 后直接推理：v4 要求二维表为 FP16，
不匹配会报错。FP16 表存储字节数是此前 FP32 的一半，不等于整个模型显存减半。

代码入口：

- [配置、分阶段图绑定](../xhmodel_merak/xh_llm/models/qwen3_5_moe/dynamic_prune.py)
- [自动校准及缓存身份](../xhmodel_merak/xh_llm/models/qwen3_5_moe/safe_calibration.py)
- [AIMER 与流式配对统计](../xhmodel_merak/xh_llm/models/qwen3_5_moe/safe_calibration_stats.py)
- [校准文件、AMP/AIMER 融合、打包与缓存](../xhmodel_merak/xh_llm/models/qwen3_5_moe/safe_prune.py)
- [CLI](../examples_merak/llm/qwen3_5/qwen3_5_dynamic_prune_workflow.py)
- 配套 SDK：`xhquant/common/moe_safe.py`、`xhquant/nn/modules/moeblock.py`、
  `xhquant/backend/xh2a/ir/operators/moeblock.py`。

## 验证与边界

本次基于 ModelZoo `942ec017`、xhquanttool `bfaf60c0` 适配。CPU 小模型回归覆盖：
prefill 不变、decode 实际 MoE 输出、图实例隔离、严格文件检查、FP16 数据保真；
SDK 覆盖独立逐 token 参考、上/下取偶数、目标重算、不安全回退、旧 ACEP、
真实量化→HMONNX 导出→重载。v2 额外覆盖融合缓存、新旧版本隔离、配对表偏移、
在线无开方、极小权重及先验乘积溢出保护。用例见当前 CI 的 `test_qwen35_safe_prune.py`。
自动校准的数值与边界回归另见
[test_qwen35_safe_calibration_stats.py](../benchmark_test/ci_test_transformer5.16.0/test_qwen35_safe_calibration_stats.py)。

历史 v1 曾直接提取研究实现做 4,800 个随机 token 比对，折叠掩码相同、FP32 gate
最大差为 0。该数据只属于 v1，不作为当前 v4 与原研究实现完全等价的证明。

v3 与修改前 v2 默认设置直接对照：4 个种子、Top-K=4/8/16、5 个 coverage、
两种 gate 总质量，共 15,360 个 token 案例（含全零 gate 行），折叠 39,532 个槽；
折叠掩码和 FP32 gate 权重全部逐位一致。针对性测试还覆盖固定损失边界、
恢复非单位 gate 总质量、拒绝非固定配置以及旧缓存保留/拒绝复用。

以上逐位一致结论仅适用于历史 FP32 版本，不是 v4 FP16 的等价性或精度结论。

2026-10-13 历史 v3 本机验证结果（v4 回归另列）：

- 职责迁移后 ModelZoo SAFE/decode-only/旧 ACEP：63 passed，含从 SDK 迁入的融合/打包测试。
- 职责迁移后 SDK SAFE/旧剪枝/schema/共享常量/量化元数据：59 passed、38 skipped（GPU）。
  真实小 MoE 的 W8 SEFP 和 W4 SSFP 均完成量化、导出、重载，并与导出前量化模型输出逐位一致。
- 历史 v1 SDK 旧 LoRA 兼容：25 passed、4 skipped（GPU）。本机缺 C++ 编译器，该组以
  `TORCH_COMPILE_DISABLE=1` 执行 eager 数值回归；不代表编译路径验收。
- scoped Ruff、`git diff --check` 通过。环境为 Python 3.10.20、Torch 2.8、
  Transformers 5.16.1；不是项目要求的完整 Python 3.12 环境。
- 本次职责迁移的 3 个 ModelZoo 改动路径经正式 CI selector 选择 17 个测试目标；
  `run.sh --collect-only` 收集到 139 个用例，但 `test_recipe_coverage.py` 因 Python 3.10
  缺 `enum.StrEnum` 收集失败，**整套 CI 未通过**。未运行大模型目标，也未为此修改无关代码或安装新依赖。
- `ocr` 未安装；已用独立只读代码审查及上述静态/数值回归补充，但不声称运行过 ocr。

2026-10-13 v4 FP16 验证结果：

- 先添加 FP16 算子类型/零路由及损失舍入用例，确认旧实现失败，再完成实现。
- ModelZoo SAFE/decode-only/旧 ACEP：68 passed；SDK SAFE/旧剪枝/schema/共享常量/
  量化元数据：61 passed、38 skipped（显式禁用 GPU，仅进行 CPU 回归）。
- 算子逐输出 dtype 检查确认 SAFE 在线浮点张量均为 FP16；覆盖零分母、损失
  `1.0001` 不可误折叠、FP16 溢出输入拒绝、旧 v1/v2/v3 缓存隔离。
- W8 SEFP、W4 SSFP 小 MoE 量化、HMONNX 导出、双 parser 常量保真通过；runtime
  与导出前量化模型输出逐位一致。独立逐 token 参考使用 FP16 舍入语义，
  掩码要求完全一致；gate 允许不同归约顺序产生的小量舍入差。
- scoped Ruff、Python 编译及 `git diff --check` 通过；独立只读审查未发现范围内问题。
- 正式 CI selector 选中 17 个目标，`run.sh --collect-only` 收集 144 个用例，但仍因
  Python 3.10 缺少 `enum.StrEnum` 收集失败；未宣称完整 CI 通过。
- 未重新测量真实数据集精度或硬件性能，历史评测分数不可直接移用为 FP16 结果。

本次变更文件：ModelZoo `safe_prune.py`、`test_qwen35_safe_prune.py` 与本文；
SDK `common/moe_safe.py`、`nn/modules/moeblock.py`、两条 ONNX parser、
`moeblock_schema.py` 与 `test_moe_safe.py`。未修改专家 GEMM 或旧一维 ACEP 计算。

参考函数来源：`moe_prune_v2/vllm_acep_plugin/src/moe_prune_vllm_acep/exfold.py`；
v1 对照文件 SHA256：`af16cc73bd30cbf491793e8cdee1a35032502da64e8ed240096a1614c38abae6`。

针对性回归命令（在各自仓库根目录，保证导入本次更新的 xhquant）：

```bash
# ModelZoo
python -m pytest -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_qwen35_safe_prune.py \
  benchmark_test/ci_test_transformer5.16.0/test_qwen35_decode_prune.py \
  benchmark_test/ci_test_transformer5.16.0/test_qwen35_dynamic_prune_acep.py
# xhquanttool
python -m pytest tests/testing/xh2a/ops/test_moe_safe.py --noconftest
```

目前接入 XH2a 常规 decode/常规导出；不支持 SAFE 的 XH3、MTP/DFlash、huge/layerwise
导出。示例沿用已有 LLM-only workflow，不能据此声称完成多模态 RealWorldQA/CC-OCR。
未跑全模型精度、芯片实机、GPU 性能或 CUDA Graph；CUDA 专项用例在无空闲 GPU 时跳过。

逻辑折叠率为被折叠 token-expert 对数 / 原 Top-K 对数，不等于硬件耗时下降比例。
静态槽后端以 `expert_id=-1` 标识无效槽，复用跳过无效专家的内核；普通批处理路径
仍按批内最大剩余 K 打包，可能保留零权重填充槽。必须另外测量实际专家计算量和延迟，
不能直接把逻辑折叠率报告成实际算力收益。


## 2026-10-19 自动校准与本机完整导出验证

默认剪枝 YAML 和独立 CLI 现在启用自动 SAFE 校准，不再要求预先生成研究产物。
新增 `safe_calibration.py`、`safe_calibration_stats.py`，在 MoE 模型包装前调用；
同步更新 CLI、两份剪枝 YAML、示例说明和当前 CI 测试/impact 声明。

本机 checkpoint 为 `/data01/datasets/Qwen3.6-35B-A3B`。使用上述连续 WikiText train
`128 × 2048` 协议、batch 1，完整导出 40 层、每层 256 专家的 text-only Prefill/Decode，
导出上下文为 2048、Prefill chunk 为 256，基础量化 `w8a8h1_sefp`，MoE 为
`w4a8h0_ssfp`，SAFE coverage 为 0.85。

产物位于 `work_dirs/qwen3_6_35b_safe_export_20261019`，验证清单为该目录的
`safe_export_verification.json`。40 张 `[256,513]` FP16 表均与本次原始校准产物重新
打包的值逐位一致；Prefill 的 40 个 MoE 均无剪枝，Decode 的 40 个 MoE 均包含
0.85 阈值和对应 SAFE 表。两张图的外部权重引用和文件长度检查通过。

- SAFE/Decode/旧 ACEP/统计公式/真实微型 hybrid MoE 自动校准：108 项 CPU 回归通过。
- 完整导出 HMONNX 用 V2 runtime 实际生成 2 个 token，覆盖 Prefill 和 SAFE Decode；
  这只属于加载与执行检查，不构成完整精度或性能评测。
- 正式 selector 选择 38 个目标；`run.sh --collect-only` 收集 357 个用例，7 个收集错误
  来自 Python 3.10 缺少 `enum.StrEnum`/`tomllib`，不宣称完整 CI 通过。`ocr` 未安装。
- 新增及改动的校准/CLI/测试通过 Ruff、Python 编译和 `git diff --check`。

本机缺少默认 `c++` 命令，运行时首次编译 MoE CUDA 辅助扩展时，需指定已有 Conda
工具链；无需修改全局环境或另装编译器。以下为本次执行检查的命令（GPU 需有足够空闲显存）：

```bash
PATH=/data01/home/zukang.xu/anaconda3/envs/xh2/bin:$PATH \
CC=/data01/home/zukang.xu/anaconda3/envs/xh2/bin/x86_64-conda-linux-gnu-gcc \
CXX=/data01/home/zukang.xu/anaconda3/envs/xh2/bin/x86_64-conda-linux-gnu-g++ \
NVCC_CCBIN=/data01/home/zukang.xu/anaconda3/envs/xh2/bin/x86_64-conda-linux-gnu-gcc \
TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TRITON_MATMUL=0 XHQUANT_HSUM_JIT=0 \
PYTHONPATH=.:/data01/home/zukang.xu/workspace/xh2/xhquanttool_3 \
python examples_merak/llm/qwen3_5/qwen3_5_dynamic_prune_hmonnx_generate.py \
  --config work_dirs/qwen3_6_35b_safe_export_20261019 \
  --device 0 --max-new-tokens 2 --min-output-tokens 2 --use-v2 --auto-offload
```
