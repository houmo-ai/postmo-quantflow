# FLUX 3 Image Klein 适配记录

实现基线：ModelZoo `a807700a`；任务分支 `feat/flux3-image-klein`。
本地 checkpoint：`/data01/datasets/flux-3-image-klein`。
参考代码：Black Forest Labs `e53f4ea697fe45db8f38fca91a90f5623f7f8f37`。
SDK：`b4c7f631c9afddfb1a9745b4a1d401e042e64476`。
运行设备：RTX 6000D，Torch 2.8.0+cu128，Transformers 5.16.1。
构建/使用命令见[示例 README](../../../examples_merak/aigc/flux3_klein/README.md)。

<a id="adaptation-overview"></a>

## 适配交付总览（2026-10-21）

本节汇总当前交付；后续各节保留初次接入、双图扩容、动态多图和扩写器的分阶段证据。
使用步骤集中维护在示例 README，模型数学及图接口集中维护在模型解释，避免混淆当前默认值
与历史验收档位。运行环境及源模型身份以上述固定版本为基线。

### 当前功能与容量

| 内容 | 当前交付 |
| --- | --- |
| 统一入口 | `flux-3-image-klein` 的 build/inference API、CLI、recipe、Manifest 及环境依赖声明 |
| 核心计算图 | `text_encode`、`vae_encode`、`denoise`、`vae_decode`；计算图数量与参考图数量独立 |
| 核心精度 | W16A16、W8A16、W8A8 SEFP，group size 64 |
| 图像生成与编辑 | 文生图、单图编辑、多参考编辑；真实验收覆盖 1024² 输出及既有非方形样例 |
| 默认多参考 | 最多八张，共享 8704 参考 token；两张各约 1 MP，超出总预算时按宽高比动态缩小 |
| 默认物理容量 | 目标 4352、单参考 4352、文本 1280、联合 14336，VAE 画布 1536² |
| 文本与采样 | Qwen3-VL 八层 hidden states、官方 80-token 补齐、双时间 flow-map、BF16 舍入与 FP32 持久 latent |
| 数值与 VAE | residual 等价缩放、防 FP16 溢出、有效区域 GroupNorm、大 head attention 与空间 mask |
| 可选扩写器 | 官方 Qwen3.5-9B，多参考有序输入，W8A16H1 SEFP；输出 caption 经 FLUX 独立重新分词 |
| 扩写计算图 | Prefill、Decode 及逐视觉档位图；默认 256/512/1024 三档，默认组合共九张计算图 |
| 扩写容量 | 读取实际导出上下文 C 与 prefill P；生成预算为 `min(组件上限, C - 输入 token 数)` |
| 已验收扩写配置 | C=8192/P=256 与 C=4096/P=128；均保持官方参考图面积及 JPEG 处理规则 |
| 2048 分辨率及更大参考容量 | 提供 YAML 与构建操作说明；相应大档位需单独导出、真实推理及 Golden，未纳入当前默认包验收 |

### 实现、使用与证据入口

| 查阅目的 | 仓库位置 |
| --- | --- |
| 文本、DiT、VAE、Host、图 ABI 与数值处理 | [模型解释](../../../xhmodel_merak/xh_other_model/models/flux3_klein/EXPLANATION.md)及[实现目录](../../../xhmodel_merak/xh_other_model/models/flux3_klein/) |
| 容量校验与动态图像预算 | [capacity.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/capacity.py)、[processor.py](../../../xhmodel_merak/xh_other_model/models/flux3_klein/processor.py) |
| 导出、运行包、采样与 Golden | 实现目录的 `workflow.py`、`build.py`、`export.py`、`bundle.py`、`pipeline.py`、`golden.py` |
| 扩写协议、独立导出、组合及全档位 Golden | 实现目录的 `upsampler.py`、`upsampler_build.py`、`upsampler_golden.py` |
| 统一 API、recipe 及产物发布接入 | [model_integrations/flux3_klein](../../../xhmodel_merak/model_integrations/flux3_klein/) |
| 三种核心精度、容量与扩写 YAML | [配置目录](../../../configs_merak/workflows/xh2a/other_models/flux3_klein/) |
| 构建、推理、尺寸、参考数量、扩写和打包 | [示例 README](../../../examples_merak/aigc/flux3_klein/README.md)及[命令入口](../../../examples_merak/aigc/flux3_klein/unified.py) |
| 原生对照、双图、动态多图及扩写验证 | 示例目录的 `validate.py`、`validate_dual.py`、`validate_multi.py`、`validate_upsampler.py` |
| 当前 CI 与影响声明 | [核心测试](../../../benchmark_test/ci_test_transformer5.16.0/test_flux3_klein.py)、[扩写测试](../../../benchmark_test/ci_test_transformer5.16.0/test_flux3_upsampler.py)、[统一接口测试](../../../benchmark_test/ci_test_transformer5.16.0/test_unified_model_flux_3_image_klein.py)及同名 `.impact.json` |
| 扩写器完整验收 | [多参考、多档位及实际容量记录](flux3-klein-prompt-upsampler-validation.md#multireference-multigear) |
| 官方参考来源与许可证 | [THIRD_PARTY_NOTICES](../../../THIRD_PARTY_NOTICES)、[官方代码许可证](../../../licenses/flux-3-image-klein/LICENSE) |

### 运行产物与验收结论

纯推理核心包为 `work_dirs/runtime_packages/flux3_klein_adaptive_{w16,w8a16,w8}`，
带扩写器的组合包为 `work_dirs/runtime_packages/flux3_klein_multigear_{w16,w8a16,w8}`。
完整目录、组件区别、Golden 位置及打包指令见[产物与打包](../../../examples_merak/aigc/flux3_klein/README.md#artifacts-and-packaging)。

| 验收阶段 | 实际通过范围 | 本地汇总索引 |
| --- | --- | --- |
| 核心动态多参考 | 三精度的 2/3/4/8 图、16 步级联、模式切换、预处理、公开 API 与 Manifest；增量 CI 819 passed | `work_dirs/adaptive_references/acceptance.json` |
| 多图扩写及可变容量 | 三精度各文生图与 1/2/4/8 图，加 4k 组件文生图/八图，共 17 个完整级联样例；同 caption 的 latent/PNG 像素一致 | `work_dirs/upsampler_multigear/acceptance.json` |
| 扩写导出及 Golden | 两套真实组件，各五图及合成/真实两例，共 20 组新增 Golden；未变化核心图按字节摘要复用既有 Golden | 同上及两套组件的 `upsampler_golden_report.json` |
| 扩写增量 CI | 181 passed，0 failed，0 skipped；之后文档 policy/recipe 门禁 133 passed | `work_dirs/upsampler_multigear/ci_complete.xml`、`policy_final.log` |

这些结论对应 RTX 6000D 上的 GPU HMONNX 功能和数值验证。八附件使用两张图片重复组成，
不作为八主体画质 benchmark。五视觉档位已覆盖配置/端口契约，真实导出验收针对默认三档；
十图、大分辨率及更多参考图各 1 MP 的大档位、NPU 板端和正式画质评测均不据此声明通过。

此前形成的算法 Word/PDF 快照保留在 `work_dirs/deliverables/flux3_klein_20261021/`，
文件为 `FLUX3_Image_Klein_适配记录.docx` 和同名 PDF。后续多图扩写和最新产物范围以本页
及扩写验收记录为准。权重、环境、Golden、压缩包与生成文档保留在 `work_dirs`，Git 提交
覆盖实现、配置、示例、测试、来源许可证和可复现的 Markdown 记录。

### 开发阶段历史索引

最终交付将核心、多参考、扩写器、测试和文档合并为一个提交。下表保留整理前的开发
阶段索引，用于关联既有验收证据；这些历史提交不再分别作为合入入口。

| 提交 | 内容 |
| --- | --- |
| `b55b9dee` | 核心四张计算图、三种精度、统一接口、原生对照及 Golden |
| `bc480243` | 可选官方扩写器、独立组件及组合构建 |
| `69b4fb4e` | 双参考图及 DiT 容量扩展 |
| `e979ee5c` | 八图数量上限与共享参考预算的动态缩放 |
| `21feaefb` | 2048 输入保留和输出尺寸的配置说明 |
| `04b19a4b` | 扩写器接入、CLI/Python 使用及实现引用 |
| `60473d29` | 多参考扩写、多视觉档位、实际上下文及完整验收 |

## 设计与边界

四张学习图：text_encode、vae_encode、denoise、vae_decode。生成核心不含 Qwen3-VL 视觉前向；
prompt upsampler 为独立可选组件，核心接口接收完整 caption，启用扩写时由组件先生成 caption。
DiT 每步接收当前及下一步时间，保留
三种 mode 的独立前置层与联合双向层，不使用 Qwen-Image 的冻结条件 KV。

默认 1MP 配置：目标 4352 token、单参考/总参考 4352/8704 token、文本 1280 token、VAE 画布 1536²、
最多八张参考图共享参考预算、联合容量 14336。两张各 1 MP 保持原有尺寸；更多参考在超出
总预算时动态缩小，小图预算优先保留，大图共享剩余容量。Host 保持单图填充、多图中心裁剪，单图面积预算从
官方 4MP 收紧到 1MP。数量、目标和文本超限，或参考图无法在最小有效尺寸下满足共享预算时明确拒绝。

文本捕获八个指定 hidden-state 索引，保留补齐 query 特征；参考实现 4096 上限与包内
8192 元数据声明的差异已显式记录。部署图不静默截断到物理容量 1280。
VAE 使用有效区域 GroupNorm，原生反射填充属于有效内容，部署额外补零不参与统计。
具体数学与 ABI 见[模型解释](../../../xhmodel_merak/xh_other_model/models/flux3_klein/EXPLANATION.md)。

## 已定位问题

1. 原生调制 `tuple(tensor.chunk(...))` 不可由 FX 追踪，部署 wrapper 改为显式三输出解包；
   数值、完整 DiT SDK 导出与执行分别验证。
2. 小模型 VAE fixture 原先单头 D=32，SDK CUDA 后端不支持该宽度；fixture 改为 D=64，
   保留真实 GroupNorm、卷积、残差及单头 attention，不 mock 后端行为。
3. RTX 6000D 的默认 D=512 Triton attention 使用 168192 字节共享内存，设备上限为 101376。
   模型图保持原始 512 维单头 QK 点积与 scale，将 Q/K/V 补零到 768，调用 SDK 已存在的
   bounded-memory 大 head 路径，输出裁回 512。不拆分 attention head，不修改 SDK 源码。
4. 真实 checkpoint 的文本投影超过 115000，后续 residual 达到数百万。直接 FP16 导出会
   饱和；统一将 residual、输入/输出投影缩小 1024 倍，LayerNorm epsilon 除以 1024²，
   保持数学等价。小真实图严格浮点对齐以及大值 SDK 回归均通过。
5. 参考时间嵌入先做 BF16 `1000*t` 再转 FP32 求三角函数；不能先把 t 转 FP32。
   Host 也保留 BF16 输出增量与 FP32 持久 latent，提供精确舍入回归。
6. SDK Golden 默认在 step 目录生成软链接，会被公共发布包校验拒绝。采集完成后将其转为
   包内普通硬链接，不放宽公共安全校验；回归覆盖 Golden 后重新从公共接口推理。
7. 当前 SDK 已移除 FlashAttention 的 `s_bits` 属性。发布包精度校验对齐当前
   Q/K/V/P 四路契约，避免拒绝新 SDK 正常导出的图；三种精度和各路错误属性均有回归覆盖。

## 验证记录

本节保留首次核心四张计算图、单参考档位的验收记录；双图、动态多参考及扩写的新增证据见后续各节。

原生 BF16 已执行 seed42、16 步的 1024² 文生图和高768/宽1376 单参考图编辑。
参考图固定使用原生文生图，两个 caption 均为完整场景描述，未运行独立 prompt upsampler。
原生逐步 latent 与图片保存于 `work_dirs/real_validation/`。

### 真实权重结果

三种精度均重新构建并执行完整四图。原生文本/DiT 为 BF16，VAE 为 FP32。seed42、16 步；编辑固定使用原生生成图，避免参考图
差异混入误差。逐步指标在 CPU FP64 中计算。以下是固定案例结果，不是画质 benchmark：

| 精度 | 文生图 RGB PSNR / dB | 编辑 RGB PSNR / dB | 文生图最终 latent cosine | 编辑最终 latent cosine |
| --- | ---: | ---: | ---: | ---: |
| W16A16 SEFP | 32.1406 | 49.024 | 0.993381 | 0.999670 |
| W8A16 SEFP | 33.0321 | 42.8884 | 0.991678 | 0.998518 |
| W8A8 SEFP | 22.8181 | 33.9196 | 0.931103 | 0.990961 |

已人工检查生成与编辑图片：红色茶壶、木桌与窗光正常，编辑保持场景并改为蓝色釉面。
W8A8 的固定样本偏差高于两档 A16，不能据此推断任意 prompt 的画质排序。

分阶段定位证据：真实文本 wrapper 对原生 cosine 0.9999942，W16 文本图对原生 cosine
0.9999961；真实 FP32 DiT wrapper 对原生 BF16 单步 cosine 0.9999379、RMSE 0.01490；
修正后的 W16 DiT 在同一原生文本输入下单步 cosine 0.9999279、RMSE 0.01876。
这些单步值与完整级联使用量化文本所得的轨迹指标分别记录。

### 产物与复现位置

所有运行产物位于任务 worktree 的 `work_dirs/`，不提交模型权重、环境或逐算子数据：

- `artifacts/flux3_klein_w16`、`artifacts/flux3_klein_w8a16`、`artifacts/flux3_klein_w8`：
  三种独立精度的正式发布根，含完整四图、processor/embedding、配置与公共 Manifest。
- `runtime_packages/flux3_klein_*`：依据 Manifest 搬迁的纯推理包，未带逐算子激活；
  真实权重公共接口和重复请求检查在这些搬迁后的目录执行。
- `validation_w16`、`validation_w8a16`、`validation_w8`：原生及 HMONNX 图片、逐步指标、
  最终 latent 与重复请求/公共接口检查结果。
- `reference_code_audit.json`：逐方法核对固定版本的官方参考；前向差异仅局部变量名和
  lambda 规范化，构造器扩展保留官方默认结构。
- `cli_infer_validation.log`：文档 CLI 的真实推理输出，与 W8A16 基线逐像素一致。
- `verification_report.json`：三档逐图 Golden、公共 Manifest 与搬迁请求检查的汇总。
- `artifact_audit.json`：12 张图的 I/O、算子、精度与存储审计。所有 Linear/Conv 的 group64、
  实际 W/A bits、FlashAttention 精度与声明一致，没有 Softmax、BfpFlashAttention 或 KVcache。
- 每个正式包的 `golden_meta_info.json` 与各图 `step_0`/`step_1`：synthetic/真实参考图
  的连通逐算子 Golden；Transformer/VAE 前向均由 HMONNX 执行，Host 从包内表查 token embedding。

### 支持与验证边界

| 能力 | 状态与范围 |
| --- | --- |
| 四图构建与统一加载 | verified：W16A16、W8A16、W8A8，所有图均使用真实 SDK 导出与执行 |
| 文生图、单参考编辑 | verified：真实权重 1024² 文生图、768×1376 编辑 |
| 不同尺寸、空参考、脏 padding、单/多参考顺序 | verified：原生浮点及小真实模型 SDK 回归 |
| 发布包搬迁、删除源权重后推理 | verified：三档小真实模型通过公共 API 执行 |
| 真实权重包搬迁与请求复现 | verified：三档公共输出逐像素一致，编辑和尺寸切换后 latent 逐值一致 |
| 默认生产多参考包、超出当前容量的请求 | unsupported：默认包只接受 1 张参考，扩容需重建 |
| Prompt upsampler、Base/FP8/NVFP4 权重 | unsupported：本次只接蒸馏 BF16 主模型和完整 caption |
| NPU 板端性能、正式图像 benchmark | 未执行：本次证据为 GPU HMONNX 功能/数值验证 |

### CI 与质量记录

本次用例位于当前 `ci_test_transformer5.16.0`，包括 19 项 FLUX 原生/SDK 测试、统一
small-model runner 用例，以及公共 recipe/build/inference/impact 回归。最终结果及 XML
见 `work_dirs/flux3_verified_ci.log`、`work_dirs/flux3_verified_ci.xml`。Ruff check/format 与两种
`git diff --check` 为提交前门禁。环境中无 `ocr` 命令，按模型/包装/运行时边界人工审查。

最终增量门禁为 **801 passed**（600.67 秒），包含 19 项 FLUX 原生/SDK 测试和统一
small-model runner 的独立进程用例。新增 Golden 后公共推理回归也单独执行通过。
容量篡改用例同时修改 text/joint 容量，保证元数据内部自洽；三档均验证物理图 shape
检查仍拒绝不匹配的声明，强化矩阵重跑 3 passed（372.87 秒）。
首轮出现依赖环境失败；在独立任务环境补齐 TorchCodec/LeRobot 相关依赖，并将 PyAV
从基础环境的 18.1 调整到 recipe 声明的 15.1 后，完整门禁重新运行全部通过。
环境调整保留在独立任务环境，公共校验继续使用现有约束。

首轮曾暴露训练权重与小随机 fixture 的动态范围差异，未把“导出通过”作为正确性证据。
失效构建已删除；定位日志和失败轨迹保留供追溯，正式包均来自修正后的重新构建。

最终汇总：12 张图 × 2 组 Golden 全部完成，逐图 I/O shape/有限值、公共 Manifest 与
三档搬迁包的重复请求校验均通过。汇总时间见 `work_dirs/verification_report.json`。

## 双参考图扩容（2026-10-19）

三档默认配置增加到最多两张参考图，每张预处理面积不超过 1048576 像素。
目标容量 4352、单参考容量 4352、参考总容量 8704、文本容量 1280、联合容量 14336；
VAE 画布仍为 1536×1536。单图保持 reflect/symmetric 填充，多图保持分别缩小与
16 网格中心裁剪；两张图分别执行 VAE encode，再按附件顺序拼接，参考编号轴为 -1、-2。

`sources.image_bundle` 现在允许只修改参考数量/总容量：复用原包文本图、两张 VAE 图、
processor、embedding 及 ONNX 外部权重，只从原 checkpoint 重新导出 DiT。
不兼容的文本/VAE 档位、精度或源配置仍明确拒绝；根运行 metadata、effective workflow、
公开 export 索引和 Manifest 由正常发布事务重新生成。旧包及其 Golden 不作修改。

本次生成的纯推理包位于 `work_dirs/runtime_packages/flux3_klein_2ref_{w16,w8a16,w8}`。
`work_dirs/dual_artifact_audit.json` 核对每档复用的 11 个运行文件，其 Manifest SHA-256
与旧单图包逐项一致。含 Golden 的独立交付副本位于
`work_dirs/artifacts/flux3_klein_2ref_{w16,w8a16,w8}`。

原生对照使用两张 1024×1024 茶壶照片、1024×1024 输出、seed42、16 步、完整双图 caption；
文本经 FLUX 处理后为 160 token。原生图像和逐步 latent 保存在 `work_dirs/dual_reference_native`。
部署对照程序为 `examples_merak.aigc.flux3_klein.validate_dual`，同时检查换序、非方形第二张
参考、双图/单图/无参考与目标尺寸切换、完整重复请求和公开接口输出。

并发环境首轮触发 tokenizer Rayon 线程池资源不足，CI 为 809 passed / 3 failed；
W16 对照也遇到相同环境错误。复验显式设置 `TOKENIZERS_PARALLELISM=false`、
`RAYON_NUM_THREADS=1`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`、
`MKL_NUM_THREADS=4`，不改变分词与推理语义。中断的部分 Golden 清理后，对改变的 DiT
重新采集单参考 synthetic 和双参考真实输入；三个未变化组件按图及外部权重的 Manifest
SHA-256 一致性复用旧包完整 `step_0`/`step_1`，使用包内普通硬链接保留资源闭包。
`golden_meta_info.json.graph_provenance` 分别记录新采集和复用的来源，不将旧组件的
真实单图样本声明为新双图采集。两个新 DiT 案例的条件均来自实际 HMONNX 文本/VAE 输出。
增量采集命令为 `PYTHONPATH=. python work_dirs/dual_capture_dit.py --precision w8a16 --device cuda:0`，
其他两档替换 precision；公开 `unified golden` 仍可对全新目的包从头采集所有组件。

双图扩展只针对 FLUX 核心。可选扩写器仍保持单参考图限制，双图请求直接提供完整 caption。

双图变更的增量 CI 已完成：**812 passed、8 warnings，659.44 秒**，记录为
`work_dirs/dual_ci_final.log` 和 `work_dirs/dual_ci_final.xml`。本轮仅补齐真实权重验收和
Golden，没有改变已通过 CI 的模型实现与测试。

### 双图真实权重功能与数值结果

三种精度均已通过 16 步双图轨迹门禁，交换参考顺序会影响输出；将第二张参考改为
1376×768 后，实际预处理为 1360×752，第一张仍为 1024×1024。切换双图/单图/无参考及
目标尺寸后，完整重复请求的 latent 逐值一致；公开 API 与直接级联的 PNG 逐像素一致。
完整报告为 `work_dirs/dual_reference_{w16,w8a16,w8}/dual_reference_report.json`。

以下对照同一份原生 BF16 双图样本，保留误差供评估，不据单个样本推断画质排名：

| 精度 | RGB PSNR / dB | 最终 latent cosine | 最终 latent RMSE |
| --- | ---: | ---: | ---: |
| W16A16 SEFP | 24.4461 | 0.963120 | 0.271936 |
| W8A16 SEFP | 23.4084 | 0.953332 | 0.305351 |
| W8A8 SEFP | 27.8467 | 0.979775 | 0.200979 |

W16 的完整验收本轮重新执行并以退出码 0 结束；W8A16/W8A8 复用已完成、对应同一
发布包的完整报告。SDK 仍为 `b4c7f631c9afddfb1a9745b4a1d401e042e64476`。


### 双图最终交付验收

2026-10-19，三档扩容包完整通过 GPU HMONNX 交付验收。汇总报告为
`work_dirs/dual_delivery_acceptance.json`，逐图审计为
`work_dirs/dual_verification_report.json`，复用文件对照为
`work_dirs/dual_artifact_audit.json`。

- 三档各四张图、每图两组 Golden，共 12 张图 / 24 组：6 组为重新导出的 DiT
  连通采集，18 组为图和外部权重摘要相同的未变化组件复用。三次采集均以退出码 0 结束。
- 每组顶层图输入输出均核对实际 ONNX shape、dtype 和有限值；Golden 不含软链接。
  三档纯推理包及含 Golden 副本的全部 Manifest 文件均重新计算 SHA-256 并通过校验。
- 新 DiT 的真实双图输入分别包含两个非零且不同的 4096-token 参考段，参考总长为
  8192，8192:8704 的尾部全零；有效前缀 Gather 顺序、mode/joint mask 和长度均通过
  保存数据的逐值检查。该单步 Golden 使用 80 个文本 token；前述 16 步数值对照使用
  完整 caption 的 160 个文本 token，两组输入分别记录，均由实际 HMONNX 文本/VAE 连接。
- 最终审计脚本首轮因向内部文件校验函数传入相对根路径而失败；文件均实际存在且位于包内。
  将临时脚本的包根规范为绝对路径后完整重跑通过，未修改产物或放宽校验。
- 最终附加 CI policy/recipe 门禁为 **133 passed、6 warnings，19.55 秒**，记录为
  `work_dirs/dual_delivery_policy.log`。7 个改动 Python 文件 Ruff check/format 全部通过，
  工作区与暂存区 `git diff --check` 均通过。模型实现自 812 项 CI 通过后未再修改。

复验最终数据审计（从任务 worktree 执行，使用本任务 Python/SDK 环境）：

```bash
PYTHONPATH=/data01/home/jing.huang/xh2modelzoo-flux3-klein \
  /data01/home/jing.huang/xh2modelzoo-flux3-klein/work_dirs/flux3_env/bin/python \
  /data01/home/jing.huang/xh2modelzoo-flux3-klein/work_dirs/finish_dual_acceptance.py
```

| 当前交付能力 | 状态与证据范围 |
| --- | --- |
| 两张参考图，每张最高约 1 MP | verified：三档真实权重，分别输入两张 1024×1024 图 |
| 不同宽高与附件顺序 | verified：混合宽高、交换顺序、逐参考预处理与有效 token 计数 |
| 单图/双图/无参考与尺寸切换 | verified：三档切换后 16 步完整请求 latent 逐值复现 |
| 统一接口和核心级联一致性 | verified：三档公开 API PNG 与直接 HMONNX 级联逐像素一致 |
| 参考容量扩展与四图完整交付 | verified：只重新导出 DiT，复用组件闭包摘要一致，24 组 Golden 审计通过 |
| 三张及以上参考图 | unsupported：当前生产档位明确拒绝，CI 验证边界 |
| 双图 prompt upsampling | unsupported：可选扩写器仍限单图，双图使用完整 caption |
| NPU 板端执行/性能及正式画质 benchmark | unsupported：本次未执行，不属于本轮 GPU HMONNX 验收结论 |


## 动态参考总预算（2026-10-20）

本轮基线为双图交付提交 `69b4fb4e`。默认 `max_references` 从 2 增加到 8，单张上限
仍为 1048576 像素，参考总容量仍为 8704，联合物理容量仍为 14336，四张计算图 ABI 不变。
旧包的数量限制来自其已发布 metadata，修改源码 YAML 不会自动改变旧包；新包通过公开
构建流程复用旧四图并重写容量、有效 workflow、export 索引和 Manifest。

### 预处理与职责

Host 先根据每张面积/边长预算计算官方式的等比缩小和 16 网格中心裁剪尺寸，统计实际 token。
总量满足预算时保持原像素处理结果；超限时按需求从小到大进行有上限的均分，小图不放大，
剩余容量由大图共享。最终从 EXIF 修正后的原图执行一次 LANCZOS 缩小，再中心裁剪；
对齐后的 token 总和再次由既有请求校验检查。极端尺寸无法满足 16 网格时明确拒绝。
该总预算策略是部署附加处理，官方多参考只限制每张约 1 MP，不共享固定 token 总上限。

单参考反射填充及裁回逻辑不变。新预处理参数在独立调用时可省略以保留原行为，实际 runtime
总是传入包内 `reference_total_capacity`。公开输出增加 `stats.geometry`，可检查实际高宽。
可选 prompt upsampler 继续只支持单参考，多图使用完整 caption。

数量上限不决定物理张量长度；只提高数量上限时所有图及外部权重保持字节一致。
参考总容量从 8704 增到 17408 时才需要重导 DiT，后者即使只传两图，投影/MLP 等仍面对
更大的物理长度。有效长度和 mask 不等于所有算子跳过 padding；相关开销及四图 1 MP、
十图共享预算、八图 1 MP 的客户操作说明均维护在示例 README。

### 验证位置

- `work_dirs/adaptive_references/red.log`：新增动态预算用例在旧实现上 7 failed。
- `work_dirs/adaptive_references/processor_green.log`：预算、像素保持、原图单次缩放、EXIF、
  宽高比、颜色顺序、极小预算拒绝和默认配置的聚焦检查，11 passed。
- `examples_merak.aigc.flux3_klein.validate_multi`：2/3/4/8 图的真实 HMONNX 验证，显式预处理
  与重复请求一致性、公开接口一致性，以及与先前同精度双图完整结果的逐值对照。
- 本轮纯推理包为 `work_dirs/runtime_packages/flux3_klein_adaptive_{w16,w8a16,w8}`。

本节为动态扩展的记录；前面 2026-10-19 的双图范围和验收数据保留为历史证据。


本轮增量 CI 为 **819 passed、8 warnings，742.41 秒**，完整记录在
`work_dirs/adaptive_references/ci.log`、`ci.xml`。三种精度的小真实模型均完成单图包→双图
DiT 扩容→仅提高数量到八张的公开构建；数量升级阶段导出调用为零，四图字节保持一致，
之后搬迁并删除源模型，通过八图动态输入、公开推理、模式切换和容量篡改拒绝。
W16 小模型另完成 synthetic / 真实八附件的四图连通逐算子 Golden。

三档真实包的全部 Manifest 文件摘要已重算通过，四图与外部权重闭包均与旧双图包一致。
`work_dirs/adaptive_references/artifact_audit.json` 记录闭包摘要与逐图 Golden I/O 审计。
`work_dirs/artifacts/flux3_klein_adaptive_{w16,w8a16,w8}` 是含既有 Golden 的独立交付副本：
共复用 12 图 / 24 组完整 Golden，不将旧双图样本宣称为新八图逐算子采集；八图请求首步的
连通顶层输入输出由新验证程序单独记录到 `validation_*/dit_io/`。


### 动态预算最终验收

三种精度均完成 2/3/4/8 张附件的 16 步真实 HMONNX 推理，并以退出码 0 结束。
前两张使用既有红/蓝茶壶参考，更多附件重复这两张图，验证数量和预算处理，不宣称八主体
画质评分。相同输入在三档得到一致的处理几何：

| 附件数量 | 每张处理尺寸 | 参考 token 合计 |
| --- | --- | ---: |
| 2 | 1024×1024 | 8192 |
| 3 | 848×848 | 8427 |
| 4 | 736×736 | 8464 |
| 8 | 512×512 | 8192 |

三档均通过双图完整 latent/PNG 与旧包逐值对齐、切换到无参考后八图完整请求复现、
显式预处理与自动预处理一致、公开 API 与直接级联 PNG 逐像素一致。八图首步保存输入的
有效长度为 `[4096, 8192, 160]`，八个参考段均非零，参考 padding 为零；全部图端口
shape/dtype/有限值、有效前缀顺序、mode/joint mask 和长度校验通过。

汇总为 `work_dirs/adaptive_references/acceptance.json`，逐档完整结果为
`work_dirs/adaptive_references/validation_{w16,w8a16,w8}/multi_reference_report.json`。
本轮不重导真实四图，复用闭包与既有 Golden 的证据见 `artifact_audit.json`；
`validation_*/dit_io/` 是新增八图连通顶层 I/O，不与复用的逐算子 Golden 混淆。
SDK revision 仍为 `b4c7f631c9afddfb1a9745b4a1d401e042e64476`。

最后文档 policy/recipe 门禁记录在 `work_dirs/adaptive_references/policy_final.log`。
7 个改动 Python 文件 Ruff check/format 和两种 `git diff --check` 为提交前检查；环境仍无
`ocr` 工具，按 Host/图 ABI、构建复用及公开接口边界人工审查。本轮证据为 GPU HMONNX
功能/请求一致性，不包含 NPU 板端或正式画质 benchmark；四图每张 1 MP 的大档位仅提供
构建操作说明，不声明已在本轮导出验收。

## 2026-10-21：扩写器多参考及多视觉档位

可选 Qwen3.5-9B 扩写组件现支持多参考编辑，默认视觉档位 256/512/1024，
每图独立选择最小可容纳档位并保持附件顺序；两张语言图加三张视觉图，与 FLUX 核心
组合共九图。上下文 C 和 prefill P 从实际导出 metadata 读取并复核物理图，
本次生成上限为 `min(组件上限, C - 模板/文字/全部图片输入 token)`，不再固定要求
8192 上下文和完整预留 3584。FLUX 文本容量 1280 仍独立检查。

已实际导出 8192/256 与 4096/128 两套组件。三种核心精度均通过文生图、1/2/4/8 图的
16 步完整级联及同 caption 的 latent/PNG 一致性；4k 包另通过文生图及八图。
新增扩写组件全部五图均有合成/真实 Golden，未变化的四张核心图按字节摘要复用已有
Golden；搬迁后的离线公开 CLI 八图推理通过。最终增量 CI 为 181 passed。

两图和八图的原始/包内 processor、M-RoPE 逐值一致，并核对真实 HMONNX 视觉特征
与原始 BF16；八附件重复两张图，属于功能与状态验证。本轮实际验收三视觉档位及最多
八附件，不声明五档真实后端、十图、NPU 板端性能或正式画质 benchmark。

当前范围、容量表、产物位置和完整证据见
[扩写器多参考与多档位验证](flux3-klein-prompt-upsampler-validation.md#multireference-multigear)，
操作方式见[示例 README 扩写器指南](../../../examples_merak/aigc/flux3_klein/README.md#prompt-upsampler)。
前面各日期的单参考扩写记录保留为其当时的验收范围。
