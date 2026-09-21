# GLM-5.3-Flash / Merak

公开部署 recipe `glm-5.3-flash` 使用 **非共享专家 W4 G64，其余 W8，激活 A16**。
GPTQModel 负责生成 packed checkpoint，Merak 负责低内存 HMONNX 导出。
完整流程见 [W4 G64 部署](w4g64.md)。KDA `forget_gate.f_a_proj/f_b_proj` 和 `b_proj`
保留 W16 A16；其他权重的 GPTQ / 导出 RTN 分工在该文档中列明。

以下保留原 **XH2a W8A16** 基线（`w8a16h0_sefp`），固定 prefill 256、decode 1、
context 4096、batch 1。输入为 `weights/GLM-5.3-Flash` 的原始 block-FP8/BF16
checkpoint；`quant: null` 表示跳过独立 GPTQ/AutoRound 阶段，W8 权重量化由导出阶段的
xhquant 完成。原始 FP8 的反量化是 checkpoint 消费步骤。

接入文本、图片和视频，类名为 `Glm5NextForConditionalGeneration`。
默认 metadata 记录 `supported_modalities: [text, image, video]`，视觉组件使用独立 ABI，
文本缓存 ABI 保持不变。当前生成支持 batch 1，不支持 beam search、PageAttention 和投机解码。

## 导出与推理

在仓库根执行。GPU 编号需要按运行时空闲情况选择；本次验证使用物理卡 0、2、3、4，
映射为进程中的 `cuda:0` 至 `cuda:3`。

```bash
conda activate xhquant_55_tf_latest_20260828
export CUDA_VISIBLE_DEVICES=0,2,3,4
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=.
python examples_merak/llm/glm5_3_flash/glm53_workflow.py \
  --model weights/GLM-5.3-Flash --output work_dirs/glm53_w8a16
```

导出只有低内存路径：先依次加载视觉 patch 投影、24 个 block 和语言投影，每个组件导出后释放；
随后主进程加载 attention、mHC、norm、embedding 和 lm_head，
FFN 保持 meta 占位；最多四个独立进程各加载一个 dense MLP / MoE，复用同一份权重
导出 prefill/decode 子图，释放后再加载下一层，最后内联回主图。
内存上界由主图权重与并发 FFN 数决定；没有 `low_memory` 开关或全模型一次加载的导出分支。
每个 worker 的 PyTorch CPU 运算使用一个线程，避免逐 FP8 block 反量化时线程过量争用。
`--max-layers 4` 可导出真实前四层，用于覆盖完整混合层组合；该产物不能代表全模型生成质量。

已有文本产物时，可复用原文件补齐视觉组件。新目录必须与原目录在同一文件系统，
文本图及权重用硬链接复用，metadata 单独原子替换，不修改原目录内容：

```bash
python examples_merak/llm/glm5_3_flash/glm53_workflow.py \
  --model weights/GLM-5.3-Flash --text-meta <文本产物>/golden_meta_info.json \
  --output work_dirs/glm53_multimodal_w8a16 --vision-max-patches 1024
```

```bash
export ENABLE_HMINFERENCE_V2=1
python examples_merak/llm/glm5_3_flash/generate.py \
  --meta <导出目录>/golden_meta_info.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32
```

图片增加 `--image /path/to/image.jpg`，视频增加 `--video /path/to/video.mp4`，
两个参数都可重复，也可混合使用。CLI 调用原生 `Glm5NextProcessor` 完成模板、resize、
归一化、patchify 和视频采帧/时间戳；`model.get_processor()` 提供相同配置。
文件视频通过 `load_glm53_video()` 显式使用 PyAV 解码，保留 GLM 原生采帧和奇数帧补齐，
避免当前 PyTorch 2.8 环境中不兼容的 TorchCodec；已有帧数组可直接交给 processor。

生成入口启用 runtime 的多 GPU 分配与输入跨设备搬运；该开关不提供专家权重的 CPU 换入换出。
完整 W8 模型的单阶段外部权重约 322.66 GB。当前 SDK 默认使用 90% 的空闲显存作
加载预算，四张 A100 80GB 的默认预算不足以直接加载完整产物；本次完整模型验证的是
导出，生成入口使用真实前四层验证，完整 45 层生成尚未验证。
每次 `generate()` 重置状态。
输入为一条未填充的多模态 token 序列，Host 在视觉特征回填后负责分块和右侧填充。
HF cache 仅作为生成协议载体，不能传入外部 `past_key_values` 续接请求；
KDA 状态和已处理长度由 Merak 维护，因此也支持前 1～3 层全为 KDA 的裁剪配置。

## 视觉组件与输入边界

视觉编码器包含 24 层、hidden 1024、16 个 head；patch 大小为 14，temporal patch 为 2，
spatial merge 为 2，输出维度为语言模型的 4096。保留 Q/K RMSNorm、二维 RoPE、
clamped SwiGLU 及 merger 中的 LayerNorm/GELU。Conv3d patch 投影和 Conv2d merge
改写为同一权重布局的 Linear，逐组件物化和导出。

`glm53_vision.abi_version=1` 记录有序的 `stem`、`block_00`～`block_23`、`merger`，
图路径相对 metadata 所在目录。默认每帧容量 1024 patches，合并后最多 256 个特征；
支持此容量内任意合法长宽比，Host 提供 merge 顺序的 RoPE 和有效 key mask，并裁去尾部填充。
一个视频 temporal group 独立经过整套视觉组件，与原生视觉 attention 的帧隔离语义一致。
禁止把一张超过容量的图切成独立 attention 窗口；超出容量或网格不匹配时直接报错。

默认 processor 的单图预算和单视频**总**预算均为 256 个合并 token。
这通过原生 resize 保持长宽比并约束开销，不强制缩放为 448×448。
原生 processor 的预算可以显式调整，但每帧仍须满足导出容量，完整 prompt 须满足文本 context。
未采样的文件视频使用原生 fps=2；已采样数组需要 `do_sample_frames=False` 和正确的 video metadata。

图像和视频都用 `image_token_id` 占位；视频起止标签之间的占位符归视频，其他归图片。
特征数与占位符数在进入文本图前校验，视觉仅在请求首轮执行，decode 继续消费已提交文本状态。
`export_vision: false` 只用于显式导出文本组件，旧文本产物仍可加载，但不能接收视觉输入。

## 模型语义与状态

本地发布配置是 **34 层 KDA + 11 层 learned C4 DSA**，不是 GDN。
DSA 位于第 3、7、…、43 层，末层 44 为 KDA。每层含四路 mHC；最后对四路做无权平均。
前 3 层为 dense MLP，其余为 288 专家、每 token 选 8 个专家的 MoE，另有共享专家。

- KDA 复用 Ling 的 chunk 分解和 Qwen3.5 的三角逆，但 Q/K 使用 GLM 的
  `x / sqrt(sum(x*x) + 1e-6)`，不能用 Ling 的 `F.normalize` 替代。
  forget gate 按 key channel 衰减，发布配置下界为 -5。
- DSA 的 raw index key **先 LayerNorm 再压缩**。四个连续 token 的 raw key
  用 `softmax(raw_gate + APE)` 做逐通道加权，无 overlap。
- 32 个索引头的 ReLU 点积分数乘可正可负的 head weight 后求和。选最多 512 个完整池，
  展开成 2048 个原 token，再附加 0～3 个未完成 tail token。
- 主 attention 使用原 token 的 normalized MLA latent，压缩池仅用于索引。
  NoPE 下把 `kv_b_proj` 的 K 因子吸收到 query，V 因子放在 attention 后，
  保持原始 256 维 Q/K 的缩放因子。
- MoE 用 sigmoid 原始分数生成归一化权重，correction bias 仅参与选专家；
  routed scaling factor 为 2.5，SwiGLU 保留发布配置的 clamp。

缓存按层交错排列，输入/输出均显式命名，metadata 的 `glm53_state_abi_version` 为 1。
前三个 HMONNX 输入沿用 Merak 的 `input_1`、`valid_length`、`current_length`，
分别对应图语义中的 `inputs_embeds`、`past_seq_length`、`current_input_length`；
其余控制和逐层缓存保留 GLM 的显式名称。
W8A16 导出接口中的浮点缓存为 FP16；PyTorch 参考图在归一化和 KDA 中间计算中使用 FP32。

| 层型 | 缓存（发布配置，batch 1） |
| --- | --- |
| KDA | conv `[1,24576,4]`；recurrent `[1,64,128,128]` |
| DSA | raw latent `[1,1,C,512]`；pool `[1,1,floor(C/4),128]`；raw index key 和 raw gate 各 `[1,4,128]` |

给定已提交长度 `L`、本次真实长度 `n`，Host 计算 `offset=4-L%4`、
`new_count=(L%4+n)//4`、`write_start=L//4`，并提供 query/pool/tail 有效性。
图只提交有效新池，边界 carry 从真实末尾截取。两个会被 `LLMCache` 原地写入的 DSA
store 在调用前克隆暂存；所有输出 shape 校验通过后统一提交，失败不污染原状态。

设计依据包括用户提供的 GPTQModel 文档
`docs/qwen3_8_flash_next_qsa_static_graph_design.md` 及
`docs/patent_assets/glm5_3_flash_dsa_static_transition.gif`。
这些文件位于相邻 GPTQModel 仓库，不作为运行依赖。

## 验证与 SDK 条件

2026-09-07 的完整多模态包入口为
[golden_meta_info.json](../../../work_dirs/glm53_multimodal_w8a16_v1/golden_meta_info.json)。
新增视觉产物为 26 个 W8A16 图，共 585,658,925 字节；全部输入/输出为 FP16，
已检查图 MD5、外部权重范围和文本文件硬链接，见
`work_dirs/glm53_validation/multimodal_export_inspection.json`。

复用的完整 45 层文本主干 W8A16 产物入口为
[golden_meta_info.json](../../../work_dirs/glm53_full_w8a16_streamed_v2/hmquant_xh2_glm5_3_flash_w8a16_256_4k_mpe1024k_20260907/golden_meta_info.json)。
两阶段图、外部权重、embedding 和 tokenizer 合计约 646.76 GB。

| 阶段 | 节点数 | 输入 / 输出 | 外部权重 |
| --- | ---: | ---: | ---: |
| prefill 256 | 21,469 | 123 / 113 | 322.66 GB |
| decode 1 | 17,253 | 123 / 113 | 322.65 GB |

两个图均有 42 个 `MoeBlock`，无剩余占位节点，全部输出为 FP16；已核对全部缓存 shape、
图 MD5 和外部数据范围，见 `work_dirs/glm53_validation/full_export_inspection.json`。
本次在物理 GPU 0、2、3、4 上导出，耗时 5441 秒；进程树采样 PSS 峰值约
185.96 GB，见 `full_export_resources_v2.json`。该记录中的 GPU 数值为整卡观测，
包含同时运行的验证任务，不能作为导出进程独占显存的峰值。

参考版本：本地 Transformers 5.16.1、PyTorch 2.8.0+cu128、xhquant 0.4.0，
环境 `xhquant_55_tf_latest_20260828`。目前的目标证据是 XH2a 图导出和 GPU HMONNX 模拟执行，
没有真机验证，也不能类推为其他 target 支持。

当前 CI 的 `test_glm53_static.py` 覆盖 learned pool 跨块、尾部填充、KDA 连续状态、
DSA 原 token 选择与并列 TopK、mHC、clamped MoE、FP8 scale 读取、失败回滚、
FX 前端，以及两层最小真实混合模型经 W8 导出、内联、重载和重复生成。
导出回归分别覆盖 KDA + DSA/MoE（两个 worker）和仅 KDA + dense MLP（串行），
均经公共 workflow 入口运行。
文本阶段的选中 CI 为 **98 passed**，耗时 173 秒；包括 `L=61, n=3, P=8, C=64`
时填充输入超过缓存末端、有效 token 恰好写满缓存的 HMONNX 执行边界。

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
python examples_merak/llm/glm5_3_flash/verify_attention.py \
  --layer 43 --device cuda:2 --output work_dirs/glm53_validation/layer_43_long.json \
  --lengths 78 1 1 1 256 256 256 256 256 256 256 256 1 55
```

真实权重验证记录保存在 `work_dirs/glm53_validation/`。第 3、43 层 DSA 和第 44 层 KDA
均验证到 2185 token，覆盖超过 2048 token 后的稀疏预算，以及连续 decode 和不完整 pool。
真实前四层的 W8 HMONNX 与 HF 原权重连续推理对照，logits 余弦相似度最低为
0.99850；聊天输入及两步 decode 对照最低为 0.99989，见
`real_4layer_w8_runtime.json` 和 `real_4layer_w8_chat_runtime.json`。
另用真实前四层 W8 HMONNX 运行 2185 个固定随机 token，覆盖超过 2048 token 的
稀疏预算后继续 decode：logits 最低余弦相似度 0.99783，三个 KDA 层的 recurrent
state 最低余弦相似度 0.99925，所有结果有限。见 `real_4layer_w8_long_runtime.json`；
同目录保留了复验脚本 `verify_real_runtime_long.py`。
这些结果验证算子、权重读取和缓存延续，不代表完整模型的生成质量。

真实多模态对照覆盖 448×448 图、多张非方形图、视频和图片/视频混合 prompt。
完整视觉编码器 W8 特征与 HF 原权重的最低余弦相似度为 **0.99631**；
接真实前四层文本 W8 图后，分块 prefill 和两步 decode logits 最低余弦为 **0.99849**，
全部结果有限，公共 `generate()` 也通过。见 `real_multimodal_w8.json`，复验入口：

```bash
python examples_merak/llm/glm5_3_flash/verify_multimodal.py \
  --meta work_dirs/glm53_multimodal_w8a16_v1/golden_meta_info.json \
  --text-meta <真实前四层文本产物>/golden_meta_info.json \
  --output work_dirs/glm53_validation/real_multimodal_w8.json
```

新增 `test_glm53_multimodal.py` 用最小真实视觉和混合文本模型覆盖帧隔离、特征回填、
原生 image/video processor、W8 导出、搬移重载、重复生成和错误请求后的重置。
多模态选中 CI 为 **103 passed**（256.80 秒），随后新增的实际 MP4 采帧/奇数帧补齐
回归与 policy gate 为 **74 passed**（14.86 秒），合计覆盖 104 个不同用例。
真实前四层多模态包的 CLI 也已通过实际 PNG + MP4 混合输入，日志为
`work_dirs/glm53_validation/real_multimodal_cli_pyav.log`。
验证日志和 SDK 补丁也保存在该目录。OCR 完成了 19 个文件中的 14 个；其余 5 个
runtime 文件在重试后仍超时，已另做源码检查和执行回归。具体覆盖和反馈处理见
`review_summary.json`，不能视为 OCR 全量审查通过。
本次多模态增量 OCR 选择 12 个文件，完成配置类和 YAML 两个文件；其余 10 个因超时或
上下文压缩阈值未完成，详见 `multimodal_review.json`。已复核源码并执行上述回归，
不将部分审查的零 finding 记录为完整审查通过。

本地 SDK 还需要三处修复（在 `/data01/home/yujy/work/xhquanttool`）：

1. `PadConcatInputChannelTo64V2` 使用 `nn.functional.gelu/silu`，避免访问不存在的
   `torch.gelu/silu`。
2. `quantization/xh2a/quantizer/_sefp_utils.py` 的 SEFP Triton 地址计算使用 int64。
   单组专家矩阵 `[288,2048,4096]` 有 2,415,919,104 个元素，会溢出 int32。
   已用该实际形状单独量化并比较末尾结果，峰值分配约 7.59 GB。
3. `backend/xh2a/ir/operators/moeblock.py` 对超过 `2**31` 个元素的专家权重使用
   已有逐专家执行路径。当前 grouped GEMM 的专家偏移使用 int32，聊天输入路由到
   第 287 个专家时已复现 CUDA 越界。逐专家切片限制单次地址范围，精度仍为 ALIGNED；
   性能会低于 grouped 路径。待该 kernel 完成 int64 地址回归后可移除此分发限制。

SDK eager MoE fast kernel 以完整 128 通道 tile 读取输入；GLM wrapper 对不满足该宽度的
小配置使用其逐专家实现，导出仍使用同一个 `MoeBlock`。这保留小测试配置的正确性。
