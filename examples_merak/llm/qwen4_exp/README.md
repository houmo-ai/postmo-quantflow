# Qwen3.8 Flash Next / Qwen4-Exp

统一模型 ID 为 `qwen3.8-flash-next`，原生架构为 `qwen4_exp`。
支持文本、图片、视频、`full` 和 `mtp` 变体。MTP 使用独立的 HC/QSA 草稿模型，
默认每轮最多提出四个候选，目标模型验证后提交；拒绝时恢复目标与草稿的
K/V、DeltaNet、PLE、QSA 和位置状态，再重放已接受前缀。

## 查询与构建

在仓库根目录运行：

```bash
python -m examples_merak.unified_api.catalog qwen3.8-flash-next --full
python -m examples_merak.llm.qwen4_exp.build \
  --speculative '{"mode":"mtp","num_draft_tokens":4}' \
  --model-dir /path/to/Qwen3.8-Flash-Next \
  --quant-checkpoint /path/to/AutoRound-GPTQModel-HF \
  --output-dir ./work_dirs/qwen38_mtp --device cuda:0
```

省略 `--quant-checkpoint` 会先量化；不传 `--speculative` 时只构建视觉和目标语言模型。
默认采用直接 RTN：`quant.method: rtn`、`iters: 0`、`disable_opt_rtn: true`，
语言和视觉旋转均关闭，不执行 AutoRound 迭代或优化 RTN 的 scale 搜索。
attention/GDN/shared expert W8、routed expert W4、group size 64 的配置保持不变；
PLE 由独立的 `export.model.ple_quant_type: sefp4` 控制。主图和视觉图的基础精度为
`w8a8h1_sefp`；MTP 沿用 Qwen3.5 的 `mtp_config.quant_scheme`，默认 W8A8，输出头位宽继承目标模型。
需要 AutoRound 时显式传入 `--quant-method autoround`，使用 200 次迭代；
独立的 `qwen3_8_flash_next_hf_autoround_xh2a_256k.yaml` 继续保留为 AutoRound 配置。
原始权重也可直接由 SDK 按上述导出配置做 SEFP RTN：构建时传入 `--quant-method none`，
此时源权重精度标记为 W16，实际图精度仍由 `export.model.quant_scheme` 决定。
现有 RTN 路径仍使用 GPTQModel 的模型加载、输入采集和打包流程，需要读取配置的样本数据。
GPTQModel 必须提供 `quantize_qwen4_exp` API，并支持关闭 scale 搜索和整数对称零点；
这两个数值契约由生产端 CPU 回归验证。
切换默认配置不会改写已有的 AutoRound checkpoint、HMONNX 或 Golden。
MTP 需要原始 `--model-dir` 保留 `mtp.*` 参数；已有量化目录可以只含目标模型。
默认 context=262144（256K）、prefill chunk=256，可用 `--context-length`、
`--prefill-chunk-length` 调整。目标与 MTP 的上下文容量相同；输出目录必须尚不存在。

### 按 block 低内存导出

统一构建的 full 和 MTP 默认按 block 低内存导出，可用 `--no-low-memory` 显式切回普通导出。
直接调用 serving 导出函数时也默认低内存；显式 `low_memory=False` 或环境变量
`HUGE_MODEL_EXPORT_ENABLED=0` 可选择普通路径。该选项控制导出阶段，
与默认 RTN 权重量化、推理/Golden 的 auto-offload 分别配置。

```bash
python -m examples_merak.llm.qwen4_exp.build \
  --model-dir /path/to/Qwen3.8-Flash-Next \
  --quant-checkpoint /path/to/RTN-or-AutoRound-GPTQModel-HF \
  --output-dir ./work_dirs/qwen38_low_memory \
  --low-memory --device cuda:0
```

需要 MTP 时增加 `--speculative '{"mode":"mtp"}'`，默认候选数为 4；
`num_draft_tokens` 可显式指定正整数，构建时会校验候选数与验证长度、上下文容量的关系。
Python 入口 `build_qwen4_exp_model(...)` 默认使用相同低内存路径，
显式 `low_memory=False` 切回普通导出。统一构建按 recipe 选择模式并在执行期间设置环境，
不依赖调用进程继承的环境变量。

低内存路径先构造 meta 骨架，每次只物化一个完整 decoder block，导出子图后释放该层权重；
每个 block 在 `--device` 指定的设备上执行 PTQ，HF 的 `device_map` 不再用于整网分配。
最后导出保留 HC、position、mask 和 head 的小骨架，并用既有拼图工具回填为完整 HMONNX。
最终图不包含 PlaceHolder。单个 block 包含该层 MoE，因此峰值仍受最大 block 大小影响。
PLE 原表通过 safetensors 按行读取并压缩到 SEFP4 资源，不随 block 全表物化；
图继续接收 Host 查表反量化后的 `ple_embeddings_N`，保持原有 Golden 和推理接口。
此模式要求 `ple_quant_type: sefp4`，FP16 PLE 可沿用普通导出。

prefill、decode 和 MTP 的 verify 分别导出，保留各自的 recurrent/cache 契约；
默认四候选对应五 token verify。
视觉在每个档位中按 stem、单个 block、merger 顺序加载、导出并释放权重，
使用 GLM5.3 同样的 QuantGraph 直接 HMONNX 接口，PTQ 明确使用请求的 CPU/CUDA 设备。
视觉无需物化整个视觉塔。MTP 草稿仅加载自身权重；目标 head 量化一次，
prefill、decode、verify 和两张草稿图复用同一固定量化 head，保持权重与 scale 一致。
分块权重读取降低权重常驻内存，但上下文缓存、单模块 PTQ 和导出器本身仍占用内存；
256K cache 容量保持原值。该实现没有启用 GPTQModel 量化阶段的 `offload_to_disk`。

`export_manifest.json` 记录 `export_mode`，resume 必须保持原来的导出模式。
文本模型与 GLM5.3 使用相同的静态 wrap → TorchFX frontend → quant → HMONNX 生命周期。
普通、分块以及 MTP 导出都捕获模型 wrapper；RMSNorm、RoPE 和缓存更新由 wrapper
直接表达，不再保留独立 Dynamo 导出器、内联恢复和全局捕获补丁。
发布时按当前 QSA 的静态几何保留原图的布局常量及退化 view，保持旧产物的完整计算图契约；
此步骤不读取旧产物、不改变活动计算连接，也不改写 SDK 调试元数据。
结构回归逐项检查算子域、全部属性、端口连接与共享关系、常量/权重内容和公开输入输出，
包括未使用的布局资源；内部名称和调试信息的差异单独报告。
新视觉契约为 `qwen4_exp_vision` ABI 2：每个 `visual_m*` 目录包含
`stem.onnx`、`block_XX.onnx` 和 `merger.onnx`，根目录 `visual_gears.json` 登记档位及组件。
发布与加载同时校验组件顺序、摘要、张量 ABI 和原生视觉配置。
旧单图视觉产物仍可加载，新导出统一使用组件契约。

新导出默认使用 `llm_cache_inplace_v1`：图显式接收 `past_key_cache_N`、
`past_value_cache_N`，形状为 `[1, KV heads, context, head_dim]`。
图上的缓存算子统一为 `KVcache`（与 DeepSeek 的 XH2 导出一致），沿 axis=2 更新缓存；
Python 模块仍为 SDK `LLMCache`，runtime 通过共享 `CacheTensor` 启用原地更新。
`llm_cache_inplace_v1` 接口标识保持兼容，已有 `LLMCache` 节点的包仍可加载；
prefill、decode 及可选 verify 共用主模型缓存，MTP 草稿另有独立缓存。
正式发布的图保留 K/V、indexer cache 输入和图内 `KVcache` 输出，但不再将
`present_past_key_cache_*`、`present_past_value_cache_*`、
`present_qsa_indexer_key_cache_*`、`present_qsa_indexer_pool_cache_*` 暴露为整图输出。
运行时直接复用原地更新的缓存；Golden 在 `runtime_before`/`runtime_after`
中保存这些缓存。该约定同时适用于目标模型和 MTP 两张草稿图。
新图不含 `PageKVCache`，旧 PageKV 包仍可加载。
SelectedKV 直接接收独立的 K/V cache，分别输出选中的 K/V；不再打包整份 cache。
K/V 各用一次 int32 Gather 取回尾部，再分别与选中数据 Concat，不再压紧搬运。
attention 使用 grouped MatMul、MaskedAdd、Softmax、MatMul，计算保留16位精度；
分段位置 mask 分别屏蔽选择区和尾部区的无效槽位，允许两区之间保留 padding。
GDR 和 PLE 的一维卷积通过 SDK 转为等价的 Conv2d。

SelectedKV 新导出增加 `qsa_indexer_pool_cache_N`，形状为
`[1, 1, floor(context/4), indexer_head_dim]`，缓存已完成块的均值、归一化和
partial RoPE 结果。每次仅从原始 indexer cache 读取本轮可能新完成的块，
按有效长度提交；单 token decode 的池化计算宽度为1块，历史压缩块直接复用。
prefill、decode、verify 和各自的 MTP 阶段共享所属模型的压缩缓存；回滚按本轮
可能覆盖的块区间备份和恢复。旧 metadata 不补新状态，非 SelectedKV 导出保留原契约。
MRoPE 三轴频率通过静态 INT32 Gather 组合，替代逐频率 Slice 后 Concat；
partial RoPE 的旋转前缀与未旋转尾部语义保持一致。新缓存与图结构需重新导出才生效。
`SparseAttentionSelectKV` 保持 pool_size=4、compression_ratio=4，这是逻辑分组规则；
indexer 按四个 token 增量池化，池化 cache 的容量为 context/4，并保留四个原始 key
作为跨 prefill/decode 的尾部状态；K/V cache 仍按完整 context 容量分配。
GQA 节点为 7 输入／2 输出，第 7 输入为 V cache；
K/V 保持 `[B,Hkv,L,D]`，选中结果分别为 `[B,Q,Hkv,K*4,D]`（K 为 candidate_topk）。
编译器需同步支持这一双 cache 契约；SDK 的既有 MLA 单 payload 接口保持兼容。
SDK 需要包含 `28d9fdb2` 的 GDRChunkScan 默认属性修复，以省略默认值8的 `matmul_input_bit`。
池化写入位置、尾部索引及 attention mask 由运行时根据请求长度在图外准备，
使用与 GLM5.3 相同的控制规则；主模型与 MTP 草稿图使用同一套规则。
导出通过 `to_export_hmonnx_v2` 的默认优化流程并设置 `save_as_external_data=True`。
为兼容当前 SDK 的 Identity 清理实现，模型导出前移除只产生 HMIdentity 的
`QIdentity` 节点，保留实际类型转换和输入输出顺序。
模型只保留专属图和状态适配，不要求通用整数 Cast／Clip parser 的新增支持。
联动变更：[SDK 47747](http://gerrit.houmo.ai/c/houmoquantization/xhquanttool/+/47747)。
当前 ModelZoo 与 SDK review 保持无 topic；验证时记录各仓库的实际 revision。
提交信息中的 `Depends-On` 本身不会触发依赖下载；本仓库 `run.sh` 使用环境中已有的 SDK。
本地使用新 SDK 通过，不代表未联动该 SDK 的远程流水已经通过。
仅更新 ModelZoo 而仍使用旧 SDK，不能满足 GQA 双 cache 的 SelectKV 契约。

普通导出由 `XHQwen4ExpModel` 管理静态 prefill/decode wrapper、缓存、frontend、量化、
HMONNX 与 metadata；workflow 负责配置和量化权重准备。默认 context 为 262144，
prefill chunk 为 256，仅生成 `prefill/` 和 `decode/`。低内存模式复用逐层导出和拼装实现；
仅供捕获使用的输入返回槽在 ONNX 序列化前移除。旧 `decode_validation` workflow 已移除。
图和 `golden_meta_info.json` 直接写入输出根目录；Golden 通过 `dump_golden` 单独执行。

新构建的 MTP 包使用运行契约 v4：`mtp_prefill/` 是与目标 prefill chunk 等长的独立草稿图，
`mtp_decode/` 是单 token 草稿图，`decode_m5/` 是默认五 token 的目标 verify 图。
加上目标 prefill/decode，共五张语言图。保留独立 decode1，以便拒绝后按原生 recurrent
语义重放；Qwen4 的 HC、QSA、PLE、MRoPE 和草稿输入结构保持原生语义。

草稿 prefill 按提示词的下一 token embedding 与前一位置的目标 HC 对齐，保留全部
hyper-connection 流。目标每完成一个提示词块即预热对应草稿块，只保留跨块所需的最后一个
HC，不积累整段提示词 HC。prefill 和 decode 共用草稿自己的 K/V、QSA 和位置状态。
短块按有效长度执行；上下文尾部不足一个静态 prefill 块时改用草稿 decode。

每轮递归生成最多 `num_draft_tokens=N` 个候选；验证图默认宽度为 `N+1`，一次处理
待提交 token 和所有候选。`export.model.verify_length` 可显式设为大于等于 `N+1` 的值，
未使用槽由有效长度屏蔽。候选数受剩余输出长度和 EOS 限制；上下文尾部放不下完整
验证图时切回单 token decode。需要新导出 m2 时同时设置 `num_draft_tokens: 1`；
运行时继续加载旧 v2 单步草稿包和 v3 单候选 m2/m4 包。
verify 的 GDR 按 token 串联 recurrent 更新并显式返回状态，使缓存提交遵循单 token decode 的计算路径；
主 prefill 继续使用 chunk 计算。导出、resume 和产物加载都会校验 verify 的这一约束。
验证事务只备份本轮静态图宽度可能覆盖的 K/V 和压缩 QSA indexer 缓存区间；
T/H/W 仅保存未完成 C4 组的组首坐标，并在回退时恢复；
GDR 循环状态及 PLE 历史完整备份。当前预分配 PageKV 使用固定 identity 页映射，
按对应物理页槽备份，跨页区间分别保存。拒绝或异常时原地恢复，再以单 token
decode 重放已提交前缀，并用真实目标 HC 恢复草稿的已提交前缀。全接受后的最后一个
移位输入也纳入草稿事务范围；备份容量不随历史上下文长度增长。草稿 runtime 的
`snapshot()` 保留完整快照语义，显式传入静态写入宽度 `write_length` 才使用局部事务。
运行 metadata 在 `spec_decode.draft` 登记两张草稿图和档位，在 `spec_decode.target`
登记 verify 图和长度。v4 另登记候选数和共享 head 的目标所有权、量化位宽；
导出、续导和加载均校验这些契约。续导复用已验证图和固定 head，不重复量化；
同时核对源 head、草稿权重内容及已完成草稿图和外部权重的摘要，不复用未完成的草稿图。
固定 head 的私有 checkpoint 只属于导出工作区，不随发布包交付。
旧导出工作区缺少这些续导记录时需要重新导出，已有 v2/v3 包仍可用于推理。
MTP 两张草稿图与主模型统一使用 SelectedKV，先选择 K/V 再执行 attention。

目标模型采用 attention/shared expert W8、routed expert W4 的混合权重；
主图、视觉图和 MTP 的基础精度均为 W8A8 SEFP。`mtp_config.quant_scheme` 可独立设置，
共享输出头的位宽必须与目标模型一致。PLE 表默认 SEFP4，可在标准导出中显式选择 `fp16`。
视觉静态 token 档位为 96/196/384/704/1536，按图片真实 patch 数选择最小可容纳档位。
图中保留头维度 72 到 128 的 padding，运行时沿原生三轴 mRoPE 位置编码注入视觉特征。

## 推理与 Golden

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_COMPILE_DISABLE=1 \
python -m examples_merak.unified_api.generate \
  --artifact-dir ./work_dirs/qwen38_mtp \
  --prompt '描述这张图片。' --image /path/to/image.jpg \
  --max-new-tokens 4 --device cuda:0 \
  --golden visual prefill decode mtp
```

也可以直接调用模型的 `generate`；此入口复用模型自己的缓存调度和 MTP 运行时：

```python
import os
os.environ["ENABLE_HMINFERENCE_V2"] = "1"
os.environ["XHQUANT_PACKED_W4"] = "1"

from xhmodel_merak.xh_llm.hmonnx import AutoLLMHONNXModel

model = AutoLLMHONNXModel.from_pretrained(
    "./work_dirs/qwen38_mtp/golden_meta_info.json",
    enable_page_attention=False, enable_cuda_graph=False,
    enable_auto_offload=True, device_map=["cuda:0", "cuda:1"],
)
model.to(device="cuda:0")
tokenizer = model.get_tokenizer()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "你好"}], tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
output_ids = model.generate(
    **inputs, max_new_tokens=8, do_sample=False,
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print(model.last_mtp_stats)
model.close()
```

模型持有视觉 encoder 和原生 processor，图片无需在调用者中手工融合 embedding：

```python
processor = model.get_processor()
inputs = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "描述图片。"},
    ]}], tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
```

同一路径接受原生 processor 生成的多图、视频及混合媒体 tensors；
视频时间戳和三轴 M-RoPE 使用原生 Qwen 语义。统一 CLI 每次接受一个媒体文件，
视频使用 `--video /path/to/video.mp4`，与 `--image` 互斥。

直接入口返回提示词与生成 token 的拼接，保持 `input_ids` 的设备和整数类型。
模型持久复用两张草稿图的 SDK 会话，每次请求前后重置草稿状态；`model.close()` 释放草稿会话。
`model.last_mtp_stats` 提供轮数、候选总数、接受/拒绝数、接受率和每轮接受数；
统一推理报告在 `stats.mtp` 返回同一统计。统计描述本次生成，不代表整网质量或加速比。
只接受无 padding 的单条请求；`attention_mask` 如提供须全为 1。
可传预融合 `inputs_embeds` 和 `[3, 1, prompt_length]` 的 `position_ids`。
`streamer` 收到提示词和完成后的生成块，结束或异常时调用 `end()`；目前不逐 token 推送。

`full` 变体选择 `visual prefill decode`；普通推理省略 `--golden`。
Golden 采集实际 SDK 执行的输入、输出、中间张量，以及 K/V、recurrent、PLE 和 QSA 缓冲的执行前后状态。
v3/v4 的 `--golden mtp` 同时要求实际执行三图，默认分别保存到 Golden 根目录的
`mtp_prefill/step_N/`、`mtp_decode/step_N/`、`decode_m5/step_N/`；旧包沿用自身 verify 档位。
三图的输入、输出及执行前后缓存各自记录，不能用单步草稿结果替代 prefill 或 verify。
视觉采集本次媒体实际执行的 stem、block、merger 会话，并保留全部档位的完整容量基线。
采集目录及完成状态记录在 `qwen4_golden_capture.json`；选定组件未执行时明确失败，
例如提示词没有可对齐的 token 对，或生成立即遇到 EOS，无法采到所选图时应调整输入后重试。
Golden 为运行输出，不作为源码提交或自动混入构建产物 Manifest。

产物包含 prefill/decode、visual_m*、可选 mtp_prefill/mtp_decode/decode_m5 图、token embedding、tokenizer/processor、
运行 metadata 和 `export_meta_info.json`；整体搬移后仍可加载。
推理当前为 batch=1、greedy，使用 SDK V2；大型目标图启用自动 offload。
NGram PLE 查表权重在 HMONNX 导出时默认使用 SEFP4：每 64 个元素共享指数，
尾数按两个 4-bit 数打包。原始 checkpoint 保持高精度，不交给 AutoRound 量化此表。
默认 SEFP4 导出将压缩表和指数保存为独立资源，主图通过四维
`ple_embeddings_<层号>` 输入接收 `[batch, sequence, ngram_heads, embedding_dim]`。
NGram 哈希、Gather 和 SEFP4 解包反量化在 host 预处理执行；新图中没有
`Qwen4PLESEFP4Gather`，也没有 PLE 查表或反量化节点。图内保留 PLE 后续计算和历史状态更新。
v3/v4 的 verify 图沿用相同 Host PLE 输入契约，根 metadata 的 `qwen4_exp_ple`
分别登记 `prefill`、`decode` 和实际 verify 档位（默认 `decode_m5`）的压缩资源。
运行时以 mmap 打开压缩表，先 Gather 压缩行，再仅反量化命中的行，结果传给图。
Flash Next 的 `[320001536, 160]` 表按 192 列填充后约 30.4 GiB，FP16 原表约 95.37 GiB。
`export.model.ple_quant_type: fp16` 可用于高精度对比；默认值为 `sefp4`。
Golden 目录附带原始 SEFP4 packed/指数二进制、资源与样本清单，以及独立的
`ple_demo.py`。该脚本只需要 NumPy，明确分为 Gather 和解包反量化两步，
默认读取保存的 NGram 索引，并逐值核对真实 SDK Golden 中的图输入：

```bash
python /path/to/golden/ple_demo.py
python /path/to/golden/ple_demo.py --manifest /path/to/golden/ple/decode_manifest.json --sample 0
```

可传 `--indices ids.npy --layer 1 --output embeddings.npy` 展示其他索引的查表结果。
只导出主模型和视觉时省略构建命令的 `--speculative`，Golden 选择 `visual prefill decode`。
公开接口支持单张图片或单个视频；多图和混合媒体通过模型的原生 processor/generate 接口使用。
LoRA 未登记为本模型能力。文本导出支持 `low_memory` 与 `standard`，视觉始终分组件导出。

Python 构建入口是 `xhmodel_merak.xh_llm.models.qwen4_exp.build_entry.build_qwen4_exp_model`，
推理仍使用 `xhmodel_merak.model_inference.inference_model`。模型实现位于
`xhmodel_merak/xh_llm/models/qwen4_exp/`，接口适配位于
`xhmodel_merak/model_integrations/qwen4_exp/`。

独立 workflow 入口仍是 `qwen4_exp_workflow.py`，默认使用 RTN unified YAML，
普通配置使用模型静态导出入口；MTP 配置显式选择 `export.model.export_mode: serving`。
历史 AutoRound YAML 也通过静态模型入口导出 prefill/decode。
`debug_scripts/export_qwen4_exp_serving_hmonnx.py` 是复用同一 serving 导出的诊断入口。

构建与推理使用共同的 PyTorch 2.8 依赖；运行时需安装与 Python/PyTorch ABI 匹配的
hsum 扩展。视频文件依赖现有 PyAV 包。旧视觉 Dynamo 路径的 PyTorch 2.11 限制
不适用于当前分组件 TorchFX/QuantGraph 导出。干净安装环境仍需独立验证。
上面的推理命令禁用 torch.compile，适用于本机缺少 C++ 编译器的情况；
构建/导出命令不能继承 `TORCH_COMPILE_DISABLE=1`。
完整模型接近 80GB 显存上限；上述推理命令启用可扩展分配器段，避免大张量分配时的显存碎片。

## 专用构建入口与 checkpoint 兼容作用域

Qwen4 的已有量化 checkpoint 构建使用以下专用脚本；每次命令启动独立进程，
在调用公共解析和执行接口之前安装 Qwen4 checkpoint 兼容补丁，成功或异常后都恢复。
补丁处理 RTN/AutoRound producer 识别和真实 HC 终端权重校验，不添加假 `norm.weight`，
不跳过公共的权重容器、层数、形状或模型身份检查。导入模型包不会安装补丁。

仅构建目标模型与视觉：

```bash
python -m examples_merak.llm.qwen4_exp.build \
  --model-dir /path/to/Qwen3.8-Flash-Next \
  --quant-checkpoint /path/to/AutoRound-GPTQModel-HF \
  --output-dir ./work_dirs/qwen38_full --device cuda:0
```

构建目标模型、视觉和 MTP：

```bash
python -m examples_merak.llm.qwen4_exp.build \
  --model-dir /path/to/Qwen3.8-Flash-Next \
  --quant-checkpoint /path/to/AutoRound-GPTQModel-HF \
  --speculative '{"mode":"mtp","num_draft_tokens":1}' \
  --output-dir ./work_dirs/qwen38_mtp --device cuda:0
```

省略 `--quant-checkpoint` 仍使用 recipe 中的量化流程。上下文、prefill、组件图精度
沿用公共构建参数，例如 `--context-length 262144 --prefill-chunk-length 256`。
Python 调用使用同一薄封装，不复制构建器：

```python
from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model

result = build_qwen4_exp_model(
    model_dir="/path/to/Qwen3.8-Flash-Next",
    quant_checkpoint="/path/to/AutoRound-GPTQModel-HF",
    output_dir="./work_dirs/qwen38_full",
    device="cuda:0",
)
```

公共 `xhmodel_merak.model_build.build_model()` 和通用 `examples_merak.unified_api.build`
入口本身不安装该补丁；需要 Qwen4 checkpoint 兼容时使用上面的专用入口。
专用入口在进程内安装临时函数替换，应在独立构建进程使用；不要与其他模型的构建
混在同一进程并发执行，也不要在补丁作用域中 fork 构建子进程。
推理仍使用公共 `inference_model()`，已有 HMONNX／Golden 的运行方式不变。
