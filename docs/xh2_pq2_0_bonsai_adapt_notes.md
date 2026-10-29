# Bonsai 2bit → XH2 PQ2_0 适配备忘

上游模型：[`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)

本地包：`/data01/datasets/Ternary-Bonsai-2-27B-mlx-2bit`

## 当前隔离后的入口

Bonsai 独立实现在 `xhmodel_merak/xh_llm/models/bonsai/`，继承 Merak Qwen3.5。
转换 CLI 已迁入 `examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py`；
MLX/HF 参数映射也随模型适配迁入 ModelZoo，xhquant 不再包含 Bonsai 专用代码。
原 `qwen3_5`、公共 `BaseModel` 与旧导出入口已恢复，Bonsai 使用自己的 registry、
GDN 数值覆盖和 runtime 子类。联调期间加入的通用 bias 修改也已撤回。
统一接口 ID 为 `bonsai-27b-phase-a`，调用示例见
[当前 README](../examples_merak/llm/bonsai/README.md)。

下文 2026-10-22 的 27B 产物和生成结果是历史记录，使用旧入口获得；
隔离后的新入口以当前 CI 的小模型验证为准，尚未重新执行 27B 整模构建和生成。

## 关键结论

1. Bonsai 做了 **离线 Hadamard fold + 在线激活旋转**，权重为 ternary g128。
2. MLX 容器是 `U32 weight + FP16 scales + FP16 biases(= -scales)`，不是 XH2 DDR 格式。
3. 编译器 `pipelined_te` 已定义 XH2 `PQ2_0` 压缩存储；最终适配仍要转码到该格式。
4. **当前实施顺序：先用现有 SSFP4 兼容，再做真 PQ2 packed。**
5. 完整存储契约见：
   [`../../xhquanttool/docs/xh2_pq2_0_weight_storage_contract.md`](../../xhquanttool/docs/xh2_pq2_0_weight_storage_contract.md)

## 当前方案（先写这个）

```text
Bonsai / PQ2 codes (2bit, K128 scale)
  → 展开为 SSFP4 尾数 {-1,0,+1,+2} + 把每个 K128 scale 复制成两个 K64
  → 复用现有 w4a8h0_ssfp / SSFP4 导出做数值与图联调   [Phase A]
  → 再打包为 XH2 PQ2_0 + S/E，并打 pq2_weight=true      [Phase B]
```

为什么可行：

- PQ2 码值落在 SSFP4 4bit 有符号范围内；
- PQ2 的 K128 可无损对齐到 SSFP4 默认 `nshare=64`；
- SSFP4 展开态接近编译器 unpack 后的 int8 尾数计算形态。

边界：

- Phase A **不是** 编译器最终 DDR 格式；
- 对外不要称 Phase A 为“已支持 PQ2”；
- Hadamard 激活变换由 zoo wrapper 插入普通 MatMul；Embedding 的逆变换在离线转换时物化。

## 编译器文档中的硬性存储点（摘录，Phase B 目标）

- 编码：`00→-1, 01→0, 10→+1, 11→+2`
- group：K128 FP16 scale
- packed shape：`[N/32, K/64, 64, 8]`，`si8` 字节载体，每 `64×32` 块 512 B
- 同字节打包 `k,k+16,k+32,k+48`
- scale 拆成 integer `S/E`；K128 复制到两个 K64
- padding：packed=`0x55`，expanded int8=`0`
- `pq2_weight` 显式启用；首期不含 FWHT / 整模型 Bonsai 导入

## Phase A 落地状态（xhquant + zoo）

已实现（SSFP4 兼容，checkpoint 契约 v2）：

- 转换核心：[`../xhmodel_merak/xh_llm/models/bonsai/bonsai_ssfp4.py`](../xhmodel_merak/xh_llm/models/bonsai/bonsai_ssfp4.py)
  - U32 unpack → mantissas `code-1`
  - K128 scale 复制为两个 K64
  - Linear 输出 HF `*.weight` + `*.quant_weight`
  - Embedding 逆旋转后只输出普通 `*.weight`，不复用旋转域的整数码
  - 完整保留 449 个浮点参数，并转换 RMSNorm/Conv1d 约定
  - metadata：`format=xh2_pq2_via_ssfp4_compat`, `pq2_weight=false`
- Hadamard 在线旋转按用户要求做成 **普通 MatMul 权重**：
  [`../xhmodel_merak/xh_llm/models/bonsai/hadamard_matmul.py`](../xhmodel_merak/xh_llm/models/bonsai/hadamard_matmul.py)
- CLI：[`../examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py`](../examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py)
- zoo 入口：[`../examples_merak/llm/bonsai/`](../examples_merak/llm/bonsai/)
- 单测：[`../benchmark_test/ci_test_transformer5.16.0/test_bonsai_conversion.py`](../benchmark_test/ci_test_transformer5.16.0/test_bonsai_conversion.py)

导出约定：

1. `quant_weight.pt` 只含 HF 可 `load_quant_weight()` 的张量；
2. `phase_a_aux.pt` 含 `w_scale_k64` / `hadamard_signs` / `hadamard_matmul.*`；
3. 复用现有 `w4a8h0_ssfp`；大模型先 `--num_blocks` smoke。
4. 整模导出要求 v2 完整 checkpoint；缺失语言参数或在线旋转 aux 时直接报错。

## 2026-10-22 乱码修复

### 1. 旋转域与残差域不能混用

设标准化 block Hadamard 为 H，固定符号对角阵为 D，R=DH。
Linear 的源包权重是旋转域的 W_rot，运行时应计算：

```text
x_rot = x D H
y = x_rot W_rot^T
```

zoo 用普通 Linear 表达旋转，其权重为 `H D`，因此
`F.linear(x, H D) = x D H`。相同输入宽度共享同一矩阵，当前包的宽度为
5120、6144、17408。只对 manifest 中有 signs 的 Linear 插入；`in_proj_a/b`
保持原来的输入域。拆分 qkv 时，q/k/v 都继承原 qkv 的整数码与同一输入旋转。

Embedding 是反方向：查表解包后必须得到 `E = E_rot H D`。
旧转换器把它与 Linear 一样只解包，没有执行逆变换；导出又直接保存这张表。
三份旧整模产物抽样与正确 Embedding 的余弦只有 0.0030875。
修复在 ModelZoo Bonsai 离线转换时按行分块物化逆变换，保持与源 runtime 相同的
FP16 解包 -> FP32 FWHT -> 乘 signs -> FP16 顺序。此后 Host 查表直接得到残差域输入。
原来的 ternary 整数码不再表示这张逆旋转后的表，因此不能继续附加 Embedding quant_weight。

### 2. 必须导入整套 Bonsai 参数

旧 full_convert 有 804 个 tensor，仅含 402 组 weight/quant_weight，完全遗漏
449 个语言模型浮点参数。strict=False 使加载继续进行，实际运行的是
Bonsai 低位权重与原 HF 浮点参数混合的模型。

v2 转换包含全部浮点参数，而且不能只改名字：

| 参数 | MLX -> HF |
|---|---|
| input/post layernorm、最终 norm、attention q/k norm | 实际乘数 gamma -> HF 参数 gamma-1，避免 wrapper 再加一次 1 |
| GatedDeltaNet gated norm | 直接保留 gamma，不减 1 |
| depthwise Conv1d | `[C,kernel,1]` -> `[C,1,kernel]` |
| in_proj_a/b、A_log、dt_bias | 保留源包数值；源 A_log 已是 log，不再取 log |

修复后的真实产物为 1252 个导出 tensor：402 个 weight、401 个 Linear
quant_weight、449 个浮点 tensor。加载时剩余 333 个 missing key 属于未替换的视觉塔，
语言模型 missing key 必须为零。metadata v2 记录完整性、浮点转换和 Embedding 逆变换状态；
旧版 checkpoint 不再被静默接受为完整模型。

### 3. delta rule 的归一化与累积精度

旧实现的 `F.normalize(x, eps)` 分母是 `max(norm(x), eps)`，HF/FLA 的分母是
`sqrt(sum(x*x)+eps)`。小幅度 Q/K 时这两者差别很大，不能通过调同一个 eps 使其等价。
本次恢复后者，并在 chunk/recurrent 扫描前按 HF 参考实现把 Q/K/V、beta、g 转为 FP32。
内部 recurrent state 保持 FP32，最终 attention output 转回输入 dtype。
这是 PyTorch wrapper 的浮点语义。本次 XH2 SSFP4 导出图的 recurrent cache
输入/输出均为 FLOAT16，量化仿真仍会产生舍入差异；不把 wrapper 对 HF 的数值回归
解释为最终导出图或芯片中的所有累加器都保留 FP32。

小幅度合成用例的输出 relative-L2 从约 6.25 降到 FP32 舍入范围；普通和小幅度输入、
FP16/FP32、跨 chunk 边界都有回归。该差异是独立数值问题，旧 Embedding 和浮点参数遗漏
则已经在真实旧产物中得到直接验证。

### 4. 导出依赖与兼容性

2026-10-22 验证时，SDK 的 fake_tensor_prop 已增加 MetaInfoPro。旧导出入口替换整个模块，
会把该接口移除，当时改为只应用共享模块参数恢复修复。
当前工作区已在 xhquant 内按模块属性去重替换参数，同时修复 SSFP 位宽推断；
zoo 直接使用配套 xhquant，不再动态替换 SDK 函数或安装全局位宽补丁。
HF decoder 将 layer_type 政名为 block_type 时，在 wrapper setup 中统一到本地 layer_type。
默认 Python 3.10 环境另有 compressed_tensors/Torch Inductor 导入问题，以及 SDK ONNX
注册表与依赖不匹配的问题。单独设置 TORCHDYNAMO_DISABLE 只解决了前者，不能作为完整导出验证。
本次实际静态导出/仿真使用本机配套的
`/data01/home/she.gao/miniconda3/envs/xhquant/bin/python` 与
`/data01/home/she.gao/xhquanttool`：Python 3.12、Torch 2.8.0、Transformers 5.5.4、
ONNXScript 0.7.0、SDK 0.9.2。两类注意力的小模型先完成真实 HMONNX 导出。
原生生成参考在默认转换环境执行（Torch 2.8.0、Transformers 5.15.0），数值公式与保存的
Bonsai 参数契约相同。未修改任何第三方安装。

量化仿真把 prefill/decode 分别固定到两张 80 GiB GPU。SDK 0.9.2 的
AutoOffload 首卡先保留一半预算，且不支持 CPU 作为计算回退设备；
`{gpu: 70GiB, cpu: 200GiB}` 会在约 35 GiB 后错误转向 CPU 并终止。
本次验证器为每张图只提供对应 GPU、预算 78 GiB，避免该 CPU 分支。
SDK 仍可能打印超过“半预算”保留量的 warning；实际每张卡约 60 GiB，
需以运行中的实际显存和执行结果核对，不能单凭 warning 判断 OOM。

### 5. 验证与产物

本次独立产物目录：`/data01/home/xuchen/xh2/work_dirs_bonsai_fixed_20261022/`。
旧产物保留。prefill 档位为 64 token，decode 为 1 token，上下文缓存 2048 token；
较长 prompt 由 runtime 分块 prefill。

权重检查对 402 个模块抽取首尾行，与源 MLX affine 解包/Embedding 逆变换对照；
449 个浮点参数逐张量检查，最大误差均为 0。原生参考模型使用 HF attention/cache，
通过独立 FP32 butterfly 做在线旋转，不使用导出 wrapper 的 dense MatMul。
这与未量化 HF 基座有明确区别，也不是在 Linux 上执行 Apple MLX runtime。

当时的 `validate_generation.py` 保存了原生生成、相同 token 历史的 HMONNX logits 对照和自由生成答案
（该历史脚本已随旧 Qwen 入口撤回，下面记录不代表新的独立入口已完成整模验证）。
测试包含算术、中文事实、英文翻译、超过一个 prefill chunk 的输入及完整的多步解释。
数值与生成结论以 `validation/` 下的 JSON 为准，导出成功本身不代表输出正确。

导出图还独立核对了每张图的 497 个 ternary Linear（包含 QKV 拆分）和 3 个共享
Hadamard 矩阵的抽样输出行，最大权重误差均为 0；整张导出 Embedding 与修复 checkpoint
逐行对照，最大误差为 0。96 个 `in_proj_a/b` 属于源包浮点 Linear，导出仍会进行常规
量化，抽样最大误差为 0.00390625；它们与 ternary payload 的原码保留分开统计。
161 个普通 Norm 的 HF offset 参数在 FP16 加载后也能精确还原源 gamma。

2026-10-22 完整 prefill/decode 导出成功，GPU ALIGNED 端到端实测：

| 输入 | HMONNX 自由生成 | 与原生 Bonsai 参考 |
|---|---|---|
| 17 + 25 | `42` | 文本完全一致 |
| 中国的首都 | `中国的首都是北京。` | 文本完全一致 |
| 你好译成英文 | `Hello` | 文本完全一致 |
| 100 token 阅读材料后求 17+25 | `42`，跨两次 prefill | 文本完全一致 |
| 三句话解释二进制并举例 | 114 token 完整解释，含 `5 = 101₂`，正常 EOS | 文本完全一致 |

固定 token 历史共比较 30 个 logits 位置，top-1 为 30/30，全部有限值；
最低 cosine 为 0.99918067，平均 KL 为 0.00071959，最大 KL 为 0.00443206。
最大 logits 绝对差为 0.611328125，因此生成一致不表示浮点 logits bit-exact。
短样本比较到 EOS，长回答比较前 16 个固定历史位置，并另做完整自由生成。
原生参考在两套 HF 环境中的 5 个样本生成 token 也完全一致。

结果文件：`validation/acceptance.json`、`hmonnx_generation.json`、
`checkpoint_check.json`、`exported_weight_check.json`、`delta_rule_check.json`。
加载入口为 `hmonnx/meta.json`；两张图的 external_data 各约 33.53 GiB，
普通 FP16 Embedding 约 2.37 GiB，属于 Phase A 的展开体积。

新增回归在当前 CI 目录的 `test_bonsai_phase_a_contract.py`，11 项通过；
xhquant 生产端 7 项通过，合计 18 项。当前仅执行定向回归；
仓库 selector 对用户指定的历史 `xh_model_zoo/`、`examples/` 路径设置了冻结门禁，
本次未修改全局 policy 或新增豁免。

验证范围是完整 64 层语言模型、2048 context 档位和上述定向样本。
未执行 Apple MLX 原生运行、XH2 实际芯片、视觉塔、262K 长上下文或完整能力 benchmark。

### 6. 各阶段的交接

```text
Bonsai MLX safetensors + hadamard.json
  │ ModelZoo Bonsai 适配：解包、K128→K64、Embedding 逆变换、浮点参数转换
  ├─ quant_weight.pt        HF 权重 + Linear 整数码 + 449 个浮点参数
  ├─ phase_a_aux.pt         scales、signs、共享 Hadamard MatMul
  └─ phase_a_metadata.json  v2 完整性及旋转契约
          │ zoo 前端：完整性检查、HF 命名、QKV 拆分保留整数码、插入在线旋转
          │ xhquant 量化图：SSFP4 权重与激活量化，保留原 ternary 数值
          │ xhquant 导出图：HMONNX 普通 Linear/MatMul、缓存输入输出
          ├─ prefill.onnx + external_data  64-token 档位
          ├─ decode.onnx + external_data   1-token 档位
          ├─ quant_embedding.pt            逆旋转后的普通 FP16 表
          └─ meta.json                     图、Embedding、cache 和 tokenizer 路径
                  │ zoo runtime：分块 prefill → 传递 KV/conv/recurrent cache → decode
                  └─ 对照独立 HF + butterfly 参考，检查 logits 与自由生成
```

## 后续 PQ2 存储工作

- Phase B：加载 `xh2_pq2_0` 产物并在导出图上标记 `pq2_weight`
- 与编译器转换脚本核对 tile 内字节地址和 S/E 的 dtype、排列、padding，再完成真正的 DDR 2bit 打包
- 当前 SSFP4 HMONNX 仍为兼容展开态，不把整模输出恢复解释为已完成 Phase B
