# Wall-OSS-0.5 构建、动作推理与开环验证

Wall-OSS-0.5 已接入 `build_model` / `inference_model`，模型 ID 为 `wall-oss-0.5`。
完整部署包包含 `visual`、`prefill`、`denoise` 三张 HMONNX 图，两路相机逐张复用视觉图。
当前导出接口为 batch 1、H32、26 维状态/动作、512 档状态字符串和默认 10 步 Flow 推理。

- [模型结构、图和缓存 ABI](../../../xhmodel_merak/xh_other_model/models/wall_oss/EXPLANATION.md)
- [适配范围与数值证据](../../../docs/modeling/structures/wall-oss.md)
- [官方 BF16 对 W8A8：1,920 组开环测试报告](reports/libero_open_loop_20261026/README.md)

本目录提供模型导出、推理和开环数值验证。报告比较同一输入下两种实现的预测差异，
不将动作误差换算成机器人任务成功率，也不提供官方量化“合格线”。

## 工作环境与精度

本次独立仓库为 `/data01/home/jing.huang/xh2modelzoo-wall-oss-05`，原始基线为 develop
`5d9c5d5e`。部署环境使用 CUDA Torch 2.8.0、Transformers 5.16.1、qwen-vl-utils 0.0.14。
以下命令从仓库根运行，假设 ModelZoo、SDK 和 CUDA Torch 已准备好：

```bash
source work_dirs/wall_env/bin/activate
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
python -m pip install -r examples_merak/vla/wall_oss/requirements-open-loop.txt
```

官方参考执行使用独立环境，实际验证过 Transformers 5.2.0、tokenizers 0.22.2、
kernels 0.11.0、peft 0.18.1、qwen-vl-utils 0.0.11 和 torchdiffeq 0.2.5。
将 [requirements-reference.txt](requirements-reference.txt) 安装到单独的参考 overlay，
不覆盖部署环境的 Transformers。本文以 `work_dirs/reference_env/bin/python` 指代该环境。
环境路径是本机示例，可替换为自己的环境；模型完整依赖以 recipe 导出清单为准。

下载用于数值参考的 Wall-X 源码，固定到已验证 revision；它不是部署推理依赖：

```bash
git clone https://github.com/X-Square-Robot/wall-x.git examples_merak/vla/wall_oss/.upstream
git -C examples_merak/vla/wall_oss/.upstream checkout --detach ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f
```

源 checkpoint 为 `x-square-robot/wall-oss-0.5`，本机目录 `/data01/datasets/wall-oss-0.5`。
官方 BF16 基线保留主干/视觉 BF16 及官方指定的 FP32 模块；旧 FP32 参考会显式将
浮点权重转换到 FP32。两种基线分开报告，不混合统计。

## 量化导出 YAML 与构建

| 精度 | YAML | export.model.quant_scheme.quant_type |
| --- | --- | --- |
| W16A16（默认） | [wall_oss_05.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05.yaml) | `w16a16_sefp` |
| W8A16 | [wall_oss_05_w8a16.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05_w8a16.yaml) | `w8a16_sefp` |
| W8A8 | [wall_oss_05_w8a8.yaml](../../../configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05_w8a8.yaml) | `w8a8_sefp` |

三份 YAML 仅部署精度不同：prefill 分块 256、KV 总容量 2048、单图 patch 容量 1024，
动作 horizon 固定 32，group size 均为 64。`quant: null` 表示跳过独立权重量化阶段；
SDK PTQ 在 `export()` 内执行，`dump_golden()` 独立执行。

统一入口通过 `--precision` 设置同一个 `export.model.quant_scheme` 字段：

```bash
CUDA_VISIBLE_DEVICES=6 python -m examples_merak.vla.wall_oss.unified build \
  --model-dir /data01/datasets/wall-oss-0.5 \
  --output-dir work_dirs/my_wall_oss_w8a8 \
  --precision w8a8_sefp --device cuda:0
```

```python
from xhmodel_merak.model_build import build_model

result = build_model(
    model_id="wall-oss-0.5",
    model_dir="/data01/datasets/wall-oss-0.5",
    output_dir="work_dirs/my_wall_package",
    deployment_precision={"policy": "w8a8_sefp"},
    device="cuda:0",
)
```

输出目录必须不存在，各精度独立导出。不能通过改已有包的 metadata 切换精度或扩大容量。
源权重、校准和 trace 保持 CPU，SDK 权重 PTQ 使用所选 GPU。

## 图像、状态与动作

相机顺序固定为 face/front、right wrist。每张图保持宽高比将长边缩放至 448，再按
官方 `smart_resize` 对齐到 28 像素网格，例如 640×480→448×336、1920×1080→448×252。
实际 grid 控制视觉位置编码、窗口范围和 token 数，物理容量不会改变图片几何。
超出 patch 或 KV 容量会报错，不静默截断指令。

| 索引 | 状态 | 动作 |
| --- | --- | --- |
| 0–2 | 左臂末端位置 | 左臂末端相对位置 |
| 3–8 | 左臂 6D 旋转 | 左臂相对 6D 旋转 |
| 9 | 左夹爪 | 左夹爪 |
| 10–12 | 右臂末端位置 | 右臂末端相对位置 |
| 13–18 | 右臂 6D 旋转 | 右臂相对 6D 旋转 |
| 19 | 右夹爪 | 右夹爪 |
| 20–22 | velocity_decomposed | velocity_decomposed |
| 23 | height | height |
| 24–25 | head_actions | head_actions |

输入状态是原始物理值，输出动作已经反归一化。`norm_key` 默认 `x2_normal`，统计量
来自 checkpoint 并随包保存；`state_mask` 与 `dof_mask` 均是 26 个 0/1，分别控制有效
状态和动作维度。状态具体含义以包内 `wall_config.json` 的原始布局为准。

输入 JSON 示例中零状态仅用于连通性检查，业务推理应使用对应机器人的真实状态：

```json
{
  "instruction": "pick up the cup",
  "images": ["/path/to/front.png", "/path/to/right_wrist.png"],
  "state": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "norm_key": "x2_normal",
  "seed": 42,
  "steps": 10
}
```

```bash
CUDA_VISIBLE_DEVICES=6 python -m examples_merak.vla.wall_oss.unified infer \
  --artifact-dir work_dirs/my_wall_oss_w8a8 \
  --input-json /path/to/input.json --output work_dirs/actions.json --device cuda:0

CUDA_VISIBLE_DEVICES=6 python -m examples_merak.vla.wall_oss.unified golden \
  --artifact-dir work_dirs/my_wall_oss_w8a8 \
  --input-json /path/to/input.json --output-dir work_dirs/my_wall_golden --device cuda:0
```

结果包括 `actions[32][26]`、horizon、有效图像网格及缓存检查。运行只读取发布包，
无需原 checkpoint 或 Wall-X 源码。Golden 包含三图连通的合成/真实输入；诊断文件
与部署产物分开保存，不随推理包发布。

## 开环数值验证

正式报告为[官方 BF16 对 W8A8 的 1,920 组配对](reports/libero_open_loop_20261026/README.md)：
40 个任务、320 个 episode、960 个观测，每观测 2 组噪声。总体最终归一化动作的
RMSE 为 0.053900、cosine 为 0.99478178；最差单组 cosine 为 0.77693389。
这些是描述性结果，不是任务准确率，也没有官方或用户确认的验收门限。

固定数据为 `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。
两路源图均为 256×256，继续使用通用包的长边 448、H32、26 维空间、512 档状态字符串、
`x2_normal` 和 10 步 Euler，不更换模型权重或归一化表。
原始状态的 XYZ 放入 10–12，轴角转旋转矩阵前两行放入 13–18，两指关节位置差放入 19。
该映射只用于数值探针，不证明机器人控制坐标和单位已完成适配。读取存储图像时不再翻转。

### 原始 120 组与浮点对照

按完整 episode、按任务固定 seed 10222 划分留出集合，每任务选一个 episode 的
25%、50%、75% 时刻。划分实现保存在 `open_loop.py`，不依赖其他训练入口。
目标运行目录必须不存在：

```bash
PYTHONPATH=. work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.open_loop prepare \
  --run-dir work_dirs/wall_libero_open_loop_20261026 \
  --data-dir data/datasets/lerobot_libero \
  --model-dir /data01/datasets/wall-oss-0.5 \
  --artifact-dir work_dirs/runtime_packages/hmquant_xh2_wall_oss_05_w8a8_p256_c2k_v1024_h32_20261024 \
  --official-source examples_merak/vla/wall_oss/.upstream

CUDA_VISIBLE_DEVICES=7 work_dirs/reference_env/bin/python -m examples_merak.vla.wall_oss.open_loop official \
  --run-dir work_dirs/wall_libero_open_loop_20261026

CUDA_VISIBLE_DEVICES=6 work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.open_loop float \
  --run-dir work_dirs/wall_libero_open_loop_20261026

CUDA_VISIBLE_DEVICES=6 work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.open_loop w8 \
  --run-dir work_dirs/wall_libero_open_loop_20261026

work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.open_loop summarize \
  --run-dir work_dirs/wall_libero_open_loop_20261026
```

旧 120 组以 FP32 为参考，用于官方→适配浮点→HMONNX 的分阶段诊断；`summarize`
保留当时的内部筛查阈值及退出码 2，不将其作为本次 BF16 主报告的合格判定。
各阶段可先加 `--ids 0,60,119` 验证代表样本，再执行全部样本。`open_loop_ablation.py`
可在相同样本和噪声上补充 W16A16/W8A16 诊断，不能把诊断子集成绩当作全量成绩。

### 官方 BF16 对 W8A8 扩充评估

保留原有 120 组观测/噪声和经审核可复用的 W8A8 预测，新增 1,800 组 W8A8，
全部 1,920 组 BF16 重新执行。每任务共 8 个 episode，每 episode 3 个时刻、2 个 seed。
两端输入指纹和实际 FP32 初始噪声相同。官方模型没有外部 noise 参数，参考入口在
原生 context 准备后替换噪声并用原生 action processor 重算初始动作 embedding，
该处理已与原生同噪声推理做逐位一致验证。

```bash
work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.bf16_open_loop prepare \
  --run-dir work_dirs/wall_libero_bf16_expanded_20261026 \
  --previous work_dirs/wall_libero_open_loop_20261026

work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.bf16_open_loop run \
  --run-dir work_dirs/wall_libero_bf16_expanded_20261026 --gpus 6,7 --detach

work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.bf16_eval_report render \
  --run-dir work_dirs/wall_libero_bf16_expanded_20261026

work_dirs/wall_env/bin/python -m examples_merak.vla.wall_oss.bf16_eval_report audit \
  --run-dir work_dirs/wall_libero_bf16_expanded_20261026
```

入口先实测每卡单/双引擎吞吐，再选择并发；SQLite WAL 独占领取样本，原子保存结果，
异常退出后恢复未完成项，避免漏测和重复计数。`--detach` 与终端分离。
原始图像、噪声、预测及日志留在 `work_dirs`；归档报告、逐样本结果及身份指纹已放入
[reports/libero_open_loop_20261026](reports/libero_open_loop_20261026/README.md)。
精确重放旧 GPU 采样噪声需原始噪声张量，不能只靠相同 seed 保证跨设备一致。
整理后的代码与历史执行代码指纹可能不同；恢复旧任务须使用冻结版本，修改后应新建运行目录。

### 单样例分阶段验证

`validate.py` 可独立执行 `official`、`float`、`deployed`，输出视觉特征和动作轨迹对比。
官方参考固定同一 Wall-X revision；参考仅恢复 meta 加载后未持久化的 RoPE buffer，
移除 SDPA 视觉接口不接受且不参与数学的 `max_seqlen` 提示，不改模型的动作网络。
具体基线精度由 `official --dtype` 指定，FP32 浮点适配对照应使用 `--dtype float32`。
历史浮点数值检查阈值属于工程诊断，不是官方量化或任务验收标准。
