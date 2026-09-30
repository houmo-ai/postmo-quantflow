# GLM-5.3-Flash / Merak

使用 `glm-5.3-flash` recipe：AutoRound 量化非共享专家为 W4 G64，其余权重为 W8，
激活为 A16；KDA 的 `forget_gate.f_a_proj/f_b_proj` 和 `b_proj` 保留 W16。
默认上下文 256K、prefill 分块 256、batch 1，支持文本、图片和视频输入。

## 环境准备

在 ModelZoo 仓库根执行，原始模型放在 `weights/GLM-5.3-Flash`。
GPU 编号按实际空闲情况替换；以下量化和导出使用一张卡。

```bash
conda activate xhquant_55_tf_latest_20260828
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 量化

配置见 [AutoRound W4 G64 YAML](../../../configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_autoround_w4g64.yaml)。
默认使用 128 条 2048-token 样本、200 次迭代、batch size 1、梯度累积 2。

```bash
python - <<'PY'
from pathlib import Path
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.workflows import AutoWorkflow

workflow = AutoWorkflow.from_config(
    model_dir=str(Path("weights/GLM-5.3-Flash").resolve()),
    config_path=get_recipe("glm-5.3-flash").default_variant.workflow_config,
)
result = workflow.quant(output_dir="work_dirs/glm53-quant", device="cuda:0")
print(result.quanted_model_dir)
PY
```

量化权重保存到 `work_dirs/glm53-quant/checkpoint`，后续导出可重复使用。

## 导出 HMONNX

```bash
python -m examples_merak.unified_api.build \
  --model-id glm-5.3-flash --model-dir weights/GLM-5.3-Flash \
  --quant-checkpoint work_dirs/glm53-quant/checkpoint \
  --output-dir work_dirs/glm53-w4g64 --device cuda:0 \
  --context-length 262144 --prefill-chunk-length 256
```

导出自动使用低内存模式，并共享 `prefill/` 和 `decode/` 的权重。
修改上下文长度后需要重新导出。输出目录必须尚不存在；以命令返回的 `output_dir` 为产物路径。

## 文本、图片和视频推理

完整模型使用多卡加载，按实际显存需求设置可见 GPU。`--device cuda` 使用全部可见卡。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GLM_ARTIFACT_DIR="$(realpath work_dirs/glm53-w4g64)"

python -m examples_merak.unified_api.generate \
  --artifact-dir "$GLM_ARTIFACT_DIR" --device cuda \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32

python -m examples_merak.unified_api.generate \
  --artifact-dir "$GLM_ARTIFACT_DIR" --device cuda \
  --image /path/to/image.jpg --prompt '描述图片中的内容。' --max-new-tokens 64

python -m examples_merak.unified_api.generate \
  --artifact-dir "$GLM_ARTIFACT_DIR" --device cuda \
  --video /path/to/video.mp4 --prompt '概括视频中的变化。' --max-new-tokens 64
```

图片和视频由 processor 自动缩放、采帧，无需手工固定长宽。
默认每帧容量为 1024 patches；单图和单视频总预算均为 256 个合并 token。
输入 token 与生成 token 的总长度不得超过导出的上下文容量。

## 搬移产物

交付整个导出目录。使用 `rsync -aH` 保留 prefill/decode 权重的硬链接，避免重复占用磁盘：

```bash
rsync -aH work_dirs/glm53-w4g64/ /path/to/deploy/glm53-w4g64/
```

推理时将 `--artifact-dir` 改为搬移后的目录即可。
