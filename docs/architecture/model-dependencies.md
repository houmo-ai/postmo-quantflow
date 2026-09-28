# Unified 模型依赖环境

状态（2026-09-28）：已实现轻量 `DependencySpec`、44 个注册变体的依赖声明查询、
`export_model_dependencies` 文件导出，以及构建产物随包发布。必要依赖的完整审计和
干净环境安装验证尚未完成；现有条目均为 `blocked_distribution`，不得当作锁文件。
回归记录需结合后续更正：Qwen3-Next 无 MTP 的旧权重不计 MTP 有效通过，
Gemma4 对齐 TorchCodec 后尚未重做完整视频推理；Macaron、Ling 问题暂缓处理。

实现位于 `model_catalog/dependencies.py` 和 `model_build/dependencies.py`；各 family
在 `plugin.py` 显式选择公共版本与自身补充项，`VariantRecipe.dependencies` 表达变体差异。
`describe_model(..., full=True)` 返回模型级 profiles 和每个 workflow variant 的合并结果，
`list_models()` 返回 profiles 名称及状态。默认 `describe_model()` 的构建参数语义保持不变。
支持 `common/build/inference` 声明；公开 profiles 另含按量化方法合并的 `quantization`
和复用 checkpoint 的 `checkpoint_build`。Gemma full 的视频依赖不注入 MTP 推理档案。
requirements 约束保留 markers/extras，合并时拒绝无条件精确版本冲突和本机路径；
完整平台与传递依赖求解交由安装验证，不声称轻量合并器能够替代 pip。

`export_model_dependencies(model_id=..., variant=..., output_dir=...)` 创建新目录，
写入 `environment.json` 和 `environments/*.txt`；拒绝覆盖既有目录。
构建在事务内写入相同文件，再生成集成清单与 Manifest，全部摘要随包保存。
查询不加载模型、不探测宿主环境；导出不执行安装。用法见统一接口示例 README。
静态源码扫描与运行进程导入记录仅作为审计辅助，不能直接推定某个包是必要依赖，
也不能代替干净环境安装验证。所有当前注册模型（包括 Pi0.5）统一声明
Transformers 5.16.1；Pi0.5 使用 LeRobot 0.6.0。公开 requirements 与 worker 前置检查统一使用已验证的精确版本，
不以版本范围暗示其他版本已验证。Python 固定为 3.12.13，GPTQModel 固定为 5.8.0，
AutoRound 固定为 0.12.0；发布包仍需包含本次验证的源码修改。安装基础 `lerobot` 包，不使用
会重新限制 Transformers 版本的 `lerobot[pi]` extra。消费端严格加载 Pi0.5 权重，
显式兼容 SigLIP 视觉键名的层级变化，不允许上游加载失败后返回未完整初始化的模型。
这一 Pi0.5 组合已通过严格加载、前缀 KV 对比及 unified 构建/推理检查。
原生参考推理的 4D mask 调用使用实例级兼容，不修改第三方包或全局 forward；
视觉投影及语言 embedding 分别保持原模型的缩放语义。
客户 DROID 固定输入/噪声 100 条平均 cosine 为 0.994276、MAE 为 0.016587；
LIBERO 100 条仿真观测平均 cosine 为 0.999177、MAE 为 0.003651，
均通过平均 cosine >= 0.99、平均 MAE <= 0.02 的数值门禁。
Docker LIBERO 闭环每侧 200 个 episode，浮点参考与 HMONNX 均为 179/200；
历史版本成绩差异尚未归因。详见 [Pi0.5 验证说明](../../examples_merak/vla/pi05/README.md)。这些证据是现有环境中的
CUDA 软件 HMONNX 精度验证，不替代依赖包的干净环境安装验证或 NPU 板端验证。

## 设计依据

- [Cog 的 `cog.yaml`](https://github.com/replicate/cog/blob/main/docs/yaml.md) 分开声明 Python 版本、pip requirements、CUDA 和系统包；requirements 支持 Git commit 固定。
- [MLflow 模型依赖文件](https://mlflow.org/docs/latest/ml/model/dependencies) 将 Python 环境信息、`requirements.txt` 和可选 lock 文件随模型保存，并支持在隔离环境恢复。
- 旧版 `ModelRecipe.worker_requirements` 只列了部分包名或宽版本范围；`quant_requirements` 按量化方法列包，但没有 Python 版本、变体差异、包来源和已验证状态。`ModelRecipe.environment` 已用于运行时环境变量，不能复用为依赖字段。
- 旧版 `worker_requirements` 按模型共享，例如 Gemma4 的视频解码包也列在只跑文本的 MTP 变体下；新的安装档案须按变体和输入能力收敛，不能原样复制共享声明。

## 公开字段与后续锁定格式

在 `describe_model(model_id, full=True)` 的完整元数据中提供版本化的 `dependencies`，并在每个变体的描述里给出已合并的安装档案。现有 `describe_model(model_id)` 默认仅描述可编辑构建参数，不混入安装档案；`list_models()` 只暴露档案名称和状态，避免将大段锁定清单放进列表响应。

```json
{
  "dependencies": {
    "schema_version": 1,
    "python": "==3.12.13",
    "profiles": {
      "full": {
        "build": {
          "pip": ["transformers==5.16.1", "torch==2.8.0"],
          "requirements_file": "environments/full-build.txt"
        },
        "inference": {
          "pip": ["transformers==5.16.1", "torch==2.8.0"],
          "requirements_file": "environments/full-inference.txt"
        },
        "status": "pending_install_validation"
      }
    },
    "platform": {"cuda": "12.8", "accelerator": "XH2a"}
  }
}
```

这里的 `pip` 只允许 PEP 508 requirement、固定提交的 VCS URL 或可访问的 wheel URL；禁止本机绝对路径和宽泛 `pip freeze`。示例只展示字段形状，不是完整 Gemma4 安装清单。Python 版本必须由环境管理器先准备，pip 只安装该解释器下的第三方包。`torch==2.8.0` 与配套的 `torchvision` 版本本身不指明 CUDA wheel 来源，最终 requirements 还需固定相应 wheel URL/校验和或受控包索引，并在目标平台验证；CUDA、驱动、芯片运行库与系统包单独声明，不伪装成 pip 包。

合并规则为模型公共依赖 + 变体依赖 + 所选量化方法依赖；`full`、`mtp`、`dflash`、`lora` 等逐个形成最终档案。构建和纯推理分开，已量化 checkpoint 的推理不强制安装量化器。推理档案必须覆盖该模型声明支持的全部输入模态；例如 WeMM 的文本通过不足以确认图片和视频依赖，视频读取后端及其版本须以真实媒体输入验证后纳入同一可安装档案。Gemma4 本轮真实 MP4 回归表明，仅凭已安装的 `torchcodec` 包名不足以保证可用：原 `torchcodec==0.10.0` 与 PyTorch 2.8.0 不匹配，后续保持 Torch 2.8.0 并对齐为 `torchcodec==0.7.0`，真实短视频解码与 12B/E2B/E4B 处理器验证通过。已撤销显式 PyAV 绕过及其无条件 worker 依赖；视频档案需固定匹配的 Torch/TorchCodec 组合并验证 FFmpeg 共享库，最终安装档案仍需在干净环境复测。删除 Recipe 的 `worker_requirements`、`quant_requirements`、`quant_checkpoint_requirements` 及对应公开字段，
以 `dependencies` 为唯一版本声明。BuildPlan 中的 `worker_requirements` 是所选变体/阶段档案的计算结果，
供执行器检查；推理直接检查该变体的 inference 档案。能力检查继续独立验证 recipe 入口是否存在。构建产物随包记录 `environment.json` 和对应 requirements 文件的**包内相对路径**及校验和，供加载方核对；不把本机路径写进产物。按现有[统一产物布局](artifact-layout.md)，`export_meta_info.json` 仅将 `environment.json` 登记为 `type: json`；requirements 文本由该 JSON 指向并纳入外层 build manifest 的文件摘要，不为它新造组件 type。

`status` 至少区分 `pending_install_validation`、`blocked_distribution`、`verified`。只有在干净的 Python 3.12.13 环境中成功执行 `pip install -r`、`pip check`、模型/变体的 `build_model -> inference_model` 后，才公开可复制的安装命令并标为 `verified`。量化回归可使用 1–2 条样本和少量迭代验证接口，但不据此宣称量化精度或吞吐。

## 当前不能直接发布安装清单的原因

本轮环境中的 Transformers 是 5.16.1，但已安装的 `xhmodel-zoo`、`xhquant-llm` 等发行包仍声明 Transformers 4.x 上界，`pip check` 有 20 余项冲突。仓库当前 `pyproject.toml` 虽已允许 `<5.17`，却仍把 `compressed-tensors`、`datasets`、`flash_attn` 等多模型依赖放在全局安装要求中；单列模型 requirements 不能消除这些发行包约束，需同步拆分或发布与本轮验证环境兼容的元数据。`gptqmodel==5.8.0` 指向本机可编辑源码，AutoRound 从其 `third_party` 目录导入；`xhquant` 从当前仓库源码导入。这些本机状态不能转成别人可用的 `pip install` 命令；可编辑源码的版本号不能唯一确定实际代码，存在未提交改动时，单独固定 HEAD 也不能重现测试环境。需先发布兼容 Transformers 5.16.1 的 wheel 或固定可访问的源码提交，再在空环境解析依赖。Kokoro 本轮实际构建发现固定源码会导入 `misaki[en]`；recipe 已将声明改成 `misaki[en,zh]==0.9.4`，由 extras 带入 `num2words`、spaCy、phonemizer 等依赖，仍需在干净环境验证安装。
