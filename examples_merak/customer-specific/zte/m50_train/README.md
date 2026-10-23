# 中兴客户模型：三个训练 Demo

本目录主要提供三个Demo：**Torch原生训练、XH2前后向＋FP32更新、XH2前后向＋FP16更新**。三者共用`train.py`，分别选择三份JSON配置即可。运行位置都是NVIDIA GPU，**没有在M50/NPU真机训练**；后两组在GPU上执行真实XH2 ALIGNED数值实现。

任务来源：[CMS-909](http://jira.houmo.ai/browse/CMS-909)。完整算子映射、反向拼接及实验分析见[飞书报告](https://houmo.feishu.cn/docx/AmIrdDV89oq0lTxoXTqcrmkrnLT)。本目录提供独立入口，不依赖 `work_dirs/zte_train/scripts/`；运行产物默认写入仓库 `work_dirs/zte_train/runs/`。

## 1. 先分清三个 Demo

| Demo | 配置文件 | 前向与反向 | 优化器和权重更新 |
|---|---|---|---|
| **1：Torch原生训练（A）** | `configs/a_fp32.json` | 原始PyTorch FP32前向，autograd计算FP32梯度 | 原始ReferenceAdam在GPU上用FP32更新 |
| **2：XH2训练＋FP32更新（B）** | `configs/b_sefp16_fp32_update.json` | XH2 ALIGNED前向和显式反向；MatMul为动态SEFP16，梯度结果为FP16 | 将梯度转FP32；GPU FP32 Adam、主权重及running统计更新，主权重再转FP16供下一轮 |
| **3：XH2训练＋FP16更新（C）** | `configs/c_sefp16_fp16_update.json` | 与Demo 2相同的XH2前后向规则，梯度结果为FP16 | 用XH2 FP16原语在GPU上更新权重、Adam状态及running统计 |

三个Demo的计算过程可以概括为：

```text
Demo 1：Torch FP32前向 → Torch autograd FP32梯度 → FP32 Adam/权重更新
Demo 2：XH2 ALIGNED前向 → XH2显式反向得到FP16梯度 → 转FP32 → FP32 Adam/主权重更新
Demo 3：XH2 ALIGNED前向 → XH2显式反向得到FP16梯度 → FP16 Adam/权重更新
```

这里“FP32/FP16更新梯度”准确说是**用梯度更新优化器状态和权重**。Demo 2没有把反向改为FP32，也没有跨batch积攒梯度；Demo 2/3改变的是后续更新阶段的精度。先用Demo 1建立基线，再用Demo 2检查XH2前后向能否学习，最后用Demo 3观察低精度更新增加的风险。

B与C的区别是更新及状态精度，不是“B的反向FP32、C的反向FP16”。MatMul包括LSTM的xK/hR、Dense和Head；前向、反向、验证推理每次都按新的归约轴重新量化，不能复用转置前的量化载荷。A中bits不构成量化开关，SEFP位宽仅对B/C有意义。

本轮固定客户语义：batch=32、输入9步×13特征、LSTM hidden=256、Dense=128、9类；variance BN的epsilon=.001，running更新系数=.01；Dropout p=.2；ReferenceAdam lr=1e-4、epsilon=1e-7；损失为NLL mean加 `.01*sum(kernel²)+.001*sum(dense²)`。这些不是本入口的配置字段。Mean保留客户Sum/Div顺序；两项L2都参与反向。

客户整数包的bFP16方案另含MAD、SR、梯度放大16、Hadamard和更新补偿；bFP24也有其block32格式。这两套客户策略不等于这里的A/B/C配置。本示例主实验使用variance、反向seed=1，无MAD、SR、Hadamard或残差补偿，不能称为完整复现客户bFP16/bFP24训练。

已有seed17、300步的数值实验中，B16总loss从3.015625下降至约.307861；C16最终约10.6484。C16标签loss下降，但权重正则项增大，因此总目标恶化。两组在容易的独立合成验证集上均为288/288分类正确，不能据此认定C的优化器正常。数据在时间步重复提供类别信号，只验证技术闭环，不能代表客户业务精度或困难时间依赖任务的能力。这些是既有报告结果，不保证不同环境复现逐bit一致。

## 2. 重点脚本与客户代码

| 文件 | 责任 |
|---|---|
| `train.py` | 轻量主入口；先解析JSON/CLI并核验客户文件哈希，再导入CUDA SDK；`--help`无需torch |
| `trainer.py` | 核心训练与恢复；浮点梯度核查、A/B/C训练、日志、checkpoint及独立恢复核验 |
| `runtime.py` | 数值适配器；直接调用XH2 ALIGNED前后向原语；保留客户Node调用边界；提供标量Adam诊断原语 |
| `common.py` | 仓库路径、客户代码加载、固定合成数据与随机状态管理 |
| `diagnose.py` | VP/NL ReLU选路与FP16 Adam最小复现，以及同梯度更新诊断 |
| `plot.py` | 多组结果绘图；分别显示标签loss、正则loss和总loss，避免误读分类与优化目标 |
| `configs/*.json` | 上述三份Demo主配置；另两份SEFP8配置仅用于可选位宽对照 |
| `customer_sources.json` | 两份客户原件的相对路径、SHA256及Jira来源，启动时强制校验 |
| `customer/torch_native/model.py` | 原始客户浮点模型与ReferenceAdam |
| `customer/torch_integer/engine/bf16_autograd.py` | 原始客户Node与Model；复用拓扑求导、九步BPTT及共享参数梯度累计 |

不是给每一层重写backward kernel：Linear反向继续用MatMul，LSTM门导数和h/c倒序传播由Mul/Sub/Add组成，偏置及共享参数梯度由Sum/Add累计。客户Node组织这些调用，runtime把数学运算接入XH2。保存每步门值、h/c与Dropout mask，反向复用同一mask。

`customer/` 仅存放CMS-909附件中的上述两份原件，已设为Git忽略。附件未提供可确认的license，因此客户原始代码不随提交分发；新checkout需从有权访问的客户材料取得文件。不要把本示例误当作客户代码再授权。可通过配置 `customer_sources` 或CLI `--customer-sources` 指定另一份同结构目录。

当前工作区已经把两份必要原件复制到上述位置，可以直接运行。新checkout可将工单的`torch_native.zip`、`torch_integer.zip`解包到`customer/`，只保留上表两份必要Python文件即可；按`customer_sources.json`核对原始字节，不修改或格式化客户文件。也支持历史工作区的双层解包布局，例如`--customer-sources work_dirs/zte_train/sources`，但不依赖那里的实验脚本。

## 3. 环境与路径约定

需要已有的CUDA、PyTorch、Triton、NumPy和可导入的 `xhquant`；绘图另需Matplotlib。本仓库通常通过 `xhquant/` 链接使用相邻xhquanttool。示例不安装依赖，不使用CPU降级路径。正式报告环境为A100、torch2.8.0+cu128、CUDA12.8、Triton3.4.0和xhquant0.4.0；迁移环境应以本次输出的环境及源码哈希为准。SDK来自wheel而没有Git信息时，会记录null与原因，实际源文件哈希仍保留。

本机已有`xh2`环境，可先执行`conda activate xh2`，再运行下述命令；也可以把`python`替换为该环境解释器的绝对路径。

确认实际导入的是对应工具链源码，而非环境中较旧的SDK副本。若仓库没有`xhquant`链接，可在训练前设置`PYTHONPATH=/实际路径/xhquanttool:$PYTHONPATH`；用`python -c "import xhquant; print(xhquant.__file__)"`核对位置。本示例使用包括`ir.operators.activate.generate_relu_lut_table`在内的现有SDK接口，不静默替换成旧接口或普通浮点算子。

以下命令从仓库根目录执行，`python`应指向已有XH2环境。也可以从任意cwd以脚本绝对路径运行；入口自行定位仓库。**所有相对output和配置内customer_sources路径统一相对仓库根，不相对当前cwd或JSON所在目录。** 相对配置文件路径也按入口约定从仓库根解析。

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py --help
```

真实MatMul路径要求在import前禁用FAST、启用Triton ALIGNED；入口负责设置并检查。不要用SEFP后普通torch.matmul或QAT浮点反向替代。默认ReLU使用正式QReLU一致的NL LUT；VP ReLU只保留为诊断选路。

## 4. 可选：单独运行梯度核验

三个Demo启动时都会自动检查客户显式反向与浮点autograd是否一致。无需手动先跑另一条训练命令；如果只想检查而不训练，可执行下面的`--check-only`。检查包括训练Dropout开启/关闭情况下的参数梯度和running统计，验证的是复用的解析求导逻辑。

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/a_fp32.json \
  --check-only --output work_dirs/zte_train/runs/check_gradients_01
```

`--output`必须是**新目录或空目录**；推荐总是取新名字，不要复用check-only或失败运行的目录。保留旧protocol与日志。check-only也会产生protocol和浮点核验结果，但不保存checkpoint；每组主实验也会执行浮点核验。

## 5. 三个 Demo 的运行命令

以下三条命令从仓库根执行，每个Demo启动独立Python进程，默认seed=17、steps=300、batch=32。按空闲GPU选择`CUDA_VISIBLE_DEVICES`；三个输出目录保持独立。首次运行只需依次执行这三条命令。

### Demo 1：Torch原生FP32训练

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/a_fp32.json \
  --output work_dirs/zte_train/runs/A_fp32_01
```

原始客户模型用Torch前向、autograd和ReferenceAdam完成训练。主训练不做XH2/SEFP量化，作为另外两个Demo的数据与训练目标基线；`bits`字段对A不生效。

### Demo 2：XH2 ALIGNED训练，GPU FP32更新

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/b_sefp16_fp32_update.json \
  --output work_dirs/zte_train/runs/B_sefp16_01
```

前向和反向均使用XH2 ALIGNED算子，Linear两侧动态SEFP16量化。梯度在XH2计算后为FP16，更新时转FP32，用FP32主权重、Adam和running状态。这个Demo主要回答“已有XH2原语拼出的前后向能否完成学习”。

### Demo 3：XH2 ALIGNED训练，GPU FP16更新

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/c_sefp16_fp16_update.json \
  --output work_dirs/zte_train/runs/C_sefp16_01
```

与Demo 2使用同一套XH2前后向，但进一步把优化器、权重和running统计更新改为FP16。该Demo用于暴露Adam下溢、舍入和权重小更新丢失；分类准确率高仍可能伴随正则Loss及总Loss上升。它仍然是GPU数值模拟，不能称为已在NPU更新。

SEFP8不属于这三个主要Demo。需要额外比较MatMul位宽时，再选择`b_sefp8_fp32_update.json`或`c_sefp8_fp16_update.json`并指定新的输出目录；这是Demo 2/3的可选位宽对照。改变seed也应另立输出目录，不能把已有单seed结果称为鲁棒性结论。

## 6. JSON配置与CLI覆盖

优先级为：**显式CLI参数 > JSON配置 > 入口默认值**。不写CLI参数不会覆盖JSON中的值。示例配置：

```json
{
  "variant": "B",
  "bits": 16,
  "seed": 17,
  "steps": 300,
  "output": "work_dirs/zte_train/runs/B_sefp16_custom_01",
  "customer_sources": "examples_merak/customer-specific/zte/m50_train/customer",
  "check_only": false,
  "relu_path": "nl",
  "diagnostic_override": null
}
```

| 字段 | 有效值/用途 |
|---|---|
| `variant` | `A`、`B`、`C` |
| `bits` | 16或8，仅B/C使用SEFP |
| `seed`、`steps` | 随机种子、训练步数；steps必须为正 |
| `output` | 新的独占输出目录或空目录；默认 `work_dirs/zte_train/runs/<name>` |
| `customer_sources` | 包含上述两份原始客户文件的目录 |
| `check_only` | true时仅执行核查，不开展主训练 |
| `relu_path` | `nl`为主实验；`vp`仅诊断 |
| `diagnostic_override` | null、`forward_matmul_fp32`、`backward_matmul_fp32` |

不要加入batch、lr、epsilon、Dropout或正则系数等字段；它们是本实验保留的客户固定语义，当前配置不支持。`diagnostic_override`只用于把指定阶段MatMul替为FP32后转half定位误差，启用时不能把结果标成完整XH2前后向。

例如同一配置只改seed、步数及输出：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/b_sefp16_fp32_update.json \
  --seed 23 --steps 30 --output work_dirs/zte_train/runs/B16_seed23_smoke_01
```

外部客户目录的根下仍需有 `torch_native/model.py` 和 `torch_integer/engine/bf16_autograd.py`：

```bash
python examples_merak/customer-specific/zte/m50_train/train.py \
  --config examples_merak/customer-specific/zte/m50_train/configs/b_sefp16_fp32_update.json \
  --customer-sources /absolute/path/to/customer \
  --check-only --output work_dirs/zte_train/runs/external_customer_check_01
```

## 7. 输出、恢复核验和读图

| 产物 | 重点内容 |
|---|---|
| `effective_config.json` | JSON、CLI、默认值合并后的配置；路径已转为绝对路径 |
| `protocol.json` | 运行前协议、有效配置、固定训练语义与判据 |
| `provenance.json` | 客户源码、SDK源码/版本、仓库版本等来源信息 |
| `float_gradient_check.json` | 显式反向对浮点autograd的核查；check-only重点检查此文件 |
| `initial_metrics.json` | 更新前验证指标 |
| `key_layer_errors.json` | 同表示值、同Dropout mask条件下关键层前向与梯度误差 |
| `first_step_diagnostics.json` | 首步梯度及更新定位 |
| `curve.csv` | 每步loss与梯度/更新统计；按记录间隔写验证指标 |
| `checkpoint.pt` | 模型、优化器、running统计、step及随机状态 |
| `summary.json` | 完成步数、环境、失败信息、实际命令及保存/恢复核验结论 |
| `code/` | 本轮实际运行代码快照，用于比较版本与追溯 |

`curve.csv`中 `train_class_loss` 是标签NLL；`train_l2_kernel`、`train_l2_dense` 是两项正则；`train_total_loss` 是实际反向的总目标。`val_ce`是验证网络输出上的监控NLL，`val_accuracy`是argmax准确率。ALIGNED Log/Exp是近似实现，输出exp后不保证严格和为1；不要把字段名CE解读为精确归一化softmax的证明。

同时观察 `gradient_finite`、`gradient_zero_fraction`、`visible_parameter_update_fraction` 和 `parameter_finite`。标签准确率提高而总loss上升时，先分开检查正则项和更新数值；仅凭 `summary.passed` 或100%分类正确不能认定完整优化目标收敛。

训练结束会保存checkpoint，再创建独立实例：先验证重载后eval完全一致，再从同一状态和同一随机流执行一步续训，与原实例结果及状态逐项比较。查看 `summary.json` 的 `reload_equal`、`resume_equal`。这是恢复能力核验；当前CLI没有对外的 `--resume` 参数，不要假设可以在已有output里自动追加训练。checkpoint仅加载可信本地材料。

三个Demo运行后，直接对比其标签loss、正则合计、总loss，以及验证与更新诊断图。`--runs`接受任意组数，后续可追加SEFP8结果；图例根据各组summary中的protocol派生，启用诊断替换的结果会显式标注。

```bash
python examples_merak/customer-specific/zte/m50_train/plot.py \
  --runs work_dirs/zte_train/runs/A_fp32_01 \
         work_dirs/zte_train/runs/B_sefp16_01 \
         work_dirs/zte_train/runs/C_sefp16_01 \
  --output work_dirs/zte_train/plots/comparison_01
```

`--runs`必填；绘图默认输出 `work_dirs/zte_train/plots`，相对输入/输出均从仓库根解析。更多选项可查看帮助：

```bash
python examples_merak/customer-specific/zte/m50_train/plot.py --help
python examples_merak/customer-specific/zte/m50_train/diagnose.py --help
```

诊断使用新目录：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python examples_merak/customer-specific/zte/m50_train/diagnose.py \
  --output work_dirs/zte_train/runs/diagnostics_01 --steps 30 --seed 17
```

诊断另支持 `--customer-sources DIR`；其 `--steps`只控制诊断步数，必须为正，不修改主训练配置。默认输出 `work_dirs/zte_train/runs/diagnostics`。

VP ReLU的小正数归零是低层clip实现的独立失败复现，不能概括默认QReLU LUT。FP16 Adam诊断分别检查二阶矩增量下溢、更新分子下溢、异常大更新及half权重吞掉小更新；影子FP32优化器只监控同一梯度，不参与真实更新。

## 8. 常见问题与范围限制

| 现象 | 处理与解释 |
|---|---|
| CUDA不可用、Triton/SDK导入失败 | 确认python属于已有XH2环境、GPU可见且xhquant链接有效；本示例不自动安装或转CPU |
| 缺ninja、hsum提示旧二进制 | 已有环境可能无法重建后继续加载旧GPU扩展；记录日志和版本，不能称已验证当前源码重编一致性。需要重建时按工具链环境规范处理 |
| 客户文件缺失 | 检查customer_sources目录与上述两级相对路径；Git忽略目录不随checkout提供 |
| 客户哈希不同 | 对照CMS-909原件和运行provenance确认来源；不要为了绕过检查改写原件或静默去掉校验。新材料应独立留档 |
| 输出目录非空 | 换新目录；不要删除既有证据后复用 |
| C总loss上升、但分类准确率高 | 拆开标签/正则loss，检查FP16更新；这是本实验要展示的失败模式之一 |
| 同seed结果不一致 | 比较SDK实际源码/二进制、配置、mask生成、RNG状态、GPU与代码快照；原始报告旧版Dropout试跑不作为正式同mask参考 |

本示例未完成产品workflow或M50执行器集成；没有测真机动态参数重打包、编译、板上更新精度、内存和吞吐。GPU测时包含Python调度、数值仿真及监控，不能作为NPU性能。长期是否允许Host/GPU更新需要结合客户最终验收，不能由B配置的技术可行性替代确认。

训练设计的详细依据保存在仓库 `work_dirs/zte_train/analysis/operator_mapping.md` 与 `full_model_report.md`，原始附件和完整性清单在同一工作区。它们用于历史证据；本目录运行不读取这些实验脚本或历史结果。

## 9. 示例验证与CI接入

CPU契约测试按仓库规则放在`benchmark_test/ci_test_transformer5.16.0/test_zte_m50_train_example.py`，包含配置覆盖/拒绝未知字段、从其他cwd启动、未安装site packages的帮助入口、客户文件布局与哈希、输出保护、数据RNG隔离和诊断图例。它还用真实小CPU tensor执行FloatRuntime矩阵乘及反向公式、广播/归约、状态复制和trainer统计/恢复比较，并检查整组改动的selector覆盖。它不依赖私有客户文件、SDK或GPU；SDK与客户代码只在实际训练时加载，不提供训练的CPU降级路径。针对性运行：

```bash
python -m pytest -q \
  -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_zte_m50_train_example.py
```

维护目录为`examples_merak/customer-specific/zte/m50_train`。旧的`examples/**`目录被全局CI冻结，因此本次评审已迁移，不保留旧路径入口，也不放宽全局policy。六个脚本和配置由同名impact声明映射到本示例契约测试；统一CI还会运行selector和recipe两项公共门禁。完整改动清单必须取修订后提交相对父提交的差异，不能只验证本地未暂存的增量。

此前整理验证中，17项示例CPU契约测试和93项CI selector测试共110项通过；A/B16/C16/B8/C8均从`/tmp`启动新入口、seed17运行300步，曲线数值、最终指标和checkpoint全部状态与迁移前归档逐项相同，独立重载及一步续训均一致。30步诊断曲线也与原归档相同；多组绘图入口已实际运行。

此前GPU验证产物保留在`work_dirs/zte_train/packaging_validation/`，其中`regression_verification.json`记录完整数值回归，`figures/`包含绘图输出。本次CI修订使用Jenkins 2395同版镜像`harbor.houmo.ai/hmquantool/dev:v1.0.5-xhquant-ubuntu24.04-py312-transformers-5.16.1-x86.64`，针对修订后完整16文件清单执行统一`run.sh`，154项全部通过（21项Demo契约、93项selector契约、40项recipe门禁）。Python3.10可运行XH2数值实验，统一CI的recipe门禁使用Python3.12环境。

检查当前评审的完整差异并运行统一CI（在上述CI环境中）：

```bash
git diff --no-renames --name-only HEAD^ HEAD > /tmp/cms909_changed_files.txt
python benchmark_test/ci_test_transformer5.16.0/select_tests.py \
  --changed-files-file /tmp/cms909_changed_files.txt --format report
CI_TEST_CHANGED_FILES_FILE=/tmp/cms909_changed_files.txt \
  ./benchmark_test/ci_test_transformer5.16.0/run.sh
```
