# Wall-OSS-0.5：1024 容量与指令截断验证

本记录归档 2026-10-29 的模型重导出与回归证据。详细结构和使用方式见
[模型说明](../../README.md)与[缓存 ABI](../../../../../xhmodel_merak/xh_other_model/models/wall_oss/EXPLANATION.md)。
结构化结果、产物 SHA-256 与来源身份见 [validation.json](validation.json)。

## 最终配置与协议

```yaml
context_length: 1024
prefill_chunk_length: 256
visual_capacity: 1024
```

三份精度 YAML 使用同一尺寸；H32 不变。双张 448×448 图像合计 512 个视觉 token，
全部非图像内容预算最多 480，动作占 32。指令、状态文本、固定模板和特殊 token
共同占用非图像预算，不能将 480 当成指令单独上限。非方图按实际 grid 计算。

参考官方 LIBERO 示例的 `data.max_length: 1024` 和截断约束；官方处理整段序列，
本实现只右截断指令，完整保留图像、有效状态、模板标记和动作位置。
通过 token offset mapping 选择原字符串完整字符边界，重建 prompt 并重新分词，
保证前缀不超过 992，然后重新生成位置编码。短指令原样保留。
新 metadata 声明 `instruction_truncation: right`，推理结果 `stats.truncation`
包含是否截断、原始/保留指令和前缀 token 数以及前缀上限。

YAML、导出和运行时统一使用 `prefill_chunk_length`。兼容读取旧 `prefill_length`，
冲突值拒绝。没有截断策略声明的旧包保持超限报错，旧包物理尺寸及文件内容不变。
默认 recipe/profile 和产物名使用 `p256-c1024-v1024-h32`。

## 实际验证结果

- 最新 develop 基线的隔离检出运行受影响 CI：18 个目标、917 passed、8 warnings、245.77 秒。
  同步前导出验证使用的检出曾通过 901 个用例，两个阶段的记录分开保存。
- 三精度与 744/1024 两种容量的真实小模型边界测试共 6 个通过，覆盖尾块和完整动作区域。
- 每种完整精度测试历史 LIBERO 样本身份 0、87、1344、1905，992+32 满容量输入，
  超长英文及中文/emoji，共 7 个输入，均为 10 步推理。
- 每精度 7 个输入与同精度 c2048 旧包的结果逐位一致，最大绝对误差和 RMSE 为 0。
  长输入对照明确使用同一截断后指令；这不是拿截断输入与原始长输入直接比较。
- 英文前缀 2699→992，中文/emoji 2999→991；Unicode、图像、状态和模板完整。
- 满容量及超长输入通过公开 `inference_model`；每精度的合成/真实相机三图 Golden 完整。
- 新包三图物理端口、FP16 接口、图位宽及 group size 64 审计通过。
- 五个旧包各 20 个 Manifest 登记文件 SHA-256 不变。

这些是有限样本的缓存/输入协议回归，不是重新运行 1920 对 BF16 开环评估，
也不代表 LIBERO 闭环成功率或官方量化验收结论。

## 构建复现

使用本仓库模型说明中的部署环境和固定官方 checkpoint，从仓库根运行。
输出目录必须不存在；GPU 编号按本机空闲设备选择。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m examples_merak.vla.wall_oss.unified build \
  --model-dir /data01/datasets/wall-oss-0.5 --precision w8a8_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/hmquant_xh2_wall_oss_05_w8a8_p256_c1024_v1024_h32_20261029
```

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m examples_merak.vla.wall_oss.unified build \
  --model-dir /data01/datasets/wall-oss-0.5 --precision w8a16_sefp --device cuda:0 \
  --output-dir work_dirs/runtime_packages/hmquant_xh2_wall_oss_05_w8a16_p256_c1024_v1024_h32_20261029
```

产物均含 visual、prefill、denoise 三张 HMONNX 图及可脱离源 checkpoint 使用的运行资源。
工作目录中的模型和 Golden 文件不随代码提交；JSON 中保存产物文件名、大小及 SHA-256。
原始运行记录位于 `work_dirs/wall_c1024_20261029/`，不能将该工作目录视作可随代码传输的依赖。

## 官方依据

- [LIBERO 配置](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/workspace/example/libero.yml)
- [训练 collator](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/data/backends/lerobot/loader.py)
- [完整序列预处理](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/data/backends/lerobot/utils.py)
