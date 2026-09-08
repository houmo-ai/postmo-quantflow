# Kokoro v1.1-zh / XH2a

当前方案固定为三张 NPU 图和一张 CPU FP32 相位图，只发布展开 LSTM 的
HMONNX。浮点 ONNX 保留标准 forward LSTM；有效前缀反转由图外输入
`reverse_indices` 和图内 Gather 表达。量化及最终 LSTM 展开交给 xhquant，
Kokoro 不修改 xhquant，也不对导出的 ONNX 做额外图优化。

## 数据流

1. CPU：ZHG2P、词表映射、NumPy voice pack 查表，生成 style。
2. NPU `text_duration`：文本编码、Duration Encoder、Duration Predictor。
3. CPU：sigmoid/sum/round 得到 duration；根据 F 选档，通过 frame-to-token
   索引 Gather 复制特征，不构造 alignment 矩阵、不做矩阵乘法。
4. NPU `frame_acoustic`：Shared LSTM、F0/Noise Predictor、Decoder；
   输出 `decoder_feature [1,512,2F]` 和 `f0 [1,2F]`。
5. CPU `phase_core`：F0 上采样、谐波频率、phase increments、CumSum、
   相位插值和 Sin，输出 `sine [1,600F,9]`，整个过程 FP32。
6. NPU `generator_istft`：SourceMerge、谐波 STFT、Generator、静态 iSTFT。
   CPU 按真实 F 裁剪音频。

不能合成单张 W16A16 图的原因是相位误差会累积。只移走 CumSum/Sin 仍不足：
NPU 上 F0 上采样后的频率换算、phase increments 量化也会引入相位漂移，
长句可能出现沙哑。CPU 相位图必须从 F0 开始；F0 Predictor 本身仍在 NPU。
Generator 中用于清浊音判定的 F0 上采样不参与相位累加，保留在 NPU。

F0 Predictor 保留 masked AdaIN；Generator 使用已适配的 mean/缩放/RMSNorm
方案，s 和 inv_s 由 CPU 按有效长度提供。Attention mask 为有效 0、
无效 -65504，wrap 中连续加两次；不使用 MaskAdd、Clip 或 -inf。
STFT 按真实长度反射补边，固定使用 CORDIC 相位表达。
SineGen 的随机声源固定 seed=1234。

## 档位与路由

| T 容量 | F 容量 | 最长音频 |
| --- | --- | --- |
| 32 | 160 | 4 秒 |
| 64 | 320 | 8 秒 |
| 128 | 640 | 16 秒 |
| 256 | 1280 | 32 秒 |

Batch=1，T 包含首尾两个 boundary token，F 为 40 Hz 帧数；
F0 长度是 2F，24 kHz 波形长度是 600F。实际语音长短由 duration 决定。
先选能容纳 token 的最小档，duration 超出 F 后重试下一对 T/F；
不导出 T/F 笛卡尔积。不满档时按真实长度 mask。超过最大档时报错，
由调用方切句；demo 不静默截断或自动切句。

## 环境与原始资源

使用安装了 xhquant 的 Python 环境，导出还需要 torch、onnx、transformers
以及锁定的 Kokoro 0.9.4 源码；自定义中文依赖 `misaki[zh]==0.9.4`。
当前本地验证环境为 Python 3.12、torch 2.8.0、transformers 5.3.0、
ONNX 1.18、ONNX Runtime 1.24.4。HMONNX demo 是 SDK 仿真，不是芯片执行。

`--model-dir` 需要 `pytorch/config.json`、
`pytorch/kokoro-v1_1-zh.pth`、`pytorch/voices/zf_001.pt` 和锁定的
`source/kokoro`。资源身份由 `assets.py` 校验，不下载或依赖第三方导出 ONNX。
原始权重不放进交付包；推理只需要随包的 config、voice.npy 和模型。

## 导出和 Golden

在仓库根目录运行：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/kokoro_workflow.py \
  --model-dir work_dirs/model_assets/kokoro \
  --output-dir work_dirs/kokoro_merak \
  --device cuda:0 --dump-golden
```

默认导出全部四档：12 张 HMONNX、12 张 NPU 浮点参考 ONNX、4 张 CPU ONNX。
只有 decomposed HMONNX，没有模式开关，环境变量不能把本 recipe 改成 native。
`--token-bucket 32` 可选择单档；`--onnx-only` 仅作浮点导出，不能和
`--dump-golden` 合用。重复发布到已有 release 目录会报错，不自动覆盖。

`quant()` 返回 skipped，PTQ 在 `export()` 调用 xhquant 时执行；
两阶段不做数值精度对比。`dump_golden()` 是独立阶段，逐档强制跑完整的
HMONNX → CPU FP32 → HMONNX 链，默认真实输入为“千里之行，始于足下。”。
下游 Golden 使用上游 HMONNX 输出，不用浮点参考代替。所有 NPU 图保存
输入、输出及逐算子数据；CPU 图保存输入/输出。已有 step_0 时拒绝覆盖。
`golden_meta_info.json` 的 complete=true 才表示全部档位完成。

## 推理

将 EXPORT_DIR 替换为导出命令最后打印的 release 目录：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/bucketed_inference_demo.py \
  --export-dir EXPORT_DIR --backend hmonnx --device cuda:0 \
  --text '你好，欢迎使用中文语音合成。' --output-wav work_dirs/kokoro.wav
```

不传 `--text` 使用内置短句。`--backend ort` 跑同一切分的浮点参考。
`--list-routes` 查看档位；`--exercise-all-routes` 用同一输入强制覆盖四档。
`--config-path` 可显式指定词表；默认从包内 hf_config 加载，不需要原始权重。
当前仅支持中文、数字及中文标点；英文会显式报错，不悄悄丢字。
超过 256 token 也会明确报错。使用 `validate_release.py` 跑十条长短句、
保存 WAV 和路由报告；精度对比在 `compare_backends.py`，不在 quant/export。

## 发布布局与命名

参照[HM 模型版本发布命名规则](https://houmo.feishu.cn/wiki/WbX8wq9tpihXeHkzAy7caqMZnUh)，
采用全小写 `hmquant_xh2_<hf模型名>_w16a16_<尺寸>_<日期>`。
非 LLM 没有 prefill/context：这里用实际 token/frame 档位替代，
不填伪造的 LLM 长度，也不使用 native/decomposed 后缀。

四档包前缀：
`hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t32-64-128-256_f160-320-640-1280_YYYYMMDD`。

```text
<release_prefix>/
  text_duration_t0032/          # 各角色＋尺寸平铺，共 16 个组件目录
    <prefix>_text_duration_t0032_with_act.onnx
    <prefix>_text_duration_t0032_external_data
    step_0/
      <prefix>_text_duration_t0032_with_act.onnx -> ../同名文件
      <prefix>_text_duration_t0032_external_data -> ../同名文件
      <prefix>_text_duration_t0032_<name>_input.npy
      <prefix>_text_duration_t0032_<name>_output.npy
      <prefix>_text_duration_t0032_with_act/    # 逐算子 Golden
  frame_acoustic_f0160/
  phase_core_f0160/             # FP32 ONNX，无量化权重文件
  generator_istft_f0160/
  ...                          # 其余三档
  reference_onnx/               # NPU 图的浮点参考
  hf_config/                   # 上游文本配置和词表
  assets/voices/zf_001.npy
  workflow.yaml
  export_meta_info.json
  golden_meta_info.json
  <release_prefix>_hmonnx.py
```

模型文件、external_data 的真实路径以 metadata 和 ONNX location 为准。
**不手工改 external_data 名称**。step 内软链接必须是包内相对链接。
运行随包 demo 仍需要匹配版本的 xh2modelzoo、xhquant 和上述依赖；
包内 config/voice 无源目录依赖。调试日志随组件保留。

压缩时从 release 的父目录运行 `zip -r -y <prefix>.zip <prefix>/`，
保留相对软链接，不能解引用复制权重。发布前核对 metadata 完整标记、
全部模型/权重存在、软链接可解析以及 ZIP 完整性。逐算子 Golden 占用较大，
应按实际输出统计存储和压缩包大小。
