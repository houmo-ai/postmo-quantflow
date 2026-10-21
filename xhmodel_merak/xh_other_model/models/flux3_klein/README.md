# FLUX 3 Image Klein

模型、Host 和图 ABI 的设计见 [EXPLANATION.md](EXPLANATION.md)。
构建、生成、编辑、Golden 与对照命令见[统一示例](../../../../examples_merak/aigc/flux3_klein/README.md)。
当前支持范围、实现文件、历史提交与验收索引见
[适配交付总览](../../../../docs/modeling/structures/flux3-image-klein.md#adaptation-overview)，
已验收核心/组合运行包及打包指令见
[产物与打包](../../../../examples_merak/aigc/flux3_klein/README.md#artifacts-and-packaging)。

官方 Qwen3.5-9B 提示词扩写器已作为可选组件接入。使用者可按
[扩写器接入步骤与 CLI/Python 示例](../../../../examples_merak/aigc/flux3_klein/README.md#prompt-upsampler)
从原始权重构建，或组合已导出的独立组件；推理时显式启用 `prompt_upsampling`。
扩写后 caption 由 FLUX 重新分词，不直接复用扩写器 hidden states。支持文生图和多参考编辑，
数量由组合产物及实际 token 预算限制。实现说明见 [EXPLANATION.md 的扩写器部分](EXPLANATION.md#可选官方扩写器)，
此前执行结果见[扩写器接入与验证记录](../../../../docs/modeling/structures/flux3-klein-prompt-upsampler-validation.md)。

核心发布包包括四张 HMONNX 图、包内 tokenizer/processor、token embedding、源配置、文本配置、
`runtime_meta_info.json`、`export_meta_info.json`、有效 workflow 和公共 Manifest。
加入扩写器后另包含 `prefill`、`decode` 和全部 `visual_m*` 档位，以及扩写器独立的运行资源。
默认视觉档位为 256/512/1024，组合后共九图；上下文与 prefill 以 Qwen3.5 实际导出 metadata 为准。
加载只消费包内资源，不读取原 checkpoint，不回退到原生学习网络。

原生参考代码固定到 Black Forest Labs `e53f4ea697fe45db8f38fca91a90f5623f7f8f37`，
以 Apache-2.0 代码许可证保留，来源与修改范围记录在根目录 `THIRD_PARTY_NOTICES`。
原生类用于严格加载 checkpoint、构造部署 wrapper 和独立数值参考，不属于部署 runtime。
