# Sol-H3 咖啡馆示例素材

这些文件供咖啡馆女孩的五模式示例使用。首尾图来自已验证样例；
两张额外参考图和视频于 2026-11-03 增强后替换，音频保持原样。
示例命令和提示词见 [Sol-H3 README](../../examples_merak/aigc/sol_h3/README.md)。
下表中的尺寸顺序为宽×高；SHA-256 对应当前随库文件。

| 文件 | 用途和尺寸 | SHA-256 |
|---|---|---|
| `cafe_first_hd.png` | 首帧、Ref2VA 第 1 图；1659×948 | `3cadcac048329047bdbebc52a9ca8750b4676889e4c0e63bd99a1ec709d79723` |
| `cafe_last_15s.png` | 尾帧；1344×768 | `b91e208dd575eb1c447c4fce1526b91c20f55693310328bf823efac140b2167e` |
| `cafe_ref_middle.png` | Ref2VA 第 2 图；1659×948，生成式修复 | `26d4a1c768d89ba7654f928f9bd36c570856760eaedc98688b90cb3a1777f5e8` |
| `cafe_ref_last.png` | Ref2VA 第 3 图；1659×948，生成式修复 | `7760febfc5ef737487ea8398760a4f709bf7e1d58733e02dd52d2b7197ef054a` |
| `cafe_reference.mp4` | Ref2VA 视频；1344×768，24 fps，120 帧，5 秒；原音轨 | `5dcd8562b2fd593957ae5e71e5a5c388de029dc40e0b3682b1e3175d4f385c77` |
| `cafe_reference.wav` | Ref2VA 独立参考音频；5 秒，32 kHz 双声道 | `95ded17f42811e09d7593c28cd9fe7f4b7454155f74121ff6134116ca2f6f1ab` |

首图来源为本地验证记录 `work_dirs/sol_h3_i2va_hd_20261102/inputs/first_restored.png`；
它是从 672×384 原图经 image_gen 生成式修复得到，细节有合成成分，并非真实高分辨率原片。
原图记录的 SHA-256 为 `f403a2f1d52ab672721e8d2cf15b07ddd9a58a4df49cb555d28e06d83795328c`。
尾图来自该首图生成的 15 秒 I2VA 样例的第 360 帧
（`work_dirs/sol_h3_15s_long_dialogue_20261102/i2va_long_s43/frame_360.png`）。

两张额外参考图、视频和音频的原始版本来自
`work_dirs/sol_h3_5s_15s_20261102/validation_cafe_20261102/inputs/`，
原文件名依次为 `middle.png`、`last.png`、`reference.mp4`、`reference.wav`。
其中后三者的更早来源记录为 `Sol-H3-assets/reference_media/realistic_precision_modes_20261022/media/`；
`middle.png` 的更早来源未在该记录中注明。这些 `work_dirs` 是本地验收记录，不是示例运行时依赖。
参考视频本身有音轨，独立音频另占一个参考输入；模板将它们描述为音色参考，输出仍需验收。

两张额外参考图原为 672×384，使用内置 image_gen 修复，以高清首图辅助人物细节，
保留各自的眼睛开合、嘴形、头部角度和手势。原始 SHA-256 分别为
`713fb5080f2d9b82c3f493ea8fb0e01493bc20180ed800fdad7dd20b852fa94a` 和
`ebea2d70cbeba9445f00bdf6a840d82b77e0aaa13d88e0f745f3b58073cf66a3`。

参考视频使用 [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) 的
[官方非 GAN 权重](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207.pth)
对完整 120 帧进行时序超分，4 倍结果缩到 2 倍，再混合 25% 原帧放大结果，
以 H.264 CRF 15 编码。全部视频帧的 PTS、duration 保持一致；
原 AAC 的 158 个包逐包保留，包含包内容、时间戳及 skip-samples 信息；独立 WAV 未处理。
权重 SHA-256 为 `0e2ff2070a160bf19b26c194f82a0ee2ca8c6c4395a2bf927c768de048f156c1`，不随库分发。
原视频 SHA-256 为 `524d99b4e6e10a99d98bb33ffda391883a3494a5a571a7c89c9ae50e3fda5251`。

增强细节由模型推断，不等于原生高清。Ref2VA 仍按固定档位缩放，增强不增加输入 token 预算；
增强后的参考素材尚未单独重跑完整 Ref2VA 质量对照。
原始文件和处理记录保存在本地 `work_dirs/sol_h3_release_cleanup_20261103/`，不影响示例独立运行。
