# 统一 API 发布产物约定

本约定覆盖 LLM/VLM、AIGC、ASR、audio、TTS、VLA，沿用
[HM 发布命名规则](https://houmo.feishu.cn/wiki/WbX8wq9tpihXeHkzAy7caqMZnUh)。
适用于 `build_model` 的交付目录；workflow 的中间工作区不直接当成交付包。

## 目录与命名

`output_dir` 就是发布根目录。组件采用小写角色名加实际输入档位，并在根目录平铺。
不添加无意义的 prefill/context 长度，不用芯片名称推断组件类型。

| 模型/组件 | 发布目录 | 说明 |
| --- | --- | --- |
| 自回归文本 | `prefill/`、`decode/` | 有实际缓存预填充/增量解码语义 |
| ASR 编码 | `encode/`、`encode_<档位>/` | 用 `encode`，不用 `encoder`；内部 Python 类名不受此限制 |
| 视觉编码 | `visual_<尺寸或token档位>/` | 用 `visual`，不用 `vision`；如 `visual_448x448x2`、`visual_m256` |
| 音频编码 | `audio_encode_<档位>/` | 多编码器用语义前缀区分，不强行叫 prefill |
| TTS | `text_duration_t0032/`、`frame_acoustic_f0160/`、`phase_core_f0160/`、`generator_istft_f0160/` | 按实际角色和 T/F 档位命名；CPU 相位图单独保留 |
| AIGC | `text_encode/`、`denoise_<档位>/`、`vae_encode/`、`vae_decode/` | 只交付实际使用的角色；VAE decode 不属于自回归 decode |
| VLA | `visual_<档位>/`、`gemma_prefill/`、`gemma_decode/`、`expert_prefill/`、`expert_decode/`、`action_in_proj/`、`action_out_proj/`、`time_mlp/` | 多主干用语义前缀区分，不用 `Other/` 聚合不相关图 |
| 投机解码 | `mtp_prefill/`、`mtp_decode/`、`decode_m2/`，或 family 既有 `mtp_draft_*/`、`dflash_draft_*/`、`dspark_*/` | 草稿预填充、草稿解码和目标验证按真实角色平铺；DSpark 保留 context、context_decode、draft_decode、markov |

每个组件目录放实际图与其 external_data 权重。图文件沿用发布前缀；不能只改权重文件名而
不更新 ONNX 引用。尺寸含义和轴顺序由模型 README 说明，不凭目录数字猜测。
同一组件存在多个部署档位时分别登记；参考浮点图不替代实际运行的 HMONNX。

Qwen4-Exp 的 MTP 运行契约 v3 使用三张独立图：`mtp_prefill/` 的序列长度与目标
prefill chunk 一致，`mtp_decode/` 的序列长度为 1，`decode_m2/` 的目标 verify
长度为 2，对应每轮一个候选。运行 metadata 的 `spec_decode.draft` 声明两张草稿图与
档位，`spec_decode.target` 声明 verify 图与档位；不能把同一单步图登记成三个角色。
目标主模型的 `prefill/`、`decode/` 仍独立交付。三张目标图的外置 PLE 资源分别在
`qwen4_exp_ple.prefill`、`decode`、`decode_m2` 登记，所有引用都相对发布根目录。
草稿图不携带目标的 PLE 表。旧 v2 的单步 MTP 包按原契约读取，不伪报为 v3。

DeepSeek V4 和 GLM5.3 Flash 的统一导出将主模型 prefill/decode 中同类型、同形状、
同内容的外部张量流式合并。两个阶段各保留 `weights.bin`，通过硬链接共享磁盘数据；
图只引用阶段内的文件，不引入根目录权重或专用加载协议。DeepSeek 的 DSpark 组件
保持独立，不参加这次整理。其他模型不自动启用该步骤。
交付使用 `tar -cf model.tar -C <父目录> <产物目录>`，解包用 `tar -xf model.tar`；
不要使用 `--hard-dereference`。普通 ZIP 不能保留这份硬链接共享关系。

```text
<output_dir>/
├── encode/<release_prefix>_encode_with_act.onnx
├── prefill/<release_prefix>_prefill_with_act.onnx
├── decode/<release_prefix>_decode_with_act.onnx
├── hf_config/                 # tokenizer、processor、模型文本配置及本地依赖
├── quant_embedding.pt         # 仅实际存在独立 embedding 时
├── quant_embedding.bin        # 构建生成的小端 FP16 原始数据
├── export_meta_info.json      # 集成商入口清单
├── runtime_meta_info.json     # family 运行契约（或既有 golden_meta_info.json）
├── effective_workflow.yaml
└── model_build_manifest.json  # 来源、参数、交付文件大小和摘要
```

构建不生成 Golden；显式生成时 `step_N/` 放在对应组件下。日志、临时子图、校准工作文件、
无运行用途的浮点参考图不进入交付。保留必要 Host 常量；不要把 voice pack、归一化统计或
lm_head 权重伪装成 tokenizer/embedding，它们仍在运行元数据与 Manifest 中登记。

Qwen4-Exp v3 的 MTP Golden 沿真实运行分别保存 `mtp_prefill/step_N/`、
`mtp_decode/step_N/` 和 `decode_m2/step_N/`，每步包含该图实际输入、输出及执行前后
缓存状态。选中 MTP 采集时，三图都执行后才能记录完成；缺少图或使用其他阶段的
数据不能作为该角色的 Golden。Host PLE 的 packed/指数二进制、索引样本和 NumPy
重放脚本随 Golden 一起保存，verify 对应 `decode_m2` 的 PLE 资源与输入样本。

## export_meta_info.json

根对象是 `组件或资源名称 -> {path, type}`；保留 `lora` 为 checkpoint 分组。
`path` 始终相对整个发布根目录，图指向具体文件、tokenizer 指向目录。
禁止绝对路径、`..`、反斜杠、软链接、空路径和不存在的资源。
组件名唯一且采用小写；checkpoint 名使用实际导出身份，不允许重复。

| type | 含义 |
| --- | --- |
| `prefill` | 实际自回归 prefill 图 |
| `decode` | 实际自回归 decode 图 |
| `hmonnx` | 其余 HMONNX 部署图，包括 encode、visual、denoise |
| `onnx` | 实际运行的标准 ONNX，如 Host FP32 相位图 |
| `tokenizer` | 完整 tokenizer/processor 配置目录 |
| `embedding` | 独立、实际使用的 embedding 文件 |
| `json` | 随包交付的运行元数据、档位描述和 JSON 配置文件 |

embedding 的 `.pt` 和构建实际生成的 `.bin` 分别登记，类型均为 `embedding`；例如
`quant_embedding` 与 `quant_embedding_bin`。公共发布器使用 embedding 转换结果确定对应路径，
不猜文件名或登记未生成的文件；LoRA 的两种格式保留在同一 checkpoint 分组内。
VLA 也采用同一规则，即使实际路径仍为 `Gemma2B/token_embedding.bin`。
`.bin` 的 shape、dtype 与字节序从 Manifest 的 `artifact.embedding` 读取。

`prefill/decode` 优先表达运行阶段，其余图用真实格式；不能从 `.onnx` 扩展名判断是否
HMONNX，也不能因为目录含 `decode` 就把 VAE 或声码器标为自回归 decode。
清单是集成入口索引，不是所有文件列表，也不含 shape、release_prefix、配置等混合字段。
完整依赖、外部权重及校验信息由运行元数据和 Manifest 承担。

发布器在内部元数据改名后逐项登记随包 JSON，包括 `runtime_meta_info.json`、
`golden_meta_info.json`、`merak_config.json`、`visual_gears.json`、组件内部 JSON 以及
`hf_config/` 下的配置/词表 JSON；tokenizer 目录条目仍保留。
根目录的清单自身与外层构建清单 `model_build_manifest.json` 不登记，后者在本清单写入后
生成并记录其摘要。组件子目录内同名的 `export_meta_info.json` 是普通运行资源，仍须登记。
JSON 条目必须指向存在、可解析的 JSON 文件，并纳入 Manifest 校验。

默认键名取去掉 `.json` 的相对路径，各目录层级以 `__` 连接并转小写；不支持的键字符转为
`_`，非字母数字开头加 `json_` 前缀。重名必须报错，不能覆盖组件或其他 JSON。
LoRA JSON 放入对应 checkpoint 分组，键相对该 checkpoint 生成，`path` 仍相对发布根目录；
family 用 `export_json_roots` 显式声明 checkpoint 的资源根目录，公共发布器不猜目录归属。

```json
{
  "encode": {"path": "encode/model_with_act.onnx", "type": "hmonnx"},
  "prefill": {"path": "prefill/model_with_act.onnx", "type": "prefill"},
  "decode": {"path": "decode/model_with_act.onnx", "type": "decode"},
  "tokenizer": {"path": "hf_config", "type": "tokenizer"},
  "quant_embedding": {"path": "quant_embedding.pt", "type": "embedding"},
  "quant_embedding_bin": {"path": "quant_embedding.bin", "type": "embedding"},
  "runtime_meta_info": {"path": "runtime_meta_info.json", "type": "json"},
  "hf_config__config": {"path": "hf_config/config.json", "type": "json"},
  "lora": {
    "checkpoint1": {
      "prefill": {"path": "checkpoint1/prefill/model_with_act.onnx", "type": "prefill"},
      "quant_embedding": {"path": "checkpoint1/quant_embedding.pt", "type": "embedding"},
      "golden_meta_info": {"path": "checkpoint1/golden_meta_info.json", "type": "json"}
    },
    "checkpoint2": {
      "prefill": {"path": "checkpoint2/prefill/model_with_act.onnx", "type": "prefill"},
      "quant_embedding": {"path": "checkpoint2/quant_embedding.pt", "type": "embedding"}
    }
  }
}
```

示例只展示结构；生成器须列出每个 checkpoint 实际使用的全部组件、tokenizer 和 embedding，
不能登记未导出的资源。共享资源允许多个条目指向同一路径。未使用 LoRA 时省略 `lora`。

## 实现边界与迁移

family 的 `validate_artifact` 返回 `export_entries`，从已校验的运行 metadata 提取真实
组件、格式和路径。公共 executor 在发布前校验并写清单，再生成 Manifest；失败不发布。
禁止公共发布器按模型名分支或扫描文件名猜组件角色。

原先占用 `export_meta_info.json` 的内部元数据在统一发布时移到 `runtime_meta_info.json`；
Manifest 的 `artifact.metadata_file` 指向真实运行契约，`artifact.export_meta_info` 指向
集成清单。底层 workflow 的工作区元数据可以保留原名称，不能将其误作公开清单。
已有自回归 `golden_meta_info.json` 继续作为其运行契约。

布局迁移必须同步更新图引用、运行元数据、build/load adapter、示例和当前 CI。
新接入执行上述布局；历史 workflow 的大小写和嵌套目录必须逐项迁移并验证，不能在文档中
宣称已全部完成。集成商始终以清单 path 读取，不能自行拼接目录或要求重新解释 Python 类名。
