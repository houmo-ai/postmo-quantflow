# Ling CI 小模型

一层真实 decoder：gated MLA＋两个 routed experts 和一个 shared expert，不构造 KDA。
hidden size 为 64，词表 128；保留 32 个 head，满足 W8/G64 的 gate 打包对齐。
测试用 seed=31 初始化权重，现场打包三个融合 Linear，不运行 GPTQ/AutoRound 训练。

这两个 Python 文件来自 [Ling-3.0-flash 的固定版本](https://huggingface.co/inclusionAI/Ling-3.0-flash/tree/e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3)。
配置源码未改；建模源码只将 FLA 导入移到 KDA 构造/执行处，并去掉未使用的 FLA 导入和
`tensor_cache` 缓存装饰器。MLA、MoE 的实现不改。上游与本地 SHA-256、修改范围见
`upstream.json`，加载前检查本地源码摘要。
`config.json` 缩到 MLA＋MoE，保留 head-wise gate、RoPE 和专家路由配置。
不包含原始权重、tokenizer 或校准数据；不读取 `LING3_FLASH_MODEL_DIR`，不联网下载。

保留真实代码是为了执行生产侧普通加载和分层加载，避免用假模型替代被测结构。
测试验证三个融合 Linear 的浮点权重和整数载荷一致，不代表完整模型精度或 KDA kernel 验证。
额外用全新子进程禁止导入 `fla`、清空 HF 缓存并禁用联网，重跑实际打包和两个加载器。
如果误构造 KDA，仍明确报缺少 `fla-core`；没有假 FLA 模块或 KDA fallback。

建模源码保留其 Apache-2.0 版权头；无单独文件声明的配置源码遵循上游 model card 的 MIT 声明。
对应文本见 [Apache-2.0](../../../../licenses/Ling-3.0-flash/LICENSE) 和
[MIT](../../../../licenses/Ling-3.0-flash/LICENSE-MIT)。
