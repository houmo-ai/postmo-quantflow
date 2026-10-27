# MiniCPM-V-4.7 CI source

The two Python files are copied unchanged from the Transformers 5.18.0 wheel.
`source.json` records the wheel and individual file SHA-256 digests. Copyright
2026 OpenBMB and the HuggingFace Inc. team; Apache-2.0, with the full license in
`licenses/Transformers/LICENSE`.

The active CI environment remains Transformers 5.16.1. `_minicpm_v47_fixture.py`
loads this model package only for its consuming test module and restores the
module and Auto registries afterwards. It does not install or upgrade packages.
Tests construct tiny native Dense/MoE and vision models with deterministic
weights; no pretrained weights or downloads are needed.

Source: https://pypi.org/project/transformers/5.18.0/
Upstream model: https://github.com/huggingface/transformers/tree/v5.18.0/src/transformers/models/minicpmv4_7
