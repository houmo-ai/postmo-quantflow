# Unified native fixtures

These JSON files describe deterministic **synthetic** models, not pretrained
weights. `_unified_small_model.py` builds native Transformers classes offline
with seed 4891. Configurations consolidate the September 2026 regression's
small Qwen, Gemma, MiniCPM and PaddleOCR models. Token IDs are remapped together
with the 512-entry synthetic tokenizer; no tokenizer or checkpoint is copied
from a workstation at test time.

- Qwen dense/MoE/Next retain linear and full attention, and MoE retains routed
  and shared experts. MTP contains a real decoder, not an empty placeholder.
- Gemma retains sliding and full attention and the registered media components.
  E2B/E4B retain per-layer embeddings and ordered assistant embeddings; 26B/31B
  have neither per-layer embedding enabled in their source architecture.
  Their sliding/full attention KV head counts remain different. Serialized
  Transformers `per_layer_config` must survive MTP contract validation and
  determine the draft graph's shared-cache inputs.
- PaddleOCR retains the 4304-wide visual MLP that exposed the packing bug.
- Auxiliary JSON stores unquantized tensor **shapes**, not old quantized data.
  Draft/LoRA tensors are generated deterministically and loaded by production
  loaders. DFlash and DFlash2 use distinct schemas.
- Ling reuses the separately attributed, pinned `../ling3_flash` source with
  both KDA and MLA layers. Its synthetic KDA `dt_bias` is explicitly initialized;
  the upstream constructor expects this parameter to come from a checkpoint.
  Calibration uses two local numeric-token rows and four AutoRound iterations.
  A poisoned-allocation regression checks finite, reproducible checkpoint weights.
  GLM reuses the native floating-model factory in
  `_glm53_packed_fixture.py`, with whitespace tokenization and two distinct
  64-token calibration rows. Unified build executes four real AutoRound
  iterations, and asserts saved loss/calibration/W4/W8 audit evidence. The
  separate packed fixture remains available only for format-consumer tests.
- Pi0.5 is constructed by `_pi05_small_fixture.py` using LeRobot 0.6.0 in the
  same Transformers 5.16.1 environment. It preserves two cameras, conditional
  expert norms, a 50-step action horizon and real normalization tensors. Only
  model width/depth and the denoising iteration budget are reduced. Its
  SentencePiece tokenizer is trained on a tiny local corpus during the test.
- Qwen-Image-2.1 is constructed by `_qwen_image21_small_fixture.py` with seed 13.
  It retains Qwen3-VL's three DeepStack branches, two complete DiT layers and
  the five-level RGBA VAE. Its 64-entry tokenizer and native processor are
  generated offline. The public build exports all six W8A8 graphs on CPU,
  then the test relocates the package, removes the source checkpoint and runs
  generation, reference editing and repeated generation. Reference dimensions
  are 32x64 and output dimensions 64x32 (height x width).

Norm vectors have small deterministic channel differences. This avoids
constant folding an all-identical synthetic vector into a scalar and keeps
the real norm input ABI visible. Tests do not disable graph optimization.

Kokoro and MOSS did not have complete synthetic-model probes in the original
regression. Their matrix entries execute the actual unified flow with
release assets; they are not counted as self-contained small-model fixtures.
Kokoro uses a SHA-256-pinned Artifactory bundle and cached download (or the
validated local override); MOSS uses the same pinned Artifactory download/cache
mechanism, with an optional `UNIFIED_CI_MOSS_SOURCE` offline override. Missing resources are failures, never successful skips. Kokoro's
production loader checks its pinned source commit and asset SHA-256 values.
Kokoro's CI fixture reduces its local profile to the smallest approved paired
route, T32/F160 (32 tokens, 160 frames / 4 seconds), while retaining all four
component roles and real build/load checks. The expected route geometry is
scoped to the worker, alongside its recipe/YAML geometry. The public default
still requires all four routes; `test_kokoro_unified_api.py` checks both that
contract and restoration after the fixture. This CI case does not establish
numerical execution coverage for the three larger gears.

Successful execution proves integration of the requested variant, not model
quality, real-size memory use, throughput or clean pip installation.
