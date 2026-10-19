# ModelZoo CI

`benchmark_test/ci_test_transformer5.16.0/run.sh` is the only active ModelZoo CI
entrypoint. It combines the retained BGE compatibility case, lightweight
recipe contracts, and SHA-pinned Transformers 5.16 block exports, then selects
only the tests affected by the current change.

## Run

```bash
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

The selector discovers local changes by default. When available, it consumes
merge-request metadata from the CI runner so the whole patchset is considered.
Developers do not need to export a base ref or a GPTQModel source directory for
the normal path.

Supported controls:

- `CI_TEST_CHANGED_FILES_FILE`: path to a newline-separated changed-path file;
  preferred in CI so large changes do not enter the process environment.
- `CI_TEST_HEAD_REF`: diff head, default `HEAD`.
- `CI_TEST_BASE_REF`: optional base branch/commit; Git discovery uses its merge
  base with the head so unrelated commits on the base branch are not selected.
- `CI_TEST_FORCE_ALL=1`: run every active CI test without bypassing frozen-path
  or fail-closed policy checks.

Use `select_tests.py --format json` (or `--format report`) to inspect every
selected test and reason without running pytest. The normal runner prints a
compact summary, including whether selection is incremental or full and the
reason for any full-suite fallback.

An explicitly supplied empty changed-path file means a known empty diff: only
the fast policy and recipe-onboarding gates run. An absent diff and unavailable
Git history mean an unknown diff: the registered suite runs in full. Invalid refs and unreadable
path lists fail with an error, rather than quietly running a different diff.
Changed-file lists must include both old and new paths for renames. Local Git
discovery disables rename collapsing to preserve both paths.

## Qwen-Image-2.1 environment preparation

Use the same configured CI Python environment as the other model cases. Before
running tests, install the selected requirements directly from the model recipe:

```bash
python benchmark_test/ci_test_transformer5.16.0/prepare_dependencies.py \
  --model-id qwen-image-2.1 --packages diffusers huggingface-hub --install
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

This installs the pinned Diffusers and compatible Hub releases using that
interpreter; it does not require a developer's conda environment name or a
`work_dirs` source checkout on `PYTHONPATH`. Without `--install`, the helper only
prints the selected requirements and command. It supplements the configured CI
environment and does not certify a complete clean installation of a blocked
dependency profile. Tests never install packages themselves. The image CLI and
dependency export checks have their own impact declaration; real three-precision
six-graph tests also run the CLI after relocating the package and deleting source
weights. Shared environment changes must be checked against the public variant
matrix, not only Qwen-Image-2.1 imports.

## Initialization state isolation

The common function-scoped fixture saves and restores the three Torch
initializers changed by GPTQModel and Transformers' legacy initialization flag.
Checkpoint loading retains its no-initialization optimization inside a test;
restoration runs on both success and failure. This isolates tests, not production
calls. A test that constructs a native model after loading a checkpoint must
request the `restore_model_initialization_state` fixture and scope that load
with `with restore_model_initialization_state():`, or use a separate worker. The unified flows already use independent workers.
Module/session fixtures and collection must not load checkpoints in the shared
pytest process: the function-scoped guard does not bracket those operations.

## Coverage policy

The selector separates global safety policy from test ownership:

- `impact_policy.json`: frozen/guarded roots, documentation handling, always-run
  policy tests and genuinely global changes (selector, runner, common workflow
  engine, base model, dependencies). Editing this file triggers the full suite.
- `<test-stem>.impact.json`: a test's explicit source/config inputs, stored next
  to its Python file. These declarations are auto-discovered only inside
  `benchmark_test/ci_test_transformer5.16.0`. Adding a model never requires
  editing global policy.

Selection uses declared subsystem boundaries, then falls back to reverse
Python imports and model-family discovery for files that have no declaration.
All declarations matching a changed file contribute their test: shared inputs
can therefore select several families. Explicit declarations suppress noisy
import/family inference for that input, so they must name all intended
representative tests. Import/name inference alone is not proof of coverage of
dynamically loaded models or YAML recipes.

Changes under guarded source/config roots fail before pytest if no relevant
test can be found, even with `--all`. Declarations are validated before any
selection: missing tests, empty inputs, duplicate JSON keys/rule IDs, invalid
paths and unknown fields are errors. A declaration cannot change global policy
or redirect itself to another test. Both incremental and full runs have a hard
test boundary: only Python tests inside `benchmark_test/ci_test_transformer5.16.0`
can be selected. `tests/`, other benchmark lanes and their declarations are
ignored; imports and model-name inference cannot enroll them or use them as
evidence of CI coverage. Put required CI coverage in the active directory.

### Onboard a model

First register a real `ModelRecipe` with a valid Workflow in
`xhmodel_merak/model_integrations/<family>/plugin.py`; a lower-level model
registration or standalone export script is not sufficient. Follow the
[integration checklist and complete example](../../xhmodel_merak/model_integrations/README.md#新模型怎样才算接入完成).

Add a runnable test, e.g. `test_new_model_export.py`, and an adjacent
`test_new_model_export.impact.json` (example placeholders below):

```json
{
  "version": 1,
  "rules": [
    {
      "id": "new-model-export",
      "patterns": [
        "xhmodel_merak/xh_llm/models/new_model/**",
        "xhmodel_merak/model_integrations/new_model/**",
        "configs_merak/workflows/**/new_model/**",
        "examples_merak/llm/new_model/**"
      ]
    }
  ]
}
```

Patterns are positive, repository-relative Python `fnmatch` globs; all matching
rules are combined. No negation/absolute paths are accepted. Keep patterns as
narrow as the test actually exercises. Add any shared helpers the test relies
on, and add the shared input to each existing affected test's declaration.
No list of target tests is needed: the declaration's filename determines its
one owner. Both `test_*.py` and `*_test.py` names are supported.
An optional `nodes` list restricts a rule to local pytest functions or parameter
instances, for example `"nodes": ["test_flow[pi0.5-libero/lerobot-h50]"]`.
The selector prefixes each node with the owning test path and sends those exact
node IDs to pytest. Nodes cannot redirect ownership to another file. Matching
rules form a union; a whole-file/function selection supersedes its individual
nodes to avoid duplicate execution. Editing/deleting the declaration or editing
the test itself still selects the whole surviving file. Unknown diffs and
forced/global full-suite selection also run the complete file.

| Change | Selected tests |
| --- | --- |
| New model + recipe + test + declaration | New target and both always-run gates |
| Edit declaration inputs | Its owner, even without a source change |
| Delete declaration | Its surviving owner; other unmapped source changes still fail closed |
| Rename declarations | Surviving owners at both old and new paths |
| Retire test and declaration together | Remaining affected tests; removed tests are not executed |
| Shared input | All declared consumers |
| Global policy/engine/environment or unknown diff | Full registered suite |

Because ownership is immutable and filename-derived, changed-path-only Jenkins
checkouts need no baseline JSON or Git history to select old/new owners. Keep
the test and declaration together when moving or retiring a target. Changing
the inputs cannot hide that target from the current run.

The old combined `impact_rules.json` schema/CLI option is removed. Existing
input-to-test edges targeting the active suite have been migrated into test-local
declarations. Mappings to tests outside this CI directory have been removed. This
one-time selector/policy refactor itself triggers full CI; subsequent model
onboarding and declaration edits remain incremental.

### Recipe onboarding gate

`test_recipe_coverage.py` is always selected, including for documentation-only
and known-empty diffs. It enumerates model packages and uses the production AST
scanners to enumerate master model types, then follows every discovered recipe's
Workflow YAML back to those types. This catches a new implementation with no
recipe even when it has an export test and a valid impact declaration. Missing
Workflow files, unknown/auxiliary targets and dynamic registration also fail.
No model imports, weights, GPU, network or Git history are needed.

`recipe_legacy_gaps.json` records the pre-gate backlog at its `frozen_at` revision:
`family/package/` is an uncovered directory, `family/package:Type` is an uncovered
master type. All entries are exact, never wildcard exemptions. Do not extend this
inventory for new work. Resolving or removing an old model requires deleting its
stale entries; CI checks that the remaining inventory matches the actual gaps.

The gate checks registration completeness, not numerical correctness or whether
a new checkpoint using an existing architecture deserves a new public model ID.
Those require the model's functional tests. Its impact declaration deliberately
does **not** cover `models/**` or all Workflow YAML: such a rule would suppress
functional import/family fallback. Do not use this cheap gate as functional
coverage. Model-specific onboarding still never changes global policy.

### Test design: complete coverage with small cases

All new tests belong in this directory, including lightweight unit contracts,
operator/structure regressions and model exports. Do not add cases to `tests/`,
`test/` or another benchmark lane. Extend an existing focused case when it
already owns the behavior; add an adjacent impact declaration for a new case.

Prefer the smallest real implementation that exercises the changed behavior:

| Behavior under test | Preferred fixture |
| --- | --- |
| Configuration, preprocessing or input ABI | Small tensors and the real loader/factory; no checkpoint or HMONNX session |
| Attention, norm, MoE or cache semantics | A minimal model built from a small config and deterministic weights; keep the actual affected layers |
| Graph lowering or export regression | A minimal model/subgraph through the real export path, with graph and output assertions appropriate to the bug |
| Checkpoint/quantized-weight compatibility | The existing immutable block artifact and shared export harness when synthetic weights cannot reproduce the format |

Small means less computation, not fewer semantics. Preserve the head grouping,
rotary dimensions, expert routing, layer kinds or cache transitions that make
the case meaningful. A hybrid model may need both attention kinds; a shared
cache bug may need multiple layers. Do not replace the operation being checked
with a mock, set zero layers, or shrink away the failing condition. Stub only
unrelated expensive boundaries, and state what the test does not cover.

Cover all affected behavior with compact parameterized cases: a valid baseline,
relevant shape/dtype boundaries, invalid inputs, and state transitions where
applicable (prefill, decode, continuation, reset, accepted/rejected verify).
Use distinguishable deterministic values so layout/expert-order errors remain
visible. Reuse fixtures and compare against a reference or public contract;
avoid repeated large-model setup, redundant cases and a Cartesian product of
unrelated dimensions. Full coverage means the changed contracts are exercised,
not that every model size is exported.

Keep PR CI fast: prefer small native models or the block harness for a necessary
real export. Full-model downloads, re-quantization or golden generation require
a concrete regression that smaller layers cannot catch; retain the existing
BGE compatibility exception. DepthSplat uses its smallest registered fixed
2v/256×256 profile to verify actual neural PTQ, Host camera preprocessing,
artifact relocation and source-independent inference together. Its immutable
137 MiB ONNX/input bundle is fetched once from the internal mirror and cached;
it does not download a checkpoint or install upstream renderer dependencies.
Markers describe categories; the selector and impact declarations determine
execution. Do not make a required case silently pass by skipping unavailable
dependencies.

Confirm selection with the actual patchset (or repeat `--changed-file` for a
small selection probe), then run the CI entrypoint from the repository root:

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

For focused debugging, preserve the runner's pytest isolation. For example,
the selector-policy suite itself is a small test with no model dependencies:

```bash
python -m pytest \
  -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py
```

Use the configured CI environment for model/SDK cases. A focused run is useful
for debugging but does not replace the selected patchset suite. Report selected
tests, outcomes, elapsed time and any unexecuted coverage separately.

### Design references

The selector is repository-specific, standard-library-only code; no
external CI action or build system is required. The design borrows these ideas:

- [dorny/paths-filter](https://github.com/dorny/paths-filter): named path filters,
  explicit change-list inputs and merge-base-aware branch comparisons.
- [Nx affected](https://nx.dev/docs/features/ci-features/affected): map changed
  inputs to owned targets and their dependents, with conservative handling of
  truly global dependency changes.
- [Tinder/bazel-diff](https://github.com/Tinder/bazel-diff#how-it-works): treat
  dependency/rule changes as target-level changes. Bazel compares graph
  snapshots; our fixed test/declaration ownership avoids needing snapshots for
  declaration edits in source-only Jenkins checkouts.

Only the refactored implementation roots `xhmodel_merak`, `examples_merak`,
and `configs_merak` participate in impact selection. The legacy roots
`xh_model_zoo`, `xh_model_zoo_develop`, `examples`, `examples_develop`, and
`configs` are frozen: any change below them is rejected before test selection,
including when `CI_TEST_FORCE_ALL=1`. Existing tests may still execute frozen
code as a compatibility contract; that does not authorize changes in those
directories.

Documentation-only changes still run the fast selector-policy and
recipe-onboarding tests, so the CI status is reported instead of disappearing. The removed `ci_test` and
`ci_test_transformer5.5.0` trees are kept as explicit tombstones in the impact
rules. Their deletion paths run both fast gates; the selector-policy test also
rejects any later recreation of either retired directory. Other deleted
guarded production/config files remain fail-closed.

## Transformers 5.16 boundary

The directory name tracks the Transformers 5.16 CI boundary. The CI runtime is
pinned to Transformers 5.16.1.

For the existing block export tests, AutoRound runs once when a block checkpoint
is published. Those tests do not quantize weights in pull-request CI. They download
the immutable checkpoint, verify the
archive SHA-256, and runs the current ModelZoo export path. It does not call
`dump_golden` or create `golden`/`step_*` payloads. The maintained real export
matrix is:

- Qwen3.5-9B: 4 text layers;
- Qwen3.5-35B-A3B: 2 representative text layers;
- Gemma4 E2B: 6 text layers;
- Gemma4 26B-A4B: 2 representative text layers.

The BGE-Reranker case downloads the full model when its cache is absent and
runs the existing W8A8 quantization, HMONNX export, and golden generation path.
The former Qwen3-0.6B and Qwen3-30B-A3B subprocess exports are retired.

For the same four Transformer models, a cheap GPTQ/AutoRound contract layer
checks YAML-to-recipe translation, Python signatures, generated command parser
compatibility, and the public workflow `quant()` / `export()` interface. The
selector maps shared family code to both representative block exports and
model-specific workflow YAML to only that model's export.

These contracts verify integration, not quantization accuracy or the effect of
every accepted argument. Algorithm and producer-side save/reload tests belong
in GPTQModel; new ModelZoo consumer tests stay in this directory. For an external
GPTQModel interface/dependency update, explicitly run the relevant contracts and
consumer tests: this repository's selector cannot detect another repository's
changes. See [development responsibilities](../../docs/architecture/quantization-boundary.md).

Ling MLA loader tests build a one-layer MLA/MoE model with deterministic small
weights and real W8/G64 packing. The [bundled fixture](fixtures/ling3_flash/README.md)
contains pinned, checksummed upstream model code and a reduced config, not
pretrained weights. Its CI-only source changes defer FLA imports to KDA, which
this fixture does not construct. No FLA installation, model directory, HF
download, or quantization training is required. Normal and streamed loading
still use the production implementations; a fresh-process regression blocks
FLA imports and repeats these paths with empty HF caches in offline mode.

MiniCPM-V-4.6 has a CPU-only text-preprocessing interface contract in
`test_minicpm_v_4_6_preprocess_contract.py`. It calls the real export/runtime
preprocessor factories with small tensors, without downloading checkpoints,
loading HMONNX sessions, or generating golden payloads. It covers embedding
padding, visual-feature scattering, sequential positions, chunk continuation,
decode, cache ordering, and the page-attention input ABI. The two MiniCPM
callers and their shared Qwen3.5 preprocessing/cache helpers select this test;
this is not a full MiniCPM vision/export/runtime correctness suite.

## Unified model and variant regressions

Each `test_unified_model_<model>.py` declares its model, preparation module and
variants in a literal `CASE_SPEC`, with its own `.impact.json`. `_unified_cases.py`
discovers these declarations without importing models. There is no manually
maintained total count or second model inventory. `test_unified_small_model_coverage.py`
compares these independent executable cases with the registered recipe variants,
checks their real pytest collection, and validates the shared Transformers 5.16.1
environment.

Add a model's case and impact beside the existing model files; reuse or add a
preparation helper in `_unified_fixtures/`. Declare every actual shared consumer.
Changing a model's case/manifest selects that model; shared build/load and runner
changes select all consumers. Report formatting only selects report contract tests.
One session-scoped scheduler owns the selected cases across all model files and
runs at most one worker per visible GPU. Each selected case runs `build_model`, moves the artifact, removes its
generated source/checkpoint directories, and runs `inference_model` in a fresh
process. Failures retain `worker.log` and `result.json` in pytest's temporary
directory, including the failing stage, traceback and installed versions.

This suite intentionally exercises quantization on newly generated tiny native
models, in addition to the existing immutable block export tests. Calibration
uses two samples and four AutoRound iterations where supported, with one held-out
MTP validation sample. Model dimensions are reduced; registered workflows,
precisions and export paths are preserved. GLM starts from a native floating
checkpoint and runs the real GPTQModel AutoRound recipe; the test checks actual
calibration tokens, tuning losses and W4/W8 module audits in the saved provenance. See [fixture details](fixtures/unified/README.md).
Macaron exposes `full` and `lora`; the LoRA case exports three adapters through
the shared Qwen pipeline. `test_qwen35_moe_lora_low_memory.py` exercises the same
three adapters with streamed export. Both paths inspect final MoeBlock operands,
compare Prefill/Decode adapter weights, reject leftover placeholders, and load
all three relocated views. `test_qwen35_moe_lora.py` additionally compares the
PEFT expert layout and round-trips W4/W8 LoRA through the SDK exporter/runtime.

These checks exercise integration contracts, not model quality or full-size
memory/performance requirements.

For source-based CI, keep the tested ModelZoo, SDK and GPTQModel checkouts on
`PYTHONPATH`. Dependency checks read ModelZoo (`xhmodel-zoo==1.0.0`) and SDK
(`xhquanttool==0.9.2`) versions from the `pyproject.toml` next to the imported
package, ahead of stale editable-install metadata. Wheel installs use their
distribution metadata. The SDK's legacy runtime `__version__` is not its package
release version. Missing GPTQModel distribution metadata does not justify
replacing the modified checkout with a public PyPI package.
Quantization also pins `datasets==3.6.0`: the CI image's 5.0.1 failed while
fingerprinting even a one-row calibration dataset (`MonthDayNano` pickling).
Install with dependency resolution and run `python -m pip check`; checking only
the direct package versions does not validate dataset creation or model execution.

Pi0.5 uses the same Python environment: Transformers 5.16.1 and the bare
`lerobot==0.6.0` package. Do not install the `lerobot[pi]` extra, whose Transformers
constraint differs. Pi0.5 weights and tokenizer are generated locally. Kokoro downloads its pinned
release inputs from the internal Artifactory on the first selected run, then
reuses a local cache. The archive includes the original checkpoint, config,
`zf_001` voice and pinned Kokoro source checkout (including its license).
`_unified_assets.py` verifies the archive SHA-256 before safe extraction and
checks the release file hashes/source commit before use. It never downloads
model weights from Hugging Face or ModelScope during CI.

- Default object: `http://10.10.1.53:8082/artifactory/model_zoo2/unified_ci_assets/kokoro/704324aeaac896e830f964f36e7a19002c134440d51c9124e582ca8460f4e13c/kokoro-ci-source-v1.tar.gz`
- Archive SHA-256: `704324aeaac896e830f964f36e7a19002c134440d51c9124e582ca8460f4e13c` (357,694,336 bytes).
- Default cache: `work_dirs/unified_ci_assets`; override with `UNIFIED_CI_ASSET_CACHE`.
- Mirror root override: `UNIFIED_CI_ARTIFACTORY`; the object path/hash remain pinned.
- Offline source override: `UNIFIED_CI_KOKORO_SOURCE`, validated with the same release checks.

MOSS also downloads a SHA-256-pinned release bundle from Artifactory on its first
selected run, using the same cache, mirror override and inter-process lock.
The bundle contains the original ModelScope weights, tokenizer/processor files,
README and Apache-2.0 license. `fixtures/moss/release.json` records the upstream
per-file revisions, sizes and SHA-256 values; downloads, cached files and local
overrides are verified. CI does not access a model hub.

- MOSS object: `http://10.10.1.53:8082/artifactory/model_zoo2/unified_ci_assets/moss/1e58b1d08c1bdd93e1ec025b0b29dad8dcc37ce29318132ca5088f97b4946dc5/moss-ci-source-v1.tar.gz`
- Archive SHA-256: `1e58b1d08c1bdd93e1ec025b0b29dad8dcc37ce29318132ca5088f97b4946dc5` (1,463,848,620 bytes).
- Extracted root: `moss-ci-source-v1`; default cache and mirror override are shared with Kokoro.

`UNIFIED_CI_MOSS_SOURCE` is now an optional offline override, pointing to the
extracted bundle directory. Missing inputs and corrupt downloads fail rather
than skip. Package-version reporting tolerates missing distribution metadata
(recorded as null); original `ModelError` codes/details are printed and saved
in `result.json` without changing test pass/fail status.
Kokoro builds only the smallest T32/F160 gear in this CI fixture, retaining all
four components and relocation/inference. The published four-gear default and
its rejection of missing gears remain covered by the fast Kokoro contract tests;
larger gears are not numerically exercised by the unified smoke test.

DepthSplat similarly downloads its pinned 137 MiB source bundle automatically.
The final 2v 256×256 pretrained ONNX graph uses Host FP32 camera matching-grid
preprocessing and Gaussian postprocessing; calibration and held-out samples
come from separate real RE10K scenes. The archive includes source/checkpoint
provenance and the upstream MIT license. Its SHA-256 and each extracted file's
size/hash are checked, including cached or overridden inputs. The full case
runs actual W8A8 PTQ, artifact relocation and HMONNX inference. It needs no
upstream renderer or manually configured model path. See
[the asset manifest and mirror details](fixtures/depthsplat/README.md).
`DEPTHSPLAT_CI_BUNDLE` remains an optional override for a complete extracted
release archive and cannot bypass the same integrity checks.

With the CI environment activated, run:

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./benchmark_test/ci_test_transformer5.16.0/run.sh -k 'unified_small'
```

Ordinary pytest schedules one isolated worker per visible GPU; do not add xdist
workers to this module. Use `-k` to narrow the model/variant selection. A worker
has a 3600-second timeout, configurable with `UNIFIED_CI_TIMEOUT`; a timeout kills
its process group before the GPU is reused. Its `.impact.json` selects parameter
nodes for model-private source/adapter changes, and all variants of the specific
model for its workflow configurations. Shared Qwen source also selects its MoE,
Qwen3-Next, Macaron and WeMM consumers. Public unified build/inference/catalog
and shared fixture changes select the entire registered case matrix. Multiple changes
are combined; `--format report` shows the exact pytest nodes and reasons.
`test_unified_small_model_coverage.py` checks registered workflow routing, shared
consumers and the actual pytest collection of every declared node. Global policy
is unchanged; changing the selector itself still triggers the full CI suite.
