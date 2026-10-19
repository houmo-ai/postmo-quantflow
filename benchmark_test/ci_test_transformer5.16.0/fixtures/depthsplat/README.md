# DepthSplat unified CI source

The unified full case downloads this immutable source automatically through
`_unified_assets._cached_source`. No developer-local path, model-hub login,
upstream Python packages, or renderer installation is needed.

- Object: `http://10.10.1.53:8082/artifactory/model_zoo2/unified_ci_assets/depthsplat/c7a459085fc5fab67230e847e7cb5a34c35219660bf69f724f38b7d3635dbf1f/depthsplat-ci-source-v1.tar.gz`
- Archive SHA-256: `c7a459085fc5fab67230e847e7cb5a34c35219660bf69f724f38b7d3635dbf1f`
- Archive root: `depthsplat-ci-source-v1/`.
- Default cache: `work_dirs/unified_ci_assets`; override with `UNIFIED_CI_ASSET_CACHE`.
- Mirror override: `UNIFIED_CI_ARTIFACTORY`, following the shared asset helper.
- Local source override: `DEPTHSPLAT_CI_BUNDLE`, pointing to the **complete
  extracted archive**. The same per-file SHA-256 and size checks apply.

`release.json` pins every file. The archive contains the final static 2-view
256×256 neural encoder ONNX, source config, disjoint real calibration and
held-out evaluation RGB/camera samples, upstream MIT license, and provenance.
Source commit: `2dad25a7b9ba7c5537aba08463ef88d8511a06be`.
Original pretrained checkpoint SHA-256:
`cfeab6b10077ad9bd4097b82b029737221bfbbbd3b4dfca5fc0feaf860276506`.
The graph and samples exactly match the Host FP32 matching-grid preprocessing
and Gaussian postprocessing evaluation recorded in the model card.

The full case performs actual W8A8 PTQ and HMONNX inference through the public
build/inference APIs, deletes the staged source, relocates the artifact, and
checks the output contract. It does not replace the network, load expected
outputs, compile for a device, or verify rendered image accuracy. The separate
model card reports the real rendered PSNR/SSIM/MSE evaluation.
