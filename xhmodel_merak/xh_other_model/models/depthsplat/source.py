# Copyright (c) 2024 Haofei Xu
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: MIT
# Adapted from DepthSplat 2dad25a7b9ba7c5537aba08463ef88d8511a06be.
# See licenses/DepthSplat/LICENSE.
"""Export pinned external DepthSplat sources without downloading weights.

Export substitutions temporarily patch upstream/PyTorch functions, so run this
module in a dedicated process, never concurrently with another model inference.
"""

import hashlib
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

PROFILES = {"2v_256x256": (2, 256, 256), "6v_512x960": (6, 512, 960)}
SOURCE_COMMIT = "2dad25a7b9ba7c5537aba08463ef88d8511a06be"
INPUT_NAMES = ["images", "intrinsics", "extrinsics", "near", "far"]
OUTPUT_NAMES = ["depths", "raw_gaussians"]


def configure_source(source_dir, dino_dir=None):
    source_dir = Path(source_dir).resolve()
    if not (source_dir / "src/model/encoder/encoder_depthsplat.py").is_file():
        raise ValueError(f"Not a DepthSplat source checkout: {source_dir}")
    for directory in (source_dir, dino_dir):
        if directory is not None:
            sys.path.insert(0, str(Path(directory).resolve()))
    for package, relative in (("src.model.encoder", "src/model/encoder"), ("src.dataset", "src/dataset")):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(source_dir / relative)]
            sys.modules[package] = module
    return source_dir


def load_encoder_config(source_dir, profile):
    source_dir = configure_source(source_dir)
    from omegaconf import OmegaConf
    from dacite import from_dict
    from src.model.encoder.encoder_depthsplat import EncoderDepthSplatCfg

    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    experiment = "re10k" if profile == "2v_256x256" else "dl3dv"
    config = OmegaConf.merge(
        OmegaConf.load(source_dir / "config/model/encoder/depthsplat.yaml"),
        OmegaConf.load(source_dir / f"config/experiment/{experiment}.yaml").model.encoder,
    )
    if profile == "6v_512x960":
        config = OmegaConf.merge(
            config,
            {
                "num_scales": 2,
                "upsample_factor": 4,
                "lowest_feature_resolution": 8,
                "monodepth_vit_type": "vitb",
                "gaussian_adapter": {"gaussian_scale_max": 0.1},
            },
        )
    return from_dict(EncoderDepthSplatCfg, OmegaConf.to_container(config, resolve=True))


def load_encoder(source_dir, dino_dir, checkpoint, profile, device):
    configure_source(source_dir, dino_dir)
    from src.model.encoder.encoder_depthsplat import EncoderDepthSplat

    config = load_encoder_config(source_dir, profile)
    original = torch.hub.load

    def local_hub(repo, name, *args, **kwargs):
        if repo != "facebookresearch/dinov2":
            raise ValueError(f"Unexpected torch.hub repository: {repo}")
        from dinov2.hub import backbones

        return getattr(backbones, name)(pretrained=False)

    torch.hub.load = local_hub
    try:
        encoder = EncoderDepthSplat(config)
    finally:
        torch.hub.load = original
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}, strict=True
    )
    return encoder.eval().to(device)


@contextmanager
def camera_hooks(captured=None, supplied=None, events=None):
    """Replace only camera-dependent inverse/Wigner operations, in call order."""
    from src.model.encoder.common import gaussian_adapter
    from e3nn.o3 import matrix_to_angles, wigner_D

    tensor_inverse, torch_inverse = torch.Tensor.inverse, torch.inverse
    original_rotate = gaussian_adapter.rotate_sh
    position = 0

    def take(value, kind, operand=None):
        nonlocal position
        if supplied is not None:
            result = supplied[position]
        else:
            result = value
            captured.append(value.detach())
        if events is not None:
            events.append(
                dict(
                    name=f"camera_{position}_{kind}",
                    kind=kind,
                    shape=list(result.shape),
                    operand_shape=list(operand.shape) if operand is not None else None,
                )
            )
        position += 1
        return result

    def inverse(matrix):
        value = None if supplied is not None else tensor_inverse(matrix)
        return take(value, "inverse", matrix)

    def rotate(coefficients, rotations):
        angles = None if supplied is not None else matrix_to_angles(rotations)
        results = []
        for degree in range(math.isqrt(coefficients.shape[-1])):
            value = None
            if supplied is None:
                with torch.device(coefficients.device):
                    value = wigner_D(degree, *angles).to(coefficients.dtype)
            matrix = take(value, f"sh{degree}", rotations)
            results.append(
                torch.matmul(matrix, coefficients[..., degree**2 : (degree + 1) ** 2].unsqueeze(-1)).squeeze(-1)
            )
        return torch.cat(results, dim=-1)

    torch.Tensor.inverse = inverse
    torch.inverse = inverse
    gaussian_adapter.rotate_sh = rotate
    try:
        yield
        if supplied is not None:
            assert position == len(supplied), (position, len(supplied))
    finally:
        torch.Tensor.inverse, torch.inverse = tensor_inverse, torch_inverse
        gaussian_adapter.rotate_sh = original_rotate


@contextmanager
def export_operations():
    """Inference-equivalent export substitutions; restored after each call."""
    from src.model.encoder.unimatch.ldm_unet import unet

    original_checkpoint = unet.checkpoint
    original_diag = torch.Tensor.diag_embed
    from src.model.encoder import encoder_depthsplat

    original_grid = encoder_depthsplat.sample_image_grid
    from src.geometry import projection

    original_einsum = projection.einsum

    def projection_matmul(left, right, equation):
        if equation == "... i j, ... j -> ... i":
            return torch.matmul(left, right.unsqueeze(-1)).squeeze(-1)
        return original_einsum(left, right, equation)

    projection.einsum = projection_matmul

    def float_grid(*args, **kwargs):
        coordinates, indices = original_grid(*args, **kwargs)
        return coordinates.float(), indices

    encoder_depthsplat.sample_image_grid = float_grid
    unet.checkpoint = lambda function, inputs, parameters, flag: function(*inputs)
    torch.Tensor.diag_embed = lambda value: value.unsqueeze(-1) * torch.eye(
        value.shape[-1], dtype=value.dtype, device=value.device
    )
    try:
        yield
    finally:
        unet.checkpoint = original_checkpoint
        torch.Tensor.diag_embed = original_diag
        encoder_depthsplat.sample_image_grid = original_grid
        projection.einsum = original_einsum


def prepare_host_cameras(images, intrinsics, extrinsics, near, far):
    """Standalone camera-only preparation for this fixed B=1,V=2,small profile.

    This does not execute the encoder and keeps all camera values sample-dependent.
    Its six outputs are checked against actual official-model operations below.
    """
    from e3nn.o3 import matrix_to_angles, wigner_D

    assert tuple(images.shape[:2]) == (1, 2)
    feature_k = intrinsics.clone()
    feature_k[..., 0, :] *= images.shape[-1]
    feature_k[..., 1, :] *= images.shape[-2]
    feature_k[..., :2, :] /= 4
    inverse_target = extrinsics.flip(1).reshape(2, 1, 4, 4).inverse()
    inverse_feature_k = feature_k.reshape(2, 3, 3).inverse()
    inverse_rays_k = intrinsics[:, :, None, None, None].inverse()
    rotations = extrinsics[:, :, None, None, None, None, :3, :3]
    angles = matrix_to_angles(rotations)
    with torch.device(images.device):
        sh_matrices = [wigner_D(degree, *angles).to(images.dtype) for degree in range(3)]
    return [inverse_target, inverse_feature_k, inverse_rays_k, *sh_matrices]


def prepare_views(example, indices, profile, source_dir):
    """Official resize/center-crop and normalized cameras, with batch dimension."""
    configure_source(source_dir)
    from torchvision.transforms import ToTensor
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views

    if profile == "2v_256x256":
        from src.dataset.dataset_re10k import DatasetRE10k as Dataset

        far = 100.0
    elif profile == "6v_512x960":
        from src.dataset.dataset_dl3dv import DatasetDL3DV as Dataset

        far = 200.0
    else:
        raise ValueError(f"Unsupported profile: {profile}")
    loader = object.__new__(Dataset)
    loader.to_tensor = ToTensor()
    indices = list(indices)
    extrinsics, intrinsics = loader.convert_poses(example["cameras"][indices])
    values = dict(
        image=loader.convert_images([example["images"][i] for i in indices]),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        near=torch.full((len(indices),), 0.5),
        far=torch.full((len(indices),), far),
        index=torch.tensor(indices),
    )
    values = apply_crop_shim_to_views(values, PROFILES[profile][1:])
    return {name: value.unsqueeze(0).numpy() for name, value in values.items()}


class _RawComplete(Exception):
    pass


@contextmanager
def _neighbors(value):
    original = torch.argsort
    if value is not None:
        torch.argsort = lambda tensor, *args, **kwargs: value
    try:
        yield
    finally:
        torch.argsort = original


class RawWrapper(torch.nn.Module):
    def __init__(self, encoder, six_views=False):
        super().__init__()
        self.encoder = encoder
        self.six_views = six_views
        self.captured = None

    def forward(self, images, intrinsics, extrinsics, near, far, *extra):
        nn_matrix, cameras = (extra[0], extra[1:]) if self.six_views else (None, extra)
        captured = {}

        def head_hook(module, args, output):
            captured["raw_gaussians"] = output

        original = self.encoder.gaussian_adapter.forward

        def adapter(*args, **kwargs):
            captured["depths"] = args[3]
            if self.six_views:
                raise _RawComplete()
            return original(*args, **kwargs)

        handle = self.encoder.gaussian_head.register_forward_hook(head_hook)
        self.encoder.gaussian_adapter.forward = adapter
        try:
            with _neighbors(nn_matrix), camera_hooks(captured=self.captured, supplied=cameras or None):
                try:
                    self.encoder(
                        dict(image=images, intrinsics=intrinsics, extrinsics=extrinsics, near=near, far=far),
                        0,
                        deterministic=True,
                    )
                except _RawComplete:
                    pass
        finally:
            handle.remove()
            self.encoder.gaussian_adapter.forward = original
        return captured["depths"], captured["raw_gaussians"]


def _cut_matching_grid(model):
    """Fixed pinned exporter boundary: fail closed if upstream graph changes."""
    import onnx

    cut = "/encoder/depth_predictor/Reshape_39_output_0"
    occurrences = sum(name == cut for node in model.graph.node for name in node.input)
    if occurrences != 1:
        raise ValueError(f"Pinned 2v matching-grid boundary changed ({occurrences} consumers)")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == cut:
                node.input[index] = "matching_grid"
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    for field in (model.graph.initializer, model.graph.input, model.graph.value_info):
        keep = [value for value in field if value.name in needed]
        del field[:]
        field.extend(keep)
    model.graph.input.append(
        onnx.helper.make_tensor_value_info("matching_grid", onnx.TensorProto.FLOAT, [2, 8192, 64, 2])
    )


def export_bundle(
    source_dir,
    dino_dir,
    checkpoint,
    profile,
    samples,
    output_dir,
    device="cuda:0",
    calibration_samples=(),
    verify=False,
):
    """Export raw neural outputs; retain original cameras for FP32 host processing.

    Six-view camera inputs are captured solely during export. Deployment inputs
    must be recomputed from each sample's cameras, as the inference CLI does.
    """
    import onnx
    from .preprocess import prepare_matching_grid

    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    if not samples:
        raise ValueError("At least one real context sample is required")
    samples = list(dict.fromkeys(Path(path).resolve() for path in [*calibration_samples, *samples]))
    if len({path.name for path in samples}) != len(samples):
        raise ValueError("Sample basenames must be unique")
    calibration = {Path(path).resolve() for path in calibration_samples}
    output_dir = Path(output_dir).resolve()
    if any(path.parent == output_dir for path in samples):
        raise ValueError("Output directory must differ from input sample directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = load_encoder(source_dir, dino_dir, checkpoint, profile, device)
    six_views = profile == "6v_512x960"
    wrapper = RawWrapper(encoder, six_views)
    records, first, names = [], None, None
    preprocessing = dict(
        type="depthsplat_matching_grid_fp32_v1", num_depth_candidates=128, feature_height=64, feature_width=64
    )
    with torch.no_grad():
        for path in samples:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["images"] = arrays.get("images", arrays.get("image"))
            expected = (1, PROFILES[profile][0], 3, *PROFILES[profile][1:])
            if arrays["images"] is None or arrays["images"].shape != expected:
                raise ValueError(f"{path}: expected images {expected}")
            for name in INPUT_NAMES:
                if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
                    raise ValueError(f"{path}: {name} must be finite float32")
            inputs = [torch.from_numpy(arrays[name]).to(device) for name in INPUT_NAMES]
            if six_views:
                xyz = inputs[2][:, :, :3, -1]
                neighbors = torch.argsort(torch.cdist(xyz, xyz, p=2))
                wrapper.captured = []
                with export_operations():
                    raw = wrapper(*inputs, neighbors)
                camera = wrapper.captured
                wrapper.captured = None
                names = INPUT_NAMES + ["nn_matrix"] + [f"camera_{i}_inverse" for i in range(len(camera))]
                values = tuple(inputs + [neighbors] + camera)
                if len(camera) != 4:
                    raise ValueError("Pinned 6v encoder must have four camera inverses")
            else:
                camera = prepare_host_cameras(*[value.cpu() for value in inputs])
                camera = [value.to(device) for value in camera]
                names = INPUT_NAMES + [
                    "camera_0_inverse",
                    "camera_1_inverse",
                    "camera_2_inverse",
                    "camera_3_sh0",
                    "camera_4_sh1",
                    "camera_5_sh2",
                ]
                values = tuple(inputs + camera)
                with export_operations():
                    raw = wrapper(*values)
            arrays.update({name: value.detach().cpu().numpy() for name, value in zip(names, values)})
            if not six_views:
                arrays["matching_grid"] = prepare_matching_grid(arrays, preprocessing)
            reference = {name: value.detach().cpu().numpy() for name, value in zip(OUTPUT_NAMES, raw)}
            np.savez_compressed(output_dir / path.name, **arrays)
            np.savez_compressed(output_dir / f"{path.stem}-raw-pytorch.npz", **reference)
            records.append(dict(sample=path.name, output_shapes={key: list(v.shape) for key, v in reference.items()}))
            if first is None:
                first = values
        with export_operations():
            torch.onnx.export(
                wrapper,
                first,
                str(output_dir / "encoder.onnx"),
                input_names=names,
                output_names=OUTPUT_NAMES,
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
    model = onnx.load(output_dir / "encoder.onnx")
    for value in model.graph.output:
        for dimension, size in zip(value.type.tensor_type.shape.dim, records[0]["output_shapes"][value.name]):
            dimension.ClearField("dim_param")
            dimension.dim_value = size
    if not six_views:
        _cut_matching_grid(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_dir / "encoder.onnx")
    input_names = [value.name for value in model.graph.input]
    host_names = ["images", "intrinsics", "extrinsics", "near", "far"]
    if not six_views:
        host_names += ["camera_3_sh0", "camera_4_sh1", "camera_5_sh2"]
    with np.load(output_dir / samples[0].name) as sample:
        host_specs = {
            name: {"shape": list(sample[name].shape), "dtype": str(sample[name].dtype)} for name in host_names
        }
    config = dict(
        model_type="DepthSplatONNX",
        profile=profile,
        onnx_file="encoder.onnx",
        input_names=input_names,
        output_names=OUTPUT_NAMES,
        host_input_specs=host_specs,
        postprocess=dict(
            type="depthsplat_gaussian_fp32_v1",
            sh_degree=2,
            gaussian_scale_min=1e-10,
            gaussian_scale_max=0.1 if six_views else 3.0,
            num_surfaces=1,
        ),
        calibration_samples=[path.name for path in samples if path in calibration],
        evaluation_samples=[path.name for path in samples if path not in calibration],
        expected_source_commit=SOURCE_COMMIT,
        source_verification="Caller must provide the pinned external source; archive revision is not independently verified.",
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    )
    if not six_views:
        config["preprocess"] = preprocessing
    else:
        config["geometry_note"] = "Two matching grids remain in graph; second depends on first-scale neural depth."
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if verify:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        session = ort.InferenceSession(
            str(output_dir / "encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for record in records:
            path = output_dir / record["sample"]
            with np.load(path) as archive:
                actual = session.run(OUTPUT_NAMES, {name: archive[name] for name in input_names})
            record["float_alignment"] = {}
            with np.load(output_dir / f"{path.stem}-raw-pytorch.npz") as reference:
                for name, value in zip(OUTPUT_NAMES, actual):
                    expected = reference[name]
                    relative = float(
                        np.linalg.norm((value - expected).ravel().astype(np.float64))
                        / max(np.linalg.norm(expected.ravel().astype(np.float64)), 1e-12)
                    )
                    record["float_alignment"][name] = {
                        "relative_l2": relative,
                        "max_abs": float(np.max(np.abs(value - expected))),
                    }
                    if not np.isfinite(value).all() or relative > 1e-3:
                        raise ValueError(f"Float export mismatch: {path.name}/{name}: {relative}")
            np.savez_compressed(output_dir / f"{path.stem}-raw-onnx.npz", **dict(zip(OUTPUT_NAMES, actual)))
    if not six_views:
        # The registered bundle contract keeps only original Host inputs.
        # Matching grids are recomputed per sample by runtime preprocessing.
        for path in samples:
            target = output_dir / path.name
            with np.load(target) as archive:
                arrays = {name: archive[name] for name in host_names}
            np.savez_compressed(target, **arrays)
    report = dict(profile=profile, input_names=input_names, samples=records, verified=verify)
    (output_dir / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
