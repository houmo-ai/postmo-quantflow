"""Paired official-float, split-float, and HMONNX action trajectory validation."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def images_for(values):
    return [Image.open(path).convert("RGB") for path in values["images"]]


def metrics(actual, expected):
    a, b = actual.double().flatten(), expected.double().flatten()
    return dict(
        max_abs=float((a - b).abs().max()),
        rmse=float((a - b).square().mean().sqrt()),
        cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    )


def load_official_model(args):
    """Load the pinned native policy once for single or multi-sample checks."""
    import subprocess

    import yaml
    from safetensors.torch import load_file
    from transformers import AutoProcessor

    source = Path(args.official_source).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f":
        raise ValueError("official source must match the validated pinned revision")
    sys.path.insert(0, str(source))
    from wall_x.model.core.action.normalizer import Normalizer
    from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    root = Path(args.model_dir)
    native = AutoProcessor.from_pretrained(root, local_files_only=True, use_fast=True)
    native.tokenizer.add_tokens(["<|propri|>", "<|action|>"])
    cfg = Qwen2_5_VLConfig.from_pretrained(root)
    cfg.update_model_config(yaml.safe_load((root / "config.yml").read_text()))
    weights = load_file(str(root / "model.safetensors"))
    cfg.vocab_size = weights["model.embed_tokens.weight"].shape[0]
    cfg._attn_implementation = "sdpa"
    cfg.vision_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen2_5_VLMoEForAction(cfg, native)
    model.set_normalizer(
        Normalizer.from_ckpt(root / "normalizer_action.pth"), Normalizer.from_ckpt(root / "normalizer_propri.pth")
    )
    missing = model.load_state_dict(weights, strict=False, assign=True)
    if set(missing.missing_keys) - {"lm_head.weight"} or missing.unexpected_keys:
        raise ValueError(str(missing))
    model.lm_head.weight = model.model.embed_tokens.weight
    del weights
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.inv_freq.is_meta:
            if hasattr(module, "rope_init_fn"):
                inv, _ = module.rope_init_fn(module.config, device="cpu")
            else:
                dim = module.inv_freq.numel() * 2
                inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
            module.inv_freq = inv
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq.clone()
    # Wall-X's SDPA vision signature omits this FlashAttention-only hint.
    for block in model.visual.blocks:
        original = block.attn.forward

        def compatible(*a, _forward=original, **kw):
            kw.pop("max_seqlen", None)
            return _forward(*a, **kw)

        block.attn.forward = compatible
    model = model.to(args.device).eval()
    if args.dtype == "bfloat16":
        model.to_bfloat16_for_selected_params()
    else:
        model.float()
    return model, native


def official(args):
    """Load published Wall-X code in its own pinned reference environment."""
    from xhmodel_merak.xh_other_model.models.wall_oss.build import package_resources
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    root, out = Path(args.model_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    package_resources(root, out / "resources")
    values = json.loads(Path(args.input_json).read_text())
    processor = WallProcessor(out / "resources")
    data = processor.prepare(
        values["instruction"],
        images_for(values),
        values["state"],
        values.get("norm_key", "x2_normal"),
        values.get("state_mask"),
    )
    model, native = load_official_model(args)
    revision = "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f"
    p = data["ids"].shape[1]
    full_ids = torch.cat(
        (data["ids"], torch.full((1, 32), native.tokenizer.convert_tokens_to_ids("<|action|>"))), 1
    ).to(args.device)
    grid = data["grid"].to(args.device)
    position, _ = model.get_rope_index(full_ids, image_grid_thw=grid)
    pixels = torch.cat([x[0, : int(g.prod())] for x, g in zip(data["patches"], data["grid"], strict=True)]).to(
        args.device
    )
    mask = values.get("dof_mask", [1] * 26)
    dof = torch.tensor(mask, device=args.device, dtype=torch.float32)[None, None].expand(1, 32, 26)
    captured = {}
    finish = model._finalize_flow_action_output

    def capture(trajectory, ctx):
        captured["trajectory"] = torch.cat((ctx["noise"][None].cpu(), trajectory.cpu()), 0)
        captured["noise"] = ctx["noise"].cpu()
        return finish(trajectory, ctx)

    model._finalize_flow_action_output = capture
    torch.cuda.manual_seed(values.get("seed", 42))
    with torch.no_grad():
        vision = model.visual(pixels.to(model.visual.dtype), grid_thw=grid)
        embeddings = model.model.embed_tokens(data["ids"].to(args.device))
        embeddings[data["ids"].to(args.device) == 151655] = vision
        prediction = model.generate_flow_action(
            input_ids=full_ids,
            action_horizon=32,
            action_dim=26,
            num_inference_timesteps=values.get("steps", 10),
            attention_mask=torch.ones_like(full_ids),
            position_ids=position,
            moe_token_types=full_ids == native.tokenizer.convert_tokens_to_ids("<|action|>"),
            pixel_values=pixels,
            image_grid_thw=grid,
            dataset_names=[values.get("norm_key", "x2_normal")],
            dof_mask=dof,
        )
    captured.update(
        embeddings=embeddings.float().cpu(),
        vision=vision.float().cpu(),
        data=data,
        values=values,
        dtype=args.dtype,
        position_ids=position.cpu(),
        actions=prediction["predict_action"].float().cpu(),
    )
    torch.save(captured, out / "official.pt")
    (out / "official_report.json").write_text(
        json.dumps(
            dict(
                complete=True,
                source_revision=revision,
                dtype=args.dtype,
                prefix_tokens=p,
                image_grids=data["grid"].tolist(),
                shape=list(captured["actions"].shape),
                compatibility=(
                    "SDPA vision drops unused max_seqlen hint; nonpersistent RoPE buffers restored after meta loading"
                ),
            ),
            indent=2,
        )
    )


def target_float(args):
    from xhmodel_merak.xh_other_model.models.wall_oss.graph import ActionDenoiser, ExpertStack
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor
    from xhmodel_merak.xh_other_model.models.wall_oss.source import load_source, load_visual
    from xhmodel_merak.xh_other_model.models.wall_oss.vision import PackedVision

    out = Path(args.output_dir)
    a = torch.load(out / "official.pt", weights_only=False)
    if a["dtype"] != "float32":
        raise ValueError("split float gate requires official FP32 reference")
    data = a["data"]
    report = {}
    with torch.no_grad():
        source = load_visual(args.model_dir)
        visual = PackedVision(source, 1024, sdk=False).to(args.device)
        native_visual = source.to(args.device)
        gc.collect()
        outputs = []
        native_outputs = []
        for patches, geometry, grid in zip(data["patches"], data["geometry"], data["grid"], strict=True):
            output = visual(patches.to(args.device), *[x.to(args.device) for x in geometry])
            outputs.append(output[:, : int(grid.prod()) // 4])
            native_outputs.append(
                native_visual(patches[0, : int(grid.prod())].to(args.device), grid[None].to(args.device)).pooler_output
            )
        features = torch.cat(outputs, 1).reshape(-1, 2048).cpu()
        report["visual"] = metrics(features, a["vision"])
        native_features = torch.cat(native_outputs, 0).cpu()
        report["native_visual_to_official"] = metrics(native_features, a["vision"])
        report["packed_visual_to_native"] = metrics(features, native_features)
        (out / "visual_parity.json").write_text(json.dumps(report, indent=2))
        # Conv3d-to-Linear and split/fused MLP GEMMs have different FP32 accumulation.
        torch.testing.assert_close(features, a["vision"], atol=2e-3, rtol=5e-4)
        embeddings = a["embeddings"].clone()
        embeddings[data["ids"] == 151655] = features
        del visual, native_visual, source
        torch.cuda.empty_cache()
        source = load_source(args.model_dir)
        prefix = ExpertStack(source, expert=0, query_capacity=256, cache_capacity=2048, sdk=False).to(args.device)
        denoise = ActionDenoiser(source, query_capacity=32, cache_capacity=2048, sdk=False).to(args.device)
        del source
        gc.collect()
        p = embeddings.shape[1]
        cos, sin = [x.to(args.device) for x in data["rope"]]
        cache = tuple(torch.zeros(1, 2, 2048, 128, device=args.device) for _ in range(72))
        for start in range(0, p, 256):
            n = min(256, p - start)
            _, cache = prefix(
                torch.nn.functional.pad(embeddings[:, start : start + n].to(args.device), (0, 0, 0, 256 - n)),
                torch.nn.functional.pad(cos[:, :, start : start + n], (0, 0, 0, 256 - n), value=1),
                torch.nn.functional.pad(sin[:, :, start : start + n], (0, 0, 0, 256 - n)),
                torch.tensor([start], device=args.device),
                torch.tensor([n], device=args.device),
                cache,
            )
        action = a["noise"].to(args.device)
        dof = (
            torch.tensor(a["values"].get("dof_mask", [1] * 26), device=args.device)[None, None]
            .expand(1, 32, 26)
            .float()
        )
        processor = WallProcessor(out / "resources")
        padding = torch.tensor(
            processor.normalize(np.zeros(26, np.float32), a["values"].get("norm_key", "x2_normal"), "action"),
            device=args.device,
        )[None, None]
        vpadding = padding - action
        steps = a["values"].get("steps", 10)
        scheduler = processor.config["model_config"]["noise_scheduler"]
        times = torch.linspace(0, 1, steps + 1, device=args.device)
        shift = scheduler.get("time_shift", 1.0)
        times = shift * times / (1 + (shift - 1) * times) * scheduler.get("s", 0.999)
        trajectory = [action.cpu().clone()]
        for t, tn in zip(times[:-1], times[1:], strict=True):
            angle = t * torch.exp(torch.arange(512, device=args.device) * (-np.log(10000) / 511))
            time = torch.cat((angle.sin(), angle.cos()))[None]
            velocity, _ = denoise(
                action,
                dof,
                time,
                cos[:, :, p:],
                sin[:, :, p:],
                torch.tensor([p], device=args.device),
                torch.tensor([32], device=args.device),
                cache,
            )
            velocity = velocity * dof + vpadding * (1 - dof)
            action = action + (tn - t) * velocity
            trajectory.append(action.cpu().clone())
        actual = torch.stack(trajectory)
        report["trajectory"] = metrics(actual, a["trajectory"])
        torch.testing.assert_close(actual, a["trajectory"], atol=2e-4, rtol=2e-3)
        torch.save(actual, out / "target_float.pt")
    report["complete"] = True
    (out / "target_float_report.json").write_text(json.dumps(report, indent=2))


def deployed(args):
    from xhmodel_merak.xh_other_model.models.wall_oss.runtime import WallRuntime

    out = Path(args.output_dir)
    a = torch.load(out / "official.pt", weights_only=False)
    expected = torch.load(out / "target_float.pt", weights_only=True)
    runtime = WallRuntime(args.artifact_dir, device=args.device)
    values = {**a["values"], "images": images_for(a["values"]), "noise": a["noise"]}
    actions, stats = runtime.predict(**values)
    trajectory = torch.stack(runtime.last_trace)
    report = dict(
        complete=True,
        precision=runtime.meta["quant_type"],
        trajectory=metrics(trajectory, expected),
        final=metrics(trajectory[-1], expected[-1]),
        stats=stats,
    )
    # A different state, geometry and mask must not contaminate the original request.
    runtime.predict(
        **{
            **values,
            "state": [0.1] * 26,
            "images": [im.transpose(Image.Transpose.ROTATE_90) for im in values["images"]],
            "dof_mask": [0] * 10 + [1] * 16,
            "state_mask": [0] * 10 + [1] * 16,
        }
    )
    repeat, _ = runtime.predict(**values)
    np.testing.assert_array_equal(actions, repeat)
    report["repeat_exact"] = True
    stem = runtime.meta["quant_type"]
    np.save(out / f"{stem}_actions.npy", actions)
    torch.save(trajectory, out / f"{stem}_trajectory.pt")
    (out / f"{stem}_report.json").write_text(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("official", "float", "deployed"):
        item = sub.add_parser(name)
        item.add_argument("--output-dir", required=True)
        item.add_argument("--device", default="cuda:0")
        if name in ("official", "float"):
            item.add_argument("--model-dir", required=True)
        if name == "official":
            item.add_argument("--official-source", required=True)
            item.add_argument("--input-json", required=True)
            item.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
        if name == "deployed":
            item.add_argument("--artifact-dir", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {"official": official, "float": target_float, "deployed": deployed}[args.stage](args)


if __name__ == "__main__":
    main()
