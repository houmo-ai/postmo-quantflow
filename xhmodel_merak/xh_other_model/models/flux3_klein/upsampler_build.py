# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build/reuse an independent prompt component and compose portable graph closures."""

import json
import shutil
import tempfile
from pathlib import Path


def validate_source(directory):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", {})
    if (config.get("model_type"), text.get("hidden_size"), text.get("num_hidden_layers")) != ("qwen3_5", 4096, 32):
        raise ValueError("prompt upsampler requires the official Qwen3.5-9B checkpoint")
    if text.get("layer_types") != ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)]:
        raise ValueError("unsupported prompt upsampler hybrid layer layout")
    prompts = json.loads((root / "system_prompts.json").read_text())
    if any(not isinstance(prompts.get(key), str) or not prompts[key].strip() for key in ("t2i", "i2i")):
        raise ValueError("prompt upsampler requires the official t2i/i2i system prompts")
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
        if not (root / name).is_file():
            raise ValueError(f"prompt upsampler resource missing: {name}")


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"prompt upsampler {name} must be a positive integer")
    return value


def visual_graphs(component, meta):
    """Bind every metadata gear to its actual component graph, including v1 bundles."""
    gears = meta["visual_config"]["gears"]
    capacities = [_positive(gear["image_token_capacity"], "visual capacity") for gear in gears]
    if not capacities or len(set(capacities)) != len(capacities):
        raise ValueError("prompt upsampler visual gears must be nonempty and unique")
    if component["schema_version"] == 1:
        if len(gears) != 1:
            raise ValueError("version 1 prompt component requires one visual graph")
        roles = ["visual"]
    else:
        roles = [f"visual_m{capacity}" for capacity in capacities]
    expected = {"prefill", "decode", *roles}
    if set(component["graphs"]) != expected or len(set(component["graphs"].values())) != len(expected):
        raise ValueError("prompt upsampler graph inventory disagrees with visual gears")
    for role, gear in zip(roles, gears, strict=True):
        if component["graphs"][role] != gear["hmonnx"]:
            raise ValueError("prompt upsampler visual graph disagrees with runtime")
    return dict(zip(roles, gears, strict=True))


def describe_component(root, *, max_new_tokens=3584):
    """Derive the portable component contract from the actual Qwen export metadata."""
    root = Path(root)
    meta = json.loads((root / "upsampler_meta_info.json").read_text())
    model = meta["model_config"]
    return dict(
        schema_version=2,
        kind="flux3_prompt_upsampler",
        context_length=_positive(model["context_max_length"], "context length"),
        prefill_chunk_length=_positive(model["prefill_chunk_length"], "prefill length"),
        max_new_tokens=_positive(max_new_tokens, "generation budget"),
        quant_type="w8a16h1_sefp",
        metadata="upsampler_meta_info.json",
        processor_config="hf_config/processor_config.json",
        system_prompts="hf_config/system_prompts.json",
        graphs={
            "prefill": meta["prefill_hmonnx"],
            "decode": meta["decode_hmonnx"],
            **{f"visual_m{gear['image_token_capacity']}": gear["hmonnx"] for gear in meta["visual_config"]["gears"]},
        },
        files=sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()),
        reference_revision="e53f4ea697fe45db8f38fca91a90f5623f7f8f37",
    )


def read_component(root):
    import onnx

    from .bundle import bundle_path

    root = Path(root).resolve()
    component = json.loads(bundle_path(root, "upsampler_component.json").read_text())
    if component.get("schema_version") not in (1, 2) or component.get("kind") != "flux3_prompt_upsampler":
        raise ValueError("unsupported prompt upsampler component identity")
    context = _positive(component.get("context_length"), "context length")
    _positive(component.get("max_new_tokens"), "generation budget")
    if component.get("quant_type") != "w8a16h1_sefp":
        raise ValueError("unsupported prompt upsampler deployment precision")
    for name in component["files"]:
        bundle_path(root, name)
    listed = set(component["files"])
    required = {
        *component["graphs"].values(),
        component["metadata"],
        component["processor_config"],
        component["system_prompts"],
    }
    if not required <= listed:
        raise ValueError("prompt upsampler inventory omits runtime files")
    meta = json.loads(bundle_path(root, component["metadata"]).read_text())
    prefill = _positive(meta["model_config"]["prefill_chunk_length"], "prefill length")
    if (
        meta["model_config"]["context_max_length"] != context
        or prefill > context
        or component.get("prefill_chunk_length", prefill) != prefill
    ):
        raise ValueError("prompt upsampler physical context/prefill metadata disagrees")
    if component["schema_version"] == 2 and "prefill_chunk_length" not in component:
        raise ValueError("prompt upsampler prefill length is missing")
    if (
        meta["prefill_hmonnx"] != component["graphs"]["prefill"]
        or meta["decode_hmonnx"] != component["graphs"]["decode"]
    ):
        raise ValueError("prompt upsampler graph inventory disagrees with runtime")
    if meta["quant_embedding"] not in listed:
        raise ValueError("prompt upsampler embedding is absent from the inventory")
    config = json.loads(bundle_path(root, str(Path(meta["hf_config"]) / "config.json")).read_text())
    text, vision = config["text_config"], config["vision_config"]
    if config.get("model_type") != "qwen3_5" or text["hidden_size"] != 4096:
        raise ValueError("prompt upsampler packaged model identity disagrees")
    visuals = visual_graphs(component, meta)
    merge = vision["spatial_merge_size"]
    if meta["visual_config"]["spatial_merge_size"] != merge:
        raise ValueError("prompt upsampler spatial merge disagrees with source")
    patch_dim = vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

    external = validate_onnx_external_locations(root, graphs=[root / path for path in component["graphs"].values()])
    if any(path.relative_to(root).as_posix() not in listed for path in external):
        raise ValueError("prompt upsampler inventory omits graph weights")
    for role, relative in component["graphs"].items():
        graph = onnx.load(str(bundle_path(root, relative)), load_external_data=False).graph
        if not graph.node or not graph.output:
            raise ValueError(f"empty prompt upsampler {role} graph")
        ports = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in graph.input}
        if role in visuals:
            gear = visuals[role]
            patches = gear["image_token_capacity"] * merge**2
            if gear["patch_token_capacity"] != patches or ports.get("pixel_values") != [1, patches, patch_dim]:
                raise ValueError("prompt upsampler physical visual capacity disagrees")
        else:
            length = prefill if role == "prefill" else 1
            if ports.get("input_1") != [1, length, text["hidden_size"]]:
                raise ValueError("prompt upsampler physical language input shape disagrees")
            caches = [shape for name, shape in ports.items() if name.endswith(("kcache_input", "vcache_input"))]
            full = text["layer_types"].count("full_attention")
            expected = [1, text["num_key_value_heads"], context, text["head_dim"]]
            if len(caches) != 2 * full or any(shape != expected for shape in caches):
                raise ValueError("prompt upsampler physical KV capacity disagrees")
            operator = "GDRChunkScan" if role == "prefill" else "GDRRecurrentScan"
            if sum(node.op_type == operator for node in graph.node) != text["layer_types"].count("linear_attention"):
                raise ValueError("prompt upsampler hybrid state operators are incomplete")
        for node in graph.node:
            if node.op_type not in ("Linear", "Conv2d"):
                continue
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            expected = dict(
                hmfp_weight_man_bit=8,
                hmfp_act_man_bit=16,
                hmfp_weight_nshare=64,
                hmfp_act_nshare=64,
                hmfp_weight_hidden_bit=1,
                hmfp_act_hidden_bit=1,
                mode=b"sefp",
            )
            if any(attributes.get(name) != value for name, value in expected.items()):
                raise ValueError("prompt upsampler physical precision disagrees with component")
    return component


def build_component(model_dir, output_dir, *, device="cuda:0", config_path=None, max_new_tokens=3584):
    from xhmodel_merak.model_build.artifacts import flatten_single_export_root
    from xhmodel_merak.model_integrations.autoregressive.build import sanitize_artifact
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    validate_source(model_dir)
    _positive(max_new_tokens, "generation budget")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    # The repository root is resolved from the installed workflow resource,
    # rather than the caller's working directory.
    from xhmodel_merak.model_catalog.resources import resolve_workflow_config

    config = config_path or resolve_workflow_config(
        "configs_merak/workflows/xh2a/other_models/flux3_klein/prompt_upsampler.yaml"
    )
    with tempfile.TemporaryDirectory(prefix=".flux3-pup-", dir=root.parent) as temporary:
        stage = Path(temporary)
        workflow = Qwen35Workflow.from_config(str(model_dir), str(config))
        workflow.export(QuantResult(raw_model_dir=str(model_dir), skipped=True), str(stage), device)
        flatten_single_export_root(stage)
        sanitize_artifact(stage, None)
        # The image runtime loads the explicit component metadata directly;
        # the standalone vLLM entry points refer to a different metadata name.
        (stage / "merak_config.json").unlink(missing_ok=True)
        (stage / "golden_meta_info.json").rename(stage / "upsampler_meta_info.json")
        shutil.copyfile(Path(model_dir) / "system_prompts.json", stage / "hf_config/system_prompts.json")
        component = describe_component(stage, max_new_tokens=max_new_tokens)
        (stage / "upsampler_component.json").write_text(json.dumps(component, indent=2) + "\n")
        read_component(stage)
        if root.exists():
            raise FileExistsError(root)
        stage.rename(root)
    return component


def copy_files(source, target, relatives):
    from .bundle import bundle_path

    for relative in sorted(set(relatives)):
        path = bundle_path(source, relative)
        destination = Path(target) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"component resource collision: {relative}")
        shutil.copyfile(path, destination)


def reuse_image_bundle(source, target, config, *, model_dir=None, device="cuda:0"):
    from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

    from .build import validate_build_config
    from .bundle import read_metadata

    source, target = Path(source).resolve(), Path(target)
    meta = read_metadata(source)
    if meta.get("prompt_upsampler"):
        raise ValueError("image_bundle must be a core-only FLUX bundle")
    capacity = validate_build_config(config)
    changed = {name for name, value in capacity.items() if meta["capacity"][name] != value}
    if changed - {"max_references", "reference_total_capacity", "joint_capacity"} or (
        meta["quant_type"] != config["model"]["quant_scheme"]["quant_type"]
    ):
        raise ValueError("image bundle capacity/precision does not match this build")
    rebuild = "reference_total_capacity" in changed
    if rebuild:
        from .source import read_source

        if model_dir is None:
            raise ValueError("DiT capacity changes require the original model_dir")
        for name, value in zip(("source_config", "text_config"), read_source(model_dir), strict=True):
            if value != json.loads((source / meta["resources"][name]).read_text()):
                raise ValueError("image bundle source configuration does not match the DiT checkpoint")
    selected = {role: path for role, path in meta["graphs"].items() if not (rebuild and role == "denoise")}
    graphs = [source / path for path in selected.values()]
    files = [*selected.values(), *meta["resources"].values(), *meta["processor_files"]]
    files.extend(
        path.relative_to(source).as_posix() for path in validate_onnx_external_locations(source, graphs=graphs)
    )
    target.mkdir(parents=True, exist_ok=False)
    copy_files(source, target, files)
    if rebuild:
        from .build import rebuild_denoise

        meta["graphs"]["denoise"] = rebuild_denoise(
            model_dir, target, source, capacity, meta["quant_type"], device=device
        )
    meta["capacity"] = capacity
    return meta


def attach_component(root, source):
    component = read_component(source)
    copy_files(source, root, [*component["files"], "upsampler_component.json"])
    return component
