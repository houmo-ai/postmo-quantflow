"""Compare official, target float and HMONNX with identical dynamic image processing."""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image


CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml"


def cases(image_path):
    image = Image.open(image_path).convert("RGB")
    red = Image.new("RGB", (320, 768), "red")
    return [
        (
            "text",
            {"text": "What is the capital of France?"},
            [{"text": "Paris is the capital of France."}, {"text": "Whales live in the ocean."}],
        ),
        (
            "long_text",
            {"text": "Which city is the capital of France?"},
            [
                {"text": "Unrelated background information. " * 90 + "Paris is the capital of France."},
                {"text": "Unrelated background information. " * 90 + "Whales live in the ocean."},
            ],
        ),
        ("text_image", {"text": "A woman and a dog playing on the beach."}, [{"image": image}, {"image": red}]),
        (
            "image_text",
            {"image": image},
            [{"text": "A woman and a dog playing on the beach."}, {"text": "A train on a snowy mountain."}],
        ),
        (
            "mixed",
            {"text": "Find the beach scene with a dog.", "image": image},
            [{"text": "A woman playing with a dog.", "image": image}, {"text": "A solid red image.", "image": red}],
        ),
        (
            "multi_image",
            {"text": "Find the image collection that includes a dog on the beach."},
            [{"image": [red, image]}, {"image": [red, red]}],
        ),
        *[
            (
                f"gear_{gear}",
                {"text": "A woman and a dog playing on the beach."},
                [{"image": image.resize(size)}, {"image": Image.new("RGB", size, "red")}],
            )
            for gear, size in [
                (96, (384, 256)),
                (196, (448, 448)),
                (384, (768, 512)),
                (704, (1024, 704)),
                (1536, (1536, 1024)),
            ]
        ],
    ]


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", default=CONFIG, help="Workflow used to construct the target floating wrapper")
    parser.add_argument("--metadata", help="golden_meta_info.json or runtime_meta_info.json")
    parser.add_argument("--image", default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-float-error", type=float, default=0.02)
    parser.add_argument("--max-quant-error", type=float, default=0.1)
    args = parser.parse_args()
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import build_messages

    samples = cases(args.image)
    script = Path(args.model_dir) / "scripts" / "qwen3_vl_reranker.py"
    spec = importlib.util.spec_from_file_location("official_qwen3_vl_reranker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.Qwen3VLReranker(
        model_name_or_path=args.model_dir,
        torch_dtype=torch.float16,
        max_length=32768,
        min_pixels=4 * 32**2,
        max_pixels=1536 * 32**2,
    )
    native.model.to(args.device)
    native.score_linear.to(args.device)
    reference, input_hashes, effective_hashes = {}, {}, {}
    for name, query, documents in samples:
        scores, hashes, processed = [], [], []
        for document in documents:
            messages = build_messages(query, document)
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image":
                        item.update(min_pixels=native.min_pixels, max_pixels=native.max_pixels)
            encoded = native.tokenize([messages]).to(args.device)
            # The bundled 4.57 script pads only input_ids. Transformers 5 adds
            # multimodal token types; preserve their values while tensorizing.
            if isinstance(encoded.get("mm_token_type_ids"), list):
                encoded["mm_token_type_ids"] = torch.tensor(encoded["mm_token_type_ids"], device=args.device)
            hashes.append(digest(encoded["input_ids"]))
            processed.append(
                {
                    key: digest(encoded[key])
                    for key in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values", "mm_token_type_ids")
                    if key in encoded
                }
            )
            scores.extend(native.compute_scores(encoded))
        reference[name], input_hashes[name] = scores, hashes
        effective_hashes[name] = processed
    del native
    gc.collect()
    torch.cuda.empty_cache()

    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config)
    target = workflow.get_model()
    target.to_wrap()
    target.to(args.device)
    target_inputs = {}
    for name, query, documents in samples:
        processor = target.get_tf_processor()
        encoded_items = [processor.apply_chat_template(build_messages(query, doc)) for doc in documents]
        hashes = [digest(item["input_ids"]) for item in encoded_items]
        if hashes != input_hashes[name]:
            raise AssertionError(f"Official and target input IDs differ for {name}")
        for index, encoded in enumerate(encoded_items):
            expected = effective_hashes[name][index]
            actual = {key: digest(encoded[key]) for key in expected}
            if actual != expected:
                raise AssertionError(f"Official and target processed tensors differ for {name}, pair {index}")
        target_inputs[name] = hashes
    floating = {}
    for name, query, documents in samples:
        floating[name] = target.rerank(query, documents)
        print(name, "official", reference[name], "target_float", floating[name], flush=True)
    float_error = max(abs(a - b) for name in reference for a, b in zip(reference[name], floating[name], strict=True))
    report = {
        "official_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "protocol": (
            "bundled official FP16 / target wrapped FP16 / HMONNX; identical dynamic-resolution inputs, 32K context"
        ),
        "input_ids_sha256": target_inputs,
        "effective_inputs_sha256": effective_hashes,
        "official": reference,
        "target_float": floating,
        "max_float_absolute_error": float_error,
    }
    del target
    gc.collect()
    torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if float_error > args.max_float_error:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        raise AssertionError(f"Float parity gate failed: {float_error}")

    if args.metadata:
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        model = AutoLLMHONNXModel.from_pretrained(args.metadata, device_map=[args.device])
        model.to(args.device)
        quantized = {name: model.rerank(query, documents) for name, query, documents in samples}
        error = max(abs(a - b) for name in reference for a, b in zip(floating[name], quantized[name], strict=True))
        rankings_match = all(
            sorted(range(len(quantized[name])), key=lambda i: -quantized[name][i])
            == sorted(range(len(floating[name])), key=lambda i: -floating[name][i])
            for name in quantized
        )
        report.update(hmonnx=quantized, max_quant_absolute_error=error, rankings_match=rankings_match)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        if error > args.max_quant_error or not rankings_match:
            raise AssertionError(f"Quantized parity gate failed: error={error}, rankings_match={rankings_match}")
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
