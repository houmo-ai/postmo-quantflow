# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import argparse

from xhmodel_merak.workflows import AutoWorkflow


def main():
    parser = argparse.ArgumentParser(description="GLM-5.3-Flash Merak streamed multimodal export")
    parser.add_argument("--model", default="weights/GLM-5.3-Flash")
    parser.add_argument(
        "--config", default="configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_w8a16.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--text-meta", help="reuse an existing text artifact in a new bundle using hard links")
    parser.add_argument("--vision-max-patches", type=int)
    args = parser.parse_args()
    workflow = AutoWorkflow.from_config(model_dir=args.model, config_path=args.config)
    if args.text_meta:
        if args.max_layers is not None:
            parser.error("--max-layers cannot change an existing --text-meta artifact")
        from xhmodel_merak.xh_llm.models.glm5_3_flash.vision_export import attach_glm53_vision

        capacity = args.vision_max_patches
        if capacity is None:
            capacity = workflow.workflow_config.export["model"].get("vision_max_patches", 1024)
        result = attach_glm53_vision(args.text_meta, args.model, args.output, capacity, device=args.device)
        print(result)
        return
    quant = workflow.quant(output_dir=args.output + "_weights", device=args.device)
    overrides = {}
    if args.max_layers is not None:
        overrides["export.model.max_layers"] = args.max_layers
    if args.vision_max_patches is not None:
        overrides["export.model.vision_max_patches"] = args.vision_max_patches
    result = workflow.export(quant, output_dir=args.output, device=args.device, config_overrides=overrides)
    print(result.work_dir)


if __name__ == "__main__":
    main()
