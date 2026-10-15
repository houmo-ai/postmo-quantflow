"""通过统一推理接口生成或编辑图片，输出实际尺寸的 RGBA 图片。"""

import argparse
import json
from pathlib import Path

from xhmodel_merak.model_inference import inference_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="build_model 返回的完整产物目录")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help="参考图路径；数量和尺寸受包内容量限制")
    parser.add_argument("--height", type=int, help="输出高度；不传时使用包内配置")
    parser.add_argument("--width", type=int, help="输出宽度；不传时使用包内配置")
    parser.add_argument("--steps", type=int, help="模型没有固定采样序列时使用的步数")
    parser.add_argument("--sigmas", type=float, nargs="+", help="显式覆盖模型的采样序列")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", help="CFG 负向提示词，可为空字符串")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", help="推理设备，例如 cuda:0")
    parser.add_argument("--output", required=True, help="保存生成或编辑的图片")
    args = parser.parse_args(argv)
    if args.guidance_scale > 1 and args.negative_prompt is None:
        parser.error("--guidance-scale above 1 requires --negative-prompt")

    inputs = dict(prompt=args.prompt, images=args.image, seed=args.seed,
                  guidance_scale=args.guidance_scale)
    for name in ("height", "width", "negative_prompt", "steps", "sigmas"):
        value = getattr(args, name)
        if value is not None:
            inputs[name] = value
    result = inference_model(artifact_dir=args.artifact_dir, inputs=inputs, device=args.device)

    from PIL import Image

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image).save(output)
    print(json.dumps(dict(output_image=str(output), width=result.width, height=result.height,
                          seed=result.seed, latency_s=result.latency_s), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
