# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint numerical diagnostic, separate from Merak build/inference adapters."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


class NumericalMonitor(TorchDispatchMode):
    """Check module boundaries and FP16 values before clamps can hide infinity."""

    def __init__(self, report):
        super().__init__()
        self.report = report

    def observe(self, name, values):
        for value in tensors(values):
            if not value.is_floating_point() or not value.numel():
                continue
            finite = bool(torch.isfinite(value).all())
            maximum = float(value.detach().abs().max())
            stat = self.report['modules'].setdefault(
                name, {'calls': 0, 'absmax': 0., 'dtype': str(value.dtype), 'finite': True}
            )
            stat['calls'] += 1
            stat['absmax'] = max(stat['absmax'], maximum) if finite else None
            stat['finite'] &= finite
            if not finite:
                event = {'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                         'step': len(self.report['steps'])}
                self.report.setdefault('first_nonfinite', event)
                raise FloatingPointError(str(event))

    def hook(self, name):
        def run(module, args, kwargs, result):
            self.observe(name + '.input', (args, kwargs))
            self.observe(name + '.output', result)
        return run

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (func in (torch.ops.aten.clamp.default, torch.ops.aten.clip.default)
                and args and args[0].dtype == torch.float16):
            self.observe('fp16_pre_clamp', args[0])
        return func(*args, **(kwargs or {}))


def validate_clipping_coverage(report, dtype, layers, steps):
    """Fail closed if instrumentation misses the pinned upstream FP16 block clips."""
    if dtype != 'float16':
        return
    expected = layers * steps
    observed = report['modules'].get('fp16_pre_clamp', {}).get('calls', 0)
    report['clipping_coverage'] = {'expected_minimum': expected, 'observed': observed}
    if observed < expected:
        raise RuntimeError(f'Incomplete FP16 clipping monitoring: {observed} < {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='weights/Qwen-Image-2.1')
    parser.add_argument('--image', help='Optional reference image for editing')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or args.steps < 1:
        parser.error('size must be a positive multiple of 32; steps must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'args': vars(args), 'torch': torch.__version__, 'modules': {}, 'steps': [], 'status': 'running'}
    monitor = NumericalMonitor(report)
    start = time.time()

    def callback(pipe, step, timestep, values):
        monitor.observe('latents', values['latents'])
        report['steps'].append({'step': step, 't': float(timestep),
                                'absmax': float(values['latents'].abs().max())})
        torch.save(values['latents'].detach().cpu(), out / f'latent_{step:02d}.pt')
        print(report['steps'][-1], flush=True)
        return values

    try:
        import diffusers
        import transformers
        from diffusers import QwenImage21Pipeline
        from PIL import Image

        report.update(transformers=transformers.__version__, diffusers=diffusers.__version__,
                      diffusers_source=diffusers.__file__)
        pipe = QwenImage21Pipeline.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), local_files_only=True
        ).to('cuda')
        for component in ('text_encoder', 'transformer', 'vae'):
            for name, module in getattr(pipe, component).named_modules():
                module.register_forward_hook(monitor.hook(component + '.' + name), with_kwargs=True)
        # Identical FP32 noise for both dtypes, cast only inside the pipeline.
        latent_shape = (1, (args.size // pipe.vae_scale_factor)**2, pipe.transformer.config.in_channels)
        latents = torch.randn(latent_shape, generator=torch.Generator('cpu').manual_seed(42))
        torch.save(latents, out / 'initial_latents.pt')
        reference = None
        if args.image:
            with Image.open(args.image) as source:
                reference = source.copy()
        with torch.inference_mode(), monitor:
            result = pipe(
                prompt='A red ceramic teapot on a wooden table, soft window light, photographic detail.',
                image=reference, width=args.size, height=args.size, output_resolution=args.size,
                num_inference_steps=args.steps, latents=latents,
                generator=torch.Generator('cuda').manual_seed(42),
                callback_on_step_end=callback, use_kv_cache=not args.no_cache,
            )
        result.images[0].save(out / 'image.png')
        validate_clipping_coverage(report, args.dtype, len(pipe.transformer.transformer_blocks), args.steps)
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
    finally:
        report['elapsed_seconds'] = time.time() - start
        report['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print('RESULT', report['status'], report['elapsed_seconds'], flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
