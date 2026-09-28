"""Shared tested package versions, explicitly selected by each family plugin.

These are candidate direct requirements. Local modified distributions and
transitive package metadata still need release and clean-install validation.
"""

from xhmodel_merak.model_catalog.dependencies import DependencySpec


GPTQ_REQUIREMENTS = ("gptqmodel==5.8.0", "datasets==3.6.0")
AUTOROUND_REQUIREMENTS = ("auto-round==0.12.0", *GPTQ_REQUIREMENTS)

IMAGE = ("pillow==11.3.0", "torchvision==0.23.0")
AUDIO = ("soundfile==0.13.1", "librosa==0.11.0")
GEMMA_VIDEO = (*IMAGE, "torchcodec==0.7.0")
QWEN_MEDIA = (*IMAGE, "qwen-vl-utils==0.0.14")
WEMM_MEDIA = (*QWEN_MEDIA, "av==15.1.0")


def runtime_dependencies(
    *,
    common: tuple[str, ...] = (),
    build: tuple[str, ...] = (),
    inference: tuple[str, ...] = (),
    system: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
    transformers: str = "5.16.1",
    quantization: tuple[tuple[str, tuple[str, ...]], ...] = (),
    checkpoint_build: tuple[str, ...] = (),
) -> DependencySpec:
    return DependencySpec(
        common=(
            "xhmodel-zoo==1.0.0", "xhquanttool==0.9.2", "torch==2.8.0", "triton==3.4.0",
            f"transformers=={transformers}", "numpy==2.2.6", "safetensors==0.8.0",
            "accelerate==1.13.0", "einops==0.8.2", "PyYAML==6.0.3", "onnx==1.18.0",
            "onnx-graphsurgeon==0.5.8", "sentencepiece==0.2.1", *common,
        ),
        build=("onnxscript==0.6.2", "onnxsim==0.6.2", "ninja==1.13.0", *build),
        inference=inference,
        quantization=quantization,
        checkpoint_build=checkpoint_build,
        system=("Linux x86_64; CUDA 12.8 compatible NVIDIA driver", *system),
        unresolved=(
            "Release the tested modified xhmodel-zoo and xhquanttool sources as immutable distributions",
            "Resolve installed distribution metadata conflicts before clean-environment validation",
            "Pin CUDA 12.8 wheel sources and transitive dependencies in the installation lock",
            *unresolved,
        ),
    )
