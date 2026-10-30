"""Public recipe for the fixed two-view DepthSplat Gaussian encoder."""

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.dependencies import DependencySpec
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._shared import variant

PROVIDER_ID = "depthsplat"
MODEL_ID = "depthsplat-gs-small-re10k-256x256-view2"


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name="gaussian-scene"), provider=PROVIDER_ID)
    for kind, target in (
        ("build", "xhmodel_merak.model_integrations.depthsplat.build"),
        ("inference", "xhmodel_merak.model_integrations.depthsplat.inference:adapter"),
    ):
        catalog.register_adapter(
            AdapterSpec(name=f"depthsplat.{kind}", kind=kind, target=target, artifact_formats=("gaussian-scene",)),
            provider=PROVIDER_ID,
        )
    catalog.register_recipe(
        ModelRecipe(
            model_id=MODEL_ID,
            display_name="DepthSplat GS Small RE10K, two views, 256x256",
            family="depthsplat",
            export_name="depthsplat_gs_small_re10k_256x256_view2",
            task="gaussian-scene-reconstruction",
            modalities=("image",),
            variants=(
                variant("full", "configs_merak/workflows/xh2a/other_models/depthsplat/depthsplat_xh2a_w8a8.yaml"),
            ),
            default_precision=ResolvedPrecision(weight="w16", activation="a16"),
            supported_precisions=(("w16", "a16"),),
            quant_methods=(),
            deployment_precision={"visual": "w8a8"},
            components=("visual",),
            golden_components=(),
            build_adapter="depthsplat.build",
            inference_adapter="depthsplat.inference",
            artifact_format="gaussian-scene",
            input_schema={
                "type": "object",
                "properties": {"sample": {"type": "string", "minLength": 1, "description": "Prepared input NPZ path"}},
                "required": ["sample"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"stats": {"type": "object"}}, "required": ["stats"]},
            artifact_name_template="hmquant_xh2_depthsplat_gs_small_re10k_{deployment_precision}_2v_256x256",
            default_profile="2v_256x256",
            profiles=("2v_256x256",),
            source_constraints=(("model_type", "DepthSplatONNX"), ("profile", "2v_256x256")),
            dependencies=DependencySpec(
                common=(
                    "torch==2.8.0",
                    "transformers==5.16.1",
                    "onnx==1.18.0",
                    "PyYAML==6.0.3",
                ),
                # NumPy/ORT/accelerate come from the prepared ModelZoo/SDK environment.
                system=(
                    "ModelZoo available as an installed package or importable source checkout",
                    "Compatible xhquant SDK supplied by the release or development environment",
                ),
                unresolved=(
                    "Clean installation with the selected xhquant SDK remains unverified",
                    "Prepared float ONNX bundle must match the declared upstream checkpoint and profile",
                ),
            ),
        ),
        provider=PROVIDER_ID,
    )


__all__ = ["PROVIDER_ID", "MODEL_ID", "register"]
