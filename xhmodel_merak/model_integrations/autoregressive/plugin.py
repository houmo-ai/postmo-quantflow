from __future__ import annotations

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._shared import AUTOREGRESSIVE_BUILD


PROVIDER_ID = "autoregressive"


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(
        ArtifactFormatSpec(
            name="autoregressive",
            validator="xhmodel_merak.model_integrations.autoregressive.artifact:validate_artifact",
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=AUTOREGRESSIVE_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.autoregressive.build",
            artifact_formats=("autoregressive",),
            source_config_for_embeddings=True,
        ),
        provider=PROVIDER_ID,
    )


__all__ = ["PROVIDER_ID", "register"]
