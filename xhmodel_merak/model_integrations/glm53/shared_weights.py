"""GLM language weights use the common stage-local hardlink layout."""

from xhmodel_merak.model_integrations.shared_weights import compact_language_weights, share_language_weights


def compact_glm53_language_weights(root):
    return compact_language_weights(root, metadata_key="glm53_shared_external_data")
