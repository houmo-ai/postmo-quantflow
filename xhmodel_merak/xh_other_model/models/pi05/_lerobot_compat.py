# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Instance-local compatibility for LeRobot's native PiGemma reference forward."""

import inspect
from types import FunctionType, MethodType


def language_prefix_embeddings(policy, tokens):
    """Apply Gemma's input scale once, independent of embedding module version."""
    import math
    from torch.nn import functional as F

    embedding = policy.model.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    # GemmaTextScaledWordEmbedding in Transformers 5.16 scales in forward;
    # the older nn.Embedding did not. Raw checkpoint weights are unscaled in
    # both versions, also matching the standalone compact deployment runtime.
    return F.embedding(tokens, embedding.weight) * math.sqrt(embedding.weight.shape[1])


def adapt_pi05_reference(policy):
    """Keep native forward math and weights; adapt only its mask API call.

    Pi0.5 supplies a complete 4D prefix/suffix mask. Transformers returns that
    mask unchanged, so its removed cache_position argument is redundant here.
    A private globals dictionary avoids modifying LeRobot or Transformers for
    other models/threads. Unsupported mask forms fail rather than guessing cache
    positions. No installed third-party files are modified.
    """
    from lerobot.policies.pi_gemma import PiGemmaModel

    original = PiGemmaModel.forward
    mask_factory = original.__globals__["create_causal_mask"]
    if "cache_position" in inspect.signature(mask_factory).parameters:
        return policy

    def complete_mask(*, cache_position, attention_mask, **kwargs):
        if attention_mask is None or attention_mask.ndim != 4:
            raise ValueError("Pi0.5 reference compatibility requires a complete 4D attention mask")
        return mask_factory(attention_mask=attention_mask, **kwargs)

    namespace = dict(original.__globals__, create_causal_mask=complete_mask)
    adapted = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    for module in policy.modules():
        if isinstance(module, PiGemmaModel):
            module.forward = MethodType(adapted, module)
    return policy
