# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Install this directory into ComfyUI/custom_nodes as a link or copy."""

from .nodes import MerakQwenImage21TurboW8A8


NODE_CLASS_MAPPINGS = {"MerakQwenImage21TurboW8A8": MerakQwenImage21TurboW8A8}
NODE_DISPLAY_NAME_MAPPINGS = {"MerakQwenImage21TurboW8A8": "Qwen Image 2.1 Turbo · Merak W8A8"}
