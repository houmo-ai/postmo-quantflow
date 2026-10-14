# Copyright (c) 2023 DeepSeek (prompt protocol and effort text)
# Copyright 2026 HOUMO AI (single-turn Jinja adaptation)
# SPDX-License-Identifier: MIT
# See licenses/DeepSeek-V4/LICENSE; source attribution is retained in the template.
"""Package the official Flash single-user encoding used by the text API."""

FLASH_CHAT_TEMPLATE = r"""{# Copyright (c) 2023 DeepSeek; Copyright 2026 HOUMO AI (Jinja adaptation).
Source: DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py
SHA256: abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
#}
{%- if messages|length != 1 or messages[0]['role'] != 'user' or messages[0]['content'] is not string -%}{{ raise_exception('DeepSeek Flash public text input requires one user string') }}{%- endif -%}
{%- set thinking = enable_thinking|default(true) -%}
{%- set effort = reasoning_effort|default('max') -%}
{%- set prefixes = {"low":"","high":"Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n","max":"Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"} -%}
{%- if thinking and effort not in prefixes -%}{{ raise_exception('Invalid DeepSeek reasoning effort') }}{%- endif -%}
{{- bos_token + (prefixes[effort] if thinking else '') + '<｜User｜>' + messages[0]['content'] + '<｜Assistant｜>' + ('<think>' if thinking else '</think>') -}}
"""


def configure_flash_chat_tokenizer(tokenizer):
    if not tokenizer.chat_template:
        if (tokenizer.bos_token, tokenizer.eos_token) != (
            '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>'
        ):
            raise ValueError('Unrecognized DeepSeek Flash tokenizer without a chat template')
        tokenizer.chat_template = FLASH_CHAT_TEMPLATE
    return tokenizer


def decode_flash_thinking_completion(tokenizer, token_ids):
    """Separate thinking tokens before returning a public text answer."""
    close_token = tokenizer.convert_tokens_to_ids('</think>')
    if close_token is None or close_token == tokenizer.unk_token_id:
        raise ValueError('DeepSeek thinking completion requires its closing token')
    try:
        boundary = token_ids.index(close_token)
    except ValueError:
        return '', {"reasoning_tokens": len(token_ids), "thinking_complete": False,
                    "reasoning_text": tokenizer.decode(token_ids, skip_special_tokens=True)}
    return tokenizer.decode(token_ids[boundary + 1:], skip_special_tokens=True), {
        "reasoning_tokens": boundary + 1, "thinking_complete": True,
        "reasoning_text": tokenizer.decode(token_ids[:boundary], skip_special_tokens=True),
    }
