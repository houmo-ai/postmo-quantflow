# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json


def prepare_source(source, root, calibration, inputs, kwargs, replacements):
    from _unified_special_fixtures import write_ling

    config = write_ling(source)
    # Match the tiny numeric vocabulary; natural language would become
    # repeated UNK tokens and be rejected by AutoRound's sample filter.
    calibration.write_text(
        "".join(
            json.dumps({"text": " ".join(str(4 + (i + row) % 6) for i in range(128))}) + "\n" for row in range(2)
        )
    )
    inputs["prompt"] = "4 5 6"
    return config
