# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed graph capacities and the independent reference preprocessing budget."""

DEFAULT_CAPACITY = dict(
    output_size=[1024, 1024],
    reference_size=[1024, 1024],
    canvas_size=[1536, 1536],
    target_capacity=4352,
    reference_capacity=4352,
    reference_total_capacity=8704,
    text_capacity=1280,
    max_references=8,
    single_reference_max_pixels=1048576,
    multi_reference_max_pixels=1048576,
)


def validate_capacity(values):
    result = {**DEFAULT_CAPACITY, **{key: values[key] for key in DEFAULT_CAPACITY if key in values}}
    for name in ("output_size", "reference_size", "canvas_size"):
        size = result[name]
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(type(n) is not int or n <= 0 or n % 16 for n in size)
        ):
            raise ValueError(f"{name} must contain positive height/width multiples of 16")
        result[name] = list(size)
    for name in set(DEFAULT_CAPACITY) - {"output_size", "reference_size", "canvas_size"}:
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if result["text_capacity"] > 4096 or result["max_references"] > 10:
        raise ValueError("reference protocol supports at most 4096 text tokens and ten references")
    if result["reference_capacity"] > result["reference_total_capacity"]:
        raise ValueError("single reference capacity exceeds total reference capacity")
    minimum_canvas = min(64, result["canvas_size"][0]) * min(64, result["canvas_size"][1])
    if result["single_reference_max_pixels"] < minimum_canvas:
        raise ValueError("single reference pixel budget cannot fit the minimum padded canvas")
    result["joint_capacity"] = result["target_capacity"] + result["reference_total_capacity"] + result["text_capacity"]
    validate_request(result, result["output_size"], [result["reference_size"]], min(80, result["text_capacity"]))
    return result


def validate_request(capacity, target_hw, reference_hws, text_length):
    if len(reference_hws) > capacity["max_references"]:
        raise ValueError("reference count exceeds the published capacity")
    if type(text_length) is not int or not 0 < text_length <= capacity["text_capacity"]:
        raise ValueError("text including template and official padding exceeds the text capacity")
    counts = []
    for index, size in enumerate([target_hw, *reference_hws]):
        if len(size) != 2 or any(type(n) is not int or n <= 0 or n % 16 for n in size):
            raise ValueError("neural image dimensions must be positive multiples of 16")
        if any(n > bound for n, bound in zip(size, capacity["canvas_size"], strict=True)):
            raise ValueError("image dimensions exceed the VAE canvas")
        count = size[0] * size[1] // 256
        if count > capacity["target_capacity" if index == 0 else "reference_capacity"]:
            raise ValueError("image token count exceeds the published capacity")
        counts.append(count)
    if sum(counts[1:]) > capacity["reference_total_capacity"]:
        raise ValueError("total reference tokens exceed the published capacity")
    if sum(counts) + text_length > capacity["joint_capacity"]:
        raise ValueError("joint sequence exceeds the published capacity")
    return counts
