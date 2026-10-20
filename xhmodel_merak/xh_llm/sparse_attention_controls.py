"""Host controls for block-sparse attention over a contiguous prefix."""

import torch


def build_sparse_attention_controls(
    past_length, current_length, input_length, context_length, selected_pools, *, pool_size=4, device="cpu",
    dtype=torch.float32,
):
    if past_length < 0 or not 0 < current_length <= input_length:
        raise ValueError("require L >= 0 and 0 < n <= fixed input length")
    if past_length + current_length > context_length:
        raise ValueError("request exceeds the configured context capacity")

    query_slots = torch.arange(input_length, device=device)
    real = query_slots < current_length
    visible = past_length + query_slots + 1
    complete = visible // pool_size
    pool_valid = torch.where(real, complete, 0).unsqueeze(0).to(torch.int32)
    tail_slots = torch.arange(pool_size - 1, device=device)
    # Streaming tail windows contain the last three tokens in chronological
    # order.  Only the rightmost visible % C4 tokens belong to the open pool.
    tail_valid = (tail_slots[None, :] >= (pool_size - 1 - visible % pool_size)[:, None]) & real[:, None]
    selected_slots = torch.arange(selected_pools * pool_size, device=device)
    selected_valid = selected_slots[None, :] < (pool_valid[0].clamp(max=selected_pools) * pool_size)[:, None]
    attention_valid = torch.cat((selected_valid, tail_valid), dim=-1)
    attention_mask = torch.zeros_like(attention_valid, dtype=dtype).masked_fill(~attention_valid, -65504.0)

    def scalar(value):
        return torch.tensor([value], device=device, dtype=torch.int32)

    return {
        "pool_offset": scalar(pool_size - past_length % pool_size),
        "pool_new_count": scalar((past_length % pool_size + current_length) // pool_size),
        "pool_write_start": scalar(past_length // pool_size),
        "query_validity": real[None].to(dtype),
        "attention_mask": attention_mask[None, :, None, :],
    }
