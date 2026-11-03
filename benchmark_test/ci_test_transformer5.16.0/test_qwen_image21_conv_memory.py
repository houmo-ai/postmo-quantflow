"""Large VAE canvases use bounded unfold with unchanged SEFP convolution results."""

import inspect

import pytest
import torch


@pytest.mark.parametrize("weight_bits,activation_bits", [(8, 8), (8, 16), (16, 16)])
@pytest.mark.parametrize("kernel,padding,stride,dilation", [(3, 1, 1, 1), (3, 2, 2, 2), (1, 0, 1, 1)])
def test_bounded_vae_convolution_matches_full_unfold(weight_bits, activation_bits, kernel, padding, stride, dilation):
    from xhquant.backend.xh2a.functions.te.conv2d import conv2d_unfold_weight_xh2a_default
    from xhquant.backend.xh2a.ir.operators import conv
    from xhquant.backend.xh2a.ir.operators._conv_bounded_unfold import bounded_unfold_dispatch_opt

    torch.manual_seed(42)
    channels, output_channels = 144, 17
    value = torch.randn(2, channels, 17, 23, device="cuda", dtype=torch.float16)
    # Padding holes remain within every row, as with a rectangular VAE canvas.
    value[..., 13:, :] = 0
    value[..., :, 19:] = 0
    weight = torch.randn(output_channels, channels, kernel, kernel, device="cuda", dtype=torch.float16) * 0.01
    bias = torch.randn(output_channels, device="cuda", dtype=torch.float16) * 0.1
    packed = conv2d_unfold_weight_xh2a_default(weight, bit=weight_bits, hidden_bit=True, stricted=True, nshare=64)
    q_weight, scale, _ = packed.unpack(is_scale=False)
    arguments = dict(
        input=value,
        q_weight=q_weight,
        w_scale=scale,
        weight=weight,
        bias=bias,
        have_bias=True,
        in_channels=channels,
        out_channels=output_channels,
        mode="sefp",
        hmfp_act_exp_bit=5,
        hmfp_act_man_bit=activation_bits,
        hmfp_act_nshare=64,
        hmfp_act_rounding="round",
        hmfp_act_hidden_bit=True,
        hmfp_act_max_exp_boost=0,
        hmfp_weight_exp_bit=5,
        hmfp_weight_man_bit=weight_bits,
        hmfp_weight_nshare=64,
        hmfp_weight_rounding="round",
        hmfp_weight_hidden_bit=True,
        hmfp_psum_dtype="float16",
        hmfp_psum_round_mode="round",
        kernel_shape=(kernel, kernel),
        pads=(padding, padding),
        strides=(stride, stride),
        dilations=(dilation, dilation),
        groups=1,
        with_relu=True,
        output_dtype="float32",
    )
    original = conv.conv2d_xh2a_default_unbounded_opt
    reference = original(**arguments)
    observed_shapes = []

    def observe(*args, **kwargs):
        import inspect

        call = inspect.signature(original).bind(*args, **kwargs)
        observed_shapes.append(tuple(call.arguments["input"].shape))
        return original(*args, **kwargs)

    import functools

    functools.update_wrapper(observe, original)
    bounded = bounded_unfold_dispatch_opt(observe, max_unfold_elements=80_000)
    actual = bounded(**arguments)
    assert len(observed_shapes) > 1 and max(shape[-2] for shape in observed_shapes) < value.shape[-2]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # SDK may wrap bounded unfold with another dispatch, such as single-input
    # VP convolution. Bounded unfold must remain in the default wrapper chain.
    assert inspect.unwrap(
        conv.conv2d_xh2a_default,
        stop=lambda function: function is conv.conv2d_xh2a_default_bounded_unfold_opt,
    ) is conv.conv2d_xh2a_default_bounded_unfold_opt
