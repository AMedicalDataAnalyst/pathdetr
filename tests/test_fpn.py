"""Tests for Feature Pyramid Network."""

import torch
from mhc_path.models.fpn import PathologyFPN


def test_fpn_output_shapes():
    in_dims = {"6": 1024, "12": 1024, "18": 1024, "24": 1024}
    fpn = PathologyFPN(in_dims=in_dims, out_dim=256, num_levels=4)

    features = {
        "6": torch.randn(2, 256, 1024),
        "12": torch.randn(2, 256, 1024),
        "18": torch.randn(2, 256, 1024),
        "24": torch.randn(2, 256, 1024),
    }
    spatial_shapes = (16, 16)
    outputs = fpn(features, spatial_shapes)

    assert len(outputs) == 4
    assert outputs[0].shape == (2, 256, 16, 16)
    assert outputs[1].shape == (2, 256, 8, 8)
    assert outputs[2].shape == (2, 256, 4, 4)
    assert outputs[3].shape == (2, 256, 2, 2)


def test_fpn_channel_dimension():
    in_dims = {"8": 768, "16": 768}
    fpn = PathologyFPN(in_dims=in_dims, out_dim=128, num_levels=3)

    features = {
        "8": torch.randn(1, 256, 768),
        "16": torch.randn(1, 256, 768),
    }
    outputs = fpn(features, (16, 16))
    for out in outputs:
        assert out.shape[1] == 128
