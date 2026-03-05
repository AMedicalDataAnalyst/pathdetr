"""Tests for DINOv3 backbone adapter."""

import torch
from mhc_path.models.backbone_adapter import DINOv3Backbone


def _make_backbone():
    """Create a backbone with fallback ViT (8-layer for speed)."""
    return DINOv3Backbone(
        model_name="dinov3_vitl16",
        frozen=True,
        output_layers=(2, 4, 6, 8),
    )


def test_output_shapes():
    bb = _make_backbone()
    x = torch.randn(1, 3, 256, 256)
    features = bb(x)
    assert len(features) == 4
    for key in ["2", "4", "6", "8"]:
        assert key in features
        assert features[key].shape[0] == 1
        assert features[key].shape[2] == bb._embed_dim


def test_all_params_frozen():
    bb = _make_backbone()
    for param in bb.backbone.parameters():
        assert not param.requires_grad


def test_feature_dims():
    bb = _make_backbone()
    dims = bb.feature_dims
    assert len(dims) == 4
    for key, dim in dims.items():
        assert dim == bb._embed_dim


def test_num_patches():
    bb = _make_backbone()
    h, w = bb.num_patches
    assert h == 16
    assert w == 16
