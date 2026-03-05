"""Tests for full mHC-Path model assembly."""

import torch
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig


def _make_model():
    """Small model using fallback ViT (8-layer) for fast tests."""
    config = MHCPathConfig(
        backbone="dinov3_vitl16",
        backbone_frozen=True,
        fpn_dim=64,
        num_queries=10,
        num_classes=5,
        num_decoder_layers=2,
        sa_n_heads=4,
        ca_n_heads=4,
        n_points=4,
        group_detr=1,
        with_segmentation=True,
        mask_upsample_factor=1,
        output_layers=(2, 4, 6, 8),
        fpn_levels=4,
    )
    return MHCPath(config)


def test_end_to_end_forward():
    model = _make_model()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert "class_logits" in out
    assert "box_coords" in out
    assert "mask_logits" in out


def test_output_shapes():
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out["class_logits"].shape == (2, 10, 5)
    assert out["box_coords"].shape == (2, 10, 4)
    B, N, H, W = out["mask_logits"].shape
    assert B == 2
    assert N == 10


def test_trainable_parameters():
    model = _make_model()
    groups = model.trainable_parameters()
    assert len(groups) == 2
    names = {g["group_name"] for g in groups}
    assert names == {"fpn", "decoder"}


def test_backbone_frozen():
    model = _make_model()
    for p in model.backbone.parameters():
        assert not p.requires_grad


def test_fpn_decoder_trainable():
    model = _make_model()
    fpn_trainable = any(p.requires_grad for p in model.fpn.parameters())
    dec_trainable = any(p.requires_grad for p in model.decoder.parameters())
    assert fpn_trainable
    assert dec_trainable
