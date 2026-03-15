"""Tests for RF-DETR decoder and deformable attention."""

import torch
from mhc_path.models.decoder import RFDETRDecoder, SegmentationHead


def _make_decoder(**kwargs):
    defaults = dict(
        d_model=64, num_queries=10, num_classes=5,
        n_decoder_layers=2, n_levels=3, sa_n_heads=4, ca_n_heads=4,
        n_points=4, dim_feedforward=128, group_detr=1,
        with_segmentation=True, mask_upsample_factor=1,
    )
    defaults.update(kwargs)
    return RFDETRDecoder(**defaults)


def _make_features(B=2, d_model=64, sizes=((8, 8), (4, 4), (2, 2))):
    return [torch.randn(B, d_model, H, W) for H, W in sizes]


def test_decoder_output_keys():
    dec = _make_decoder()
    dec.eval()
    feats = _make_features()
    out = dec(feats)
    assert "class_logits" in out
    assert "box_coords" in out
    assert "mask_logits" in out
    assert "aux_outputs" not in out  # eval mode


def test_decoder_output_shapes():
    dec = _make_decoder(num_queries=10, num_classes=5)
    dec.eval()
    out = dec(_make_features(B=2))
    assert out["class_logits"].shape == (2, 10, 5)
    assert out["box_coords"].shape == (2, 10, 4)


def test_decoder_box_format():
    dec = _make_decoder()
    dec.eval()
    out = dec(_make_features())
    boxes = out["box_coords"]
    # Boxes should be in [0, 1] range (sigmoid output)
    assert (boxes >= 0).all() and (boxes <= 1).all()


def test_decoder_aux_outputs_in_train():
    dec = _make_decoder(n_decoder_layers=3)
    dec.train()
    out = dec(_make_features())
    assert "aux_outputs" in out
    assert len(out["aux_outputs"]) == 3


def test_decoder_no_segmentation():
    dec = _make_decoder(with_segmentation=False)
    dec.eval()
    out = dec(_make_features())
    assert "mask_logits" not in out


def test_segmentation_head_output_shape():
    head = SegmentationHead(d_model=64, pixel_dim=32, upsample_factor=1)
    queries = torch.randn(2, 10, 64)
    pixels = torch.randn(2, 64, 8, 8)
    masks, _ = head(queries, pixels)
    assert masks.shape == (2, 10, 8, 8)


def test_segmentation_head_upsample():
    head = SegmentationHead(d_model=64, pixel_dim=32, upsample_factor=4)
    queries = torch.randn(2, 10, 64)
    pixels = torch.randn(2, 64, 8, 8)
    masks, _ = head(queries, pixels)
    assert masks.shape == (2, 10, 32, 32)


def test_decoder_mask_shape_with_upsample():
    dec = _make_decoder(mask_upsample_factor=4)
    dec.eval()
    out = dec(_make_features(B=1, sizes=((16, 16), (8, 8), (4, 4))))
    assert out["mask_logits"].shape == (1, 10, 64, 64)
