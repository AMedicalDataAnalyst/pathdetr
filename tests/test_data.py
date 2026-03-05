"""Tests for dataset and augmentation modules."""

import torch
import pytest

from mhc_path.data.dataset import (
    DetectionTarget,
    detection_collate_fn,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
)
from mhc_path.data.gpu_augmentation import GPUPathologyAugPipeline, GPUStainAugmentation
from mhc_path.data.stain_augmentation import (
    StainStatsExtractor,
    RandStainAugmentation,
    rgb_to_od,
    od_to_rgb,
    rgb_to_hed,
    hed_to_rgb,
)


# ---------------------------------------------------------------------------
# Box conversions
# ---------------------------------------------------------------------------


def test_xyxy_to_cxcywh_roundtrip():
    boxes = torch.tensor([[10, 20, 30, 40], [0, 0, 100, 100]], dtype=torch.float32)
    cxcywh = xyxy_to_cxcywh(boxes)
    back = cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(boxes, back, atol=1e-5)


# ---------------------------------------------------------------------------
# Detection collation
# ---------------------------------------------------------------------------


def test_detection_collate_fn_shapes():
    batch = [
        {
            "image": torch.rand(3, 256, 256),
            "target": DetectionTarget(
                boxes=torch.rand(3, 4),
                labels=torch.tensor([0, 1, 2]),
                masks=None,
            ),
            "stain_stats": None,
        },
        {
            "image": torch.rand(3, 256, 256),
            "target": DetectionTarget(
                boxes=torch.rand(5, 4),
                labels=torch.tensor([0, 1, 2, 3, 4]),
                masks=None,
            ),
            "stain_stats": None,
        },
    ]
    out = detection_collate_fn(batch)
    assert out["images"].shape == (2, 3, 256, 256)
    assert out["boxes"].shape == (2, 5, 4)  # padded to max 5
    assert out["labels"].shape == (2, 5)
    assert out["num_objects"].tolist() == [3, 5]
    assert out["masks"] is None


def test_detection_collate_fn_with_masks():
    batch = [
        {
            "image": torch.rand(3, 256, 256),
            "target": DetectionTarget(
                boxes=torch.rand(2, 4),
                labels=torch.tensor([0, 1]),
                masks=torch.ones(2, 256, 256, dtype=torch.bool),
            ),
            "stain_stats": None,
        },
    ]
    out = detection_collate_fn(batch)
    assert out["masks"] is not None
    assert out["masks"].shape == (1, 2, 256, 256)


# ---------------------------------------------------------------------------
# Stain augmentation
# ---------------------------------------------------------------------------


def test_stain_stats_extractor():
    image = torch.rand(3, 64, 64)
    extractor = StainStatsExtractor(color_spaces=("HED",))
    stats = extractor(image)
    assert stats.foreground_mask.shape == (1, 64, 64)
    assert "HED" in stats.channel_means
    assert stats.channel_means["HED"].shape == (3,)


def test_rand_stain_aug_shape():
    image = torch.rand(3, 64, 64)
    aug = RandStainAugmentation(color_spaces=("HED",), p=1.0)
    out = aug(image)
    assert out.shape == (3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_rgb_od_roundtrip():
    img = torch.rand(3, 16, 16).clamp(0.01, 1.0)
    od = rgb_to_od(img)
    back = od_to_rgb(od)
    assert torch.allclose(img, back, atol=1e-4)


# ---------------------------------------------------------------------------
# GPU augmentation pipeline
# ---------------------------------------------------------------------------


def test_gpu_aug_pipeline_shapes():
    pipeline = GPUPathologyAugPipeline(
        target_size=256, stain_aug=False, geometric=True,
    )
    batch = {
        "images": torch.rand(2, 3, 256, 256),
        "boxes": torch.rand(2, 5, 4),
        "labels": torch.zeros(2, 5, dtype=torch.int64),
        "num_objects": torch.tensor([3, 5]),
        "masks": None,
    }
    out = pipeline(batch)
    assert out["images"].shape == (2, 3, 256, 256)
    assert out["boxes"].shape == (2, 5, 4)


def test_gpu_stain_aug_output_range():
    aug = GPUStainAugmentation(p=1.0)
    images = torch.rand(4, 3, 64, 64)
    out = aug(images)
    assert out.shape == (4, 3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
