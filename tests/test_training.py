"""Tests for loss computation and training engine."""

import torch
from mhc_path.training.losses import (
    DetectionLoss, DetectionTarget, HungarianMatcher,
    sigmoid_focal_loss_onehot, dice_loss,
)


def test_focal_loss_shape():
    logits = torch.randn(20, 5)
    targets = torch.zeros(20, 5)
    targets[0, 2] = 1.0
    targets[1, 0] = 1.0
    loss = sigmoid_focal_loss_onehot(logits, targets, num_boxes=2.0)
    assert loss.shape == ()
    assert loss.item() > 0


def test_dice_loss_perfect():
    pred = torch.full((3, 8, 8), 10.0)  # sigmoid -> ~1.0
    target = torch.ones(3, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() < 0.01


def test_dice_loss_worst():
    pred = torch.full((3, 8, 8), -10.0)  # sigmoid -> ~0.0
    target = torch.ones(3, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() > 0.9


def test_matcher_output_format():
    matcher = HungarianMatcher()
    outputs = {
        "pred_logits": torch.randn(2, 10, 5),
        "pred_boxes": torch.rand(2, 10, 4).sigmoid(),
    }
    targets = [
        DetectionTarget(boxes=torch.rand(3, 4), labels=torch.tensor([0, 1, 2])),
        DetectionTarget(boxes=torch.rand(2, 4), labels=torch.tensor([0, 3])),
    ]
    indices = matcher.forward(outputs, targets)
    assert len(indices) == 2
    assert len(indices[0][0]) == 3  # 3 targets matched
    assert len(indices[1][0]) == 2  # 2 targets matched


def test_detection_loss_keys():
    matcher = HungarianMatcher()
    criterion = DetectionLoss(num_classes=5, matcher=matcher)
    outputs = {
        "pred_logits": torch.randn(2, 10, 5),
        "pred_boxes": torch.rand(2, 10, 4).sigmoid(),
    }
    targets = [
        DetectionTarget(boxes=torch.rand(3, 4), labels=torch.tensor([0, 1, 2])),
        DetectionTarget(boxes=torch.rand(2, 4), labels=torch.tensor([0, 3])),
    ]
    losses = criterion(outputs, targets)
    assert "cls" in losses
    assert "bbox" in losses
    assert "mask" in losses
    assert "dice" in losses
    assert "total" in losses


def test_detection_loss_backward():
    matcher = HungarianMatcher()
    criterion = DetectionLoss(num_classes=5, matcher=matcher)
    criterion.train()
    outputs = {
        "pred_logits": torch.randn(1, 10, 5, requires_grad=True),
        "pred_boxes": torch.rand(1, 10, 4, requires_grad=True).sigmoid(),
    }
    targets = [
        DetectionTarget(boxes=torch.rand(2, 4), labels=torch.tensor([0, 1])),
    ]
    losses = criterion(outputs, targets)
    losses["total"].backward()
    assert outputs["pred_logits"].grad is not None


def test_detection_loss_with_masks():
    matcher = HungarianMatcher()
    criterion = DetectionLoss(num_classes=5, matcher=matcher, mask_loss_resolution=16)
    outputs = {
        "pred_logits": torch.randn(1, 10, 5),
        "pred_boxes": torch.rand(1, 10, 4).sigmoid(),
        "pred_masks": torch.randn(1, 10, 16, 16),
    }
    targets = [
        DetectionTarget(
            boxes=torch.rand(2, 4),
            labels=torch.tensor([0, 1]),
            masks=torch.randint(0, 2, (2, 16, 16)).float(),
        ),
    ]
    losses = criterion(outputs, targets)
    assert losses["mask"].item() > 0
    assert losses["dice"].item() > 0
