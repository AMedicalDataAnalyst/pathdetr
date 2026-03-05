"""Tests for evaluation metrics on synthetic data."""

import torch
from mhc_path.training.losses import DetectionTarget
from mhc_path.evaluation.metrics import DetectionMetrics, PanopticQuality


def _make_perfect_predictions():
    """Create predictions that perfectly match targets (boxes in xyxy)."""
    targets = [
        DetectionTarget(
            boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
            labels=torch.tensor([0, 1]),
        ),
    ]
    # xyxy format: cxcywh (0.5,0.5,0.2,0.2) -> (0.4,0.4,0.6,0.6)
    predictions = [
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6], [0.25, 0.25, 0.35, 0.35]]),
            "scores": torch.tensor([0.95, 0.90]),
            "labels": torch.tensor([0, 1]),
        },
    ]
    return predictions, targets


def test_detection_metrics_compute():
    dm = DetectionMetrics(num_classes=5)
    preds, tgts = _make_perfect_predictions()
    dm.update(preds, tgts)
    results = dm.compute()
    assert "mAP@50" in results
    assert results["mAP@50"] >= 0.0


def test_detection_metrics_has_f1d():
    dm = DetectionMetrics(num_classes=5)
    preds, tgts = _make_perfect_predictions()
    dm.update(preds, tgts)
    results = dm.compute()
    assert "F1d@50" in results


def test_detection_metrics_reset():
    dm = DetectionMetrics(num_classes=5)
    preds, tgts = _make_perfect_predictions()
    dm.update(preds, tgts)
    dm.reset()
    results = dm.compute()
    assert results["mAP@50"] == 0.0


def test_pq_compute():
    pq = PanopticQuality(num_classes=5)
    preds, tgts = _make_perfect_predictions()
    pq.update(preds, tgts)
    results = pq.compute()
    assert "PQ" in results
    assert "DQ" in results
    assert "SQ" in results
    for cls in range(5):
        assert f"PQ_class{cls}" in results


def test_pq_no_predictions():
    pq = PanopticQuality(num_classes=5)
    targets = [
        DetectionTarget(
            boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            labels=torch.tensor([0]),
        ),
    ]
    predictions = [
        {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)},
    ]
    pq.update(predictions, targets)
    results = pq.compute()
    assert results["PQ_class0"] == 0.0


def test_pq_reset():
    pq = PanopticQuality(num_classes=5)
    preds, tgts = _make_perfect_predictions()
    pq.update(preds, tgts)
    pq.reset()
    results = pq.compute()
    assert results["PQ"] == 0.0
