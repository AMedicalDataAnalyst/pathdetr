"""Pure-function box operations: format conversions and IoU variants."""

from __future__ import annotations

import math

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def _pairwise_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def _pairwise_enclosing(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Returns (enclosing_x1, enclosing_y1, enclosing_x2, enclosing_y2) for all pairs."""
    ex1 = torch.min(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    ey1 = torch.min(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    ex2 = torch.max(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    ey2 = torch.max(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    return ex1, ey1, ex2, ey2


def _pairwise_center_dist_sq(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    cx1 = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) * 0.5
    dx = cx1.unsqueeze(1) - cx2.unsqueeze(0)
    dy = cy1.unsqueeze(1) - cy2.unsqueeze(0)
    return dx * dx + dy * dy


def _safe_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (iou, union, area1, area2) with safe handling of degenerate boxes."""
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    inter = _pairwise_intersection(boxes1, boxes2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    eps = 1e-7
    iou = inter / (union + eps)
    iou = torch.where(union > 0, iou, torch.zeros_like(iou))
    return iou, union, area1, area2


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, _union, _a1, _a2 = _safe_iou(boxes1, boxes2)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union, _a1, _a2 = _safe_iou(boxes1, boxes2)
    ex1, ey1, ex2, ey2 = _pairwise_enclosing(boxes1, boxes2)
    enclosing_area = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)
    eps = 1e-7
    giou = iou - (enclosing_area - union) / (enclosing_area + eps)
    return giou


def distance_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, _union, _a1, _a2 = _safe_iou(boxes1, boxes2)
    center_dist_sq = _pairwise_center_dist_sq(boxes1, boxes2)
    ex1, ey1, ex2, ey2 = _pairwise_enclosing(boxes1, boxes2)
    diagonal_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    eps = 1e-7
    diou = iou - center_dist_sq / (diagonal_sq + eps)
    return diou


def complete_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, _union, _a1, _a2 = _safe_iou(boxes1, boxes2)
    center_dist_sq = _pairwise_center_dist_sq(boxes1, boxes2)
    ex1, ey1, ex2, ey2 = _pairwise_enclosing(boxes1, boxes2)
    diagonal_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    eps = 1e-7

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    atan1 = torch.atan2(w1, h1 + eps)
    atan2 = torch.atan2(w2, h2 + eps)
    diff = atan1.unsqueeze(1) - atan2.unsqueeze(0)
    v = (4.0 / (math.pi ** 2)) * (diff ** 2)

    alpha = v / ((1.0 - iou) + v + eps)
    alpha = alpha.detach()

    ciou = iou - center_dist_sq / (diagonal_sq + eps) - alpha * v
    return ciou


def _elementwise_iou_components(
    pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (iou, union, inter) element-wise for matched pairs."""
    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0)
    area_t = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) * (
        target_xyxy[:, 3] - target_xyxy[:, 1]
    ).clamp(min=0)

    ix1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    iy1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    ix2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    iy2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    union = area_p + area_t - inter
    eps = 1e-7
    iou = inter / (union + eps)
    iou = torch.where(union > 0, iou, torch.zeros_like(iou))
    return iou, union, inter


def complete_box_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_xyxy = cxcywh_to_xyxy(pred)
    target_xyxy = cxcywh_to_xyxy(target)

    iou, _union, _inter = _elementwise_iou_components(pred_xyxy, target_xyxy)

    cx_p = pred[:, 0]
    cy_p = pred[:, 1]
    cx_t = target[:, 0]
    cy_t = target[:, 1]
    center_dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

    ex1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    ey1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    ex2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    ey2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    diagonal_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    eps = 1e-7
    w_p = pred[:, 2].clamp(min=0)
    h_p = pred[:, 3].clamp(min=0)
    w_t = target[:, 2].clamp(min=0)
    h_t = target[:, 3].clamp(min=0)

    atan_p = torch.atan2(w_p, h_p + eps)
    atan_t = torch.atan2(w_t, h_t + eps)
    v = (4.0 / (math.pi ** 2)) * (atan_t - atan_p) ** 2

    alpha = v / ((1.0 - iou) + v + eps)
    alpha = alpha.detach()

    ciou = iou - center_dist_sq / (diagonal_sq + eps) - alpha * v
    return 1.0 - ciou


def distance_box_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise DIoU loss for matched pred/target pairs (cxcywh format).

    DIoU = IoU - center_dist^2 / diagonal^2. Simpler than CIoU (no aspect
    ratio term), better suited for detection-only tasks where aspect ratio
    matching is less critical.
    """
    pred_xyxy = cxcywh_to_xyxy(pred)
    target_xyxy = cxcywh_to_xyxy(target)

    iou, _union, _inter = _elementwise_iou_components(pred_xyxy, target_xyxy)

    cx_p = pred[:, 0]
    cy_p = pred[:, 1]
    cx_t = target[:, 0]
    cy_t = target[:, 1]
    center_dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

    ex1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    ey1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    ex2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    ey2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    diagonal_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    eps = 1e-7
    diou = iou - center_dist_sq / (diagonal_sq + eps)
    return 1.0 - diou


def generalized_box_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_xyxy = cxcywh_to_xyxy(pred)
    target_xyxy = cxcywh_to_xyxy(target)

    iou, union, _inter = _elementwise_iou_components(pred_xyxy, target_xyxy)

    ex1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    ey1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    ex2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    ey2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enclosing_area = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    eps = 1e-7
    giou = iou - (enclosing_area - union) / (enclosing_area + eps)
    return 1.0 - giou
