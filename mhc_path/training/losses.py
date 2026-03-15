"""Loss functions for DETR-style detection: focal, dice, CIoU, and Hungarian matching.

Key design decision: the Hungarian matcher uses GIoU for the cost matrix (stable
assignment early in training), while the actual box regression loss uses CIoU
(faster convergence via center-distance and aspect-ratio gradients).

Changes from RF-DETR reference:
- logsigmoid for numerical stability in matcher (matching reference)
- NaN/Inf handling in cost matrix (matching reference)
- Group DETR support in matcher (matching reference)
- Loss normalization by num_boxes (sum/num_boxes) instead of mean()
- Auxiliary loss support at intermediate decoder layers
- CIoU for box regression loss (spec choice; RF-DETR uses GIoU)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mhc_path.models.box_utils import (
    cxcywh_to_xyxy,
    complete_box_iou_loss,
    generalized_box_iou,
    generalized_box_iou_loss,
)
from mhc_path.models.util import get_world_size, is_dist_avail_and_initialized


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class DetectionTarget:
    """Per-image detection target in DETR-compatible format."""

    boxes: torch.Tensor    # (N, 4) cxcywh normalized
    labels: torch.Tensor   # (N,) int64
    masks: Optional[torch.Tensor] = None  # (N, H, W) binary or None


# ---------------------------------------------------------------------------
# Standalone loss functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss_onehot(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss with float one-hot targets (standard DETR formulation).

    Unmatched queries receive all-zeros targets, pushing all C foreground
    sigmoids toward 0 — no fake background dimension needed.

    Args:
        logits: (N, C) raw classification logits.
        targets: (N, C) float one-hot targets (all-zeros for background).
        num_boxes: normalisation denominator (number of matched boxes).
        alpha: Balancing factor for the positive class.
        gamma: Focusing parameter to down-weight easy examples.

    Returns:
        Scalar focal loss normalised by num_boxes.
    """
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * focal_weight * bce
    return loss.mean(1).sum() / num_boxes


def dice_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """Dice loss with internal sigmoid activation.

    Args:
        pred_masks: (N, H, W) raw mask logits (sigmoid applied internally).
        target_masks: (N, H, W) binary target masks.

    Returns:
        Scalar mean dice loss.
    """
    pred = pred_masks.sigmoid().flatten(1)   # (N, H*W)
    target = target_masks.flatten(1).float()  # (N, H*W)

    eps = 1e-6
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Hungarian bipartite matcher with Group DETR support
# ---------------------------------------------------------------------------

class HungarianMatcher:
    """Bipartite matching using GIoU for the cost matrix.

    Uses logsigmoid for numerical stability (matching RF-DETR reference).
    Supports Group DETR: splits queries into groups, runs independent
    matching per group, then concatenates indices.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: list[DetectionTarget],
        group_detr: int = 1,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal bipartite matching between predictions and targets.

        Args:
            outputs: dict with keys:
                - "pred_logits": (B, Q, C) classification logits
                - "pred_boxes":  (B, Q, 4) predicted boxes in cxcywh normalised
            targets: list of DetectionTarget, one per batch element.
            group_detr: number of query groups for Group DETR matching.

        Returns:
            List of (pred_indices, target_indices) tuples, one per batch element.
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten across batch for efficient computation
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)  # (B*Q, C)
        out_prob = flat_pred_logits.sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (B*Q, 4)

        tgt_ids = torch.cat([t.labels for t in targets])
        tgt_bbox = torch.cat([t.boxes for t in targets])

        if tgt_ids.numel() == 0:
            return [(
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
            )] * batch_size

        # --- Classification cost (focal-based with logsigmoid) ---
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-flat_pred_logits))
        )
        pos_cost_class = (
            alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        )
        class_cost = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # --- L1 box cost ---
        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1)

        # --- GIoU cost ---
        pred_xyxy = cxcywh_to_xyxy(out_bbox)
        tgt_xyxy = cxcywh_to_xyxy(tgt_bbox)
        giou_matrix = generalized_box_iou(pred_xyxy, tgt_xyxy)
        giou_cost = -giou_matrix

        # --- Combined cost ---
        C = (
            self.cost_class * class_cost
            + self.cost_bbox * bbox_cost
            + self.cost_giou * giou_cost
        )
        C = C.view(batch_size, num_queries, -1).float().cpu()

        # NaN/Inf handling (matching RF-DETR reference)
        max_cost = C.max() if C.numel() > 0 else 0
        C[C.isinf() | C.isnan()] = max_cost * 2

        sizes = [len(t.boxes) for t in targets]

        # Group DETR matching
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)

        indices: list[tuple] = []
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C_g.split(sizes, -1))
            ]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([ind1[0], ind2[0] + g_num_queries * g_i]),
                        np.concatenate([ind1[1], ind2[1]]),
                    )
                    for ind1, ind2 in zip(indices, indices_g)
                ]

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# ---------------------------------------------------------------------------
# Combined detection + segmentation loss
# ---------------------------------------------------------------------------

_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "cls": 2.0,
    "bbox": 5.0,
    "mask": 5.0,
    "dice": 2.0,
    "mask_boundary": 1.0,
}


# Sobel kernels for boundary loss
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0


def mask_boundary_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """Sobel gradient loss on predicted masks vs target masks.

    Encourages sharp boundaries without requiring a separate HV head.
    MSE of Sobel gradients, masked to target foreground pixels.

    Args:
        pred_masks: (N, H, W) raw mask logits (sigmoid applied internally).
        target_masks: (N, H, W) binary target masks.

    Returns:
        Scalar boundary loss.
    """
    device = pred_masks.device
    sobel_x = _SOBEL_X.to(device)
    sobel_y = _SOBEL_Y.to(device)

    pred = pred_masks.sigmoid().unsqueeze(1)   # (N, 1, H, W)
    tgt = target_masks.float().unsqueeze(1)    # (N, 1, H, W)

    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    tgt_gx = F.conv2d(tgt, sobel_x, padding=1)
    tgt_gy = F.conv2d(tgt, sobel_y, padding=1)

    fg = (target_masks > 0.5).unsqueeze(1).float()  # (N, 1, H, W)
    n_fg = fg.sum().clamp(min=1.0)

    diff_x = ((pred_gx - tgt_gx) ** 2) * fg
    diff_y = ((pred_gy - tgt_gy) ** 2) * fg

    return (diff_x.sum() + diff_y.sum()) / n_fg


_BOX_LOSS_FNS = {
    "ciou": complete_box_iou_loss,
    "giou": generalized_box_iou_loss,
}


class DetectionLoss(nn.Module):
    """Combined detection + segmentation loss with auxiliary loss support.

    Box regression uses CIoU (default) or GIoU loss. Classification uses
    sigmoid focal loss. Mask loss uses BCE + Dice. Loss normalisation:
    sum / num_boxes (matching RF-DETR).
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        loss_weights: Optional[dict[str, float]] = None,
        group_detr: int = 1,
        box_loss_type: str = "ciou",
        mask_loss_resolution: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_weights = dict(loss_weights) if loss_weights else dict(_DEFAULT_LOSS_WEIGHTS)
        self.group_detr = group_detr
        self.mask_loss_resolution = mask_loss_resolution
        if box_loss_type not in _BOX_LOSS_FNS:
            raise ValueError(f"box_loss_type must be one of {list(_BOX_LOSS_FNS)}, got {box_loss_type!r}")
        self._box_loss_fn = _BOX_LOSS_FNS[box_loss_type]

    def forward(
        self,
        outputs: dict,
        targets: list[DetectionTarget],
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components and the weighted total.

        Args:
            outputs: dict with keys:
                - "pred_logits": (B, Q, C) classification logits
                - "pred_boxes":  (B, Q, 4) predicted boxes (cxcywh normalised)
                - "pred_masks":  (B, Q, H, W) mask logits (optional)
                - "aux_outputs": list of dicts (optional)
            targets: list of DetectionTarget, one per batch element.

        Returns:
            Dict with keys "cls", "bbox", "mask", "dice", "mask_boundary", "total".
        """
        group_detr = self.group_detr if self.training else 1
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher.forward(outputs_no_aux, targets, group_detr=group_detr)

        # Compute num_boxes for normalisation
        num_boxes = sum(len(t.labels) for t in targets)
        num_boxes = num_boxes * group_detr
        num_boxes_t = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs["pred_logits"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_t)
        num_boxes = torch.clamp(num_boxes_t / get_world_size(), min=1).item()

        loss_cls = self._classification_loss(outputs_no_aux, targets, indices, num_boxes)
        loss_bbox = self._bbox_loss(outputs_no_aux, targets, indices, num_boxes)
        loss_mask, loss_dice, loss_boundary = self._mask_loss(outputs_no_aux, targets, indices)

        total = (
            self.loss_weights.get("cls", 0.0) * loss_cls
            + self.loss_weights.get("bbox", 0.0) * loss_bbox
            + self.loss_weights.get("mask", 0.0) * loss_mask
            + self.loss_weights.get("dice", 0.0) * loss_dice
            + self.loss_weights.get("mask_boundary", 0.0) * loss_boundary
        )

        # Auxiliary losses at intermediate decoder layers
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_pred = {
                    "pred_logits": aux_out["class_logits"],
                    "pred_boxes": aux_out["box_coords"],
                }
                aux_indices = self.matcher.forward(aux_pred, targets, group_detr=group_detr)
                aux_cls = self._classification_loss(aux_pred, targets, aux_indices, num_boxes)
                aux_bbox = self._bbox_loss(aux_pred, targets, aux_indices, num_boxes)
                total = total + (
                    self.loss_weights.get("cls", 0.0) * aux_cls
                    + self.loss_weights.get("bbox", 0.0) * aux_bbox
                )

        return {
            "cls": loss_cls,
            "bbox": loss_bbox,
            "mask": loss_mask,
            "dice": loss_dice,
            "mask_boundary": loss_boundary,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Private loss components
    # ------------------------------------------------------------------

    def _classification_loss(
        self,
        outputs: dict,
        targets: list[DetectionTarget],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> torch.Tensor:
        """Focal loss over all queries; unmatched queries get all-zeros target."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C)
        device = pred_logits.device
        batch_size, num_queries, _ = pred_logits.shape

        # Unmatched queries get class index = num_classes (background)
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.int64,
            device=device,
        )

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b].labels[tgt_idx].to(device)

        # One-hot over C+1 classes, then drop the background column.
        # Unmatched queries (class=num_classes) become all-zeros in C dims,
        # pushing all foreground sigmoids toward 0.
        target_onehot = F.one_hot(target_classes, self.num_classes + 1)
        target_onehot = target_onehot[..., :self.num_classes].float()  # (B, Q, C)

        loss = sigmoid_focal_loss_onehot(
            pred_logits.reshape(-1, self.num_classes),
            target_onehot.reshape(-1, self.num_classes),
            num_boxes=num_boxes,
        )
        return loss

    def _bbox_loss(
        self,
        outputs: dict,
        targets: list[DetectionTarget],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> torch.Tensor:
        """IoU-based + L1 box loss over matched pairs, normalised by num_boxes."""
        device = outputs["pred_boxes"].device
        matched_pred_boxes: list[torch.Tensor] = []
        matched_tgt_boxes: list[torch.Tensor] = []

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred_boxes.append(outputs["pred_boxes"][b][pred_idx])
            matched_tgt_boxes.append(targets[b].boxes[tgt_idx].to(device))

        if not matched_pred_boxes:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pred_cat = torch.cat(matched_pred_boxes, dim=0)
        tgt_cat = torch.cat(matched_tgt_boxes, dim=0)

        iou_loss = self._box_loss_fn(pred_cat, tgt_cat)
        l1 = F.l1_loss(pred_cat, tgt_cat, reduction="none").sum(dim=-1)

        return (iou_loss.sum() + l1.sum()) / num_boxes

    def _mask_loss(
        self,
        outputs: dict,
        targets: list[DetectionTarget],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BCE + dice + boundary loss over matched mask pairs."""
        device = outputs["pred_logits"].device

        if "pred_masks" not in outputs:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero.clone(), zero.clone()

        matched_pred_masks: list[torch.Tensor] = []
        matched_tgt_masks: list[torch.Tensor] = []

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            tgt = targets[b]
            if tgt.masks is None:
                continue
            matched_pred_masks.append(outputs["pred_masks"][b][pred_idx])
            matched_tgt_masks.append(tgt.masks[tgt_idx].to(device))

        if not matched_pred_masks:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero.clone(), zero.clone()

        pred_cat = torch.cat(matched_pred_masks, dim=0)
        tgt_cat = torch.cat(matched_tgt_masks, dim=0)

        # Resize both to intermediate resolution to save memory during backward
        res = (self.mask_loss_resolution, self.mask_loss_resolution)
        if pred_cat.shape[-2:] != res:
            pred_cat = F.interpolate(
                pred_cat.unsqueeze(1), size=res,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        if tgt_cat.shape[-2:] != res:
            tgt_cat = F.interpolate(
                tgt_cat.unsqueeze(1).float(), size=res,
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        mask_bce = F.binary_cross_entropy_with_logits(
            pred_cat, tgt_cat.float(), reduction="mean",
        )
        mask_dice = dice_loss(pred_cat, tgt_cat)
        loss_boundary = mask_boundary_loss(pred_cat, tgt_cat)

        return mask_bce, mask_dice, loss_boundary
