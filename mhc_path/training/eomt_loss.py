"""EoMT mask-classification loss: mask-based Hungarian matching + CE/BCE/Dice.

Follows Mask2Former/EoMT loss protocol:
  1. Hungarian matching using point-sampled mask costs (BCE + Dice) + class cost
  2. Supervised losses on matched pairs: CE (class) + BCE (mask) + Dice (mask)
  3. Multi-layer supervision: same loss computed at each decoder block output

Reference: Cheng et al., "Masked-attention Mask Transformer", CVPR 2022.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Point sampling (following Mask2Former / EoMT)
# ---------------------------------------------------------------------------

def _sample_points_uniformly(n_points: int, device: torch.device) -> torch.Tensor:
    """Sample uniform random points in [0, 1]^2. Returns (1, n_points, 2)."""
    return torch.rand(1, n_points, 2, device=device)


def _sample_at_points(
    masks: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Sample mask values at given points using grid_sample.

    Args:
        masks: (N, 1, H, W) or (N, H, W) mask logits.
        points: (1, P, 2) or (N, P, 2) points in [0, 1].

    Returns:
        (N, P) sampled values.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    # grid_sample expects coords in [-1, 1]
    grid = points * 2 - 1  # [0,1] -> [-1,1]
    if grid.ndim == 2:
        grid = grid.unsqueeze(0)
    grid = grid.unsqueeze(1)  # (N, 1, P, 2)
    if grid.shape[0] == 1 and masks.shape[0] > 1:
        grid = grid.expand(masks.shape[0], -1, -1, -1)
    sampled = F.grid_sample(masks, grid, mode="bilinear", align_corners=False)
    return sampled.squeeze(1).squeeze(1)  # (N, P)


def _importance_sample_points(
    mask_logits: torch.Tensor,
    n_points: int,
    oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
) -> torch.Tensor:
    """Importance sampling: prefer uncertain points (|logit| near 0).

    Args:
        mask_logits: (N, H, W) logits for matched masks.
        n_points: total points to sample.
        oversample_ratio: oversample candidates before selecting.
        importance_ratio: fraction of points chosen by uncertainty.

    Returns:
        (N, n_points, 2) sampled point coordinates.
    """
    N = mask_logits.shape[0]
    n_candidate = int(n_points * oversample_ratio)
    n_important = int(n_points * importance_ratio)
    n_random = n_points - n_important

    # Candidate points
    candidate_pts = torch.rand(N, n_candidate, 2, device=mask_logits.device)
    candidate_vals = _sample_at_points(mask_logits, candidate_pts)

    # Most uncertain = smallest |logit|
    uncertainty = candidate_vals.abs()
    _, topk_idx = uncertainty.topk(n_important, dim=1, largest=False)

    important_pts = candidate_pts.gather(
        1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))

    # Random fill
    random_pts = torch.rand(N, n_random, 2, device=mask_logits.device)

    return torch.cat([important_pts, random_pts], dim=1)


# ---------------------------------------------------------------------------
# Mask-based Hungarian matcher
# ---------------------------------------------------------------------------

class MaskHungarianMatcher:
    """Bipartite matching using mask costs (BCE + Dice) + class cost."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        n_match_points: int = 12544,
    ) -> None:
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.n_match_points = n_match_points

    @torch.no_grad()
    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        gt_masks: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal matching.

        Args:
            mask_logits: (B, Q, H_mask, W_mask) predicted mask logits.
            class_logits: (B, Q, num_classes+1) including "no object".
            gt_masks: list of (N_i, H_img, W_img) binary GT masks.
            gt_labels: list of (N_i,) class labels.

        Returns:
            List of (pred_indices, gt_indices) per batch element.
        """
        B, Q = mask_logits.shape[:2]
        device = mask_logits.device

        # Uniform sample points for matching
        match_points = _sample_points_uniformly(self.n_match_points, device)

        indices = []
        for b in range(B):
            n_gt = len(gt_labels[b])
            if n_gt == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device),
                ))
                continue

            # Class cost: negative softmax probability of the GT class
            probs = class_logits[b].softmax(-1)  # (Q, C+1)
            class_cost = -probs[:, gt_labels[b]]  # (Q, N_gt)

            # Sample predicted masks at match points
            pred_sampled = _sample_at_points(
                mask_logits[b], match_points)  # (Q, P)

            # Sample GT masks at match points (resize GT to [0,1] coords)
            gt_m = gt_masks[b].float()  # (N_gt, H, W)
            gt_sampled = _sample_at_points(gt_m, match_points)  # (N_gt, P)

            # Mask BCE cost: pairwise
            pred_sig = pred_sampled.sigmoid()
            # (Q, 1, P) vs (1, N_gt, P)
            bce_cost = F.binary_cross_entropy_with_logits(
                pred_sampled[:, None, :].expand(-1, n_gt, -1),
                gt_sampled[None, :, :].expand(Q, -1, -1),
                reduction="none",
            ).mean(-1)  # (Q, N_gt)

            # Dice cost: pairwise
            num = 2 * (pred_sig[:, None, :] * gt_sampled[None, :, :]).sum(-1)
            den = pred_sig[:, None, :].sum(-1) + gt_sampled[None, :, :].sum(-1)
            dice_cost = 1.0 - (num + 1) / (den + 1)  # (Q, N_gt)

            # Combined cost
            C = (self.cost_class * class_cost
                 + self.cost_mask * bce_cost
                 + self.cost_dice * dice_cost)

            pred_idx, gt_idx = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=device),
                torch.tensor(gt_idx, dtype=torch.long, device=device),
            ))

        return indices


# ---------------------------------------------------------------------------
# EoMT Loss
# ---------------------------------------------------------------------------

class EoMTLoss(nn.Module):
    """Mask-classification loss for EoMT.

    Components:
      - Softmax CE for classification (with down-weighted "no object" class)
      - Point-sampled BCE for masks
      - Point-sampled Dice for masks
      - Multi-layer supervision (same loss at each decoder block output)
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
        n_loss_points: int = 12544,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.n_loss_points = n_loss_points

        # CE with down-weighted "no object" class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = MaskHungarianMatcher()

    def _single_layer_loss(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        gt_masks: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss for one layer's output."""
        B, Q = mask_logits.shape[:2]
        device = mask_logits.device

        # Match
        indices = self.matcher.forward(mask_logits, class_logits, gt_masks, gt_labels)

        # --- Classification loss ---
        # Target: "no object" for unmatched queries
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = gt_labels[b][gt_idx]

        cls_loss = F.cross_entropy(
            class_logits.view(-1, self.num_classes + 1),
            target_classes.view(-1),
            weight=self.empty_weight.to(class_logits.device),
        )

        # --- Mask losses (only on matched pairs) ---
        all_pred_masks = []
        all_gt_masks = []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            all_pred_masks.append(mask_logits[b, pred_idx])  # (K, H, W)
            all_gt_masks.append(gt_masks[b][gt_idx].float())  # (K, H_gt, W_gt)

        if not all_pred_masks:
            zero = mask_logits.sum() * 0
            return {"cls": cls_loss, "mask": zero, "dice": zero,
                    "total": self.class_weight * cls_loss}

        pred_m = torch.cat(all_pred_masks, dim=0)  # (K_total, H, W)
        gt_m = torch.cat(all_gt_masks, dim=0)      # (K_total, H_gt, W_gt)

        # Importance-sample points for loss
        sample_pts = _importance_sample_points(
            pred_m, self.n_loss_points)  # (K, P, 2)

        pred_sampled = _sample_at_points(pred_m, sample_pts)  # (K, P)
        gt_sampled = _sample_at_points(gt_m, sample_pts)      # (K, P)

        # BCE mask loss
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_sampled, gt_sampled, reduction="mean")

        # Dice mask loss
        pred_sig = pred_sampled.sigmoid()
        num = 2 * (pred_sig * gt_sampled).sum(-1)
        den = pred_sig.sum(-1) + gt_sampled.sum(-1)
        dice_loss = (1.0 - (num + 1) / (den + 1)).mean()

        total = (self.class_weight * cls_loss
                 + self.mask_weight * mask_loss
                 + self.dice_weight * dice_loss)

        return {
            "cls": cls_loss,
            "mask": mask_loss,
            "dice": dice_loss,
            "total": total,
        }

    def forward(
        self,
        outputs: dict,
        targets,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-layer loss.

        Args:
            outputs: dict with 'mask_logits', 'class_logits',
                     optionally 'aux_outputs' (list of same).
            targets: list of DetectionTarget (with .masks and .labels),
                     or tuple of (gt_masks, gt_labels).

        Returns:
            dict with loss components and 'total'.
        """
        # Extract GT masks and labels from targets
        if isinstance(targets, (list, tuple)) and hasattr(targets[0], 'masks'):
            gt_masks = [t.masks for t in targets]
            gt_labels = [t.labels for t in targets]
        else:
            gt_masks, gt_labels = targets

        # Get mask logits — handle both raw and normalized key names
        mask_key = "mask_logits" if "mask_logits" in outputs else "pred_masks"
        cls_key = "class_logits" if "class_logits" in outputs else "pred_logits"

        # Final layer loss
        losses = self._single_layer_loss(
            outputs[mask_key], outputs[cls_key],
            gt_masks, gt_labels,
        )

        # Auxiliary layer losses (same weight, summed)
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"]:
                aux_mask_key = "mask_logits" if "mask_logits" in aux else "pred_masks"
                aux_cls_key = "class_logits" if "class_logits" in aux else "pred_logits"
                aux_losses = self._single_layer_loss(
                    aux[aux_mask_key], aux[aux_cls_key],
                    gt_masks, gt_labels,
                )
                losses["total"] = losses["total"] + aux_losses["total"]

        return losses
