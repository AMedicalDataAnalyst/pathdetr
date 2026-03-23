"""DN-DETR-style denoising training for RF-DETR decoder.

During training, perturbed GT boxes and labels are prepended as extra
"denoising" queries. The decoder must reconstruct the clean GT from
the noised version. This accelerates convergence because it provides
direct supervision signal without Hungarian matching.

At inference, denoising queries are not used.

Reference: Li et al., "DN-DETR: Accelerate DETR Training by Introducing
Query DeNoising", CVPR 2022.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_path.models.util import inverse_sigmoid


@dataclass
class DenoisingOutput:
    """Output of denoising query generation."""
    dn_queries: torch.Tensor           # (B, N_dn, d_model)
    dn_reference_points: torch.Tensor  # (B, N_dn, 4)
    dn_labels: torch.Tensor            # (B, N_dn) — clean GT labels
    dn_boxes: torch.Tensor             # (B, N_dn, 4) — clean GT boxes
    dn_mask: torch.Tensor              # (N_dn + N_q, N_dn + N_q) attention mask
    n_dn: int                          # number of denoising queries per sample


class DenoisingGenerator(nn.Module):
    """Generate denoising queries from ground truth.

    For each GT instance, creates ``num_dn_groups`` noised copies with
    perturbed boxes and optionally flipped labels.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 5,
        num_dn_groups: int = 5,
        box_noise_scale: float = 0.4,
        label_noise_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_dn_groups = num_dn_groups
        self.box_noise_scale = box_noise_scale
        self.label_noise_ratio = label_noise_ratio

        # Learnable label embedding for denoising queries
        self.label_embed = nn.Embedding(num_classes, d_model)

    def forward(
        self,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        num_queries: int,
    ) -> Optional[DenoisingOutput]:
        """Generate denoising queries from GT.

        Args:
            gt_boxes: List of (N_i, 4) tensors in cxcywh normalised format.
            gt_labels: List of (N_i,) class label tensors.
            num_queries: Number of learnable queries (for attention mask sizing).

        Returns:
            DenoisingOutput or None if no GT objects in the batch.
        """
        batch_size = len(gt_boxes)
        device = gt_boxes[0].device

        # Find max GT count to pad to uniform size
        n_gts = [len(b) for b in gt_boxes]
        max_gt = max(n_gts) if n_gts else 0
        if max_gt == 0:
            return None

        # Cap total dn queries to avoid OOM on dense images
        max_dn = 100
        effective_groups = min(self.num_dn_groups, max(1, max_dn // max_gt))
        n_dn_per_sample = max_gt * effective_groups

        # Pad GT to uniform size
        padded_boxes = torch.zeros(batch_size, max_gt, 4, device=device)
        padded_labels = torch.zeros(batch_size, max_gt, dtype=torch.long, device=device)
        valid_mask = torch.zeros(batch_size, max_gt, dtype=torch.bool, device=device)

        for i, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            n = len(boxes)
            if n > 0:
                padded_boxes[i, :n] = boxes
                padded_labels[i, :n] = labels
                valid_mask[i, :n] = True

        # Repeat for dn groups: (B, max_gt * effective_groups, ...)
        boxes_rep = padded_boxes.repeat(1, effective_groups, 1)   # (B, N_dn, 4)
        labels_rep = padded_labels.repeat(1, effective_groups)    # (B, N_dn)
        valid_rep = valid_mask.repeat(1, effective_groups)        # (B, N_dn)

        # --- Box noise (following DN-DETR: additive, center uses half-wh) ---
        cx, cy, w, h = boxes_rep.unbind(-1)
        noise = torch.rand_like(boxes_rep) * 2 - 1  # uniform [-1, 1]
        # DN-DETR: diff[:2] = wh/2 (center), diff[2:] = wh (size)
        noised_cx = cx + noise[..., 0] * (w / 2) * self.box_noise_scale
        noised_cy = cy + noise[..., 1] * (h / 2) * self.box_noise_scale
        noised_w = w + noise[..., 2] * w * self.box_noise_scale
        noised_h = h + noise[..., 3] * h * self.box_noise_scale
        noised_boxes = torch.stack([noised_cx, noised_cy, noised_w, noised_h], dim=-1)
        noised_boxes = noised_boxes.clamp(min=0.0, max=1.0)

        # --- Label noise: flip to random class with probability ---
        label_noise_mask = torch.rand_like(labels_rep.float()) < self.label_noise_ratio
        random_labels = torch.randint_like(labels_rep, 0, self.num_classes)
        noised_labels = torch.where(label_noise_mask, random_labels, labels_rep)

        # --- Build denoising queries from label embeddings ---
        dn_queries = self.label_embed(noised_labels)  # (B, N_dn, d_model)
        dn_reference_points = noised_boxes             # (B, N_dn, 4)

        # Zero out padding positions
        dn_queries = dn_queries * valid_rep.unsqueeze(-1).float()
        dn_reference_points = dn_reference_points * valid_rep.unsqueeze(-1).float()

        # --- Attention mask ---
        # Denoising queries can attend to each other within the same group,
        # but NOT across groups and NOT to/from learnable queries.
        total_q = n_dn_per_sample + num_queries
        attn_mask = torch.zeros(total_q, total_q, dtype=torch.bool, device=device)

        # DN-DETR: learnable queries CANNOT see dn queries (prevents leakage),
        # but dn queries CAN see learnable queries (richer context).
        attn_mask[n_dn_per_sample:, :n_dn_per_sample] = True

        # Block attention between different dn groups
        for i in range(effective_groups):
            for j in range(effective_groups):
                if i != j:
                    attn_mask[i * max_gt:(i + 1) * max_gt,
                              j * max_gt:(j + 1) * max_gt] = True

        return DenoisingOutput(
            dn_queries=dn_queries,
            dn_reference_points=dn_reference_points,
            dn_labels=labels_rep,
            dn_boxes=boxes_rep,
            dn_mask=attn_mask,
            n_dn=n_dn_per_sample,
        )


def split_denoising_outputs(
    outputs: dict[str, torch.Tensor],
    n_dn: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split decoder outputs into denoising and matching components.

    Args:
        outputs: Decoder output dict with class_logits, box_coords, etc.
        n_dn: Number of denoising queries.

    Returns:
        (matching_outputs, denoising_outputs) — each a dict with the same keys.
    """
    matching = {}
    denoising = {}

    # Support both raw keys (class_logits/box_coords) and normalized (pred_logits/pred_boxes)
    _SPLIT_KEYS = [
        "class_logits", "box_coords", "mask_logits",
        "pred_logits", "pred_boxes", "pred_masks",
    ]
    for key in _SPLIT_KEYS:
        if key in outputs:
            denoising[key] = outputs[key][:, :n_dn]
            matching[key] = outputs[key][:, n_dn:]

    if "aux_outputs" in outputs:
        matching["aux_outputs"] = []
        denoising["aux_outputs"] = []
        for aux in outputs["aux_outputs"]:
            m_aux, d_aux = {}, {}
            for key in _SPLIT_KEYS:
                if key in aux:
                    d_aux[key] = aux[key][:, :n_dn]
                    m_aux[key] = aux[key][:, n_dn:]
            matching["aux_outputs"].append(m_aux)
            denoising["aux_outputs"].append(d_aux)

    return matching, denoising


def denoising_loss(
    dn_outputs: dict[str, torch.Tensor],
    dn_labels: torch.Tensor,
    dn_boxes: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Compute denoising reconstruction loss (no Hungarian matching needed).

    Args:
        dn_outputs: dict with class_logits (B, N_dn, C) and box_coords (B, N_dn, 4).
        dn_labels: (B, N_dn) clean GT class labels.
        dn_boxes: (B, N_dn, 4) clean GT boxes in cxcywh.

    Returns:
        dict with 'dn_loss_cls' and 'dn_loss_box' scalar tensors.
    """
    pred_logits = dn_outputs.get("pred_logits", dn_outputs.get("class_logits"))
    pred_boxes = dn_outputs.get("pred_boxes", dn_outputs.get("box_coords"))

    # Valid mask: non-padding positions (boxes with non-zero area)
    valid = (dn_boxes[..., 2] > 0) & (dn_boxes[..., 3] > 0)

    # Classification: sigmoid focal loss (alpha=0.25, gamma=2.0, matching DN-DETR)
    alpha, gamma = 0.25, 2.0
    target_onehot = F.one_hot(dn_labels, num_classes).float()  # (B, N_dn, C)
    prob = pred_logits.sigmoid()
    focal_weight = target_onehot * alpha * (1 - prob) ** gamma + \
                   (1 - target_onehot) * (1 - alpha) * prob ** gamma
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_onehot, reduction="none")
    cls_loss = (focal_weight * bce).sum(-1)  # (B, N_dn)
    cls_loss = (cls_loss * valid.float()).sum() / valid.float().sum().clamp(min=1)

    # Box: L1 + GIoU
    from mhc_path.models.box_utils import generalized_box_iou_loss

    pred_flat = pred_boxes[valid]    # (N_valid, 4)
    target_flat = dn_boxes[valid]    # (N_valid, 4)

    if pred_flat.numel() == 0:
        box_loss = pred_boxes.sum() * 0
    else:
        l1_loss = F.l1_loss(pred_flat, target_flat, reduction="mean")
        giou_loss = generalized_box_iou_loss(pred_flat, target_flat).mean()
        box_loss = 2.0 * l1_loss + 1.0 * giou_loss

    return {
        "dn_loss_cls": cls_loss,
        "dn_loss_box": box_loss,
    }
