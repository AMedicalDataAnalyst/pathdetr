"""Slicing Aided Hyper Inference (SAHI) for small object detection.

Implements the inference-time slicing strategy from Akyon et al. (2022):
split each image into overlapping crops, run the detector on each crop
plus the full image, remap predictions to full-image coordinates, and
merge via NMS to remove duplicates.

This is inference-only — no training changes required. Works with any
model that accepts (B, 3, H, W) and returns pred_logits + pred_boxes
(+ optional pred_masks).

Reference: https://arxiv.org/abs/2202.06934
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_path.models.box_utils import cxcywh_to_xyxy, xyxy_to_cxcywh


@dataclass
class SAHIConfig:
    """Configuration for sliced inference.

    Attributes:
        slice_size: Height and width of each crop in pixels.
        overlap_ratio: Fraction of overlap between adjacent crops (0-1).
        include_full_image: Run detector on the full image too (recommended).
        score_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for cross-slice NMS merging.
        image_size: Expected input image size (assumes square).
    """
    slice_size: int = 128
    overlap_ratio: float = 0.25
    include_full_image: bool = True
    score_threshold: float = 0.1
    nms_iou_threshold: float = 0.5
    image_size: int = 256


def _compute_slices(
    image_size: int, slice_size: int, overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Compute (y1, x1, y2, x2) crop coordinates with overlap.

    Returns a list of pixel-coordinate crop windows that tile the image
    with the requested overlap.
    """
    if slice_size >= image_size:
        return [(0, 0, image_size, image_size)]

    stride = max(1, int(slice_size * (1 - overlap_ratio)))
    slices = []
    for y in range(0, image_size, stride):
        for x in range(0, image_size, stride):
            y2 = min(y + slice_size, image_size)
            x2 = min(x + slice_size, image_size)
            y1 = y2 - slice_size  # snap to keep slice_size consistent
            x1 = x2 - slice_size
            slices.append((y1, x1, y2, x2))

    # Deduplicate (can happen at boundaries)
    return list(dict.fromkeys(slices))


def _normalize_output(raw: dict) -> dict:
    """Normalise model output keys."""
    out: dict[str, torch.Tensor] = {}
    out["pred_logits"] = raw.get("pred_logits", raw.get("class_logits"))
    out["pred_boxes"] = raw.get("pred_boxes", raw.get("box_coords"))
    if "pred_masks" in raw:
        out["pred_masks"] = raw["pred_masks"]
    elif "mask_logits" in raw:
        out["pred_masks"] = raw["mask_logits"]
    return out


def _extract_predictions(
    outputs: dict, score_threshold: float, include_masks: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract (boxes_xyxy, scores, labels, masks) from model output for one image.

    Boxes are returned in normalised xyxy format [0,1].
    """
    logits = outputs["pred_logits"][0]  # (Q, C)
    scores, labels = logits.sigmoid().max(dim=-1)
    boxes = cxcywh_to_xyxy(outputs["pred_boxes"][0])  # (Q, 4) normalised xyxy

    masks = None
    if include_masks and "pred_masks" in outputs:
        mask_logits = outputs["pred_masks"][0]  # (Q, h, w)
        masks = mask_logits  # keep as logits, upsample later

    keep = scores >= score_threshold
    return boxes[keep], scores[keep], labels[keep], masks[keep] if masks is not None else None


def _remap_boxes_to_full(
    boxes: torch.Tensor,
    y1: int, x1: int, y2: int, x2: int,
    image_size: int,
) -> torch.Tensor:
    """Remap normalised [0,1] boxes from crop coords to full-image coords.

    Args:
        boxes: (N, 4) xyxy in [0,1] relative to the crop.
        y1, x1, y2, x2: Crop window in pixels.
        image_size: Full image size in pixels.
    """
    crop_w = x2 - x1
    crop_h = y2 - y1
    # Scale from [0,1] in crop space to pixel coords in crop
    boxes_px = boxes.clone()
    boxes_px[:, 0] = boxes[:, 0] * crop_w + x1
    boxes_px[:, 1] = boxes[:, 1] * crop_h + y1
    boxes_px[:, 2] = boxes[:, 2] * crop_w + x1
    boxes_px[:, 3] = boxes[:, 3] * crop_h + y1
    # Normalise to full image [0,1]
    boxes_px /= image_size
    return boxes_px


def _remap_masks_to_full(
    mask_logits: torch.Tensor,
    y1: int, x1: int, y2: int, x2: int,
    image_size: int,
) -> torch.Tensor:
    """Place crop-space mask logits into full-image canvas.

    Args:
        mask_logits: (N, h, w) mask logits from the crop.
        y1, x1, y2, x2: Crop window in pixels.
        image_size: Full image size in pixels.

    Returns:
        (N, image_size, image_size) mask logits in full-image space,
        with -10 (strong negative) outside the crop region.
    """
    n = mask_logits.shape[0]
    if n == 0:
        return torch.zeros(0, image_size, image_size, device=mask_logits.device)

    # Upsample to crop pixel size
    crop_h, crop_w = y2 - y1, x2 - x1
    up = F.interpolate(
        mask_logits.unsqueeze(1), size=(crop_h, crop_w),
        mode="bilinear", align_corners=False,
    ).squeeze(1)  # (N, crop_h, crop_w)

    # Place into full canvas
    canvas = torch.full(
        (n, image_size, image_size), -10.0,
        device=mask_logits.device, dtype=mask_logits.dtype,
    )
    canvas[:, y1:y2, x1:x2] = up
    return canvas


def _batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Per-class NMS. Returns indices to keep."""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    # Offset boxes by class to prevent cross-class suppression
    max_coord = boxes.max()
    offsets = labels.float() * (max_coord + 1)
    shifted = boxes + offsets[:, None]
    return torch.ops.torchvision.nms(shifted, scores, iou_threshold)


def _nms_fallback(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Pure-torch NMS fallback (no torchvision dependency)."""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(order)):
        idx = order[i].item()
        if suppressed[idx]:
            continue
        keep.append(idx)
        # Compute IoU with remaining
        xx1 = torch.max(boxes[idx, 0], boxes[order[i + 1:], 0])
        yy1 = torch.max(boxes[idx, 1], boxes[order[i + 1:], 1])
        xx2 = torch.min(boxes[idx, 2], boxes[order[i + 1:], 2])
        yy2 = torch.min(boxes[idx, 3], boxes[order[i + 1:], 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
        area_j = (boxes[order[i + 1:], 2] - boxes[order[i + 1:], 0]) * \
                 (boxes[order[i + 1:], 3] - boxes[order[i + 1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        # Suppress same-class detections with high IoU
        for j_off in range(len(iou)):
            j_idx = order[i + 1 + j_off].item()
            if labels[idx] == labels[j_idx] and iou[j_off] > iou_threshold:
                suppressed[j_idx] = True

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def nms_merge(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """NMS with torchvision fallback to pure-torch."""
    try:
        return _batched_nms(boxes, scores, labels, iou_threshold)
    except (RuntimeError, AttributeError):
        return _nms_fallback(boxes, scores, labels, iou_threshold)


@torch.no_grad()
def sahi_predict_single(
    model: nn.Module,
    image: torch.Tensor,
    config: SAHIConfig,
) -> dict[str, torch.Tensor]:
    """Run SAHI on a single image.

    Args:
        model: Detection model accepting (1, 3, H, W) input.
        image: (3, H, W) input tensor (already on device).
        config: SAHI configuration.

    Returns:
        Dictionary with ``boxes`` (N, 4) xyxy normalised, ``scores`` (N,),
        ``labels`` (N,), and optionally ``masks`` (N, image_size, image_size).
    """
    device = image.device
    H, W = image.shape[1], image.shape[2]
    has_masks = True  # try to get masks, gracefully handle if model doesn't produce them

    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []

    slices = _compute_slices(H, config.slice_size, config.overlap_ratio)

    # --- Run on each slice ---
    for (y1, x1, y2, x2) in slices:
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)  # (1, 3, sh, sw)
        # Resize crop to model's expected input size if needed
        if crop.shape[2] != H or crop.shape[3] != W:
            crop = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)

        raw = model(crop)
        outputs = _normalize_output(raw)

        boxes, scores, labels, masks = _extract_predictions(
            outputs, config.score_threshold, has_masks)

        if len(boxes) == 0:
            continue

        # Remap boxes from crop space to full-image space
        boxes = _remap_boxes_to_full(boxes, y1, x1, y2, x2, H)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        if masks is not None:
            full_masks = _remap_masks_to_full(masks, y1, x1, y2, x2, H)
            all_masks.append(full_masks)

    # --- Run on full image ---
    if config.include_full_image:
        raw = model(image.unsqueeze(0))
        outputs = _normalize_output(raw)
        boxes, scores, labels, masks = _extract_predictions(
            outputs, config.score_threshold, has_masks)

        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            if masks is not None:
                full_masks = F.interpolate(
                    masks.unsqueeze(1), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                all_masks.append(full_masks)

    # --- Merge ---
    if not all_boxes:
        result: dict[str, torch.Tensor] = {
            "boxes": torch.zeros(0, 4, device=device),
            "scores": torch.zeros(0, device=device),
            "labels": torch.zeros(0, dtype=torch.long, device=device),
        }
        if has_masks:
            result["masks"] = torch.zeros(0, H, W, dtype=torch.bool, device=device)
        return result

    merged_boxes = torch.cat(all_boxes, dim=0)
    merged_scores = torch.cat(all_scores, dim=0)
    merged_labels = torch.cat(all_labels, dim=0)

    # NMS to remove cross-slice duplicates
    keep = nms_merge(merged_boxes, merged_scores, merged_labels,
                     config.nms_iou_threshold)

    result = {
        "boxes": merged_boxes[keep],
        "scores": merged_scores[keep],
        "labels": merged_labels[keep],
    }

    if all_masks:
        merged_masks = torch.cat(all_masks, dim=0)
        # For duplicate predictions, take the mask with highest score (NMS already did this)
        result["masks"] = (merged_masks[keep].sigmoid() > 0.5)

    return result


@torch.no_grad()
def sahi_evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: SAHIConfig,
    device: torch.device,
) -> list[tuple[dict, list]]:
    """Run SAHI inference on a dataloader, returning predictions and targets.

    Args:
        model: Detection model.
        dataloader: Validation/test dataloader.
        config: SAHI configuration.
        device: CUDA device.

    Returns:
        List of (predictions_dict, targets_list) per batch. Each
        predictions_dict has keys boxes/scores/labels/masks.
    """
    from mhc_path.training.losses import DetectionTarget

    model.eval()
    results = []

    for batch in dataloader:
        images = batch["images"].to(device)
        boxes_batch = batch["boxes"]
        labels_batch = batch["labels"]
        num_obj = batch["num_objects"]
        masks_batch = batch.get("masks")

        batch_preds = []
        batch_targets = []

        for i in range(images.shape[0]):
            # SAHI predict
            pred = sahi_predict_single(model, images[i], config)
            batch_preds.append(pred)

            # Build target
            n = num_obj[i].item()
            tgt_boxes = boxes_batch[i, :n].to(device) if not isinstance(boxes_batch, list) \
                else boxes_batch[i][:n].to(device)
            tgt_labels = labels_batch[i, :n].to(device) if not isinstance(labels_batch, list) \
                else labels_batch[i][:n].to(device)
            tgt_masks = None
            if masks_batch is not None:
                tgt_masks = masks_batch[i, :n].to(device) if not isinstance(masks_batch, list) \
                    else masks_batch[i][:n].to(device)
            batch_targets.append(DetectionTarget(
                boxes=tgt_boxes, labels=tgt_labels, masks=tgt_masks))

        results.append((batch_preds, batch_targets))

    return results
