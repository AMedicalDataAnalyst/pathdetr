"""Watershed post-processing using HV distance maps.

Applies Sobel edge detection on predicted HV maps, then uses marker-controlled
watershed to refine instance boundaries. This follows the HoVer-Net paradigm
of using gradient information for instance separation.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def composite_hv_maps(
    pred_masks: torch.Tensor,
    pred_hv_maps: torch.Tensor,
    pred_scores: torch.Tensor,
    image_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite per-query HV maps into a global image-level map.

    Uses winner-take-all by confidence score to assign each pixel to
    the highest-confidence query that claims it.

    Args:
        pred_masks: (N_q, H, W) binary masks.
        pred_hv_maps: (N_q, 2, H, W) HV distance maps in [-1, 1].
        pred_scores: (N_q,) confidence scores.
        image_size: target spatial resolution.

    Returns:
        global_hv: (2, H, W) composited HV map.
        global_mask: (H, W) binary foreground mask.
        instance_map: (H, W) int32, query-level instance IDs (0 = background).
    """
    N = pred_masks.shape[0]
    H = W = image_size

    global_hv = np.zeros((2, H, W), dtype=np.float32)
    global_mask = np.zeros((H, W), dtype=np.float32)
    instance_map = np.zeros((H, W), dtype=np.int32)
    confidence_map = np.zeros((H, W), dtype=np.float32)

    # Sort by score descending — lowest-confidence painted first, overwritten
    order = pred_scores.argsort(descending=True).cpu().numpy()

    masks_np = pred_masks.cpu().numpy().astype(np.float32)
    hv_np = pred_hv_maps.cpu().numpy().astype(np.float32)
    scores_np = pred_scores.cpu().numpy()

    # Paint in reverse order so highest-confidence wins
    for rank, qi in enumerate(reversed(order)):
        m = masks_np[qi]  # (H, W)
        fg = m > 0.5
        if not fg.any():
            continue

        score = scores_np[qi]
        inst_id = N - rank  # unique positive ID

        update = fg & (score >= confidence_map)
        global_hv[0][update] = hv_np[qi, 0][update]
        global_hv[1][update] = hv_np[qi, 1][update]
        global_mask[update] = 1.0
        instance_map[update] = inst_id
        confidence_map[update] = score

    return global_hv, global_mask, instance_map


def watershed_post_process(
    global_hv: np.ndarray,
    global_mask: np.ndarray,
    instance_map: np.ndarray,
    energy_threshold: float = 0.4,
    min_area: int = 10,
) -> np.ndarray:
    """Refine instance segmentation using Sobel gradients + watershed.

    Args:
        global_hv: (2, H, W) composited HV maps.
        global_mask: (H, W) binary foreground.
        instance_map: (H, W) initial instance assignments.
        energy_threshold: threshold on Sobel energy for marker detection.
        min_area: minimum instance area in pixels.

    Returns:
        refined_map: (H, W) int32, refined instance IDs (0 = background).
    """
    H, W = global_mask.shape
    fg = global_mask > 0.5

    if not fg.any():
        return np.zeros((H, W), dtype=np.int32)

    # Sobel gradients of HV maps
    sobelh_x = ndimage.sobel(global_hv[0], axis=1)
    sobelh_y = ndimage.sobel(global_hv[0], axis=0)
    sobelv_x = ndimage.sobel(global_hv[1], axis=1)
    sobelv_y = ndimage.sobel(global_hv[1], axis=0)

    # Gradient magnitude (energy)
    energy = np.sqrt(sobelh_x**2 + sobelh_y**2 + sobelv_x**2 + sobelv_y**2)

    # Markers: low gradient energy = interior of instances
    markers = np.zeros_like(global_mask, dtype=np.int32)
    interior = fg & (energy < energy_threshold)

    # Label connected components as markers
    labeled, n_markers = ndimage.label(interior)
    markers = labeled

    if n_markers == 0:
        # Fall back to initial instance map
        return instance_map

    # Watershed using energy as the landscape
    # scipy doesn't have watershed, use marker-controlled distance approach
    # Instead, use the energy map with markers for a simple segmentation
    from skimage.segmentation import watershed as sk_watershed

    # Invert energy for watershed (higher = more likely boundary)
    energy_uint = np.clip(energy * 255, 0, 255).astype(np.uint8)

    ws = sk_watershed(energy_uint, markers=markers, mask=fg)

    # Remove small instances
    refined = np.zeros_like(ws, dtype=np.int32)
    for inst_id in range(1, ws.max() + 1):
        area = (ws == inst_id).sum()
        if area >= min_area:
            refined[ws == inst_id] = inst_id

    # Re-label to sequential IDs
    unique_ids = np.unique(refined)
    unique_ids = unique_ids[unique_ids > 0]
    relabeled = np.zeros_like(refined)
    for new_id, old_id in enumerate(unique_ids, start=1):
        relabeled[refined == old_id] = new_id

    return relabeled
