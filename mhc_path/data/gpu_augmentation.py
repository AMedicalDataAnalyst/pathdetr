"""Step 2b: GPU Batch Augmentation.

GPU-side batched augmentation module. Runs in the training loop after the
DataLoader, before the model forward pass. Includes HistoRotate (rotate
oversized tile then center-crop), RandStainNA++ stain perturbation, and
geometric augmentations. All operations use pure PyTorch (no kornia dependency).

Depends on: Step 2a (color space conversions from stain_augmentation).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_path.data.stain_augmentation import (
    StainStats,
    _FORWARD_CONVERTERS,
    _INVERSE_CONVERTERS,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _rotation_matrix_2d(angles: torch.Tensor) -> torch.Tensor:
    """Build 2x3 affine matrices for rotation about the center.

    Args:
        angles: (B,) rotation angles in radians.

    Returns:
        (B, 2, 3) affine matrices suitable for F.affine_grid.
    """
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros_like(angles)

    # [cos, -sin, 0]
    # [sin,  cos, 0]
    row1 = torch.stack([cos_a, -sin_a, zeros], dim=-1)  # (B, 3)
    row2 = torch.stack([sin_a, cos_a, zeros], dim=-1)    # (B, 3)
    return torch.stack([row1, row2], dim=1)               # (B, 2, 3)


def _center_crop(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """Center-crop spatial dimensions to target_size via tensor slicing.

    Args:
        x: (B, C, H, W) tensor where H, W >= target_size.

    Returns:
        (B, C, target_size, target_size) tensor.
    """
    _, _, h, w = x.shape
    y0 = (h - target_size) // 2
    x0 = (w - target_size) // 2
    return x[:, :, y0 : y0 + target_size, x0 : x0 + target_size]


def _transform_boxes_rotation(
    boxes: torch.Tensor, angles: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """Transform CXCYWH normalized boxes under rotation about image center.

    Args:
        boxes: (N, 4) in [cx, cy, bw, bh] normalized [0, 1].
        angles: scalar angle in radians (single rotation for the image).
        h: image height.
        w: image width.

    Returns:
        (N, 4) transformed boxes in [cx, cy, bw, bh] normalized.
    """
    if boxes.numel() == 0:
        return boxes

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Convert to pixel coords centered at image center
    cx_px = boxes[:, 0] * w - w / 2.0
    cy_px = boxes[:, 1] * h - h / 2.0

    # Rotate centers
    new_cx_px = cos_a * cx_px - sin_a * cy_px
    new_cy_px = sin_a * cx_px + cos_a * cy_px

    # Back to normalized coords
    new_cx = (new_cx_px + w / 2.0) / w
    new_cy = (new_cy_px + h / 2.0) / h

    # For axis-aligned bounding boxes, rotation changes the effective w/h
    bw = boxes[:, 2]
    bh = boxes[:, 3]
    abs_cos = torch.abs(cos_a)
    abs_sin = torch.abs(sin_a)
    new_bw = bw * abs_cos + bh * abs_sin
    new_bh = bw * abs_sin + bh * abs_cos

    return torch.stack([new_cx, new_cy, new_bw, new_bh], dim=-1)


def _transform_boxes_flip(
    boxes: torch.Tensor, horizontal: bool, vertical: bool
) -> torch.Tensor:
    """Transform CXCYWH normalized boxes under flip.

    Args:
        boxes: (N, 4) in [cx, cy, bw, bh] normalized [0, 1].
        horizontal: whether horizontal flip was applied.
        vertical: whether vertical flip was applied.

    Returns:
        (N, 4) transformed boxes.
    """
    if boxes.numel() == 0:
        return boxes

    result = boxes.clone()
    if horizontal:
        result[:, 0] = 1.0 - boxes[:, 0]
    if vertical:
        result[:, 1] = 1.0 - boxes[:, 1]
    return result


# ── HistoRotate ───────────────────────────────────────────────────────


class HistoRotate(nn.Module):
    """Rotation augmentation for histopathology (Alfasly et al., CVPR 2024).

    Rotates oversized tiles by a random angle, then center-crops to
    target_size. The source tile must be at least ``target_size * sqrt(2)``
    to avoid border artifacts after arbitrary rotation.

    Mode selection:
        - ``"auto"``: if ``input_size >= 4 * target_size`` use continuous
          360-degree rotation, otherwise use discrete 90-degree multiples.
        - ``"continuous"``: always use full 360-degree rotation.
        - ``"discrete"``: always use 90-degree multiples only.
    """

    _DISCRETE_ANGLES = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)

    def __init__(self, target_size: int, mode: str = "auto") -> None:
        super().__init__()
        if mode not in ("auto", "continuous", "discrete"):
            raise ValueError(f"Invalid mode: {mode!r}")
        self.target_size = target_size
        self.mode = mode

    def _choose_angles(self, batch_size: int, ratio: float,
                       device: torch.device) -> torch.Tensor:
        """Sample rotation angles for each image in the batch.

        Args:
            batch_size: number of images.
            ratio: input_size / target_size.
            device: target device.

        Returns:
            (B,) tensor of angles in radians.
        """
        if self.mode == "discrete" or (self.mode == "auto" and ratio < 4.0):
            indices = torch.randint(0, 4, (batch_size,), device=device)
            angle_choices = torch.tensor(
                self._DISCRETE_ANGLES, dtype=torch.float32, device=device
            )
            return angle_choices[indices]
        else:
            return torch.rand(batch_size, device=device) * 2.0 * math.pi

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate and center-crop.

        Args:
            x: (B, C, H_oversized, W_oversized) input tensor.

        Returns:
            Tuple of (output images, angles):
                - images: (B, C, target_size, target_size) output tensor.
                - angles: (B,) rotation angles in radians.
        """
        b, c, h, w = x.shape
        ratio = min(h, w) / self.target_size

        angles = self._choose_angles(b, ratio, x.device)

        # Build affine grid and sample
        theta = _rotation_matrix_2d(angles)  # (B, 2, 3)
        theta = theta.to(dtype=x.dtype, device=x.device)

        grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)
        rotated = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )

        return _center_crop(rotated, self.target_size), angles


# ── GPUStainAugmentation ─────────────────────────────────────────────


class GPUStainAugmentation(nn.Module):
    """RandStainNA++ on GPU: Reinhard normalization to a virtual template.

    For each image, converts to LAB, computes per-channel mean/std,
    samples a virtual stain template from dataset-level statistics,
    and applies Reinhard normalization: (x - μ_src) × (σ_tgt / σ_src) + μ_tgt.

    Requires pre-computed dataset statistics (JSON file from
    ``compute_stain_stats.py``).

    Parameters
    ----------
    stats_file : str or Path
        JSON file with keys: mean_of_means, std_of_means, mean_of_stds,
        std_of_stds (each a 3-element list for LAB channels).
    std_hyper : float
        Scaling factor for distribution width. 0 = use dataset statistics
        as-is. Positive = wider distribution (more diverse augmentation).
        Negative = narrower (more conservative).
    color_space : str
        Color space for normalization ('LAB', 'HSV', or 'HED').
    p : float
        Probability of applying augmentation per image.
    """

    def __init__(
        self,
        stats_file: Optional[str | Path] = None,
        std_hyper: float = 0.0,
        color_space: str = "LAB",
        p: float = 0.8,
    ) -> None:
        super().__init__()
        if color_space not in _FORWARD_CONVERTERS:
            raise ValueError(f"Unsupported color space: {color_space}")
        self.color_space = color_space
        self.std_hyper = std_hyper
        self.p = p

        if stats_file is not None:
            with open(stats_file) as f:
                cfg = json.load(f)
            self.register_buffer(
                "mean_of_means", torch.tensor(cfg["mean_of_means"], dtype=torch.float32))
            self.register_buffer(
                "std_of_means", torch.tensor(cfg["std_of_means"], dtype=torch.float32))
            self.register_buffer(
                "mean_of_stds", torch.tensor(cfg["mean_of_stds"], dtype=torch.float32))
            self.register_buffer(
                "std_of_stds", torch.tensor(cfg["std_of_stds"], dtype=torch.float32))
        else:
            # Sensible defaults for H&E in LAB (can be overridden)
            self.register_buffer(
                "mean_of_means", torch.tensor([65.0, 22.0, -11.0]))
            self.register_buffer(
                "std_of_means", torch.tensor([13.0, 9.0, 8.0]))
            self.register_buffer(
                "mean_of_stds", torch.tensor([15.0, 8.0, 7.5]))
            self.register_buffer(
                "std_of_stds", torch.tensor([4.5, 3.0, 2.5]))

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        stain_stats: Optional[list[StainStats]] = None,
    ) -> torch.Tensor:
        """Apply RandStainNA++ to a batch.

        Args:
            images: (B, C, H, W) float tensor in [0, 1].
            stain_stats: ignored (kept for API compatibility).

        Returns:
            (B, C, H, W) augmented tensor in [0, 1].
        """
        b, c, h, w = images.shape
        device = images.device

        apply_mask = torch.rand(b, device=device) < self.p
        if not apply_mask.any():
            return images

        forward_fn = _FORWARD_CONVERTERS[self.color_space]
        inverse_fn = _INVERSE_CONVERTERS[self.color_space]

        # Batched color conversion: (B, 3, H, W)
        converted = forward_fn(images)

        # Per-image statistics: (B, 3)
        flat = converted.reshape(b, 3, -1)  # (B, 3, N)
        img_means = flat.mean(dim=2)  # (B, 3)
        img_stds = flat.std(dim=2).clamp(min=1e-4)  # (B, 3)

        # Sample virtual templates for all images at once: (B, 3)
        scale = 1.0 + self.std_hyper
        tar_means = torch.normal(
            self.mean_of_means.unsqueeze(0).expand(b, -1),
            self.std_of_means.unsqueeze(0).expand(b, -1) * scale,
        )
        tar_stds = torch.normal(
            self.mean_of_stds.unsqueeze(0).expand(b, -1),
            self.std_of_stds.unsqueeze(0).expand(b, -1) * scale,
        ).clamp(min=1e-4)

        # Batched Reinhard: (x - μ_src) × (σ_tgt / σ_src) + μ_tgt
        ratio = (tar_stds / img_stds).unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        shift = (tar_means - img_means * tar_stds / img_stds).unsqueeze(-1).unsqueeze(-1)
        normalized = converted * ratio + shift

        # Batched inverse conversion
        result = inverse_fn(normalized).clamp(0.0, 1.0)

        # Apply per-image mask (skip images where p test failed)
        mask = apply_mask.reshape(b, 1, 1, 1)
        return torch.where(mask, result, images)


# ── GPUPathologyAugPipeline ──────────────────────────────────────────


class GPUPathologyAugPipeline(nn.Module):
    """Full GPU augmentation pipeline for histopathology training.

    Composes HistoRotate, stain perturbation, and geometric augmentations
    (horizontal/vertical flip). Inserted between the DataLoader and the
    model forward pass.

    Modes:
        - ``"detection"``: HistoRotate + flip. Updates boxes and masks
          consistently with all geometric transforms.
        - ``"ssl"``: HistoRotate per crop (independent). No box/mask
          handling. Full 360-degree rotation for local crops.
    """

    def __init__(
        self,
        target_size: int = 256,
        histo_rotate: bool = True,
        stain_aug: bool = True,
        geometric: bool = True,
        dab_jitter: bool = False,
        mode: str = "detection",
        stain_stats_file: Optional[str | Path] = None,
        stain_std_hyper: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("detection", "ssl"):
            raise ValueError(f"Invalid mode: {mode!r}")

        self.target_size = target_size
        self.mode = mode
        self.geometric = geometric

        self.rotator: Optional[HistoRotate] = None
        if histo_rotate:
            self.rotator = HistoRotate(target_size, mode="auto")

        self.stain: Optional[GPUStainAugmentation] = None
        if stain_aug:
            self.stain = GPUStainAugmentation(
                stats_file=stain_stats_file,
                std_hyper=stain_std_hyper,
            )

        self.dab_jitter = dab_jitter

        # Flip probabilities
        self.hflip_p = 0.5
        self.vflip_p = 0.5

    def _apply_detection(self, batch: dict) -> dict:
        """Detection-mode augmentation pipeline.

        Expects batch keys:
            - ``"images"``: (B, C, H, W)
            - ``"boxes"`` (optional): list of (N_i, 4) tensors in CXCYWH normalized
            - ``"masks"`` (optional): (B, 1, H, W) or (B, H, W) binary masks
            - ``"stain_stats"`` (optional): list[StainStats]
        """
        images = batch["images"]
        b, c, h_orig, w_orig = images.shape

        # --- HistoRotate ---
        if self.rotator is not None:
            images, angles = self.rotator(images)
            _, _, h, w = images.shape

            # Transform boxes to match rotation + center-crop
            if "boxes" in batch and batch["boxes"] is not None:
                new_boxes = []
                keep_masks: list[torch.Tensor] = []
                boxes_raw = batch["boxes"]
                num_obj = batch.get("num_objects")
                is_list = isinstance(boxes_raw, (list, tuple))

                for i in range(b):
                    if is_list:
                        box_tensor = boxes_raw[i]
                    else:
                        n_i = num_obj[i].item() if num_obj is not None else boxes_raw.shape[1]
                        box_tensor = boxes_raw[i, :n_i]
                    box_tensor = box_tensor.to(images.device)
                    if box_tensor.numel() == 0:
                        new_boxes.append(box_tensor)
                        keep_masks.append(torch.ones(0, dtype=torch.bool,
                                                     device=box_tensor.device))
                        continue
                    # Rotate boxes in the original (pre-crop) coordinate space
                    rotated = _transform_boxes_rotation(
                        box_tensor, angles[i], h_orig, w_orig,
                    )
                    # Adjust for center-crop: remap from oversized to cropped coords
                    x_off = (w_orig - w) / (2.0 * w_orig)
                    y_off = (h_orig - h) / (2.0 * h_orig)
                    rotated[:, 0] = (rotated[:, 0] - x_off) * (w_orig / w)
                    rotated[:, 1] = (rotated[:, 1] - y_off) * (h_orig / h)
                    rotated[:, 2] = rotated[:, 2] * (w_orig / w)
                    rotated[:, 3] = rotated[:, 3] * (h_orig / h)
                    # Filter boxes mostly outside [0, 1] after transform
                    cx, cy, bw, bh = rotated[:, 0], rotated[:, 1], rotated[:, 2], rotated[:, 3]
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    inside_center = (cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)
                    clipped_area = (x2.clamp(0, 1) - x1.clamp(0, 1)) * \
                                   (y2.clamp(0, 1) - y1.clamp(0, 1))
                    orig_area = (bw * bh).clamp(min=1e-6)
                    keep = inside_center & (clipped_area / orig_area > 0.25)
                    rotated = rotated[keep]
                    rotated[:, 0] = rotated[:, 0].clamp(0, 1)
                    rotated[:, 1] = rotated[:, 1].clamp(0, 1)
                    rotated[:, 2] = rotated[:, 2].clamp(min=1e-4)
                    rotated[:, 3] = rotated[:, 3].clamp(min=1e-4)
                    new_boxes.append(rotated)
                    keep_masks.append(keep)
                batch["boxes"] = new_boxes

                # Filter labels to match filtered boxes
                if "labels" in batch and batch["labels"] is not None:
                    labels = batch["labels"]
                    new_labels = []
                    for i, keep in enumerate(keep_masks):
                        keep_cpu = keep.cpu()
                        if isinstance(labels, (list, tuple)):
                            new_labels.append(labels[i][keep_cpu])
                        else:
                            n_i = num_obj[i].item() if num_obj is not None else labels.shape[1]
                            new_labels.append(labels[i, :n_i][keep_cpu])
                    batch["labels"] = new_labels

                # Update num_objects
                if "num_objects" in batch:
                    batch["num_objects"] = torch.tensor(
                        [b.shape[0] for b in batch["boxes"]],
                        device=images.device,
                    )
        else:
            h, w = h_orig, w_orig

        # --- Stain augmentation ---
        if self.stain is not None:
            stain_stats = batch.get("stain_stats")
            images = self.stain(images, stain_stats)

        # --- Geometric: flips ---
        if self.geometric:
            do_hflip = torch.rand(1).item() < self.hflip_p
            do_vflip = torch.rand(1).item() < self.vflip_p

            if do_hflip:
                images = torch.flip(images, dims=[-1])
            if do_vflip:
                images = torch.flip(images, dims=[-2])

            # Update boxes
            if "boxes" in batch and batch["boxes"] is not None:
                new_boxes = []
                for box_tensor in batch["boxes"]:
                    new_boxes.append(
                        _transform_boxes_flip(box_tensor, do_hflip, do_vflip)
                    )
                batch["boxes"] = new_boxes

            # Store flip flags so masks can be flipped lazily per-object
            # during target building (avoids flipping the full padded tensor)
            if "masks" in batch and batch["masks"] is not None:
                if do_hflip or do_vflip:
                    masks = batch["masks"]
                    num_obj = batch.get("num_objects")
                    if num_obj is not None:
                        # Flip only valid (non-padded) masks per image
                        flip_dims = []
                        if do_hflip:
                            flip_dims.append(-1)
                        if do_vflip:
                            flip_dims.append(-2)
                        for i_img in range(masks.shape[0]):
                            n = num_obj[i_img].item()
                            if n > 0:
                                masks[i_img, :n] = torch.flip(
                                    masks[i_img, :n], dims=flip_dims)
                    else:
                        if do_hflip:
                            masks = torch.flip(masks, dims=[-1])
                        if do_vflip:
                            masks = torch.flip(masks, dims=[-2])
                    # Crop masks to match image size if needed
                    if masks.shape[-2:] != images.shape[-2:]:
                        masks = _center_crop(
                            masks if masks.ndim == 4 else masks.unsqueeze(1),
                            self.target_size,
                        )
                        if batch["masks"].ndim == 3:
                            masks = masks.squeeze(1)
                    batch["masks"] = masks

        batch["images"] = images
        return batch

    def _apply_ssl(self, batch: dict) -> dict:
        """SSL-mode augmentation pipeline.

        Expects batch keys:
            - ``"global_crops"``: list of (B, C, H, W) tensors
            - ``"local_crops"`` (optional): list of (B, C, H, W) tensors
            - ``"stain_stats"`` (optional): list[StainStats]
        """
        if self.rotator is not None:
            # Global crops: use the configured mode (discard angles for SSL)
            if "global_crops" in batch:
                new_global = []
                for crop in batch["global_crops"]:
                    rotated_crop, _ = self.rotator(crop)
                    new_global.append(rotated_crop)
                batch["global_crops"] = new_global

            # Local crops: always use continuous 360-degree rotation
            if "local_crops" in batch:
                local_rotator = HistoRotate(
                    self.target_size, mode="continuous"
                )
                new_local = []
                for crop in batch["local_crops"]:
                    rotated_crop, _ = local_rotator(crop)
                    new_local.append(rotated_crop)
                batch["local_crops"] = new_local

        # Stain augmentation on all crops
        if self.stain is not None:
            stain_stats = batch.get("stain_stats")
            for key in ("global_crops", "local_crops"):
                if key in batch:
                    new_crops = []
                    for crop in batch[key]:
                        new_crops.append(self.stain(crop, stain_stats))
                    batch[key] = new_crops

        return batch

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        """Augment images and transform targets consistently.

        Args:
            batch: dictionary with images and optional targets.

        Returns:
            Augmented batch dictionary.
        """
        if self.mode == "detection":
            return self._apply_detection(batch)
        else:
            return self._apply_ssl(batch)
