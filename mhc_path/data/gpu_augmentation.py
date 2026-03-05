"""GPU batch augmentation for detection training.

GPU-side batched augmentation module. Runs in the training loop after the
DataLoader, before the model forward pass. Includes RandStainNA++ stain
perturbation and geometric augmentations (flips). All operations use pure
PyTorch (no kornia dependency).

Depends on: stain_augmentation (color space conversions).
"""

from __future__ import annotations

import json
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


def _transform_boxes_flip(
    boxes: torch.Tensor, horizontal: bool, vertical: bool
) -> torch.Tensor:
    """Transform CXCYWH normalized boxes under flip."""
    if boxes.numel() == 0:
        return boxes

    result = boxes.clone()
    if horizontal:
        result[:, 0] = 1.0 - boxes[:, 0]
    if vertical:
        result[:, 1] = 1.0 - boxes[:, 1]
    return result


# ── GPUStainAugmentation ─────────────────────────────────────────────


class GPUStainAugmentation(nn.Module):
    """RandStainNA++ on GPU: Reinhard normalization to a virtual template.

    For each image, converts to LAB, computes per-channel mean/std,
    samples a virtual stain template from dataset-level statistics,
    and applies Reinhard normalization: (x - mu_src) * (sigma_tgt / sigma_src) + mu_tgt.

    Parameters
    ----------
    stats_file : str or Path
        JSON file with keys: mean_of_means, std_of_means, mean_of_stds,
        std_of_stds (each a 3-element list for LAB channels).
    std_hyper : float
        Scaling factor for distribution width.
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
            # Sensible defaults for H&E in LAB
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

        converted = forward_fn(images)

        flat = converted.reshape(b, 3, -1)
        img_means = flat.mean(dim=2)
        img_stds = flat.std(dim=2).clamp(min=1e-4)

        scale = 1.0 + self.std_hyper
        tar_means = torch.normal(
            self.mean_of_means.unsqueeze(0).expand(b, -1),
            self.std_of_means.unsqueeze(0).expand(b, -1) * scale,
        )
        tar_stds = torch.normal(
            self.mean_of_stds.unsqueeze(0).expand(b, -1),
            self.std_of_stds.unsqueeze(0).expand(b, -1) * scale,
        ).clamp(min=1e-4)

        ratio = (tar_stds / img_stds).unsqueeze(-1).unsqueeze(-1)
        shift = (tar_means - img_means * tar_stds / img_stds).unsqueeze(-1).unsqueeze(-1)
        normalized = converted * ratio + shift

        result = inverse_fn(normalized).clamp(0.0, 1.0)

        mask = apply_mask.reshape(b, 1, 1, 1)
        return torch.where(mask, result, images)


# ── GPUPathologyAugPipeline ──────────────────────────────────────────


class GPUPathologyAugPipeline(nn.Module):
    """GPU augmentation pipeline for detection training.

    Composes stain perturbation and geometric augmentations (horizontal/vertical
    flip). Inserted between the DataLoader and the model forward pass.
    """

    def __init__(
        self,
        target_size: int = 256,
        stain_aug: bool = True,
        geometric: bool = True,
        stain_stats_file: Optional[str | Path] = None,
        stain_std_hyper: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.geometric = geometric

        self.stain: Optional[GPUStainAugmentation] = None
        if stain_aug:
            self.stain = GPUStainAugmentation(
                stats_file=stain_stats_file,
                std_hyper=stain_std_hyper,
            )

        self.hflip_p = 0.5
        self.vflip_p = 0.5

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        """Augment images and transform targets consistently."""
        images = batch["images"]

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
                boxes_raw = batch["boxes"]
                if isinstance(boxes_raw, (list, tuple)):
                    batch["boxes"] = [
                        _transform_boxes_flip(box_t, do_hflip, do_vflip)
                        for box_t in boxes_raw
                    ]
                else:
                    # Padded tensor (B, N_max, 4)
                    num_obj = batch.get("num_objects")
                    new_boxes = boxes_raw.clone()
                    if do_hflip:
                        new_boxes[:, :, 0] = 1.0 - boxes_raw[:, :, 0]
                    if do_vflip:
                        new_boxes[:, :, 1] = 1.0 - boxes_raw[:, :, 1]
                    batch["boxes"] = new_boxes

            # Flip masks
            if "masks" in batch and batch["masks"] is not None:
                if do_hflip or do_vflip:
                    masks = batch["masks"]
                    num_obj = batch.get("num_objects")
                    if num_obj is not None:
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
                    batch["masks"] = masks

        batch["images"] = images
        return batch
