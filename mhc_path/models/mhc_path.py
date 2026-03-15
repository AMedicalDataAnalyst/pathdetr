"""Full mHC-Path model: ViT backbone -> FPN -> RF-DETR decoder.

Assembles the complete detection/segmentation pipeline with frozen
(or partially unfrozen) ViT backbone, multi-scale FPN, and RF-DETR
decoder with deformable attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mhc_path.models.backbone_adapter import DINOv3Backbone
from mhc_path.models.decoder import RFDETRDecoder
from mhc_path.models.fpn import PathologyFPN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MHCPathConfig:
    """Configuration for the full mHC-Path model.

    Attributes:
        backbone: Model variant key (e.g. ``"dinov3_vitl16"``, ``"phikon_v2"``).
        backbone_frozen: Whether to freeze backbone weights.
        fpn_dim: Channel dimension shared across FPN output levels.
        num_queries: Number of learnable object queries.
        num_classes: Number of detection classes.
        num_decoder_layers: Number of stacked decoder layers.
        sa_n_heads: Number of self-attention heads in decoder.
        ca_n_heads: Number of cross-attention heads in decoder.
        n_points: Sampling points per head per level in deformable attention.
        group_detr: Number of query groups for Group DETR training.
        with_segmentation: Whether to produce per-query mask logits.
        mask_upsample_factor: Upsample pixel features before dot-product mask head.
        with_pixel_decoder: Use multi-scale pixel decoder with ViT skip connections.
        large_kernel: Use large-kernel depthwise convs in pixel decoder.
        large_kernel_size: Kernel size for large-kernel blocks.
        output_layers: Backbone layers to tap for multi-scale features.
        fpn_levels: Number of FPN output pyramid levels.
    """

    backbone: str = "dinov3_vitl16"
    backbone_frozen: bool = True
    fpn_dim: int = 256
    num_queries: int = 100
    num_classes: int = 5
    num_decoder_layers: int = 6
    sa_n_heads: int = 8
    ca_n_heads: int = 8
    n_points: int = 4
    group_detr: int = 1
    with_segmentation: bool = True
    mask_upsample_factor: int = 1
    with_pixel_decoder: bool = False
    large_kernel: bool = False
    large_kernel_size: int = 13
    output_layers: tuple[int, ...] = (8, 16, 24, 32)
    fpn_levels: int = 4


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MHCPath(nn.Module):
    """Full mHC-Path detection/segmentation model.

    Composes a ViT backbone, a pathology FPN, and an RF-DETR decoder.
    """

    def __init__(self, config: MHCPathConfig) -> None:
        super().__init__()
        self.config = config

        # --- Backbone ---
        self.backbone = DINOv3Backbone(
            model_name=config.backbone,
            frozen=config.backbone_frozen,
            output_layers=config.output_layers,
        )

        # --- FPN ---
        self.fpn = PathologyFPN(
            in_dims=self.backbone.feature_dims,
            out_dim=config.fpn_dim,
            num_levels=config.fpn_levels,
        )

        # --- Decoder ---
        feat_dims = self.backbone.feature_dims
        backbone_dim = next(iter(feat_dims.values())) if feat_dims else 1024
        self.decoder = RFDETRDecoder(
            d_model=config.fpn_dim,
            num_queries=config.num_queries,
            num_classes=config.num_classes,
            n_decoder_layers=config.num_decoder_layers,
            n_levels=config.fpn_levels,
            sa_n_heads=config.sa_n_heads,
            ca_n_heads=config.ca_n_heads,
            n_points=config.n_points,
            group_detr=config.group_detr,
            with_segmentation=config.with_segmentation,
            mask_upsample_factor=config.mask_upsample_factor,
            with_pixel_decoder=config.with_pixel_decoder,
            backbone_dim=backbone_dim,
            backbone_layers=config.output_layers,
            large_kernel=config.large_kernel,
            large_kernel_size=config.large_kernel_size,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """End-to-end forward: images -> detections (+ optional masks).

        Args:
            images: ``(B, 3, H, W)`` input tensor.

        Returns:
            Dictionary with keys ``"class_logits"``, ``"box_coords"``,
            and optionally ``"mask_logits"`` and ``"aux_outputs"``.
        """
        multi_layer_features = self.backbone(images)
        spatial_shapes = self.backbone.num_patches
        fpn_features = self.fpn(multi_layer_features, spatial_shapes)

        backbone_features = None
        patch_grid = None
        if self.config.with_pixel_decoder:
            backbone_features = multi_layer_features
            patch_grid = spatial_shapes

        return self.decoder(fpn_features, backbone_features, patch_grid)

    @classmethod
    def from_pretrained_backbone(cls, config: MHCPathConfig) -> "MHCPath":
        """Create model and load pretrained backbone weights."""
        return cls(config)

    def trainable_parameters(self) -> list[dict]:
        """Return parameter groups for differential learning rates."""
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        fpn_params = list(self.fpn.parameters())
        decoder_params = list(self.decoder.parameters())

        return [
            {"params": backbone_params, "group_name": "backbone"},
            {"params": fpn_params, "group_name": "fpn"},
            {"params": decoder_params, "group_name": "decoder"},
        ]
