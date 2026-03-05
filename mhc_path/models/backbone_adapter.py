"""Frozen DINOv3 backbone adapter with fallback ViT for testing.

Wraps DINOv3 (or a fallback ViT) as a multi-scale feature extractor,
exposing intermediate transformer layer outputs for building a feature
pyramid. All backbone parameters are frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fallback ViT built from standard PyTorch modules
# ---------------------------------------------------------------------------

class _FallbackViT(nn.Module):
    """Minimal ViT for environments where timm/DINOv3 is unavailable."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 1024,
        num_layers: int = 32,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        import copy
        self.blocks = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + self._interpolate_pos(x.shape[1], H, W)
        else:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def _interpolate_pos(self, num_patches: int, h: int, w: int) -> torch.Tensor:
        old_len = self.pos_embed.shape[1]
        old_side = int(math.sqrt(old_len))
        pos = self.pos_embed.reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)
        pos = nn.functional.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1)


# ---------------------------------------------------------------------------
# Model variant configs
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS: dict[str, dict] = {
    "dinov3_vitl16": {
        "timm_name": "vit_large_patch16_dinov3.lvd1689m",
        "embed_dim": 1024, "num_layers": 24, "num_heads": 16, "patch_size": 16,
    },
    "dinov3_vitb16": {
        "timm_name": "vit_base_patch16_dinov3.lvd1689m",
        "embed_dim": 768, "num_layers": 12, "num_heads": 12, "patch_size": 16,
    },
    "dinov3_vitg14": {
        "timm_name": "vit_giant_patch14_dinov2.lvd142m",
        "embed_dim": 1536, "num_layers": 40, "num_heads": 24, "patch_size": 14,
    },
}


# ---------------------------------------------------------------------------
# Main backbone adapter
# ---------------------------------------------------------------------------

class DINOv3Backbone(nn.Module):
    """Frozen multi-scale feature extractor wrapping DINOv3 (or fallback ViT).

    All parameters are frozen. Uses forward hooks on specified transformer
    blocks to capture intermediate representations for feature pyramid
    construction.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        frozen: bool = True,
        output_layers: tuple[int, ...] = (6, 12, 18, 24),
        **kwargs,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._output_layers = output_layers
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._is_timm = False

        cfg = _VARIANT_CONFIGS.get(model_name, _VARIANT_CONFIGS["dinov3_vitl16"])
        self._embed_dim = cfg["embed_dim"]
        self._patch_size = cfg["patch_size"]
        self._num_layers = cfg["num_layers"]

        self.backbone = self._load_backbone(model_name, cfg)
        self._register_hooks()

        if frozen:
            self._freeze_backbone()

    def _load_backbone(self, model_name: str, cfg: dict) -> nn.Module:
        timm_name = cfg.get("timm_name", model_name)
        try:
            import timm
            model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            self._is_timm = True
            self._embed_dim = model.embed_dim
            self._num_layers = len(model.blocks)
            return model
        except Exception:
            return _FallbackViT(
                embed_dim=cfg["embed_dim"],
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                patch_size=cfg["patch_size"],
            )

    def _get_blocks(self) -> nn.ModuleList:
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        raise AttributeError("Cannot locate transformer blocks on backbone")

    def _register_hooks(self) -> None:
        blocks = self._get_blocks()
        for idx in self._output_layers:
            block_idx = idx - 1
            if block_idx < 0 or block_idx >= len(blocks):
                continue
            layer_key = str(idx)
            hook = blocks[block_idx].register_forward_hook(
                self._make_hook(layer_key),
            )
            self._hooks.append(hook)

    def _make_hook(self, key: str):
        def hook_fn(_module: nn.Module, _input, output: torch.Tensor) -> None:
            self._features[key] = output
        return hook_fn

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-layer features."""
        if self._is_timm:
            return self._forward_timm(x)
        return self._forward_fallback(x)

    def _forward_timm(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        indices = [i - 1 for i in self._output_layers]
        intermediates = self.backbone.forward_intermediates(
            x, indices=indices, intermediates_only=True,
        )
        if isinstance(intermediates, tuple):
            intermediates = intermediates[-1] if isinstance(intermediates[-1], list) else list(intermediates)
        features: dict[str, torch.Tensor] = {}
        for layer_idx, feat in zip(self._output_layers, intermediates):
            B, C, H, W = feat.shape
            features[str(layer_idx)] = feat.flatten(2).transpose(1, 2)
        return features

    def _forward_fallback(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.backbone.pos_embed.shape[1]:
            x = x + self.backbone._interpolate_pos(x.shape[1], H, W)
        else:
            x = x + self.backbone.pos_embed

        output_set = set(self._output_layers)
        features: dict[str, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.backbone.blocks, start=1):
            x = block(x)
            if layer_idx in output_set:
                features[str(layer_idx)] = x

        return features

    @property
    def feature_dims(self) -> dict[str, int]:
        return {str(idx): self._embed_dim for idx in self._output_layers}

    @property
    def num_patches(self) -> tuple[int, int]:
        return (256 // self._patch_size, 256 // self._patch_size)

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
