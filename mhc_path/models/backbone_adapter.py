"""DINOv3 backbone adapter with fallback ViT for testing.

Wraps DINOv3 (or a fallback ViT) as a multi-scale feature extractor,
exposing intermediate transformer layer outputs for building a feature
pyramid.
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
        # Build layers individually so we can tap intermediate outputs via hooks
        self.blocks = nn.ModuleList(
            [self._clone_layer(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _clone_layer(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
        """Deep-copy a layer template by re-creating it with the same config."""
        import copy
        return copy.deepcopy(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Interpolate positional embedding if resolution differs from default
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + self._interpolate_pos(x.shape[1], H, W)
        else:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def _interpolate_pos(self, num_patches: int, h: int, w: int) -> torch.Tensor:
        """Bicubic interpolation of positional embeddings for variable resolution."""
        old_len = self.pos_embed.shape[1]
        old_side = int(math.sqrt(old_len))
        pos = self.pos_embed.reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)
        pos = nn.functional.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1)


# ---------------------------------------------------------------------------
# Model variant configs
# ---------------------------------------------------------------------------

# Maps our names to timm/HuggingFace model identifiers and architecture params
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
    "phikon_v2": {
        "hf_name": "owkin/phikon-v2",
        "embed_dim": 1024, "num_layers": 24, "num_heads": 16, "patch_size": 16,
    },
}


# ---------------------------------------------------------------------------
# Main backbone adapter
# ---------------------------------------------------------------------------

class DINOv3Backbone(nn.Module):
    """Multi-scale feature extractor wrapping DINOv3 (or fallback ViT).

    Registers forward hooks on specified transformer blocks to capture
    intermediate representations for feature pyramid construction.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        frozen: bool = True,
        output_layers: tuple[int, ...] = (6, 12, 18, 24),
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._output_layers = output_layers
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._is_timm = False
        self._is_hf = False

        cfg = _VARIANT_CONFIGS.get(model_name, _VARIANT_CONFIGS["dinov3_vitl16"])
        self._embed_dim = cfg["embed_dim"]
        self._patch_size = cfg["patch_size"]
        self._num_layers = cfg["num_layers"]

        self.backbone = self._load_backbone(model_name, cfg)
        if not self._is_hf:
            self._register_hooks()

        if frozen:
            self.freeze_backbone()

    def _load_backbone(self, model_name: str, cfg: dict) -> nn.Module:
        """Try HuggingFace, timm, then fall back to built-in ViT."""
        # HuggingFace path (e.g. Phikon-v2)
        hf_name = cfg.get("hf_name")
        if hf_name is not None:
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(hf_name)
                self._is_hf = True
                self._embed_dim = model.config.hidden_size
                self._num_layers = model.config.num_hidden_layers
                self._patch_size = model.config.patch_size
                return model
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load HF model %s: %s. Trying timm/fallback.", hf_name, e
                )

        # timm path (DINOv3/DINOv2)
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
        """Return the sequential list of transformer blocks from the backbone."""
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        # HuggingFace DINOv2 stores layers in encoder.layer
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        raise AttributeError("Cannot locate transformer blocks on backbone")

    def _register_hooks(self) -> None:
        blocks = self._get_blocks()
        for idx in self._output_layers:
            # Convert 1-indexed layer spec to 0-indexed block list
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
        """Extract multi-layer features.

        Uses timm's ``forward_intermediates`` for pretrained models (handles
        RoPE, cls/reg tokens, etc.), HuggingFace ``output_hidden_states``
        for HF models (e.g. Phikon-v2), or manual block iteration for the
        fallback ViT.
        """
        if self._is_hf:
            return self._forward_hf(x)
        if self._is_timm:
            return self._forward_timm(x)
        return self._forward_fallback(x)

    def _forward_timm(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Use timm's forward_intermediates for proper DINOv3 inference."""
        # timm uses 0-indexed block indices
        indices = [i - 1 for i in self._output_layers]
        intermediates = self.backbone.forward_intermediates(
            x, indices=indices, intermediates_only=True,
        )
        # intermediates_only=True returns a list of (B, C, H, W) tensors
        if isinstance(intermediates, tuple):
            intermediates = intermediates[-1] if isinstance(intermediates[-1], list) else list(intermediates)
        features: dict[str, torch.Tensor] = {}
        for layer_idx, feat in zip(self._output_layers, intermediates):
            # forward_intermediates returns (B, C, H, W) spatial tensors;
            # reshape to (B, N, C) token format to match FPN input
            B, C, H, W = feat.shape
            features[str(layer_idx)] = feat.flatten(2).transpose(1, 2)
        return features

    def _forward_hf(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from HuggingFace DINOv2-style models."""
        outputs = self.backbone(x, output_hidden_states=True)
        # hidden_states[0] = embedding, [1]..[N] = layer outputs
        # Each is (B, num_patches+1, embed_dim) — position 0 is CLS token
        features: dict[str, torch.Tensor] = {}
        for layer_idx in self._output_layers:
            if layer_idx > len(outputs.hidden_states) - 1:
                continue
            feat = outputs.hidden_states[layer_idx]
            # Strip CLS token to get patch-only features: (B, N, C)
            features[str(layer_idx)] = feat[:, 1:, :]
        return features

    def _forward_fallback(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Manual block iteration for the fallback ViT."""
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
        """Patch grid size for the last forward pass's input (or default 256x256)."""
        return (256 // self._patch_size, 256 // self._patch_size)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n(self, n: int) -> int:
        """Unfreeze the last *n* transformer blocks and the final LayerNorm.

        Returns the count of newly unfrozen parameters.
        """
        blocks = self._get_blocks()
        total = len(blocks)
        count = 0
        for i, block in enumerate(blocks):
            if i >= total - n:
                for p in block.parameters():
                    p.requires_grad = True
                    count += p.numel()
        # Final LayerNorm follows the last block — must also be trainable
        # timm: backbone.norm, HF DINOv2: backbone.layernorm
        for norm_name in ("norm", "layernorm"):
            norm = getattr(self.backbone, norm_name, None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True
                    count += p.numel()
        return count
