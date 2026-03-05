"""Feature Pyramid Network adapter for pathology transformers.

Converts DINOv3's single-scale multi-layer features into a multi-scale
feature pyramid suitable for the deformable attention decoder.  Each
backbone layer's flattened token sequence is reshaped to a spatial grid,
projected to a common channel dimension, and fused via top-down lateral
connections.  Additional coarser scales are generated with stride-2
convolutions.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnGelu(nn.Module):
    """Conv2d -> BatchNorm2d -> GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Lateral projection: 1x1 conv that maps each backbone dim to out_dim
# ---------------------------------------------------------------------------

class LateralProjection(nn.Module):
    """1x1 convolution that projects a backbone feature to the FPN dim."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = ConvBnGelu(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Top-down fusion: upsample coarser level and add to the current level
# ---------------------------------------------------------------------------

class TopDownFusion(nn.Module):
    """Fuse a coarser (top) feature with a finer (lateral) feature.

    The coarser map is upsampled to match the finer spatial size, then
    element-wise added.  A 3x3 conv smooths aliasing artifacts.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.smooth = ConvBnGelu(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(
        self, lateral: torch.Tensor, top_down: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse *lateral* (finer) and *top_down* (coarser) feature maps."""
        upsampled = nn.functional.interpolate(
            top_down,
            size=lateral.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.smooth(lateral + upsampled)


# ---------------------------------------------------------------------------
# Coarser-scale generator: stride-2 conv to create additional FPN levels
# ---------------------------------------------------------------------------

class DownsampleLevel(nn.Module):
    """Stride-2 3x3 convolution producing a coarser FPN level."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.down = ConvBnGelu(out_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# ---------------------------------------------------------------------------
# Main FPN
# ---------------------------------------------------------------------------

class PathologyFPN(nn.Module):
    """Multi-scale Feature Pyramid Network for pathology transformers.

    Takes multi-layer features from a ViT backbone (each as a flattened
    token sequence), reshapes them to spatial grids, projects to a common
    channel dimension, fuses via top-down lateral connections, and
    optionally generates additional coarser scales with stride-2 convs.

    Parameters
    ----------
    in_dims:
        Mapping ``{layer_key: embed_dim}`` for each backbone layer to tap.
        The keys are sorted to determine ordering (finest -> coarsest
        corresponds to *earliest -> latest* transformer layer by convention).
    out_dim:
        Channel dimension shared across all FPN output levels.
    num_levels:
        Total number of output pyramid levels.  If ``num_levels`` exceeds
        the number of backbone layers the extra coarser levels are produced
        via stride-2 convolutions on the finest (first) fused level.
    """

    def __init__(
        self,
        in_dims: dict[str, int],
        out_dim: int = 256,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.num_levels = num_levels

        # Sort layer keys so that the *smallest* index (earliest / finest
        # layer) comes first.  The backbone adapter uses string keys like
        # "8", "16", "24".
        self._sorted_keys: list[str] = sorted(in_dims.keys(), key=lambda k: int(k))

        # --- lateral projections (one per backbone layer) ---
        self.lateral_projs = nn.ModuleDict(
            OrderedDict(
                (key, LateralProjection(in_dims[key], out_dim))
                for key in self._sorted_keys
            )
        )

        # --- top-down fusion modules (one fewer than backbone layers) ---
        # Fuses from deepest (coarsest semantically) back to earliest.
        num_fusions = max(len(self._sorted_keys) - 1, 0)
        self.fusions = nn.ModuleList(
            [TopDownFusion(out_dim) for _ in range(num_fusions)]
        )

        # --- downsample levels to build the multi-scale pyramid ---
        # ViT layers all share the same spatial resolution, so the pyramid
        # is built by taking the fused finest-level feature and
        # progressively halving it with stride-2 convolutions.
        num_downsamples = max(num_levels - 1, 0)
        self.downsamples = nn.ModuleList(
            [DownsampleLevel(out_dim) for _ in range(num_downsamples)]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_to_spatial(
        tokens: torch.Tensor, h: int, w: int,
    ) -> torch.Tensor:
        """Reshape ``(B, N, D)`` token sequence to ``(B, D, H, W)``."""
        B, N, D = tokens.shape
        assert N == h * w, (
            f"Token count {N} does not match spatial shape {h}x{w}={h * w}"
        )
        return tokens.permute(0, 2, 1).reshape(B, D, h, w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: dict[str, torch.Tensor],
        spatial_shapes: tuple[int, int],
    ) -> list[torch.Tensor]:
        """Build the feature pyramid.

        Args:
            features: Multi-layer backbone features ``{layer_key: (B, N, D)}``.
            spatial_shapes: ``(H_patches, W_patches)`` grid dimensions.

        Returns:
            List of ``(B, out_dim, H_i, W_i)`` feature maps at decreasing
            resolutions.  The first element has the finest spatial resolution
            (equal to *spatial_shapes*); subsequent elements are
            progressively halved.
        """
        h, w = spatial_shapes

        # 1. Reshape to spatial grids and apply lateral 1x1 projections.
        #    Ordered from earliest layer (finest semantically) to deepest.
        laterals: list[torch.Tensor] = []
        for key in self._sorted_keys:
            spatial = self._reshape_to_spatial(features[key], h, w)
            laterals.append(self.lateral_projs[key](spatial))

        # 2. Top-down pathway: fuse from deepest back to earliest.
        #    All laterals share the same spatial size (ViT), so the
        #    interpolation inside TopDownFusion is effectively a no-op on
        #    size but the addition + smoothing 3x3 conv still contributes.
        num_layers = len(laterals)
        for i in range(num_layers - 2, -1, -1):
            laterals[i] = self.fusions[i](laterals[i], laterals[i + 1])

        # 3. Build multi-scale pyramid starting from the fused finest level.
        #    Level 0 = fused finest (full resolution).
        #    Levels 1..num_levels-1 are produced by successive stride-2 convs.
        outputs: list[torch.Tensor] = [laterals[0]]
        for down in self.downsamples:
            outputs.append(down(outputs[-1]))

        # 4. Trim to exactly num_levels.
        outputs = outputs[: self.num_levels]

        return outputs
