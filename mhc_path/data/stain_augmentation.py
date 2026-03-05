"""Step 2a: CPU Stain Preprocessing.

CPU-side stain statistics extraction and per-image augmentation.
Runs in DataLoader workers; prepares metadata for GPU augmentation (Step 2b).
Also provides CPU-only fallback augmentation for testing/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ── Stain deconvolution matrix (scikit-image convention) ──────────────

# Each row is a stain vector in OD space: [Hematoxylin, Eosin, DAB].
_STAIN_MATRIX = torch.tensor(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=torch.float32,
)
_STAIN_MATRIX_INV = torch.linalg.inv(_STAIN_MATRIX)

_OD_EPSILON = 1e-6
_RGB_EPSILON = 1e-6


# ── Color-space conversions (pure functions, reused by GPU module) ────


def rgb_to_od(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB [0, 1] to optical density. Shape-preserving."""
    return -torch.log10(rgb.clamp(min=_OD_EPSILON))


def od_to_rgb(od: torch.Tensor) -> torch.Tensor:
    """Convert optical density back to RGB [0, 1]. Shape-preserving."""
    return torch.pow(torch.tensor(10.0), -od).clamp(0.0, 1.0)


def rgb_to_hed(image: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> HED via optical density deconvolution. Accepts (3,H,W) or (B,3,H,W)."""
    od = rgb_to_od(image)
    # (..., 3, H, W) -> (..., H*W, 3) matmul -> (..., 3, H, W)
    spatial = od.shape[-2:]
    od_flat = od.flatten(-2).transpose(-1, -2)  # (..., N, 3)
    hed_flat = od_flat @ _STAIN_MATRIX_INV.to(od.device)
    return hed_flat.transpose(-1, -2).unflatten(-1, spatial)


def hed_to_rgb(hed: torch.Tensor) -> torch.Tensor:
    """HED -> RGB [0,1]. Accepts (3,H,W) or (B,3,H,W)."""
    spatial = hed.shape[-2:]
    hed_flat = hed.flatten(-2).transpose(-1, -2)  # (..., N, 3)
    od_flat = hed_flat @ _STAIN_MATRIX.to(hed.device)
    od = od_flat.transpose(-1, -2).unflatten(-1, spatial)
    return od_to_rgb(od)


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> HSV. Accepts (3,H,W) or (B,3,H,W)."""
    r = image.select(-3, 0)
    g = image.select(-3, 1)
    b = image.select(-3, 2)
    maxc = torch.max(image, dim=-3).values
    minc = torch.min(image, dim=-3).values
    diff = maxc - minc

    v = maxc
    s = torch.where(maxc > 0, diff / (maxc + _RGB_EPSILON), torch.zeros_like(maxc))

    h = torch.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > 0)
    mask_g = (maxc == g) & (diff > 0) & ~mask_r
    mask_b = (diff > 0) & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4.0
    h = h / 6.0
    h = h % 1.0

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """HSV -> RGB [0,1]. Accepts (3,H,W) or (B,3,H,W)."""
    h = hsv.select(-3, 0) * 6.0
    s = hsv.select(-3, 1)
    v = hsv.select(-3, 2)
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = torch.zeros_like(hsv)
    for idx, (r_val, g_val, b_val) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = i == idx
        rgb.select(-3, 0)[mask] = r_val[mask]
        rgb.select(-3, 1)[mask] = g_val[mask]
        rgb.select(-3, 2)[mask] = b_val[mask]

    return rgb.clamp(0.0, 1.0)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> LAB. Accepts (3, H, W) or (B, 3, H, W)."""
    # Channel dim: -3 works for both (3,H,W) and (B,3,H,W)
    linear = torch.where(
        image > 0.04045,
        torch.pow((image + 0.055) / 1.055, 2.4),
        image / 12.92,
    )

    r = linear.select(-3, 0)
    g = linear.select(-3, 1)
    b = linear.select(-3, 2)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def _f(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, torch.pow(t, 1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    l_val = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return torch.stack([l_val, a_val, b_val], dim=-3)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """LAB -> RGB [0,1]. Accepts (3, H, W) or (B, 3, H, W)."""
    l_val = lab.select(-3, 0)
    a_val = lab.select(-3, 1)
    b_val = lab.select(-3, 2)

    fy = (l_val + 16.0) / 116.0
    fx = a_val / 500.0 + fy
    fz = fy - b_val / 200.0

    delta = 6.0 / 29.0

    def _finv(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, torch.pow(t, 3.0), 3 * delta**2 * (t - 4.0 / 29.0))

    x = 0.95047 * _finv(fx)
    y = 1.0 * _finv(fy)
    z = 1.08883 * _finv(fz)

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    linear = torch.stack([r, g, b], dim=-3).clamp(0.0, 1.0)

    srgb = torch.where(
        linear > 0.0031308,
        1.055 * torch.pow(linear.clamp(min=1e-10), 1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return srgb.clamp(0.0, 1.0)


_FORWARD_CONVERTERS: dict[str, callable] = {
    "HED": rgb_to_hed,
    "HSV": rgb_to_hsv,
    "LAB": rgb_to_lab,
}
_INVERSE_CONVERTERS: dict[str, callable] = {
    "HED": hed_to_rgb,
    "HSV": hsv_to_rgb,
    "LAB": lab_to_rgb,
}


# ── Otsu thresholding ────────────────────────────────────────────────


def _otsu_threshold(gray: torch.Tensor) -> float:
    """Compute Otsu threshold on a 1D tensor of grayscale values in [0,1]."""
    values = gray.flatten().float()
    nbins = 256
    hist = torch.histc(values, bins=nbins, min=0.0, max=1.0)
    bin_edges = torch.linspace(0.0, 1.0, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    sum_total = (hist * bin_centers).sum()

    w0 = torch.tensor(0.0)
    sum0 = torch.tensor(0.0)
    best_idx = 0
    best_variance = -1.0

    for i in range(nbins):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mu0 = sum0 / w0
        mu1 = (sum_total - sum0) / w1
        variance = (w0 * w1 * (mu0 - mu1) ** 2).item()
        if variance > best_variance:
            best_variance = variance
            best_idx = i

    # Threshold at upper edge of the winning bin to sit between the two populations
    return bin_edges[best_idx + 1].item()


# ── StainStats dataclass ─────────────────────────────────────────────


@dataclass
class StainStats:
    foreground_mask: torch.Tensor  # (1, H, W) bool
    channel_means: dict[str, torch.Tensor]  # color_space -> (C,) per-channel mean
    channel_stds: dict[str, torch.Tensor]  # color_space -> (C,) per-channel std


# ── StainStatsExtractor ──────────────────────────────────────────────


class StainStatsExtractor:
    """Runs on CPU in DataLoader workers. Computes per-image stain statistics."""

    def __init__(self, color_spaces: tuple[str, ...] = ("HED", "HSV", "LAB")) -> None:
        for cs in color_spaces:
            if cs not in _FORWARD_CONVERTERS:
                raise ValueError(f"Unsupported color space: {cs}")
        self.color_spaces = color_spaces

    def __call__(self, image: torch.Tensor) -> StainStats:
        """Compute foreground mask and per-channel stats for each color space.

        Args:
            image: (3, H, W) float tensor in [0, 1].
        """
        gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        thresh = _otsu_threshold(gray)
        # Foreground = darker than threshold (tissue is darker than background)
        foreground_mask = (gray < thresh).unsqueeze(0)  # (1, H, W)

        # Fallback: if mask is empty, treat entire image as foreground
        if foreground_mask.sum() == 0:
            foreground_mask = torch.ones_like(foreground_mask, dtype=torch.bool)

        fg_flat = foreground_mask.squeeze(0)  # (H, W)

        channel_means: dict[str, torch.Tensor] = {}
        channel_stds: dict[str, torch.Tensor] = {}

        for cs in self.color_spaces:
            converted = _FORWARD_CONVERTERS[cs](image)  # (3, H, W)
            means = torch.zeros(3)
            stds = torch.zeros(3)
            for c in range(3):
                fg_pixels = converted[c][fg_flat]
                means[c] = fg_pixels.mean()
                stds[c] = fg_pixels.std() if fg_pixels.numel() > 1 else torch.tensor(0.0)
            channel_means[cs] = means
            channel_stds[cs] = stds

        return StainStats(
            foreground_mask=foreground_mask,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )


# ── RandStainAugmentation ───────────────────────────────────────────


class RandStainAugmentation:
    """CPU fallback: applies stain augmentation per-sample.

    For production training, use GPUStainAugmentation (Step 2b) instead.
    This exists for testing and environments without GPU.
    """

    def __init__(
        self,
        color_spaces: tuple[str, ...] = ("HED", "HSV", "LAB"),
        sigma_range: tuple[float, float] = (0.05, 0.2),
        separate_foreground: bool = True,
        p: float = 0.8,
    ) -> None:
        self.color_spaces = color_spaces
        self.sigma_range = sigma_range
        self.separate_foreground = separate_foreground
        self.p = p
        self._stats_extractor = StainStatsExtractor(color_spaces)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random stain augmentation. (3, H, W) -> (3, H, W)."""
        if torch.rand(1).item() >= self.p:
            return image

        # Pick a random color space
        idx = torch.randint(len(self.color_spaces), (1,)).item()
        cs = self.color_spaces[idx]

        forward = _FORWARD_CONVERTERS[cs]
        inverse = _INVERSE_CONVERTERS[cs]

        converted = forward(image)  # (3, H, W)

        # Random sigma for each channel
        sigma = (
            torch.rand(3) * (self.sigma_range[1] - self.sigma_range[0])
            + self.sigma_range[0]
        )
        perturbation = torch.randn(3) * sigma  # per-channel shift

        if self.separate_foreground:
            stats = self._stats_extractor(image)
            fg_mask = stats.foreground_mask.squeeze(0)  # (H, W)
            result = converted.clone()
            for c in range(3):
                result[c][fg_mask] = converted[c][fg_mask] + perturbation[c]
        else:
            result = converted + perturbation.reshape(3, 1, 1)

        rgb_out = inverse(result)
        return rgb_out.clamp(0.0, 1.0)


# ── DABIntensityJitter ───────────────────────────────────────────────


class DABIntensityJitter:
    """IHC-specific: jitter the DAB channel intensity independently."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.7, 1.3),
        p: float = 0.5,
    ) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Scale the DAB channel (index 2 in HED space). (3, H, W) -> (3, H, W)."""
        if torch.rand(1).item() >= self.p:
            return image

        hed = rgb_to_hed(image)

        scale = (
            torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        )
        hed[2] = hed[2] * scale  # DAB is channel 2

        rgb_out = hed_to_rgb(hed)
        return rgb_out.clamp(0.0, 1.0)
