"""DETR decoder diagnostics: attention maps, backbone stats, query analysis.

Provides hook-based utilities for extracting internal model statistics
without modifying forward pass code. All functions are designed for
offline analysis (not in the training hot loop).

Depends on:
    - mhc_path.models.ms_deform_attn (MSDeformAttn module)
    - mhc_path.models.box_utils (cxcywh_to_xyxy)
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.nn as nn

from mhc_path.models.ms_deform_attn import MSDeformAttn


# ---------------------------------------------------------------------------
# Attention map extraction
# ---------------------------------------------------------------------------

class AttentionHook:
    """Captures attention weights from MSDeformAttn layers via forward hooks."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self.captured: dict[str, dict[str, torch.Tensor]] = {}

    def register(self, model: nn.Module) -> None:
        """Register hooks on all MSDeformAttn modules in the model."""
        for name, module in model.named_modules():
            if isinstance(module, MSDeformAttn):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook_fn(module: MSDeformAttn, inputs: tuple, output: torch.Tensor) -> None:
            query = inputs[0]
            # Recompute attention weights (same as forward) to capture them
            attn_raw = module.attention_weights(query)
            B, Len_q = attn_raw.shape[:2]
            n_lp = module.n_heads * module.n_levels * module.n_points
            attn_weights = attn_raw.view(B, Len_q, module.n_heads, n_lp // module.n_heads)
            attn_weights = torch.softmax(attn_weights, dim=-1)

            # Compute summary statistics (not full maps)
            entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean()
            max_weight = attn_weights.max(dim=-1).values.mean()

            self.captured[name] = {
                "entropy": entropy.detach().cpu(),
                "max_weight": max_weight.detach().cpu(),
                "n_heads": torch.tensor(module.n_heads),
                "n_levels": torch.tensor(module.n_levels),
                "n_points": torch.tensor(module.n_points),
            }
        return hook_fn

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Clear captured data."""
        self.captured.clear()


@torch.no_grad()
def extract_attention_stats(
    model: nn.Module,
    images: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Run a forward pass and return attention statistics per layer.

    Args:
        model: the full MHCPath model (or any model containing MSDeformAttn).
        images: (B, 3, H, W) input tensor.

    Returns:
        Dict mapping layer name to {"entropy": float, "max_weight": float}.
    """
    hook = AttentionHook()
    hook.register(model)
    was_training = model.training
    model.eval()

    try:
        model(images)
    finally:
        hook.remove()
        model.train(was_training)

    results: dict[str, dict[str, float]] = {}
    for name, data in hook.captured.items():
        results[name] = {
            "entropy": data["entropy"].item(),
            "max_weight": data["max_weight"].item(),
        }
    return results


# ---------------------------------------------------------------------------
# Backbone feature statistics
# ---------------------------------------------------------------------------

class BackboneFeatureHook:
    """Captures output statistics from backbone feature layers."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self.stats: dict[str, dict[str, float]] = {}

    def register(self, backbone: nn.Module, layer_names: Optional[list[str]] = None) -> None:
        """Register hooks on backbone output layers.

        If layer_names is None, hooks all children with 'block' or 'layer' in the name.
        """
        for name, module in backbone.named_modules():
            if layer_names is not None:
                if name not in layer_names:
                    continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook_fn(module: nn.Module, inputs: tuple, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                t = output.detach().float()
                self.stats[name] = {
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "sparsity": (t.abs() < 1e-6).float().mean().item(),
                    "l2_norm": t.norm(2).item(),
                }
        return hook_fn

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.stats.clear()


@torch.no_grad()
def compute_backbone_stats(
    model: nn.Module,
    images: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Run a forward pass and return feature statistics from the backbone.

    Hooks onto the model's backbone (expects model.backbone attribute).

    Args:
        model: the full MHCPath model.
        images: (B, 3, H, W) input tensor.

    Returns:
        Dict mapping layer name to {mean, std, sparsity, l2_norm}.
    """
    if not hasattr(model, "backbone"):
        return {}

    hook = BackboneFeatureHook()
    hook.register(model.backbone)
    was_training = model.training
    model.eval()

    try:
        model(images)
    finally:
        hook.remove()
        model.train(was_training)

    return dict(hook.stats)


# ---------------------------------------------------------------------------
# Unmatched query analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def unmatched_query_stats(
    outputs: dict[str, torch.Tensor],
    matched_indices: list[tuple[torch.Tensor, torch.Tensor]],
    conf_threshold: float = 0.3,
) -> dict[str, float]:
    """Analyse queries that have high confidence but weren't matched.

    Args:
        outputs: model outputs with "pred_logits" (B, Q, C).
        matched_indices: list of (pred_idx, gt_idx) from Hungarian matching.
        conf_threshold: threshold for "high confidence".

    Returns:
        Dict with n_unmatched_high_conf, n_matched, n_total_queries.
    """
    logits = outputs["pred_logits"]
    B, Q, _ = logits.shape
    max_scores = logits.sigmoid().max(dim=-1).values  # (B, Q)

    n_matched = 0
    n_high_conf_unmatched = 0

    for b in range(B):
        matched_set = set()
        if b < len(matched_indices):
            pred_idx = matched_indices[b][0]
            matched_set = set(pred_idx.tolist()) if len(pred_idx) > 0 else set()
        n_matched += len(matched_set)

        for q in range(Q):
            if q not in matched_set and max_scores[b, q].item() > conf_threshold:
                n_high_conf_unmatched += 1

    return {
        "n_unmatched_high_conf": float(n_high_conf_unmatched),
        "n_matched": float(n_matched),
        "n_total_queries": float(B * Q),
    }


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    images: torch.Tensor,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> dict[str, float]:
    """Measure inference time per image using CUDA events.

    Args:
        model: the model to benchmark.
        images: (B, 3, H, W) a single batch of images.
        n_warmup: warmup iterations (not timed).
        n_measure: measurement iterations.

    Returns:
        Dict with mean_ms, std_ms, n_images.
    """
    model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    # Warmup
    for _ in range(n_warmup):
        model(images)

    if use_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_measure):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            model(images)
            times.append((time.perf_counter() - t0) * 1000.0)

    times_t = torch.tensor(times)
    return {
        "mean_ms": times_t.mean().item(),
        "std_ms": times_t.std().item(),
        "n_images": float(images.shape[0]),
        "ms_per_image": times_t.mean().item() / images.shape[0],
    }
