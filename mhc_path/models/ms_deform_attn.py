"""Multi-Scale Deformable Attention Module.

Faithful port of RF-DETR's MSDeformAttn (pure-PyTorch fallback path).
Uses bilinear grid_sample per level — no custom CUDA kernels needed.

Ported from:
  rfdetr/models/ops/modules/ms_deform_attn.py
  rfdetr/models/ops/functions/ms_deform_attn_func.py
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


# ---------------------------------------------------------------------------
# Pure-PyTorch deformable attention core
# ---------------------------------------------------------------------------

def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch multi-scale deformable attention via grid_sample.

    Args:
        value: ``(B, n_heads, head_dim, N)`` projected values (flattened).
        value_spatial_shapes: ``(n_levels, 2)`` — ``[(H_0, W_0), …]``.
        sampling_locations: ``(B, Len_q, n_heads, n_levels, n_points, 2)``
            in ``[0, 1]`` normalised coordinates.
        attention_weights: ``(B, Len_q, n_heads, n_levels * n_points)``
            softmaxed weights.

    Returns:
        ``(B, Len_q, n_heads * head_dim)`` attended output.
    """
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, _, L, P, _ = sampling_locations.shape
    value_list = value.split([int(H * W) for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module.

    Parameters:
        d_model: Hidden dimension.
        n_levels: Number of feature levels.
        n_heads: Number of attention heads.
        n_points: Sampling points per head per level.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the "
                "dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        # Radial grid bias initialisation
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(N, Length_query, C)``
            reference_points: ``(N, Length_query, n_levels, 2)`` or ``(…, 4)``
            input_flatten: ``(N, sum(H_l * W_l), C)``
            input_spatial_shapes: ``(n_levels, 2)``
            input_level_start_index: ``(n_levels,)``
            input_padding_mask: ``(N, sum(H_l * W_l))`` or ``None``

        Returns:
            ``(N, Length_query, C)``
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2,
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
