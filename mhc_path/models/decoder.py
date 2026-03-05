"""RF-DETR-style decoder with deformable attention for pathology detection.

Faithful port of RF-DETR's TransformerDecoder with:
- MSDeformAttn cross-attention (radial grid init)
- Post-norm decoder layers
- Query positional embeddings from sine-encoded reference points
- Iterative 4D box refinement (cx, cy, w, h)
- Group DETR support for training acceleration
- Auxiliary losses at intermediate layers
- Focal bias + bbox zero initialization

Ported from:
  rfdetr/models/transformer.py
  rfdetr/models/lwdetr.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_path.models.ms_deform_attn import MSDeformAttn
from mhc_path.models.util import inverse_sigmoid


# ---------------------------------------------------------------------------
# Helpers ported from RF-DETR transformer.py
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-layer perceptron (FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor: torch.Tensor, d_model: int = 256) -> torch.Tensor:
    """Generate sine positional embeddings from reference point coordinates.

    Args:
        pos_tensor: ``(B, N, 2)`` or ``(B, N, 4)`` normalised positions.
        d_model: model dimension (must be divisible by number of coords).

    Returns:
        ``(B, N, d_model)`` sine embeddings.
    """
    scale = 2 * math.pi
    dim_per_coord = d_model // pos_tensor.shape[-1]
    dim_t = torch.arange(dim_per_coord, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_per_coord)

    parts = []
    for i in range(pos_tensor.shape[-1]):
        x_embed = pos_tensor[..., i:i+1] * scale  # (B, N, 1)
        embed = x_embed / dim_t  # (B, N, dim_per_coord)
        embed_sin = embed[..., 0::2].sin()
        embed_cos = embed[..., 1::2].cos()
        embed = torch.stack([embed_sin, embed_cos], dim=-1).flatten(-2)
        parts.append(embed)

    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Position encoding for multi-scale features
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2-D sine positional encoding for spatial feature maps."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, H, W)`` feature tensor.
            mask: ``(B, H, W)`` bool mask (True = padding). None = no padding.

        Returns:
            ``(B, num_pos_feats*2, H, W)`` positional encoding.
        """
        B, _, H, W = x.shape
        if mask is None:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # Normalise to [0, 2*pi]
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, C/2)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)
        return pos


# ---------------------------------------------------------------------------
# Decoder layer: post-norm with Group DETR self-attention
# ---------------------------------------------------------------------------

class DeformableDecoderLayer(nn.Module):
    """Single decoder layer with post-norm (matching RF-DETR reference).

    Self-attention -> cross-attention (MSDeformAttn) -> FFN.
    Query positional embeddings are added to Q/K in self-attention and Q
    in cross-attention, following the reference implementation.
    """

    def __init__(
        self,
        d_model: int = 256,
        sa_n_heads: int = 8,
        ca_n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            d_model, sa_n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention to pixel features
        self.cross_attn = MSDeformAttn(
            d_model=d_model, n_heads=ca_n_heads,
            n_levels=n_levels, n_points=n_points,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
        group_detr: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model).
            query_pos: (B, N_q, d_model) positional embedding for queries.
            reference_points: (B, N_q, n_levels, 4) normalised reference boxes.
            input_flatten: (B, sum(H_l*W_l), d_model) flattened multi-scale values.
            input_spatial_shapes: (n_levels, 2) H,W per level.
            input_level_start_index: (n_levels,) start indices in flattened dim.
            input_padding_mask: (B, sum(H_l*W_l)) or None.
            group_detr: number of query groups for Group DETR.

        Returns:
            (B, N_q, d_model) updated query embeddings.
        """
        # --- Self-attention (post-norm) ---
        # Group DETR: partition queries into groups, run self-attn per group
        if group_detr > 1 and self.training:
            B, N_q, C = query.shape
            g_nq = N_q // group_detr
            q_groups = query.reshape(B * group_detr, g_nq, C)
            qp_groups = query_pos.reshape(B * group_detr, g_nq, C)
            q_with_pos = q_groups + qp_groups
            sa_out, _ = self.self_attn(q_with_pos, q_with_pos, q_groups)
            sa_out = sa_out.reshape(B, N_q, C)
        else:
            q_with_pos = query + query_pos
            sa_out, _ = self.self_attn(q_with_pos, q_with_pos, query)

        query = self.norm1(query + self.dropout1(sa_out))

        # --- Deformable cross-attention (post-norm) ---
        ca_out = self.cross_attn(
            query + query_pos, reference_points,
            input_flatten, input_spatial_shapes,
            input_level_start_index, input_padding_mask,
        )
        query = self.norm2(query + self.dropout2(ca_out))

        # --- FFN (post-norm) ---
        ffn_out = self.linear2(self.dropout_ffn(self.activation(self.linear1(query))))
        query = self.norm3(query + self.dropout3(ffn_out))

        return query


# ---------------------------------------------------------------------------
# Detection / Segmentation heads
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """Classification head for detection queries.

    Single Linear layer with focal loss bias initialization.
    Box regression is handled by per-layer bbox_embed MLPs in the decoder.
    """

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.class_head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # Focal loss bias initialization: bias = -log((1 - prior) / prior)
        prior_prob = 0.1
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_value)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeddings: (B, N_q, d_model).

        Returns:
            class_logits: (B, N_q, num_classes).
        """
        return self.class_head(query_embeddings)


class SegmentationHead(nn.Module):
    """Generates per-query masks via dot product between projected queries
    and projected pixel features.

    When ``upsample_factor > 1``, pixel features are upsampled via transposed
    convolutions before the dot product so masks are produced at higher
    native resolution (e.g. factor=4 turns 16×16 → 64×64).
    """

    def __init__(
        self, d_model: int, pixel_dim: int = 64, upsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(d_model, pixel_dim)

        layers: list[nn.Module] = [
            nn.Conv2d(d_model, pixel_dim, kernel_size=1),
            nn.GroupNorm(1, pixel_dim),
        ]
        # Upsample via transposed convolutions (each doubles spatial dims)
        f = upsample_factor
        while f > 1:
            layers.extend([
                nn.ConvTranspose2d(pixel_dim, pixel_dim, kernel_size=2, stride=2),
                nn.GroupNorm(1, pixel_dim),
                nn.ReLU(inplace=True),
            ])
            f //= 2

        self.pixel_proj = nn.Sequential(*layers)

    def forward(
        self, query_embeddings: torch.Tensor, pixel_features: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product mask generation.

        Args:
            query_embeddings: (B, N_q, d_model).
            pixel_features: (B, C, H, W) finest-level feature map.

        Returns:
            mask_logits: (B, N_q, H_up, W_up).
        """
        B, N_q, _ = query_embeddings.shape

        q_proj = self.query_proj(query_embeddings)
        p_proj = self.pixel_proj(pixel_features)

        _, _, H, W = p_proj.shape
        p_flat = p_proj.flatten(2)
        mask_logits = torch.bmm(q_proj, p_flat)
        return mask_logits.view(B, N_q, H, W)


# ---------------------------------------------------------------------------
# Full RF-DETR decoder
# ---------------------------------------------------------------------------

class RFDETRDecoder(nn.Module):
    """Full RF-DETR decoder with iterative box refinement, auxiliary losses,
    4D reference points, Group DETR, and position-aware query embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 100,
        num_classes: int = 10,
        n_decoder_layers: int = 6,
        n_levels: int = 4,
        sa_n_heads: int = 8,
        ca_n_heads: int = 8,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        group_detr: int = 1,
        with_segmentation: bool = True,
        mask_upsample_factor: int = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.n_levels = n_levels
        self.group_detr = group_detr
        self.with_segmentation = with_segmentation

        # Learnable 4D reference point embeddings (cx, cy, w, h)
        self.refpoint_embed = nn.Embedding(num_queries, 4)

        # Query positional embedding: sine -> MLP
        self.ref_point_head = MLP(d_model, d_model, d_model, num_layers=2)

        # Learnable query content embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Input projection per level
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.GroupNorm(1, d_model),
            )
            for _ in range(n_levels)
        ])

        # Position encoding for features
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # Decoder layers
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=d_model,
                sa_n_heads=sa_n_heads,
                ca_n_heads=ca_n_heads,
                n_levels=n_levels,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_decoder_layers)
        ])

        # Per-layer bbox refinement heads (shared class head, per-layer box head)
        self.detection_head = DetectionHead(d_model, num_classes)
        self.bbox_embed = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3)
            for _ in range(n_decoder_layers)
        ])
        self._init_bbox_embeds()

        self.final_norm = nn.LayerNorm(d_model)

        # Segmentation
        self.segmentation_head: Optional[SegmentationHead] = None
        if with_segmentation:
            self.segmentation_head = SegmentationHead(
                d_model, upsample_factor=mask_upsample_factor,
            )

        self._reset_parameters()

    def _init_bbox_embeds(self) -> None:
        """Zero-init last layer of each per-layer bbox refinement MLP."""
        for embed in self.bbox_embed:
            nn.init.zeros_(embed.layers[-1].weight)
            nn.init.zeros_(embed.layers[-1].bias)

    def _reset_parameters(self) -> None:
        # Grid init: spread queries uniformly so after .sigmoid() they
        # cover [0.05, 0.95] — prevents mode collapse from random clustering.
        nq = self.num_queries
        side = int(math.ceil(math.sqrt(nq)))
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0.05, 0.95, side),
            torch.linspace(0.05, 0.95, side),
            indexing="xy",
        ), dim=-1).reshape(-1, 2)[:nq]  # (nq, 2) for cx, cy
        inv_grid = torch.log(grid / (1 - grid))  # inverse sigmoid
        wh = torch.full((nq, 2), -2.2)  # sigmoid(-2.2) ≈ 0.10
        init = torch.cat([inv_grid, wh], dim=-1)  # (nq, 4)
        self.refpoint_embed.weight.data.copy_(init)

    def _prepare_features(
        self, multi_scale_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Project features and flatten for MSDeformAttn input.

        Returns:
            input_flatten: (B, sum(H_l*W_l), d_model)
            input_spatial_shapes: (n_levels, 2) as int64
            input_level_start_index: (n_levels,)
            input_padding_mask: None (we don't pad)
            projected_features: list of (B, d_model, H_l, W_l) for segmentation
        """
        projected_features: list[torch.Tensor] = []
        flatten_list: list[torch.Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []

        for lvl_idx, feat in enumerate(multi_scale_features):
            proj = self.input_proj[lvl_idx](feat)
            projected_features.append(proj)

            B, C, H, W = proj.shape
            spatial_shapes.append((H, W))

            # Add positional encoding
            pos = self.pos_encoder(proj)  # (B, C, H, W)
            flatten_list.append((proj + pos).flatten(2).transpose(1, 2))  # (B, H*W, C)

        input_flatten = torch.cat(flatten_list, dim=1)  # (B, sum(H*W), C)

        device = input_flatten.device
        input_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
        input_level_start_index = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            input_spatial_shapes.prod(1).cumsum(0)[:-1],
        ])

        return input_flatten, input_spatial_shapes, input_level_start_index, None, projected_features

    def forward(
        self, multi_scale_features: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the decoder on multi-scale backbone features.

        Args:
            multi_scale_features: list of L tensors (B, C, H_l, W_l).

        Returns:
            dict with keys:
                'class_logits': (B, num_queries, num_classes)
                'box_coords': (B, num_queries, 4)
                'mask_logits': (B, num_queries, H_0, W_0) if with_segmentation
                'aux_outputs': list of dicts with 'class_logits', 'box_coords'
        """
        B = multi_scale_features[0].shape[0]
        device = multi_scale_features[0].device

        (input_flatten, input_spatial_shapes,
         input_level_start_index, input_padding_mask,
         projected_features) = self._prepare_features(multi_scale_features)

        # 4D reference points from learnable embeddings
        refpoint = self.refpoint_embed.weight.sigmoid()  # (N_q, 4)

        # Group DETR: replicate queries and reference points
        group_detr = self.group_detr if self.training else 1
        if group_detr > 1:
            refpoint = refpoint.unsqueeze(0).expand(group_detr, -1, -1).flatten(0, 1)  # (G*N_q, 4)
            content = self.query_embed.weight.unsqueeze(0).expand(group_detr, -1, -1).flatten(0, 1)
        else:
            content = self.query_embed.weight

        reference_points = refpoint.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, 4)
        queries = content.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, d_model)

        n_levels = input_spatial_shapes.shape[0]
        aux_outputs: list[dict[str, torch.Tensor]] = []
        final_box_coords: Optional[torch.Tensor] = None

        # Iterative refinement through decoder layers
        for lid, layer in enumerate(self.layers):
            # Reference points for cross-attention: expand to per-level
            ref_pts_ca = reference_points[:, :, None, :].expand(-1, -1, n_levels, -1)

            # Query positional embedding from sine-encoded reference points
            query_pos = self.ref_point_head(
                gen_sineembed_for_position(reference_points, self.d_model)
            )

            queries = layer(
                queries, query_pos, ref_pts_ca,
                input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask,
                group_detr=group_detr,
            )

            # Iterative box refinement: predict delta and update reference points
            box_delta = self.bbox_embed[lid](queries)
            new_reference_points = (inverse_sigmoid(reference_points) + box_delta).sigmoid()

            # Auxiliary losses: prediction at this intermediate layer
            if self.training:
                layer_cls = self.detection_head(self.final_norm(queries))
                aux_outputs.append({
                    "class_logits": layer_cls,
                    "box_coords": new_reference_points,
                })

            # Keep gradient-attached coords from last layer for final output
            if lid == len(self.layers) - 1:
                final_box_coords = new_reference_points

            reference_points = new_reference_points.detach()

        queries = self.final_norm(queries)

        # Final detection outputs — single refinement, no recomputation
        class_logits = self.detection_head(queries)
        box_coords = final_box_coords

        # Collapse Group DETR back to base queries for inference
        if group_detr > 1:
            class_logits = class_logits[:, :self.num_queries]
            box_coords = box_coords[:, :self.num_queries]
            queries = queries[:, :self.num_queries]
            reference_points = reference_points[:, :self.num_queries]

        outputs: dict[str, torch.Tensor] = {
            "class_logits": class_logits,
            "box_coords": box_coords,
        }

        if self.training and aux_outputs:
            outputs["aux_outputs"] = aux_outputs

        if self.segmentation_head is not None:
            outputs["mask_logits"] = self.segmentation_head(
                queries, projected_features[0],
            )

        return outputs
