"""EoMT-style encoder-only decoder for instance segmentation.

Injects learnable queries into the final N blocks of a ViT backbone.
Queries attend jointly with patch tokens via standard self-attention,
eliminating the need for a separate decoder or FPN.

Outputs per-query masks and class logits — no box predictions.
Boxes can be derived from masks at evaluation time if needed.

Reference: Kerssies et al., "Your ViT is Secretly an Image Segmentation
Model", CVPR 2025.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EoMTDecoder(nn.Module):
    """Encoder-only decoder: queries injected into final ViT blocks.

    Outputs: class logits (including "no object") and mask logits per query.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_queries: int = 100,
        num_classes: int = 5,
        num_inject_blocks: int = 4,
        num_upscale_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_inject_blocks = num_inject_blocks

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Class head: +1 for "no object" class (softmax CE, not focal sigmoid)
        self.class_head = nn.Linear(embed_dim, num_classes + 1)

        # Mask head: 3-layer MLP, then einsum with upscaled patch features
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Upscale patch features from patch grid to higher resolution
        upscale_layers: list[nn.Module] = []
        for _ in range(num_upscale_blocks):
            upscale_layers.extend([
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
                nn.GELU(),
            ])
        self.upscale = nn.Sequential(*upscale_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        for m in self.mask_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def predict(
        self,
        tokens: torch.Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks and classes from a single block's output.

        Args:
            tokens: (B, N_q + N_patches, embed_dim) combined tokens.
            patch_grid: (H_patches, W_patches).

        Returns:
            (mask_logits, class_logits):
                mask_logits: (B, Q, H_up, W_up)
                class_logits: (B, Q, num_classes + 1)
        """
        B = tokens.shape[0]
        H, W = patch_grid

        query_tokens = tokens[:, :self.num_queries, :]
        patch_tokens = tokens[:, self.num_queries:, :]

        # Class prediction (includes "no object" class)
        class_logits = self.class_head(query_tokens)

        # Mask prediction: project queries, upscale patches, dot product
        query_proj = self.mask_head(query_tokens)
        patch_spatial = patch_tokens.transpose(1, 2).reshape(B, -1, H, W)
        patch_upscaled = self.upscale(patch_spatial)
        mask_logits = torch.einsum("bqc,bchw->bqhw", query_proj, patch_upscaled)

        return mask_logits, class_logits

    def forward(
        self,
        final_tokens: torch.Tensor,
        patch_grid: tuple[int, int],
        aux_block_outputs: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Decode all layers' outputs.

        Args:
            final_tokens: (B, N_q + N_patches, embed_dim) from last ViT block.
            patch_grid: (H_patches, W_patches).
            aux_block_outputs: Intermediate block outputs for auxiliary losses.

        Returns:
            dict with:
                'mask_logits': (B, Q, H_up, W_up) from final layer
                'class_logits': (B, Q, num_classes + 1) from final layer
                'aux_outputs': list of dicts with mask_logits + class_logits
        """
        mask_logits, class_logits = self.predict(final_tokens, patch_grid)

        outputs: dict = {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
        }

        # Auxiliary losses from intermediate blocks
        if self.training and aux_block_outputs:
            aux_outputs = []
            for aux_tokens in aux_block_outputs:
                aux_mask, aux_cls = self.predict(aux_tokens, patch_grid)
                aux_outputs.append({
                    "mask_logits": aux_mask,
                    "class_logits": aux_cls,
                })
            outputs["aux_outputs"] = aux_outputs

        return outputs


def masks_to_boxes(mask_logits: torch.Tensor) -> torch.Tensor:
    """Derive bounding boxes from mask logits for evaluation.

    Uses probability-weighted spatial moments.

    Args:
        mask_logits: (B, Q, H, W) raw mask logits.

    Returns:
        (B, Q, 4) boxes in cxcywh normalised format.
    """
    B, Q, H, W = mask_logits.shape
    probs = mask_logits.sigmoid()

    ys = torch.linspace(0, 1, H, device=mask_logits.device)
    xs = torch.linspace(0, 1, W, device=mask_logits.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    probs_flat = probs.view(B, Q, -1)
    total = probs_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    weights = probs_flat / total

    gx = grid_x.reshape(-1)
    gy = grid_y.reshape(-1)

    cx = (weights * gx).sum(dim=-1)
    cy = (weights * gy).sum(dim=-1)
    var_x = (weights * (gx - cx.unsqueeze(-1)) ** 2).sum(dim=-1)
    var_y = (weights * (gy - cy.unsqueeze(-1)) ** 2).sum(dim=-1)
    w = (2.0 * var_x.sqrt()).clamp(min=0.01)
    h = (2.0 * var_y.sqrt()).clamp(min=0.01)

    return torch.stack([cx, cy, w, h], dim=-1)
