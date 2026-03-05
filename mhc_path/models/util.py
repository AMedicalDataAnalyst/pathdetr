"""Shared utilities: NestedTensor, distributed helpers, and misc functions.

Ported from RF-DETR (Roboflow) util/misc.py. Only the subset needed by
the detection pipeline is included here — no MetricLogger, no ONNX export.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# ---------------------------------------------------------------------------
# NestedTensor
# ---------------------------------------------------------------------------

class NestedTensor:
    """Batched images + padding mask.

    Attributes:
        tensors: ``(B, C, H, W)`` padded image batch.
        mask: ``(B, H, W)`` boolean mask — ``True`` for padding pixels.
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> Tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """Pad a list of ``(C, H_i, W_i)`` tensors to uniform size with a mask."""
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: Tuple[int, ...] = (1,)) -> List[Tensor]:
    """Top-k accuracy between predicted logits and integer labels."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def interpolate(
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """``F.interpolate`` wrapper with empty-batch safety."""
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
