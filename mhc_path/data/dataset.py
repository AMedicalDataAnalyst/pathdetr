"""Step 3: Detection & Segmentation Dataset.

PyTorch Datasets serving tile images with COCO-format detection and segmentation
annotations for DETR-style training. Supports CoNSeP, PanNuke, and Lizard via
the canonical class mapping from Step 0. GPU augmentation (Step 2b) happens
outside the dataset in the training loop -- the dataset returns raw images
plus optional precomputed stain stats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mhc_path.config.class_maps import (
    CONSEP_SOURCE_NAMES,
    LIZARD_SOURCE_NAMES,
    PANNUKE_SOURCE_NAMES,
    ClassMap,
    get_class_map,
)
from mhc_path.data.stain_augmentation import StainStats, StainStatsExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-name lookup per dataset
# ---------------------------------------------------------------------------

_DATASET_SOURCE_NAMES: dict[str, tuple[str, ...]] = {
    "consep": CONSEP_SOURCE_NAMES,
    "pannuke": PANNUKE_SOURCE_NAMES,
    "lizard": LIZARD_SOURCE_NAMES,
}


# ---------------------------------------------------------------------------
# Detection target dataclass
# ---------------------------------------------------------------------------


@dataclass
class DetectionTarget:
    """Per-image detection targets for DETR-style training.

    Parameters
    ----------
    boxes : torch.Tensor
        (N, 4) bounding boxes in center-x, center-y, width, height format,
        normalized to [0, 1] by image dimensions.
    labels : torch.Tensor
        (N,) int64 canonical class indices.
    masks : torch.Tensor or None
        (N, H, W) binary masks per instance, or ``None`` if unavailable.
    """

    boxes: torch.Tensor
    labels: torch.Tensor
    masks: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Box-format conversions
# ---------------------------------------------------------------------------


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) boxes from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Mask decoding utilities
# ---------------------------------------------------------------------------


def _decode_rle(rle: dict[str, Any], height: int, width: int) -> torch.Tensor:
    """Decode COCO-style run-length encoding to a binary (H, W) mask.

    Parameters
    ----------
    rle : dict
        Must contain ``"counts"`` (list of int) and ``"size"`` ([H, W]).
    height : int
        Expected mask height.
    width : int
        Expected mask width.

    Returns
    -------
    torch.Tensor
        (H, W) bool tensor.
    """
    counts = rle["counts"]
    if isinstance(counts, str):
        # Compressed RLE string (pycocotools format) -- decode manually
        # This handles the compact COCO RLE encoding
        import pycocotools.mask as mask_util  # type: ignore[import-untyped]

        decoded = mask_util.decode(rle)
        return torch.from_numpy(decoded.astype(bool)).squeeze(-1)

    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    val = 0  # RLE starts with background count
    for count in counts:
        flat[pos : pos + count] = val
        pos += count
        val = 1 - val
    mask = flat.reshape((height, width), order="F")  # column-major (COCO)
    return torch.from_numpy(mask.astype(bool))


def _decode_polygon(
    segmentation: list[list[float]], height: int, width: int
) -> torch.Tensor:
    """Rasterize COCO polygon segmentation to a binary (H, W) mask.

    Uses a simple scanline fill without external dependencies.

    Parameters
    ----------
    segmentation : list of list of float
        COCO polygon format: [[x1, y1, x2, y2, ...], ...].
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    torch.Tensor
        (H, W) bool tensor.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        # Convert flat list to (M, 2) array of (x, y) pairs
        pts = np.array(polygon, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        if n < 3:
            continue
        # Scanline rasterization
        ys = pts[:, 1]
        y_min = max(0, int(np.floor(ys.min())))
        y_max = min(height - 1, int(np.ceil(ys.max())))
        for y in range(y_min, y_max + 1):
            intersections: list[float] = []
            yf = float(y) + 0.5
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i, 1], pts[j, 1]
                if (y0 <= yf < y1) or (y1 <= yf < y0):
                    x0, x1 = pts[i, 0], pts[j, 0]
                    t = (yf - y0) / (y1 - y0)
                    x_intersect = x0 + t * (x1 - x0)
                    intersections.append(x_intersect)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(np.floor(intersections[k])))
                x_end = min(width, int(np.ceil(intersections[k + 1])))
                mask[y, x_start:x_end] = 1
    return torch.from_numpy(mask.astype(bool))


def decode_mask(
    annotation: dict[str, Any], height: int, width: int
) -> Optional[torch.Tensor]:
    """Decode a COCO annotation's segmentation field to a binary mask.

    Supports RLE (``"counts"`` key) and polygon formats. Returns ``None``
    if the annotation has no segmentation field.

    Parameters
    ----------
    annotation : dict
        Single COCO annotation dict.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    torch.Tensor or None
        (H, W) bool tensor, or ``None`` if no segmentation data.
    """
    seg = annotation.get("segmentation")
    if seg is None:
        return None

    if isinstance(seg, dict) and "counts" in seg:
        return _decode_rle(seg, height, width)

    if isinstance(seg, list) and len(seg) > 0:
        return _decode_polygon(seg, height, width)

    return None


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def _load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image file as a (3, H, W) float32 tensor in [0, 1].

    Uses PIL to avoid hard torchvision dependency in the data module.
    """
    from PIL import Image  # local import to keep top-level lean

    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# PathologyDetectionDataset
# ---------------------------------------------------------------------------


class PathologyDetectionDataset(Dataset):
    """COCO-format detection and segmentation dataset for pathology tiles.

    Reads a COCO JSON annotation file and corresponding images. Source class
    labels are remapped to the canonical 5-class taxonomy via the ClassMap
    for the specified ``dataset_name``.

    No augmentation is applied. Raw images are returned alongside optional
    precomputed stain statistics (for the GPU augmentation pipeline).

    Parameters
    ----------
    annotation_file : str
        Path to COCO-format JSON annotation file.
    image_dir : str
        Directory containing the tile images referenced by the annotations.
    dataset_name : str
        One of ``"consep"``, ``"pannuke"``, ``"lizard"``. Selects the class
        mapping via :func:`get_class_map`.
    stain_stats_extractor : StainStatsExtractor or None
        If provided, computes per-image stain statistics on CPU.
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        dataset_name: str = "consep",
        stain_stats_extractor: Optional[StainStatsExtractor] = None,
    ) -> None:
        self._image_dir = Path(image_dir)
        self._class_map: ClassMap = get_class_map(dataset_name)
        self._source_names: tuple[str, ...] = _DATASET_SOURCE_NAMES[dataset_name]
        self._stain_stats_extractor = stain_stats_extractor

        # Load COCO JSON
        with open(annotation_file, "r") as f:
            coco: dict[str, Any] = json.load(f)

        # Build image index: image_id -> image metadata
        self._images: list[dict[str, Any]] = coco["images"]
        self._image_id_to_meta: dict[int, dict[str, Any]] = {
            img["id"]: img for img in self._images
        }

        # Group annotations by image_id
        self._anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for img in self._images:
            self._anns_by_image[img["id"]] = []
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self._anns_by_image:
                self._anns_by_image[img_id].append(ann)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load image, decode annotations, remap labels, compute stain stats.

        Returns
        -------
        dict
            ``"image"`` : (3, H, W) float32 tensor in [0, 1].
            ``"target"`` : :class:`DetectionTarget`.
            ``"stain_stats"`` : :class:`StainStats` or ``None``.
        """
        img_meta = self._images[idx]
        img_id = img_meta["id"]
        height = img_meta["height"]
        width = img_meta["width"]

        # Load image
        file_name = img_meta["file_name"]
        image_path = str(self._image_dir / file_name)
        image = _load_image_as_tensor(image_path)

        # Parse annotations
        anns = self._anns_by_image.get(img_id, [])
        boxes_list: list[torch.Tensor] = []
        labels_list: list[int] = []
        masks_list: list[torch.Tensor] = []
        has_masks = False

        for ann in anns:
            # COCO bbox: [x, y, w, h] (top-left corner)
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue  # skip degenerate boxes

            # Convert to xyxy then to cxcywh normalized
            x1 = x / width
            y1 = y / height
            x2 = (x + bw) / width
            y2 = (y + bh) / height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            nw = x2 - x1
            nh = y2 - y1
            boxes_list.append(torch.tensor([cx, cy, nw, nh], dtype=torch.float32))

            # category_ids are already canonical (converter maps raw → canonical)
            labels_list.append(ann["category_id"])

            # Decode mask
            mask = decode_mask(ann, height, width)
            if mask is not None:
                has_masks = True
                masks_list.append(mask)

        # Assemble tensors
        if len(boxes_list) > 0:
            boxes = torch.stack(boxes_list, dim=0)  # (N, 4)
            labels = torch.tensor(labels_list, dtype=torch.int64)  # (N,)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if has_masks and len(masks_list) == len(boxes_list):
            masks: Optional[torch.Tensor] = torch.stack(masks_list, dim=0)
        else:
            masks = None

        target = DetectionTarget(boxes=boxes, labels=labels, masks=masks)

        # Stain stats
        stain_stats: Optional[StainStats] = None
        if self._stain_stats_extractor is not None:
            stain_stats = self._stain_stats_extractor(image)

        return {
            "image": image,
            "target": target,
            "stain_stats": stain_stats,
        }


# ---------------------------------------------------------------------------
# SSLTileDataset
# ---------------------------------------------------------------------------


class SSLTileDataset(Dataset):
    """Unlabelled tile dataset for Phase 1 SSL pretraining.

    Returns oversized tiles. Multi-cropping and HistoRotate happen on GPU
    in the training engine (Step 10) via GPUPathologyAugPipeline -- not here.

    Parameters
    ----------
    tile_dir : str
        Directory containing tile images (PNG/JPEG).
    stain_stats_extractor : StainStatsExtractor or None
        If provided, computes per-image stain statistics on CPU.
    """

    _EXTENSIONS: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    def __init__(
        self,
        tile_dir: str,
        stain_stats_extractor: Optional[StainStatsExtractor] = None,
    ) -> None:
        self._tile_dir = Path(tile_dir)
        self._stain_stats_extractor = stain_stats_extractor

        # Collect all image files, sorted for reproducibility
        self._paths: list[Path] = sorted(
            p
            for p in self._tile_dir.iterdir()
            if p.suffix.lower() in self._EXTENSIONS
        )
        if len(self._paths) == 0:
            logger.warning("SSLTileDataset: no images found in %s", tile_dir)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a single tile image.

        Returns
        -------
        dict
            ``"image"`` : (3, H, W) float32 tensor in [0, 1].
            ``"stain_stats"`` : :class:`StainStats` or ``None``.
        """
        image = _load_image_as_tensor(str(self._paths[idx]))

        stain_stats: Optional[StainStats] = None
        if self._stain_stats_extractor is not None:
            stain_stats = self._stain_stats_extractor(image)

        return {
            "image": image,
            "stain_stats": stain_stats,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def detection_collate_fn(
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Collate a batch of detection samples with variable object counts.

    Images are stacked into a (B, 3, H, W) tensor (assumes same spatial size).
    Boxes and labels are padded to the maximum object count in the batch, with
    a ``num_objects`` tensor indicating the valid count per sample. Masks are
    similarly padded when present.

    Parameters
    ----------
    batch : sequence of dict
        Output of :meth:`PathologyDetectionDataset.__getitem__`.

    Returns
    -------
    dict
        ``"images"`` : (B, 3, H, W) float32 tensor.
        ``"boxes"`` : (B, N_max, 4) float32 tensor (zero-padded).
        ``"labels"`` : (B, N_max) int64 tensor (zero-padded).
        ``"masks"`` : (B, N_max, H, W) bool tensor, or ``None``.
        ``"num_objects"`` : (B,) int64 tensor.
        ``"stain_stats"`` : list of :class:`StainStats` or ``None``.
    """
    images = torch.stack([s["image"] for s in batch], dim=0)
    targets: list[DetectionTarget] = [s["target"] for s in batch]

    num_objects_list = [t.boxes.shape[0] for t in targets]
    max_objects = max(num_objects_list) if num_objects_list else 0
    # Ensure at least 1 slot to avoid zero-dim tensors
    max_objects = max(max_objects, 1)

    b = len(batch)
    padded_boxes = torch.zeros(b, max_objects, 4, dtype=torch.float32)
    padded_labels = torch.zeros(b, max_objects, dtype=torch.int64)
    num_objects = torch.tensor(num_objects_list, dtype=torch.int64)

    # Check for masks
    any_masks = any(t.masks is not None for t in targets)
    padded_masks: Optional[torch.Tensor] = None
    if any_masks:
        # Determine mask spatial dims from the first available mask
        mh, mw = 0, 0
        for t in targets:
            if t.masks is not None and t.masks.numel() > 0:
                mh, mw = t.masks.shape[1], t.masks.shape[2]
                break
        if mh > 0 and mw > 0:
            padded_masks = torch.zeros(
                b, max_objects, mh, mw, dtype=torch.bool
            )

    for i, t in enumerate(targets):
        n = t.boxes.shape[0]
        if n > 0:
            padded_boxes[i, :n] = t.boxes
            padded_labels[i, :n] = t.labels
            if padded_masks is not None and t.masks is not None:
                padded_masks[i, :n] = t.masks

    # Stain stats as list (variable-shaped foreground masks)
    stain_stats = [s["stain_stats"] for s in batch]
    if all(ss is None for ss in stain_stats):
        stain_stats_out: Any = None
    else:
        stain_stats_out = stain_stats

    return {
        "images": images,
        "boxes": padded_boxes,
        "labels": padded_labels,
        "masks": padded_masks,
        "num_objects": num_objects,
        "stain_stats": stain_stats_out,
    }


def ssl_collate_fn(
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Collate a batch of SSL tile samples.

    Parameters
    ----------
    batch : sequence of dict
        Output of :meth:`SSLTileDataset.__getitem__`.

    Returns
    -------
    dict
        ``"images"`` : (B, 3, H, W) float32 tensor.
        ``"stain_stats"`` : list of :class:`StainStats` or ``None``.
    """
    images = torch.stack([s["image"] for s in batch], dim=0)

    stain_stats = [s["stain_stats"] for s in batch]
    if all(ss is None for ss in stain_stats):
        stain_stats_out: Any = None
    else:
        stain_stats_out = stain_stats

    return {
        "images": images,
        "stain_stats": stain_stats_out,
    }
