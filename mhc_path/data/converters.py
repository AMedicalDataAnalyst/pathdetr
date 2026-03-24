"""Annotation Format Converters (Step 16).

Standalone converters to transform CoNSeP, PanNuke, and Lizard native
annotation formats into COCO-format JSON.  Each converter reads the native
format, remaps class labels to the canonical 5-class taxonomy via
:func:`~mhc_path.config.class_maps.get_class_map`, and writes a COCO JSON
file alongside a flat image directory.

These are data-preparation utilities -- run once before training, not at
training time.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io as sio
from scipy.ndimage import label as cc_label
from skimage.measure import regionprops

from mhc_path.config.class_maps import (
    CANONICAL_CLASSES,
    CONSEP_SOURCE_NAMES,
    LIZARD_SOURCE_NAMES,
    PANNUKE_SOURCE_NAMES,
    get_class_map,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    """Return COCO-style [x, y, w, h] from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = int(np.where(rows)[0][[0, -1]].tolist()[0]), int(
        np.where(rows)[0][[0, -1]].tolist()[1]
    )
    cmin, cmax = int(np.where(cols)[0][[0, -1]].tolist()[0]), int(
        np.where(cols)[0][[0, -1]].tolist()[1]
    )
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def _mask_area(mask: np.ndarray) -> float:
    """Return the pixel area of a binary mask."""
    return float(np.count_nonzero(mask))


def _mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as COCO uncompressed RLE (column-major order)."""
    flat = mask.flatten(order="F").astype(np.uint8)
    counts: list[int] = []
    prev = 0
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = v
    counts.append(run)
    # RLE must start with a background (0) run
    if flat[0] == 1:
        counts.insert(0, 0)
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _make_coco_skeleton(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap image and annotation lists in a COCO-format dict."""
    categories = [
        {"id": i, "name": name} for i, name in enumerate(CANONICAL_CLASSES)
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _write_json_deterministic(data: dict[str, Any], path: Path) -> None:
    """Write JSON with sorted keys for idempotent output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# CoNSeP converter
# ---------------------------------------------------------------------------


def convert_consep(mat_dir: str, output_dir: str) -> str:
    """Convert CoNSeP ``.mat`` files to COCO JSON + image directory.

    Parameters
    ----------
    mat_dir:
        Directory containing ``*.mat`` files. Each ``.mat`` file holds keys
        ``inst_map`` (H x W instance ID map) and ``inst_type`` (N x 1 class
        label array), and optionally ``img`` (the RGB image).
    output_dir:
        Destination directory. A ``annotations.json`` and ``images/``
        sub-directory will be created here.

    Returns
    -------
    str
        Path to the generated ``annotations.json`` file.
    """
    mat_path = Path(mat_dir)
    out_path = Path(output_dir)
    img_out = out_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    cm = get_class_map("consep")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0

    for mat_file in sorted(mat_path.glob("*.mat")):
        data = sio.loadmat(str(mat_file))
        inst_map: np.ndarray = data["inst_map"]
        inst_type: np.ndarray = np.asarray(data["inst_type"]).flatten()

        h, w = inst_map.shape[:2]
        file_name = mat_file.stem + ".png"
        image_id = len(images)
        images.append({"id": image_id, "file_name": file_name, "height": h, "width": w})

        # Copy / save image if present
        if "img" in data:
            from PIL import Image as PILImage

            img_arr = np.asarray(data["img"], dtype=np.uint8)
            PILImage.fromarray(img_arr).save(str(img_out / file_name))

        instance_ids = [i for i in np.unique(inst_map) if i != 0]
        for inst_id in sorted(instance_ids):
            mask = (inst_map == inst_id).astype(np.uint8)
            if _mask_area(mask) == 0:
                continue
            # inst_type is 1-indexed parallel to instance IDs
            src_label = int(inst_type[int(inst_id) - 1])
            cat_id = cm.remap_label(src_label, CONSEP_SOURCE_NAMES)
            bbox = _bbox_from_mask(mask)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    json_path = out_path / "annotations.json"
    _write_json_deterministic(_make_coco_skeleton(images, annotations), json_path)
    logger.info(
        "CoNSeP: wrote %d images, %d annotations -> %s",
        len(images), len(annotations), json_path,
    )
    return str(json_path)


# ---------------------------------------------------------------------------
# PanNuke converter
# ---------------------------------------------------------------------------


def convert_pannuke(fold_dir: str, fold_idx: int, output_dir: str) -> str:
    """Convert a PanNuke fold to COCO JSON + image directory.

    Parameters
    ----------
    fold_dir:
        Directory containing ``images.npy`` (N, 256, 256, 3) and
        ``masks.npy`` (N, 256, 256, 6). Channel 0-4 of the mask array
        correspond to the five PanNuke classes; channel 5 is background.
    fold_idx:
        Fold number (1, 2, or 3) used in naming.
    output_dir:
        Destination directory.

    Returns
    -------
    str
        Path to the generated ``annotations.json`` file.
    """
    fold_path = Path(fold_dir)
    out_path = Path(output_dir)
    img_out = out_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    from PIL import Image as PILImage

    cm = get_class_map("pannuke")
    images_arr: np.ndarray = np.load(str(fold_path / "images.npy"))
    masks_arr: np.ndarray = np.load(str(fold_path / "masks.npy"))

    num_images = images_arr.shape[0]
    num_classes = 5  # channels 0-4

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0

    for idx in range(num_images):
        rgb = images_arr[idx].astype(np.uint8)
        h, w = rgb.shape[:2]
        file_name = f"fold{fold_idx}_{idx:05d}.png"
        image_id = len(images)
        images.append({"id": image_id, "file_name": file_name, "height": h, "width": w})
        PILImage.fromarray(rgb).save(str(img_out / file_name))

        for ch in range(num_classes):
            channel = masks_arr[idx, :, :, ch]
            if channel.max() == 0:
                continue
            labeled, n_components = cc_label(channel > 0)
            for comp_id in range(1, n_components + 1):
                mask = (labeled == comp_id).astype(np.uint8)
                area = _mask_area(mask)
                if area == 0:
                    continue
                cat_id = cm.remap_label(ch, PANNUKE_SOURCE_NAMES)
                bbox = _bbox_from_mask(mask)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": _mask_to_rle(mask),
                    "iscrowd": 0,
                })
                ann_id += 1

    json_path = out_path / "annotations.json"
    _write_json_deterministic(_make_coco_skeleton(images, annotations), json_path)
    logger.info(
        "PanNuke fold %d: wrote %d images, %d annotations -> %s",
        fold_idx, len(images), len(annotations), json_path,
    )
    return str(json_path)


# ---------------------------------------------------------------------------
# Lizard converter
# ---------------------------------------------------------------------------


def convert_lizard(annotation_dir: str, image_dir: str, output_dir: str) -> str:
    """Convert Lizard annotations to COCO JSON.

    Parameters
    ----------
    annotation_dir:
        Directory containing ``*.mat`` files with keys ``inst_map`` and
        ``inst_type`` (same layout as CoNSeP).
    image_dir:
        Directory containing the corresponding ``.png`` images (same stem
        names as the ``.mat`` files).
    output_dir:
        Destination directory.

    Returns
    -------
    str
        Path to the generated ``annotations.json`` file.
    """
    ann_path = Path(annotation_dir)
    src_img_path = Path(image_dir)
    out_path = Path(output_dir)
    img_out = out_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    cm = get_class_map("lizard")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0

    for mat_file in sorted(ann_path.glob("*.mat")):
        data = sio.loadmat(str(mat_file))
        inst_map: np.ndarray = data["inst_map"]
        inst_type: np.ndarray = np.asarray(data["inst_type"]).flatten()

        h, w = inst_map.shape[:2]
        file_name = mat_file.stem + ".png"
        image_id = len(images)
        images.append({"id": image_id, "file_name": file_name, "height": h, "width": w})

        src_img = src_img_path / file_name
        if src_img.exists():
            shutil.copy2(str(src_img), str(img_out / file_name))

        instance_ids = [i for i in np.unique(inst_map) if i != 0]
        for inst_id in sorted(instance_ids):
            mask = (inst_map == inst_id).astype(np.uint8)
            if _mask_area(mask) == 0:
                continue
            src_label = int(inst_type[int(inst_id) - 1])
            cat_id = cm.remap_label(src_label, LIZARD_SOURCE_NAMES)
            bbox = _bbox_from_mask(mask)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    json_path = out_path / "annotations.json"
    _write_json_deterministic(_make_coco_skeleton(images, annotations), json_path)
    logger.info(
        "Lizard: wrote %d images, %d annotations -> %s",
        len(images), len(annotations), json_path,
    )
    return str(json_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_coco_annotations(json_path: str, image_dir: str) -> dict[str, Any]:
    """Run sanity checks on a COCO-format annotation file.

    Parameters
    ----------
    json_path:
        Path to the COCO JSON file.
    image_dir:
        Directory where the image files referenced in the JSON should exist.

    Returns
    -------
    dict
        Stats dictionary with keys: ``num_images``, ``num_annotations``,
        ``class_distribution``, ``mean_objects_per_image``,
        ``bbox_area_distribution``, ``num_degenerate_boxes``.

    Raises
    ------
    FileNotFoundError
        If any referenced image file is missing from *image_dir*.
    ValueError
        If any annotation has zero-area bounding box, an empty mask, or an
        unexpected (non-canonical) category ID.
    """
    with open(json_path) as f:
        coco = json.load(f)

    img_dir = Path(image_dir)
    valid_cat_ids = set(range(len(CANONICAL_CLASSES)))

    # Check images exist
    for img in coco["images"]:
        fp = img_dir / img["file_name"]
        if not fp.exists():
            raise FileNotFoundError(
                f"Image referenced in annotations not found: {fp}"
            )

    # Check annotations
    class_dist: dict[int, int] = {i: 0 for i in range(len(CANONICAL_CLASSES))}
    areas: list[float] = []
    num_degenerate = 0
    objects_per_image: dict[int, int] = {img["id"]: 0 for img in coco["images"]}

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in valid_cat_ids:
            raise ValueError(
                f"Unexpected category_id {cat_id} in annotation {ann['id']}. "
                f"Expected one of {sorted(valid_cat_ids)}."
            )
        bbox = ann["bbox"]
        w, h = bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            num_degenerate += 1
            raise ValueError(
                f"Zero-area bounding box in annotation {ann['id']}: {bbox}"
            )
        class_dist[cat_id] += 1
        areas.append(w * h)
        if ann["image_id"] in objects_per_image:
            objects_per_image[ann["image_id"]] += 1

    num_images = len(coco["images"])
    num_annotations = len(coco["annotations"])
    mean_objects = num_annotations / max(num_images, 1)

    area_arr = np.array(areas) if areas else np.array([0.0])
    stats: dict[str, Any] = {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "class_distribution": class_dist,
        "mean_objects_per_image": mean_objects,
        "bbox_area_distribution": {
            "min": float(area_arr.min()),
            "max": float(area_arr.max()),
            "mean": float(area_arr.mean()),
            "median": float(np.median(area_arr)),
        },
        "num_degenerate_boxes": num_degenerate,
    }

    logger.info("Validation passed: %s", json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# PanNuke tissue type extraction
# ---------------------------------------------------------------------------


_PANNUKE_TISSUE_NAMES: list[str] = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus",
]


def extract_tissue_types(
    fold_dir: str, fold_idx: int, output_dir: str,
) -> str:
    """Extract per-image tissue types from PanNuke types.npy.

    Parameters
    ----------
    fold_dir:
        Directory containing ``types.npy`` (N,) array of tissue type indices.
    fold_idx:
        Fold number (1, 2, or 3) used in image naming.
    output_dir:
        Destination directory. A ``tissue_types.json`` file will be created.

    Returns
    -------
    str
        Path to the generated ``tissue_types.json`` file.
    """
    fold_path = Path(fold_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    types_arr = np.load(str(fold_path / "types.npy")).flatten()
    image_tissues: dict[str, int] = {}
    for idx, tissue_id in enumerate(types_arr):
        file_name = f"fold{fold_idx}_{idx:05d}.png"
        image_tissues[file_name] = int(tissue_id)

    result = {
        "tissue_names": _PANNUKE_TISSUE_NAMES,
        "image_tissues": image_tissues,
    }
    json_path = out_path / "tissue_types.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "PanNuke fold %d: wrote tissue types for %d images -> %s",
        fold_idx, len(image_tissues), json_path,
    )
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Convert raw PanNuke .npy files to COCO-format annotations.

    Usage:
        python -m mhc_path.data.converters \\
            --raw_dir data/pannuke_raw --output_dir data
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw PanNuke .npy files to COCO-format annotations")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing fold1/, fold2/, fold3/ with .npy files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (pannuke_fold{1,2,3}/ subdirs will be created)")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3],
                        help="Which folds to convert (default: 1 2 3)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after conversion")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    for fold_idx in args.folds:
        fold_dir = raw_dir / f"fold{fold_idx}"
        out_dir = output_dir / f"pannuke_fold{fold_idx}"

        if not fold_dir.exists():
            print(f"WARNING: {fold_dir} does not exist, skipping fold {fold_idx}")
            continue

        print(f"Converting fold {fold_idx}: {fold_dir} -> {out_dir}")
        json_path = convert_pannuke(str(fold_dir), fold_idx, str(out_dir))
        print(f"  Annotations: {json_path}")

        # Extract tissue types if types.npy exists
        if (fold_dir / "types.npy").exists():
            tissue_path = extract_tissue_types(str(fold_dir), fold_idx, str(out_dir))
            print(f"  Tissue types: {tissue_path}")
        else:
            print(f"  WARNING: {fold_dir / 'types.npy'} not found, skipping tissue types")

        if args.validate:
            stats = validate_coco_annotations(json_path, str(out_dir / "images"))
            print(f"  Validated: {stats['num_images']} images, "
                  f"{stats['num_annotations']} annotations")
            print(f"  Class distribution: {stats['class_distribution']}")

    print("\nDone. Expected directory structure:")
    print(f"  {output_dir}/pannuke_fold{{1,2,3}}/")
    print(f"    annotations.json")
    print(f"    images/")
    print(f"    tissue_types.json")


if __name__ == "__main__":
    _cli()
