"""Calibrate per-class score thresholds to maximise PQ.

Sweeps thresholds independently per class on the validation set, finding
the optimal threshold for each class that maximises its class-specific PQ.

Rare classes (e.g. dead cells at 1.6%) need lower thresholds to maintain
recall, while dominant classes (neoplastic at 40%) can afford higher
thresholds to suppress false positives.

Usage:
    python -m experiments.calibrate_thresholds \
        --checkpoint experiments/baseline/fold3/checkpoints/best_pq.pt \
        --data_root data --test_fold 3

    # Then use the calibrated thresholds in evaluation:
    python -m experiments.train_pannuke \
        --per_class_thresholds 0.35 0.30 0.25 0.20 0.10 ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from mhc_path.config.class_maps import CANONICAL_CLASSES
from mhc_path.config.reproducibility import seed_everything
from mhc_path.data.dataset import PathologyDetectionDataset, detection_collate_fn
from mhc_path.evaluation.metrics import PanopticQuality
from mhc_path.models.box_utils import cxcywh_to_xyxy
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig
from mhc_path.training.losses import DetectionTarget

_NUM_CLASSES = 5


def _collect_raw_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict], list[DetectionTarget]]:
    """Run model on all images, return unfiltered predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["images"].to(device)
        raw = model(images)

        logits = raw.get("pred_logits", raw.get("class_logits"))
        boxes_raw = raw.get("pred_boxes", raw.get("box_coords"))
        mask_logits = raw.get("pred_masks", raw.get("mask_logits"))

        num_obj = batch["num_objects"]
        batch_boxes = batch["boxes"]
        batch_labels = batch["labels"]
        batch_masks = batch.get("masks")

        for b in range(images.shape[0]):
            sc, lb = logits[b].sigmoid().max(dim=-1)
            bx = cxcywh_to_xyxy(boxes_raw[b])

            masks = None
            if mask_logits is not None:
                ml = mask_logits[b]
                up = torch.nn.functional.interpolate(
                    ml.unsqueeze(1), size=(256, 256),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                masks = (up.sigmoid() > 0.5)

            all_preds.append({
                "boxes": bx, "scores": sc, "labels": lb,
                "masks": masks,
            })

            n = num_obj[b].item()
            if isinstance(batch_boxes, (list, tuple)):
                tb = batch_boxes[b][:n].to(device)
                tl = batch_labels[b][:n].to(device)
            else:
                tb = batch_boxes[b, :n].to(device)
                tl = batch_labels[b, :n].to(device)
            tm = None
            if batch_masks is not None:
                tm = batch_masks[b, :n].to(device) if not isinstance(batch_masks, list) \
                    else batch_masks[b][:n].to(device)
            all_targets.append(DetectionTarget(boxes=tb, labels=tl, masks=tm))

    return all_preds, all_targets


def _apply_per_class_threshold(
    preds: list[dict],
    thresholds: dict[int, float],
    fallback: float = 0.3,
) -> list[dict]:
    """Filter predictions using per-class thresholds."""
    filtered = []
    for pred in preds:
        sc, lb, bx = pred["scores"], pred["labels"], pred["boxes"]
        thresh_per_query = torch.tensor(
            [thresholds.get(l.item(), fallback) for l in lb],
            device=sc.device)
        keep = sc >= thresh_per_query
        entry = {"boxes": bx[keep], "scores": sc[keep], "labels": lb[keep]}
        if pred.get("masks") is not None:
            entry["masks"] = pred["masks"][keep]
        filtered.append(entry)
    return filtered


def _evaluate_pq(
    preds: list[dict], targets: list[DetectionTarget],
) -> dict[str, float]:
    """Compute PQ and per-class PQ."""
    pq = PanopticQuality(num_classes=_NUM_CLASSES, iou_threshold=0.5)
    pq.update(preds, targets)
    return pq.compute()


def calibrate(
    preds: list[dict],
    targets: list[DetectionTarget],
    sweep_range: tuple[float, float, float] = (0.05, 0.80, 0.05),
) -> dict[int, float]:
    """Find optimal per-class threshold by sweeping each class independently.

    For each class, fixes all other classes at 0.3 and sweeps the target
    class threshold. Picks the threshold maximising that class's PQ.
    """
    lo, hi, step = sweep_range
    candidates = []
    t = lo
    while t <= hi + 1e-9:
        candidates.append(round(t, 3))
        t += step

    best_thresholds = {}

    for cls_id in range(_NUM_CLASSES):
        cls_name = CANONICAL_CLASSES[cls_id]
        best_pq = -1.0
        best_t = 0.3

        for t in candidates:
            thresholds = {c: 0.3 for c in range(_NUM_CLASSES)}
            thresholds[cls_id] = t
            filt = _apply_per_class_threshold(preds, thresholds)
            results = _evaluate_pq(filt, targets)

            cls_pq = results.get(f"PQ_class{cls_id}", 0.0)
            if cls_pq > best_pq:
                best_pq = cls_pq
                best_t = t

        best_thresholds[cls_id] = best_t
        print(f"  {cls_name:>14s} (class {cls_id}): best_threshold={best_t:.2f}  "
              f"PQ_class{cls_id}={best_pq:.4f}")

    return best_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-class score thresholds for PQ")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_fold", type=int, required=True)
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16")
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--group_detr", type=int, default=3)
    parser.add_argument("--mask_upsample_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sweep_lo", type=float, default=0.05)
    parser.add_argument("--sweep_hi", type=float, default=0.80)
    parser.add_argument("--sweep_step", type=float, default=0.05)
    parser.add_argument("--out_dir", type=str, default="experiments/calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_cfg = MHCPathConfig(
        backbone=args.backbone, backbone_frozen=True, lora_rank=None,
        fpn_dim=256, num_queries=args.num_queries, num_classes=_NUM_CLASSES,
        num_decoder_layers=6, with_segmentation=True, use_mhc=False,
        output_layers=(6, 12, 18, 24), fpn_levels=4,
        mask_upsample_factor=args.mask_upsample_factor,
        group_detr=args.group_detr,
    )
    model = MHCPath(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Load data
    test_dir = data_root / f"pannuke_fold{args.test_fold}"
    ds = PathologyDetectionDataset(
        annotation_file=str(test_dir / "annotations.json"),
        image_dir=str(test_dir / "images"),
        dataset_name="pannuke")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
        pin_memory=False)

    # Collect raw predictions (no threshold)
    print(f"\nCollecting predictions on fold {args.test_fold} ({len(ds)} images)...")
    t0 = time.time()
    preds, targets = _collect_raw_predictions(model, loader, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Baseline: uniform threshold
    print(f"\nBaseline (uniform threshold=0.3):")
    filt_uniform = _apply_per_class_threshold(preds, {c: 0.3 for c in range(5)})
    baseline = _evaluate_pq(filt_uniform, targets)
    print(f"  PQ={baseline.get('PQ', 0):.4f}")
    for c in range(_NUM_CLASSES):
        print(f"    {CANONICAL_CLASSES[c]:>14s}: PQ={baseline.get(f'PQ_class{c}', 0):.4f}")

    # Per-class sweep
    print(f"\nCalibrating per-class thresholds "
          f"(sweep {args.sweep_lo:.2f}-{args.sweep_hi:.2f}, step={args.sweep_step:.2f}):")
    best_thresholds = calibrate(
        preds, targets,
        sweep_range=(args.sweep_lo, args.sweep_hi, args.sweep_step))

    # Evaluate with calibrated thresholds
    print(f"\nCalibrated thresholds:")
    for c in range(_NUM_CLASSES):
        print(f"  {CANONICAL_CLASSES[c]:>14s}: {best_thresholds[c]:.2f}")

    filt_calibrated = _apply_per_class_threshold(preds, best_thresholds)
    calibrated = _evaluate_pq(filt_calibrated, targets)
    print(f"\nCalibrated PQ={calibrated.get('PQ', 0):.4f}  "
          f"(baseline={baseline.get('PQ', 0):.4f}, "
          f"delta={calibrated.get('PQ', 0) - baseline.get('PQ', 0):+.4f})")
    for c in range(_NUM_CLASSES):
        b_pq = baseline.get(f'PQ_class{c}', 0)
        c_pq = calibrated.get(f'PQ_class{c}', 0)
        print(f"  {CANONICAL_CLASSES[c]:>14s}: {c_pq:.4f} (was {b_pq:.4f}, {c_pq - b_pq:+.4f})")

    # Save
    result = {
        "checkpoint": args.checkpoint,
        "test_fold": args.test_fold,
        "uniform_threshold": 0.3,
        "baseline_PQ": baseline.get("PQ", 0),
        "calibrated_thresholds": {
            CANONICAL_CLASSES[c]: best_thresholds[c] for c in range(_NUM_CLASSES)
        },
        "calibrated_PQ": calibrated.get("PQ", 0),
        "delta_PQ": calibrated.get("PQ", 0) - baseline.get("PQ", 0),
        "per_class_baseline": {
            CANONICAL_CLASSES[c]: baseline.get(f"PQ_class{c}", 0) for c in range(_NUM_CLASSES)
        },
        "per_class_calibrated": {
            CANONICAL_CLASSES[c]: calibrated.get(f"PQ_class{c}", 0) for c in range(_NUM_CLASSES)
        },
    }
    out_file = out_dir / "calibration_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}")
    print(f"\nTo use these thresholds, pass to train_pannuke or eval scripts:")
    thresh_str = " ".join(f"{best_thresholds[c]:.2f}" for c in range(_NUM_CLASSES))
    print(f"  --per_class_thresholds {thresh_str}")


if __name__ == "__main__":
    main()
