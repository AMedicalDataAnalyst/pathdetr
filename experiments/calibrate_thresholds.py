"""Per-class score threshold sweep on an existing checkpoint.

Sweeps score thresholds independently per class, optimizing per-class PQ.
The resulting thresholds can be used for evaluation via --per_class_thresholds.

Usage:
  python -m experiments.calibrate_thresholds \
      --checkpoint experiments/exp_phikon_v2/checkpoints/best_pq.pt \
      --data_root data --test_fold 3 \
      --lo 0.1 --hi 0.75 --step 0.05
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mhc_path.config.class_maps import CANONICAL_CLASSES
from mhc_path.data.dataset import PathologyDetectionDataset, detection_collate_fn
from mhc_path.evaluation.metrics import PanopticQuality, TissuePanopticQuality
from mhc_path.models.box_utils import cxcywh_to_xyxy
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig
from mhc_path.training.losses import DetectionTarget

_NUM_CLASSES = 5
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def collect_raw_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict], list[DetectionTarget]]:
    """Run model on all batches, collect unfiltered predictions + targets."""
    model.eval()
    all_preds: list[dict] = []
    all_targets: list[DetectionTarget] = []

    for batch in loader:
        images = batch["images"].to(device)
        raw = model(images)

        # Normalize output keys
        outputs: dict[str, torch.Tensor] = {}
        key_map = {
            "class_logits": "pred_logits",
            "box_coords": "pred_boxes",
            "mask_logits": "pred_masks",
        }
        for k, v in raw.items():
            outputs[key_map.get(k, k)] = v

        B = outputs["pred_logits"].shape[0]
        has_masks = "pred_masks" in outputs

        for b in range(B):
            sc, lb = outputs["pred_logits"][b].sigmoid().max(dim=-1)
            bx = cxcywh_to_xyxy(outputs["pred_boxes"][b])

            pred = {"boxes": bx.cpu(), "scores": sc.cpu(), "labels": lb.cpu()}
            if has_masks:
                # Store logits at native res (e.g. 64×64) as float16 to save memory
                # (~0.78 MB/image vs ~6.25 MB for 256×256 binary)
                pred["mask_logits"] = outputs["pred_masks"][b].cpu().half()
            all_preds.append(pred)

            all_targets.append(DetectionTarget(
                boxes=batch["boxes"][b],
                labels=batch["labels"][b],
                masks=batch["masks"][b] if "masks" in batch else None,
            ))

    return all_preds, all_targets


def evaluate_with_thresholds(
    all_preds: list[dict],
    all_targets: list[DetectionTarget],
    thresholds: list[float],
    tissue_ids: Optional[list[int]] = None,
) -> dict[str, float]:
    """Evaluate PQ (and mPQ if tissue_ids provided) with per-class thresholds."""
    pq = PanopticQuality(num_classes=_NUM_CLASSES, iou_threshold=0.5)
    tissue_pq: Optional[TissuePanopticQuality] = None
    if tissue_ids is not None:
        tissue_pq = TissuePanopticQuality(num_classes=_NUM_CLASSES)
    thresh_t = torch.tensor(thresholds)

    for i, (pred, tgt) in enumerate(zip(all_preds, all_targets)):
        per_class_thresh = thresh_t[pred["labels"]]
        keep = pred["scores"] >= per_class_thresh
        filt = {
            "boxes": pred["boxes"][keep],
            "scores": pred["scores"][keep],
            "labels": pred["labels"][keep],
        }
        if "mask_logits" in pred:
            ml = pred["mask_logits"][keep].float()
            if len(ml) > 0:
                ml_up = F.interpolate(
                    ml.unsqueeze(1), size=(256, 256),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                filt["masks"] = (ml_up.sigmoid() > 0.5)
        pq.update([filt], [tgt])
        if tissue_pq is not None:
            tissue_pq.update([filt], [tgt], [tissue_ids[i]])

    results = pq.compute()
    if tissue_pq is not None:
        results.update(tissue_pq.compute())
    return results


def sweep_thresholds(
    all_preds: list[dict],
    all_targets: list[DetectionTarget],
    tissue_ids: Optional[list[int]] = None,
    lo: float = 0.1,
    hi: float = 0.75,
    step: float = 0.05,
    checkpoint_path: Optional[Path] = None,
) -> tuple[list[float], dict[str, float]]:
    """Find optimal per-class thresholds by greedy coordinate descent on mPQ.

    Saves intermediate results after each class to checkpoint_path (if given)
    so progress is not lost on crash.
    """
    candidates = np.arange(lo, hi + step / 2, step).tolist()
    best_thresholds = [0.3] * _NUM_CLASSES
    use_mpq = tissue_ids is not None
    metric_name = "mPQ" if use_mpq else "PQ"

    for cls_id in range(_NUM_CLASSES):
        best_score = -1.0
        best_t = 0.3

        for t in candidates:
            trial = list(best_thresholds)
            trial[cls_id] = t
            results = evaluate_with_thresholds(
                all_preds, all_targets, trial, tissue_ids=tissue_ids)
            score = results.get(metric_name, 0.0)

            if score > best_score:
                best_score = score
                best_t = t

        best_thresholds[cls_id] = best_t
        print(f"  {CANONICAL_CLASSES[cls_id]:15s}: best threshold={best_t:.2f}, "
              f"{metric_name}={best_score:.4f}")

        if checkpoint_path is not None:
            _save_intermediate(checkpoint_path, best_thresholds, cls_id, metric_name)

    final = evaluate_with_thresholds(
        all_preds, all_targets, best_thresholds, tissue_ids=tissue_ids)
    return best_thresholds, final


def _save_intermediate(
    path: Path, thresholds: list[float], last_cls: int, metric: str,
) -> None:
    """Save partial sweep progress so it survives crashes."""
    intermediate = {
        "status": "in_progress",
        "completed_classes": last_cls + 1,
        "metric": metric,
        "thresholds_so_far": dict(zip(CANONICAL_CLASSES[:last_cls + 1],
                                       thresholds[:last_cls + 1])),
    }
    with open(path, "w") as f:
        json.dump(intermediate, f, indent=2)
    print(f"    (saved intermediate to {path})")


def main():
    parser = argparse.ArgumentParser(description="Per-class score threshold sweep")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--test_fold", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lo", type=float, default=0.1, help="Lower bound of sweep")
    parser.add_argument("--hi", type=float, default=0.75, help="Upper bound of sweep")
    parser.add_argument("--step", type=float, default=0.05, help="Sweep step size")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON (default: <checkpoint_dir>/../threshold_sweep.json)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)

    # Reconstruct model from checkpoint config
    raw_cfg = ckpt["config"]
    cfg_dict = raw_cfg.get("model", raw_cfg)  # handle nested or flat config
    model_cfg = MHCPathConfig(
        backbone=cfg_dict.get("backbone", "dinov3_vitl16"),
        backbone_frozen=True,
        lora_rank=None,
        fpn_dim=cfg_dict.get("fpn_dim", 256),
        num_queries=cfg_dict.get("num_queries", 100),
        num_classes=cfg_dict.get("num_classes", 5),
        num_decoder_layers=cfg_dict.get("num_decoder_layers", 6),
        with_segmentation=cfg_dict.get("with_segmentation", True),
        use_mhc=False,
        output_layers=tuple(cfg_dict.get("output_layers", (6, 12, 18, 24))),
        fpn_levels=cfg_dict.get("fpn_levels", 4),
        mask_upsample_factor=cfg_dict.get("mask_upsample_factor", 1),
        with_pixel_decoder=cfg_dict.get("with_pixel_decoder", False),
        large_kernel=cfg_dict.get("large_kernel", False),
        large_kernel_size=cfg_dict.get("large_kernel_size", 13),
    )
    model = MHCPath(model_cfg).to(_DEVICE)
    # strict=False: frozen backbone weights are not saved in checkpoint
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load test data
    data_root = Path(args.data_root)
    test_fold_dir = data_root / f"pannuke_fold{args.test_fold}"
    test_ds = PathologyDetectionDataset(
        annotation_file=str(test_fold_dir / "annotations.json"),
        image_dir=str(test_fold_dir / "images"),
        dataset_name="pannuke",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
        pin_memory=False,
    )

    # Load tissue type mapping for mPQ
    tissue_ids: Optional[list[int]] = None
    tissue_file = test_fold_dir / "tissue_types.json"
    if tissue_file.exists():
        with open(tissue_file) as f:
            tissue_data = json.load(f)
        tissue_map = tissue_data["image_tissues"]
        tissue_ids = []
        for img_meta in test_ds._images:
            fname = img_meta["file_name"]
            tissue_ids.append(tissue_map.get(fname, 0))
        print(f"Loaded tissue types for {len(tissue_ids)} images")
    else:
        print(f"WARNING: {tissue_file} not found — optimizing on PQ instead of mPQ")

    metric_name = "mPQ" if tissue_ids is not None else "PQ"
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Config: {cfg_dict.get('backbone', 'dinov3_vitl16')}, "
          f"queries={cfg_dict.get('num_queries', 100)}")
    print(f"Test fold {args.test_fold}: {len(test_ds)} images")
    print(f"Optimizing: {metric_name}")
    print(f"Sweep: {args.lo:.2f} to {args.hi:.2f}, step {args.step:.2f}")
    print()

    # Collect predictions
    t0 = time.time()
    print("Collecting predictions...")
    all_preds, all_targets = collect_raw_predictions(
        model, test_loader, torch.device(_DEVICE))
    print(f"  {len(all_preds)} images, {time.time() - t0:.1f}s")

    # Baseline
    print(f"\nBaseline (uniform threshold=0.3):")
    baseline = evaluate_with_thresholds(
        all_preds, all_targets, [0.3] * _NUM_CLASSES, tissue_ids=tissue_ids)
    print(f"  {metric_name}={baseline.get(metric_name, baseline['PQ']):.4f}, "
          f"PQ={baseline['PQ']:.4f}, DQ={baseline['DQ']:.4f}, SQ={baseline['SQ']:.4f}")
    for c in range(_NUM_CLASSES):
        print(f"    {CANONICAL_CLASSES[c]:15s}: PQ={baseline[f'PQ_class{c}']:.4f} "
              f"DQ={baseline[f'DQ_class{c}']:.4f} SQ={baseline[f'SQ_class{c}']:.4f}")

    # Sweep
    out_path = Path(args.out) if args.out else ckpt_path.parent.parent / "threshold_sweep.json"
    print(f"\nPer-class threshold sweep (optimizing {metric_name}):")
    best_thresholds, final = sweep_thresholds(
        all_preds, all_targets, tissue_ids=tissue_ids,
        lo=args.lo, hi=args.hi, step=args.step,
        checkpoint_path=out_path)

    print(f"\nOptimal: {dict(zip(CANONICAL_CLASSES, best_thresholds))}")
    print(f"\nResults with optimal thresholds:")
    if metric_name == "mPQ":
        print(f"  mPQ={final['mPQ']:.4f} (was {baseline['mPQ']:.4f}, "
              f"delta={final['mPQ'] - baseline['mPQ']:+.4f})")
    print(f"  PQ={final['PQ']:.4f} (was {baseline['PQ']:.4f}, "
          f"delta={final['PQ'] - baseline['PQ']:+.4f})")
    print(f"  DQ={final['DQ']:.4f} (was {baseline['DQ']:.4f}, "
          f"delta={final['DQ'] - baseline['DQ']:+.4f})")
    for c in range(_NUM_CLASSES):
        print(f"    {CANONICAL_CLASSES[c]:15s}: PQ={final[f'PQ_class{c}']:.4f} "
              f"(was {baseline[f'PQ_class{c}']:.4f})")

    # Save final results (overwrites intermediate checkpoint)
    result = {
        "checkpoint": str(ckpt_path),
        "test_fold": args.test_fold,
        "optimized_metric": metric_name,
        "sweep_range": {"lo": args.lo, "hi": args.hi, "step": args.step},
        "baseline_uniform_0.3": {
            k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        "optimal_thresholds": dict(zip(CANONICAL_CLASSES, best_thresholds)),
        "optimal_thresholds_list": best_thresholds,
        "results_with_optimal": {
            k: v for k, v in final.items() if isinstance(v, (int, float))},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
