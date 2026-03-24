"""Evaluate a trained checkpoint with and without SAHI.

Runs standard evaluation and SAHI-enhanced evaluation side by side,
comparing PQ, mAP, F1d, and mIoU. Tests multiple slice sizes to find
the optimal SAHI configuration.

Usage:
    python -m experiments.eval_sahi \
        --checkpoint experiments/baseline/fold3/checkpoints/best_pq.pt \
        --data_root data --test_fold 3 \
        --slice_sizes 96 128 192

    python -m experiments.eval_sahi \
        --checkpoint experiments/phikon_v2/fold3/checkpoints/best_pq.pt \
        --data_root data --test_fold 3 \
        --backbone phikon_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from mhc_path.config.reproducibility import seed_everything
from mhc_path.data.dataset import PathologyDetectionDataset, detection_collate_fn
from mhc_path.evaluation.metrics import (
    DetectionMetrics,
    PanopticQuality,
    TissuePanopticQuality,
    SegmentationMetrics,
    _mask_iou_matrix,
)
from mhc_path.evaluation.sahi import SAHIConfig, sahi_predict_single
from mhc_path.models.box_utils import cxcywh_to_xyxy
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig
from mhc_path.training.losses import DetectionTarget

_NUM_CLASSES = 5
_IOU_THRESHOLDS = (0.3, 0.5, 0.75)


def _normalize_output(raw: dict) -> dict:
    out: dict[str, torch.Tensor] = {}
    out["pred_logits"] = raw.get("pred_logits", raw.get("class_logits"))
    out["pred_boxes"] = raw.get("pred_boxes", raw.get("box_coords"))
    if "pred_masks" in raw:
        out["pred_masks"] = raw["pred_masks"]
    elif "mask_logits" in raw:
        out["pred_masks"] = raw["mask_logits"]
    return out


def _standard_predict(
    model: torch.nn.Module, image: torch.Tensor, score_threshold: float,
) -> dict:
    """Standard single-pass prediction for one image."""
    raw = model(image.unsqueeze(0))
    outputs = _normalize_output(raw)
    logits = outputs["pred_logits"][0]
    scores, labels = logits.sigmoid().max(dim=-1)
    boxes = cxcywh_to_xyxy(outputs["pred_boxes"][0])

    masks = None
    if "pred_masks" in outputs:
        mask_logits = outputs["pred_masks"][0]
        up = torch.nn.functional.interpolate(
            mask_logits.unsqueeze(1), size=(256, 256),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        masks = (up.sigmoid() > 0.5)

    keep = scores >= score_threshold
    result: dict = {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
    if masks is not None:
        result["masks"] = masks[keep]
    return result


def _build_targets(batch: dict, device: torch.device) -> list[DetectionTarget]:
    boxes, labels = batch["boxes"], batch["labels"]
    num_obj = batch["num_objects"]
    masks = batch.get("masks")
    targets = []
    for i in range(batch["images"].shape[0]):
        n = num_obj[i].item()
        if isinstance(boxes, (list, tuple)):
            b = boxes[i][:n].to(device)
            l = labels[i][:n].to(device)
        else:
            b = boxes[i, :n].to(device)
            l = labels[i, :n].to(device)
        m = None
        if masks is not None:
            m = masks[i, :n].to(device) if not isinstance(masks, list) else masks[i][:n].to(device)
        targets.append(DetectionTarget(boxes=b, labels=l, masks=m))
    return targets


@torch.no_grad()
def evaluate_method(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    sahi_config: Optional[SAHIConfig] = None,
    tissue_ids: Optional[list[int]] = None,
) -> dict:
    """Evaluate with either standard or SAHI inference."""
    model.eval()
    det_metrics = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    pq_metric = PanopticQuality(num_classes=_NUM_CLASSES, iou_threshold=0.5)
    tissue_pq: Optional[TissuePanopticQuality] = None
    if tissue_ids is not None:
        tissue_pq = TissuePanopticQuality(num_classes=_NUM_CLASSES)
    seg_metric = SegmentationMetrics(num_classes=_NUM_CLASSES)

    img_cursor = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = _build_targets(batch, device)
        bs = images.shape[0]

        preds_list = []
        for i in range(bs):
            if sahi_config is not None:
                pred = sahi_predict_single(model, images[i], sahi_config)
            else:
                pred = _standard_predict(model, images[i], score_threshold)
            preds_list.append(pred)

        # Detection metrics (unfiltered mAP needs all preds)
        det_metrics.update(preds_list, targets)
        pq_metric.update(preds_list, targets)

        if tissue_pq is not None and tissue_ids is not None:
            batch_tissues = tissue_ids[img_cursor:img_cursor + bs]
            tissue_pq.update(preds_list, targets, batch_tissues)
            img_cursor += bs

        # Segmentation metrics
        for pred_d, tgt in zip(preds_list, targets):
            if "masks" in pred_d and tgt.masks is not None:
                pm, pl = pred_d["masks"], pred_d["labels"]
                tm, tl = tgt.masks, tgt.labels
                if len(pm) > 0 and len(tm) > 0:
                    iou_mat = _mask_iou_matrix(pm.to(tm.device), tm)
                    matched_pm, matched_tm = [], []
                    matched_pl, matched_tl = [], []
                    used_gt = set()
                    order = pred_d["scores"].argsort(descending=True)
                    for pi in order:
                        pc = pl[pi].item()
                        best_iou, best_gi = -1.0, -1
                        for gi in range(len(tl)):
                            if gi in used_gt or tl[gi].item() != pc:
                                continue
                            val = iou_mat[pi, gi].item()
                            if val > best_iou:
                                best_iou, best_gi = val, gi
                        if best_gi >= 0 and best_iou >= 0.5:
                            used_gt.add(best_gi)
                            matched_pm.append(pm[pi])
                            matched_tm.append(tm[best_gi])
                            matched_pl.append(pl[pi])
                            matched_tl.append(tl[best_gi])
                    if matched_pm:
                        seg_metric.update(
                            torch.stack(matched_pm).float(),
                            torch.stack(matched_tm).float(),
                            torch.stack(matched_pl),
                            torch.stack(matched_tl),
                        )

    results = {}
    det_results = det_metrics.compute()
    results.update(det_results)
    pq_results = pq_metric.compute()
    results.update(pq_results)
    if tissue_pq is not None:
        results.update(tissue_pq.compute())
    seg_results = seg_metric.compute()
    results.update(seg_results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate with and without SAHI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_fold", type=int, required=True)
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16")
    parser.add_argument("--num_queries", type=int, default=300)
    parser.add_argument("--group_detr", type=int, default=3)
    parser.add_argument("--mask_upsample_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--slice_sizes", type=int, nargs="+", default=[96, 128, 192],
                        help="Slice sizes to test (default: 96 128 192)")
    parser.add_argument("--overlap_ratios", type=float, nargs="+", default=[0.25],
                        help="Overlap ratios to test (default: 0.25)")
    parser.add_argument("--nms_iou", type=float, default=0.5,
                        help="IoU threshold for SAHI cross-slice NMS (default: 0.5)")
    parser.add_argument("--out_dir", type=str, default="experiments/sahi_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
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
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # --- Test data ---
    test_fold_dir = data_root / f"pannuke_fold{args.test_fold}"
    test_ds = PathologyDetectionDataset(
        annotation_file=str(test_fold_dir / "annotations.json"),
        image_dir=str(test_fold_dir / "images"),
        dataset_name="pannuke")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
        pin_memory=False)

    # Load tissue types
    tissue_ids: Optional[list[int]] = None
    tissue_file = test_fold_dir / "tissue_types.json"
    if tissue_file.exists():
        with open(tissue_file) as f:
            td = json.load(f)
        tissue_map = td["image_tissues"]
        tissue_ids = [tissue_map.get(img["file_name"], 0) for img in test_ds._images]

    # --- Standard evaluation ---
    print("\n--- Standard evaluation (no SAHI) ---")
    t0 = time.time()
    standard_results = evaluate_method(
        model, test_loader, device, args.score_threshold,
        tissue_ids=tissue_ids)
    std_time = time.time() - t0
    _print_results("Standard", standard_results, std_time)

    # --- SAHI evaluations ---
    all_results = {"standard": {
        "results": {k: v for k, v in standard_results.items() if isinstance(v, (int, float))},
        "time_seconds": std_time,
    }}

    for slice_size in args.slice_sizes:
        for overlap in args.overlap_ratios:
            sahi_cfg = SAHIConfig(
                slice_size=slice_size,
                overlap_ratio=overlap,
                include_full_image=True,
                score_threshold=args.score_threshold,
                nms_iou_threshold=args.nms_iou,
                image_size=256,
            )
            label = f"SAHI s={slice_size} o={overlap:.0%}"
            print(f"\n--- {label} ---")
            t0 = time.time()
            sahi_results = evaluate_method(
                model, test_loader, device, args.score_threshold,
                sahi_config=sahi_cfg, tissue_ids=tissue_ids)
            sahi_time = time.time() - t0
            _print_results(label, sahi_results, sahi_time)

            key = f"sahi_s{slice_size}_o{int(overlap*100)}"
            all_results[key] = {
                "config": {"slice_size": slice_size, "overlap_ratio": overlap,
                           "nms_iou": args.nms_iou},
                "results": {k: v for k, v in sahi_results.items()
                            if isinstance(v, (int, float))},
                "time_seconds": sahi_time,
            }

            # Delta from standard
            for metric in ["PQ", "mPQ", "mAP@50", "F1d@30", "mIoU"]:
                std_val = standard_results.get(metric, 0.0)
                sahi_val = sahi_results.get(metric, 0.0)
                delta = sahi_val - std_val
                sign = "+" if delta >= 0 else ""
                print(f"  {metric}: {sign}{delta:.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SAHI COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'PQ':>8} {'mPQ':>8} {'mAP@50':>8} {'F1d@30':>8} {'mIoU':>8} {'Time':>8}")
    print("-" * 70)
    for key, data in all_results.items():
        r = data["results"]
        t = data["time_seconds"]
        print(f"{key:<25} {r.get('PQ', 0):.4f}   {r.get('mPQ', 0):.4f}   "
              f"{r.get('mAP@50', 0):.4f}   {r.get('F1d@30', 0):.4f}   "
              f"{r.get('mIoU', 0):.4f}   {t:.0f}s")
    print("=" * 70)

    # Save
    out_file = out_dir / "sahi_comparison.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


def _print_results(label: str, results: dict, elapsed: float) -> None:
    print(f"  {label} ({elapsed:.1f}s):")
    print(f"    PQ={results.get('PQ', 0):.4f}  mPQ={results.get('mPQ', 0):.4f}  "
          f"mAP@50={results.get('mAP@50', 0):.4f}  F1d@30={results.get('F1d@30', 0):.4f}  "
          f"mIoU={results.get('mIoU', 0):.4f}")


if __name__ == "__main__":
    main()
