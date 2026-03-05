"""Full PanNuke training run with comprehensive metrics and diagnostics.

Supports the official PanNuke 3-fold cross-validation protocol: train on two
folds, validate on a held-out portion, and test on the third fold.  Repeat for
all three fold rotations and aggregate.

Config derived from 5-factor factorial results:
  Frozen DINOv3-L, AdamW for FPN+decoder, StainAug+geometric
  CIoU box loss, num_queries=100
  200 epochs, batch_size=32, validate every 10 epochs + at {1, 5}
  lr_fpn=5e-4, lr_decoder=1e-3, cosine LR with 10-epoch warmup
  clip_grad_norm=1.0, EMA decay=0.998

Usage (3-fold CV — recommended):
  python -m experiments.train_pannuke --train_folds 1 2 --test_fold 3
  python -m experiments.train_pannuke --train_folds 1 3 --test_fold 2
  python -m experiments.train_pannuke --train_folds 2 3 --test_fold 1

Legacy (single fold, internal train/val split, no test set):
  python -m experiments.train_pannuke --data_dir data/pannuke_fold3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from mhc_path.config.class_maps import CANONICAL_CLASSES
from mhc_path.config.reproducibility import seed_everything
from mhc_path.data.dataset import PathologyDetectionDataset, detection_collate_fn
from mhc_path.data.gpu_augmentation import GPUPathologyAugPipeline
from mhc_path.evaluation.detr_diagnostics import (
    extract_attention_stats,
    measure_inference_time,
    compute_backbone_stats,
)
from mhc_path.evaluation.metrics import (
    DetectionMetrics,
    PanopticQuality,
    TissuePanopticQuality,
    SegmentationMetrics,
    _mask_iou_matrix,
    expected_calibration_error,
)
from mhc_path.models.box_utils import cxcywh_to_xyxy
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig
from mhc_path.training.detection_engine import (
    DetectionConfig,
    DetectionEngine,
    _EMAModel,
)
from mhc_path.training.losses import DetectionLoss, DetectionTarget, HungarianMatcher


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "experiments" / "pannuke_split1"

_VAL_FRACTION = 0.1
_N_EPOCHS = 200
_BATCH_SIZE = 32
_VAL_EPOCHS = {1, 5} | set(range(9, _N_EPOCHS, 10))  # 1, 5, then every 10
_WARMUP_EPOCHS = 10
_NUM_CLASSES = 5
_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BASE_LR_FPN = 5e-4
_BASE_LR_DECODER = 1e-3
_EMA_DECAY = 0.998
_NUM_QUERIES = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_fold_annotations(
    data_root: Path, fold_indices: list[int],
) -> dict:
    """Merge COCO annotations from multiple PanNuke folds.

    ``file_name`` is updated to include the relative path from *data_root*
    (e.g. ``pannuke_fold1/images/fold1_00000.png``) so that passing
    ``image_dir=data_root`` resolves all images.
    """
    merged: dict = {"images": [], "annotations": [], "categories": None}
    img_id_offset = 0
    ann_id_offset = 0

    for fold_idx in fold_indices:
        fold_dir = data_root / f"pannuke_fold{fold_idx}"
        ann_file = fold_dir / "annotations.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Fold {fold_idx} annotations not found at {ann_file}. "
                f"Run convert_pannuke() first."
            )
        with open(ann_file) as f:
            coco = json.load(f)

        if merged["categories"] is None:
            merged["categories"] = coco["categories"]

        old_to_new_img: dict[int, int] = {}
        for img in coco["images"]:
            new_id = img["id"] + img_id_offset
            old_to_new_img[img["id"]] = new_id
            merged["images"].append({
                **img,
                "id": new_id,
                "file_name": f"pannuke_fold{fold_idx}/images/{img['file_name']}",
            })

        for ann in coco["annotations"]:
            merged["annotations"].append({
                **ann,
                "id": ann["id"] + ann_id_offset,
                "image_id": old_to_new_img[ann["image_id"]],
            })

        img_id_offset += len(coco["images"])
        ann_id_offset += len(coco["annotations"])

    return merged


def _split_train_val(
    coco: dict, val_fraction: float, seed: int, out_dir: Path,
) -> tuple[str, str]:
    """Split a merged COCO dict into train and val JSON files."""
    rng = np.random.RandomState(seed)
    ids = np.array([img["id"] for img in coco["images"]])
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val].tolist())
    train_ids = set(ids[n_val:].tolist())

    id2img = {img["id"]: img for img in coco["images"]}

    def _save(subset_ids: set[int], suffix: str) -> str:
        imgs = [id2img[i] for i in sorted(subset_ids)]
        anns = [a for a in coco["annotations"] if a["image_id"] in subset_ids]
        p = out_dir / f"annotations_{suffix}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"images": imgs, "annotations": anns,
                        "categories": coco["categories"]}, f)
        return str(p)

    return _save(train_ids, "train"), _save(val_ids, "val")


def _split_coco_json(
    ann_file: Path, out_dir: Path, n_train: int, n_val: int, seed: int = 42,
) -> tuple[str, str]:
    """Legacy: split a single-fold COCO JSON into train/val by count."""
    with open(ann_file) as f:
        coco = json.load(f)
    rng = np.random.RandomState(seed)
    ids = [img["id"] for img in coco["images"]]
    chosen = rng.choice(ids, size=min(n_train + n_val, len(ids)), replace=False)
    rng.shuffle(chosen)
    train_ids = set(chosen[:n_train].tolist())
    val_ids = set(chosen[n_train:n_train + n_val].tolist())
    id2img = {img["id"]: img for img in coco["images"]}

    def _save(subset_ids: set[int], suffix: str) -> str:
        imgs = [id2img[i] for i in sorted(subset_ids)]
        anns = [a for a in coco["annotations"] if a["image_id"] in subset_ids]
        p = out_dir / f"annotations_{suffix}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"images": imgs, "annotations": anns,
                        "categories": coco["categories"]}, f)
        return str(p)

    return _save(train_ids, "train"), _save(val_ids, "val")


def _cosine_lr_with_warmup(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _grad_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {"fpn": 0.0, "decoder": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.data.norm(2).item()
        lo = name.lower()
        if "fpn" in lo:
            norms["fpn"] = max(norms["fpn"], n)
        else:
            norms["decoder"] = max(norms["decoder"], n)
    return norms


def _prediction_stats(outputs: dict) -> dict[str, float]:
    scores = outputs["pred_logits"].sigmoid().max(dim=-1).values
    boxes = outputs["pred_boxes"]
    return {
        "conf_mean": scores.mean().item(),
        "conf_std": scores.std().item(),
        "n_above_0.1": (scores > 0.1).sum().item(),
        "n_above_0.3": (scores > 0.3).sum().item(),
        "n_above_0.5": (scores > 0.5).sum().item(),
        "box_cx_mean": boxes[:, :, 0].mean().item(),
        "box_cy_mean": boxes[:, :, 1].mean().item(),
        "box_w_mean": boxes[:, :, 2].mean().item(),
        "box_h_mean": boxes[:, :, 3].mean().item(),
        "box_cx_std": boxes[:, :, 0].std().item(),
        "box_cy_std": boxes[:, :, 1].std().item(),
        "box_w_std": boxes[:, :, 2].std().item(),
        "box_h_std": boxes[:, :, 3].std().item(),
    }


def _outputs_to_predictions(
    outputs: dict, score_threshold: float = 0.0,
    include_masks: bool = False,
) -> list[dict]:
    has_masks = include_masks and "pred_masks" in outputs
    preds = []
    for b in range(outputs["pred_logits"].shape[0]):
        sc, lb = outputs["pred_logits"][b].sigmoid().max(dim=-1)
        boxes = cxcywh_to_xyxy(outputs["pred_boxes"][b])

        if has_masks:
            mask_logits = outputs["pred_masks"][b]  # (Q, h, w)
            upsampled = torch.nn.functional.interpolate(
                mask_logits.unsqueeze(1), size=(256, 256),
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (Q, 256, 256)
            binary_masks = (upsampled.sigmoid() > 0.5)
        else:
            binary_masks = None

        if score_threshold > 0.0:
            keep = sc >= score_threshold
            sc, lb, boxes = sc[keep], lb[keep], boxes[keep]
            if binary_masks is not None:
                binary_masks = binary_masks[keep]

        entry: dict = {"boxes": boxes, "scores": sc, "labels": lb}
        if binary_masks is not None:
            entry["masks"] = binary_masks
        preds.append(entry)
    return preds


def _normalize_output(raw: dict) -> dict:
    out: dict[str, torch.Tensor] = {}
    out["pred_logits"] = raw.get("pred_logits", raw.get("class_logits"))
    out["pred_boxes"] = raw.get("pred_boxes", raw.get("box_coords"))
    if "pred_masks" in raw:
        out["pred_masks"] = raw["pred_masks"]
    elif "mask_logits" in raw:
        out["pred_masks"] = raw["mask_logits"]
    if "aux_outputs" in raw:
        out["aux_outputs"] = raw["aux_outputs"]
    return out


def _batch_to_targets(batch: dict, device: torch.device) -> list[DetectionTarget]:
    boxes, labels = batch["boxes"], batch["labels"]
    num_obj = batch["num_objects"]
    masks = batch.get("masks")
    targets: list[DetectionTarget] = []
    if isinstance(boxes, (list, tuple)):
        for i, box_t in enumerate(boxes):
            n = num_obj[i].item()
            lbl = labels[i][:n] if isinstance(labels, list) else labels[i, :n]
            m = None
            if masks is not None:
                m = masks[i][:n] if isinstance(masks, list) else masks[i, :n]
                m = m.to(device)
            targets.append(DetectionTarget(
                boxes=box_t[:n].to(device), labels=lbl.to(device), masks=m))
    else:
        for i in range(boxes.shape[0]):
            n = num_obj[i].item()
            m = masks[i, :n].to(device) if masks is not None else None
            targets.append(DetectionTarget(
                boxes=boxes[i, :n].to(device), labels=labels[i, :n].to(device),
                masks=m))
    return targets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    score_threshold: float = 0.0,
    tissue_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Comprehensive evaluation with all metrics.

    ``score_threshold`` filters low-confidence predictions for PQ and F1d
    (which are hurt by false-positive padding queries).  mAP, ECE, and
    prediction stats always use the *unfiltered* predictions so that the
    full precision-recall curve is preserved.

    ``tissue_ids``: optional list of tissue type ints (one per dataset image,
    in dataset order). When provided, mPQ and bPQ are computed per the
    official PanNuke protocol (per-tissue stratification).
    """
    model.eval()
    det_metrics = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    det_metrics_filt = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    pq_metric = PanopticQuality(num_classes=_NUM_CLASSES, iou_threshold=0.5)
    tissue_pq: Optional[TissuePanopticQuality] = None
    if tissue_ids is not None:
        tissue_pq = TissuePanopticQuality(num_classes=_NUM_CLASSES)
    seg_metric = SegmentationMetrics(num_classes=_NUM_CLASSES)

    running_loss: dict[str, float] = {}
    count = 0
    all_pred_stats: dict[str, float] = {}
    n_stats = 0
    img_cursor = 0

    for batch in val_loader:
        images = batch["images"].to(device)
        raw = model(images)
        outputs = _normalize_output(raw)
        targets = _batch_to_targets(batch, device)

        losses = criterion(outputs, targets)
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            running_loss[k] = running_loss.get(k, 0.0) + val
        count += 1

        # Unfiltered preds for mAP (needs full P-R curve)
        preds_all = _outputs_to_predictions(outputs, score_threshold=0.0)
        det_metrics.update(preds_all, targets)

        # Filtered preds with masks for PQ, F1d, segmentation metrics
        has_masks = "pred_masks" in outputs
        preds_filt = _outputs_to_predictions(
            outputs, score_threshold=score_threshold, include_masks=has_masks)
        det_metrics_filt.update(preds_filt, targets)
        pq_metric.update(preds_filt, targets)

        # Tissue-stratified PQ
        if tissue_pq is not None and tissue_ids is not None:
            bs = outputs["pred_logits"].shape[0]
            batch_tissues = tissue_ids[img_cursor:img_cursor + bs]
            tissue_pq.update(preds_filt, targets, batch_tissues)
            img_cursor += bs

        # Segmentation metrics
        if has_masks:
            for pred_d, tgt in zip(preds_filt, targets):
                if "masks" in pred_d and tgt.masks is not None:
                    pm, pl = pred_d["masks"], pred_d["labels"]
                    tm, tl = tgt.masks, tgt.labels
                    if len(pm) > 0 and len(tm) > 0:
                        iou_mat = _mask_iou_matrix(pm.to(tm.device), tm)
                        matched_p_masks, matched_t_masks = [], []
                        matched_p_labels, matched_t_labels = [], []
                        used_gt = set()
                        order = pred_d["scores"].argsort(descending=True)
                        for pi in order:
                            pc = pl[pi].item()
                            best_iou, best_gi = -1.0, -1
                            for gi in range(len(tl)):
                                if gi in used_gt:
                                    continue
                                if tl[gi].item() != pc:
                                    continue
                                val = iou_mat[pi, gi].item()
                                if val > best_iou:
                                    best_iou = val
                                    best_gi = gi
                            if best_gi >= 0 and best_iou >= 0.5:
                                used_gt.add(best_gi)
                                matched_p_masks.append(pm[pi])
                                matched_t_masks.append(tm[best_gi])
                                matched_p_labels.append(pl[pi])
                                matched_t_labels.append(tl[best_gi])
                        if matched_p_masks:
                            seg_metric.update(
                                torch.stack(matched_p_masks).float(),
                                torch.stack(matched_t_masks).float(),
                                torch.stack(matched_p_labels),
                                torch.stack(matched_t_labels),
                            )

        # Prediction stats (unfiltered — diagnostic)
        ps = _prediction_stats(outputs)
        for k2, v2 in ps.items():
            all_pred_stats[k2] = all_pred_stats.get(k2, 0.0) + v2
        n_stats += 1

    # Aggregate
    results: dict[str, Any] = {}

    for k, v in running_loss.items():
        results[f"val_{k}"] = v / max(count, 1)

    # mAP from unfiltered preds
    det_results = det_metrics.compute()
    for k, v in det_results.items():
        if k.startswith("mAP") or k.startswith("AP_class"):
            results[k] = v

    # F1d, P, R from filtered preds
    det_results_filt = det_metrics_filt.compute()
    for k, v in det_results_filt.items():
        if not (k.startswith("mAP") or k.startswith("AP_class")):
            results[k] = v

    # PQ
    pq_results = pq_metric.compute()
    results.update(pq_results)

    # Official mPQ / bPQ (tissue-stratified)
    if tissue_pq is not None:
        tpq_results = tissue_pq.compute()
        results.update(tpq_results)

    # Segmentation IoU
    seg_results = seg_metric.compute()
    results.update(seg_results)

    # Confusion matrix at IoU=0.3
    cm = det_metrics.confusion_matrix(iou_threshold=0.3)
    results["confusion_matrix_30"] = cm.tolist()

    # ECE
    t30_idx = det_metrics._find_threshold_index(0.3)
    if t30_idx is not None:
        all_tp_list = []
        all_sc_list = []
        for cls_id in range(_NUM_CLASSES):
            if det_metrics._scores[cls_id]:
                all_sc_list.extend(det_metrics._scores[cls_id])
                all_tp_list.extend(det_metrics._tp[t30_idx][cls_id])
        if all_sc_list:
            ece_scores = torch.cat(all_sc_list)
            ece_tp = torch.cat(all_tp_list)
            results["ECE"] = expected_calibration_error(ece_scores, ece_tp)
        else:
            results["ECE"] = 0.0
    else:
        results["ECE"] = 0.0

    for k, v in all_pred_stats.items():
        results[k] = v / max(n_stats, 1)

    results["score_threshold"] = score_threshold
    return results


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: _EMAModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": {k: v for k, v in metrics.items()
                    if not isinstance(v, (list, dict))},
    }, str(path))


# ---------------------------------------------------------------------------
# Post-training diagnostics
# ---------------------------------------------------------------------------

def _run_post_training(
    model_cfg: MHCPathConfig,
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Run post-training diagnostics: queries sweep, timing, attention stats."""
    results: dict[str, Any] = {}

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # --- num_queries sweep ---
    for nq in [50, 100]:
        cfg = MHCPathConfig(
            backbone=model_cfg.backbone,
            backbone_frozen=True,
            fpn_dim=model_cfg.fpn_dim,
            num_queries=nq,
            num_classes=_NUM_CLASSES,
            num_decoder_layers=model_cfg.num_decoder_layers,
            with_segmentation=model_cfg.with_segmentation,
            output_layers=model_cfg.output_layers,
            fpn_levels=model_cfg.fpn_levels,
        )
        sweep_model = MHCPath(cfg).to(device)
        state = ckpt["model_state_dict"]
        filtered = {k: v for k, v in state.items()
                    if k in sweep_model.state_dict()
                    and v.shape == sweep_model.state_dict()[k].shape}
        sweep_model.load_state_dict(filtered, strict=False)
        sweep_model.eval()

        criterion = DetectionLoss(
            num_classes=_NUM_CLASSES,
            matcher=HungarianMatcher(),
        )
        sweep_results = full_evaluate(sweep_model, val_loader, criterion, device,
                                      score_threshold=0.3)
        results[f"queries_{nq}"] = {
            "mAP@30": sweep_results.get("mAP@30", 0.0),
            "mAP@50": sweep_results.get("mAP@50", 0.0),
            "PQ": sweep_results.get("PQ", 0.0),
            "F1d@30": sweep_results.get("F1d@30", 0.0),
        }
        del sweep_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Inference timing ---
    model = MHCPath(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_batch = next(iter(val_loader))
    sample_images = sample_batch["images"][:4].to(device)
    timing = measure_inference_time(model, sample_images, n_warmup=10, n_measure=100)
    results["inference_timing"] = timing

    # --- Attention statistics ---
    attn_stats = extract_attention_stats(model, sample_images)
    results["attention_stats"] = attn_stats

    # --- Backbone feature statistics ---
    bb_stats = compute_backbone_stats(model, sample_images)
    results["backbone_stats"] = {k: v for k, v in bb_stats.items()}

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full PanNuke training run")

    # Multi-fold interface (primary)
    parser.add_argument("--data_root", type=str, default=str(_DEFAULT_DATA_ROOT),
                        help="Root dir containing pannuke_fold{1,2,3}/")
    parser.add_argument("--train_folds", type=int, nargs="+", default=[1, 2],
                        help="Fold indices for training (default: 1 2)")
    parser.add_argument("--test_fold", type=int, default=3,
                        help="Fold index held out for testing (default: 3)")
    parser.add_argument("--val_fraction", type=float, default=_VAL_FRACTION,
                        help="Fraction of training data for validation (default: 0.1)")

    # Legacy single-fold interface
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Legacy: single fold dir (overrides multi-fold args)")

    # Common
    parser.add_argument("--out_dir", type=str, default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=_N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--num_queries", type=int, default=_NUM_QUERIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Min score for PQ/F1d predictions (default: 0.3)")
    parser.add_argument("--skip_post_training", action="store_true")
    parser.add_argument("--mask_loss_resolution", type=int, default=128,
                        help="Resolution for mask loss computation (default: 128)")
    parser.add_argument("--mask_upsample_factor", type=int, default=4,
                        help="Upsample pixel features before dot-product mask head (default: 4)")
    parser.add_argument("--lr_fpn", type=float, default=_BASE_LR_FPN,
                        help=f"Base LR for FPN params (default: {_BASE_LR_FPN})")
    parser.add_argument("--lr_decoder", type=float, default=_BASE_LR_DECODER,
                        help=f"Base LR for decoder params (default: {_BASE_LR_DECODER})")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    t0 = time.time()

    out_dir = Path(args.out_dir)
    n_epochs = args.epochs
    legacy_mode = args.data_dir is not None
    test_loader: DataLoader | None = None

    # ---------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------
    if legacy_mode:
        data_dir = Path(args.data_dir)
        ann_file = data_dir / "annotations.json"
        image_dir = str(data_dir / "images")
        if not ann_file.exists():
            raise FileNotFoundError(
                f"PanNuke annotations not found at {ann_file}")

        with open(ann_file) as f:
            coco = json.load(f)
        n_total = len(coco["images"])
        n_val = max(1, int(n_total * args.val_fraction))
        n_train = n_total - n_val

        split_dir = data_dir / "splits"
        train_json, val_json = _split_coco_json(
            ann_file, split_dir, n_train, n_val, args.seed)

        train_ds = PathologyDetectionDataset(
            annotation_file=train_json, image_dir=image_dir,
            dataset_name="pannuke")
        val_ds = PathologyDetectionDataset(
            annotation_file=val_json, image_dir=image_dir,
            dataset_name="pannuke")

        fold_desc = f"legacy single-fold: {data_dir.name}"

    else:
        data_root = Path(args.data_root)

        merged_coco = _merge_fold_annotations(data_root, args.train_folds)

        split_dir = out_dir / "splits"
        train_json, val_json = _split_train_val(
            merged_coco, args.val_fraction, args.seed, split_dir)

        image_dir = str(data_root)

        train_ds = PathologyDetectionDataset(
            annotation_file=train_json, image_dir=image_dir,
            dataset_name="pannuke")
        val_ds = PathologyDetectionDataset(
            annotation_file=val_json, image_dir=image_dir,
            dataset_name="pannuke")

        # Test fold
        test_fold_dir = data_root / f"pannuke_fold{args.test_fold}"
        test_ann = test_fold_dir / "annotations.json"
        if not test_ann.exists():
            raise FileNotFoundError(
                f"Test fold annotations not found at {test_ann}. "
                f"Run convert_pannuke() first.")
        test_ds = PathologyDetectionDataset(
            annotation_file=str(test_ann),
            image_dir=str(test_fold_dir / "images"),
            dataset_name="pannuke")
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=detection_collate_fn,
            pin_memory=False)

        fold_desc = (
            f"train folds {args.train_folds}, test fold {args.test_fold}"
        )

    n_train_images = len(train_ds)

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    print("=" * 70)
    print("mHC-Path Full PanNuke Training Run")
    print(f"Folds: {fold_desc}")
    print(f"  Train: {len(train_ds)} images, Val: {len(val_ds)} images"
          + (f", Test: {len(test_ds)} images" if test_loader else ""))
    print(f"Config: Frozen DINOv3-L + FPN + RF-DETR, AdamW, StainAug+geometric")
    print(f"  CIoU box loss, num_queries={args.num_queries}")
    print(f"  {n_epochs} epochs, batch_size={args.batch_size}")
    print(f"  lr_fpn={args.lr_fpn}, lr_decoder={args.lr_decoder}")
    print(f"  score_threshold={args.score_threshold} (for PQ/F1d)")
    print(f"  mask_loss_resolution={args.mask_loss_resolution}, mask_upsample_factor={args.mask_upsample_factor}")
    print(f"  warmup={_WARMUP_EPOCHS} epochs, EMA decay={_EMA_DECAY}")
    print(f"Device: {_DEVICE}")
    print("=" * 70)

    # --- Model ---
    model_cfg = MHCPathConfig(
        backbone="dinov3_vitl16", backbone_frozen=True,
        fpn_dim=256, num_queries=args.num_queries, num_classes=_NUM_CLASSES,
        num_decoder_layers=6, with_segmentation=True,
        output_layers=(6, 12, 18, 24), fpn_levels=4,
        mask_upsample_factor=args.mask_upsample_factor,
    )
    model = MHCPath(model_cfg).to(_DEVICE)

    # --- Augmentation: StainAug ON, geometric ON ---
    gpu_aug = GPUPathologyAugPipeline(
        target_size=256, stain_aug=True, geometric=True,
    ).to(_DEVICE)

    # --- Loss ---
    matcher = HungarianMatcher()
    criterion = DetectionLoss(
        num_classes=_NUM_CLASSES, matcher=matcher,
        mask_loss_resolution=args.mask_loss_resolution,
    )

    # --- Engine ---
    det_cfg = DetectionConfig(
        epochs=n_epochs, batch_size=args.batch_size,
        lr_fpn=args.lr_fpn, lr_decoder=args.lr_decoder,
        weight_decay=0.01, clip_grad_norm=1.0, ema_decay=_EMA_DECAY,
        use_amp=True,
        output_dir=str(out_dir / "checkpoints"),
        num_workers=args.num_workers, log_interval=1000,
    )
    det_metrics = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    engine = DetectionEngine(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        gpu_aug=gpu_aug, config=det_cfg, criterion=criterion, metrics=det_metrics,
    )

    # Stash initial LR on each param group for per-group cosine scheduling
    for pg in engine.optimizer.param_groups:
        pg["_base_lr"] = pg["lr"]

    # --- Logging setup ---
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log = open(out_dir / "train_log.jsonl", "w")
    val_log = open(out_dir / "val_log.jsonl", "w")

    best_map30 = -1.0
    best_pq = -1.0
    best_map30_epoch = 0
    best_pq_epoch = 0

    full_config: dict[str, Any] = {
        "model": {
            "backbone": model_cfg.backbone,
            "backbone_frozen": model_cfg.backbone_frozen,
            "fpn_dim": model_cfg.fpn_dim,
            "num_queries": model_cfg.num_queries,
            "num_classes": model_cfg.num_classes,
            "num_decoder_layers": model_cfg.num_decoder_layers,
            "mask_upsample_factor": model_cfg.mask_upsample_factor,
        },
        "training": {
            "epochs": n_epochs,
            "batch_size": args.batch_size,
            "lr_fpn": args.lr_fpn,
            "lr_decoder": args.lr_decoder,
            "warmup_epochs": _WARMUP_EPOCHS,
            "ema_decay": _EMA_DECAY,
            "clip_grad_norm": 1.0,
            "box_loss": "ciou",
            "mask_loss_resolution": args.mask_loss_resolution,
        },
        "augmentation": {
            "stain_aug": True,
            "geometric": True,
        },
        "seed": args.seed,
        "folds": {
            "train_folds": args.train_folds if not legacy_mode else None,
            "test_fold": args.test_fold if not legacy_mode else None,
            "data_dir": args.data_dir if legacy_mode else None,
            "val_fraction": args.val_fraction,
        },
    }

    if _DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ===================================================================
    # Training loop
    # ===================================================================
    for epoch in range(n_epochs):
        # --- LR schedule (per-group) ---
        lr_mult = _cosine_lr_with_warmup(epoch, _WARMUP_EPOCHS, n_epochs)
        cur_lr_fpn = args.lr_fpn * lr_mult
        cur_lr_decoder = args.lr_decoder * lr_mult
        for pg in engine.optimizer.param_groups:
            pg["lr"] = pg["_base_lr"] * lr_mult

        # --- Train ---
        ep_t0 = time.time()
        tm = engine.train_epoch(epoch)
        ep_time = time.time() - ep_t0

        gnorms = _grad_norms_by_group(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if _DEVICE == "cuda" else 0.0

        train_rec: dict[str, Any] = {
            "epoch": epoch,
            "train_loss_total": tm.get("total", 0.0),
            "train_loss_cls": tm.get("cls", 0.0),
            "train_loss_bbox": tm.get("bbox", 0.0),
            "train_loss_mask": tm.get("mask", 0.0),
            "train_loss_dice": tm.get("dice", 0.0),
            "grad_norm_fpn": gnorms["fpn"],
            "grad_norm_decoder": gnorms["decoder"],
            "lr_fpn": cur_lr_fpn,
            "lr_decoder": cur_lr_decoder,
            "epoch_time_seconds": ep_time,
            "peak_gpu_memory_gb": peak_mem,
        }
        train_log.write(json.dumps(train_rec) + "\n")
        train_log.flush()

        # --- Validate ---
        if epoch in _VAL_EPOCHS or epoch == n_epochs - 1:
            if _DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_results = full_evaluate(
                model, engine.val_loader, criterion, torch.device(_DEVICE),
                score_threshold=args.score_threshold)

            val_rec: dict[str, Any] = {"epoch": epoch}
            for k, v in val_results.items():
                if isinstance(v, (int, float)):
                    val_rec[k] = v
                elif isinstance(v, list):
                    val_rec[k] = v

            for cls_id, cls_name in enumerate(CANONICAL_CLASSES):
                for thresh in [0.3, 0.5, 0.75]:
                    key = f"AP_class{cls_id}@{thresh:.0%}"
                    t_label = f"{int(thresh * 100)}"
                    if key in val_results:
                        val_rec[f"AP_{cls_name}@{t_label}"] = val_results[key]

            val_log.write(json.dumps(val_rec) + "\n")
            val_log.flush()

            # --- Checkpoint: best mAP@30 ---
            cur_map30 = val_results.get("mAP@30", 0.0)
            if cur_map30 > best_map30:
                best_map30 = cur_map30
                best_map30_epoch = epoch
                _save_checkpoint(
                    ckpt_dir / "best_map30.pt",
                    model, engine.ema, engine.optimizer,
                    full_config, epoch, val_results,
                )

            # --- Checkpoint: best PQ ---
            cur_pq = val_results.get("PQ", 0.0)
            if cur_pq > best_pq:
                best_pq = cur_pq
                best_pq_epoch = epoch
                _save_checkpoint(
                    ckpt_dir / "best_pq.pt",
                    model, engine.ema, engine.optimizer,
                    full_config, epoch, val_results,
                )

            print(
                f"[Epoch {epoch:3d}/{n_epochs}] "
                f"loss={tm.get('total', 0.0):.4f} "
                f"(cls={tm.get('cls', 0.0):.4f} bbox={tm.get('bbox', 0.0):.4f}) | "
                f"mAP@30={cur_map30:.4f} mAP@50={val_results.get('mAP@50', 0.0):.4f} "
                f"PQ={cur_pq:.4f} F1d@30={val_results.get('F1d@30', 0.0):.4f} "
                f"mIoU={val_results.get('mIoU', 0.0):.4f} | "
                f"ECE={val_results.get('ECE', 0.0):.4f} | "
                f"{ep_time:.1f}s"
            )
        else:
            print(
                f"[Epoch {epoch:3d}/{n_epochs}] "
                f"loss={tm.get('total', 0.0):.4f} "
                f"(cls={tm.get('cls', 0.0):.4f} bbox={tm.get('bbox', 0.0):.4f}) | "
                f"gnorm fpn={gnorms['fpn']:.3f} dec={gnorms['decoder']:.3f} | "
                f"{ep_time:.1f}s"
            )

    # --- Final checkpoint ---
    final_val = full_evaluate(
        model, engine.val_loader, criterion, torch.device(_DEVICE),
        score_threshold=args.score_threshold)
    _save_checkpoint(
        ckpt_dir / "final.pt",
        model, engine.ema, engine.optimizer,
        full_config, n_epochs - 1, final_val,
    )

    train_log.close()
    val_log.close()

    # ===================================================================
    # Test fold evaluation
    # ===================================================================
    test_results: dict[str, Any] | None = None
    if test_loader is not None:
        print("\n" + "=" * 70)
        print(f"Evaluating on held-out test fold {args.test_fold}")
        print("=" * 70)

        best_ckpt = ckpt_dir / "best_map30.pt"
        if best_ckpt.exists():
            ckpt = torch.load(
                str(best_ckpt), map_location=_DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best checkpoint (epoch {ckpt['epoch']})")

        # Load tissue type mapping for mPQ/bPQ
        test_tissue_ids: Optional[list[int]] = None
        tissue_file = data_root / f"pannuke_fold{args.test_fold}" / "tissue_types.json"
        if tissue_file.exists():
            with open(tissue_file) as f:
                tissue_data = json.load(f)
            tissue_map = tissue_data["image_tissues"]
            test_tissue_ids = []
            for img_meta in test_ds._images:
                fname = img_meta["file_name"]
                test_tissue_ids.append(tissue_map.get(fname, 0))
            print(f"Loaded tissue types for {len(test_tissue_ids)} test images")

        test_results = full_evaluate(
            model, test_loader, criterion, torch.device(_DEVICE),
            score_threshold=args.score_threshold,
            tissue_ids=test_tissue_ids)

        serialisable = {
            k: v for k, v in test_results.items()
            if isinstance(v, (int, float, list))
        }
        serialisable["checkpoint"] = "best_map30.pt"
        serialisable["test_fold"] = args.test_fold
        serialisable["train_folds"] = args.train_folds
        with open(out_dir / "test_results.json", "w") as f:
            json.dump(serialisable, f, indent=2)

        print(f"  mAP@30:  {test_results.get('mAP@30', 0.0):.4f}")
        print(f"  mAP@50:  {test_results.get('mAP@50', 0.0):.4f}")
        print(f"  PQ:      {test_results.get('PQ', 0.0):.4f}")
        print(f"  mPQ:     {test_results.get('mPQ', 0.0):.4f}")
        print(f"  bPQ:     {test_results.get('bPQ', 0.0):.4f}")
        print(f"  F1d@30:  {test_results.get('F1d@30', 0.0):.4f}")
        print(f"  mIoU:    {test_results.get('mIoU', 0.0):.4f}")
        print(f"  ECE:     {test_results.get('ECE', 0.0):.4f}")
        print(f"Saved to {out_dir / 'test_results.json'}")

    # ===================================================================
    # Post-training diagnostics
    # ===================================================================
    wall_clock = time.time() - t0

    if not args.skip_post_training:
        print("\n" + "=" * 70)
        print("Post-training diagnostics")
        print("=" * 70)

        best_ckpt = ckpt_dir / "best_map30.pt"
        if best_ckpt.exists():
            post_results = _run_post_training(
                model_cfg, best_ckpt, engine.val_loader,
                torch.device(_DEVICE), out_dir,
            )
        else:
            post_results = {}

        post_results["wall_clock_minutes"] = wall_clock / 60.0
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if _DEVICE == "cuda" else 0.0
        post_results["peak_gpu_memory_gb"] = peak_mem

        with open(out_dir / "post_training.json", "w") as f:
            json.dump(post_results, f, indent=2, default=str)

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Folds: {fold_desc}")
    print(f"Best val mAP@30: {best_map30:.4f} (epoch {best_map30_epoch})")
    print(f"Best val PQ:     {best_pq:.4f} (epoch {best_pq_epoch})")
    print(f"Final val mAP@30: {final_val.get('mAP@30', 0.0):.4f}")
    print(f"Final val mAP@50: {final_val.get('mAP@50', 0.0):.4f}")
    print(f"Final val PQ:     {final_val.get('PQ', 0.0):.4f}")
    print(f"Final val F1d@30: {final_val.get('F1d@30', 0.0):.4f}")
    print(f"Final val mIoU:   {final_val.get('mIoU', 0.0):.4f}")
    print(f"Final val ECE:    {final_val.get('ECE', 0.0):.4f}")
    if test_results is not None:
        print(f"--- Test fold {args.test_fold} ---")
        print(f"Test mAP@30: {test_results.get('mAP@30', 0.0):.4f}")
        print(f"Test mAP@50: {test_results.get('mAP@50', 0.0):.4f}")
        print(f"Test PQ:     {test_results.get('PQ', 0.0):.4f}")
        print(f"Test mPQ:    {test_results.get('mPQ', 0.0):.4f}")
        print(f"Test bPQ:    {test_results.get('bPQ', 0.0):.4f}")
        print(f"Test F1d@30: {test_results.get('F1d@30', 0.0):.4f}")
        print(f"Test mIoU:   {test_results.get('mIoU', 0.0):.4f}")
        print(f"Test ECE:    {test_results.get('ECE', 0.0):.4f}")
    print(f"Wall clock:   {wall_clock / 60:.1f} min")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
