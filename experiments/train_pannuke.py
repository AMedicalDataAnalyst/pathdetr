"""Full PanNuke training run with comprehensive metrics and diagnostics.

Supports the official PanNuke 3-fold cross-validation protocol: train on two
folds, validate on a held-out portion, and test on the third fold.  Repeat for
all three fold rotations and aggregate with ``aggregate_pannuke_cv.py``.

Config derived from factorial experiment results:
  ON:  StainAug, geometric aug, AdamW for FPN+decoder
  GIoU box loss, BCE+Dice mask loss
  200 epochs, batch_size=64, validate every 10 epochs + at {1, 5}
  lr_fpn=5e-4, lr_decoder=1e-3, cosine LR with 10-epoch warmup
  clip_grad_norm=1.0, EMA decay=0.998, num_queries=300

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
from mhc_path.models.mhc_path import MHCPath, MHCPathConfig, MHCPathEoMT
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
_BATCH_SIZE = 64
_VAL_EPOCHS = {1, 5} | set(range(9, _N_EPOCHS, 10))  # 1, 5, then every 10
_WARMUP_EPOCHS = 10
_NUM_CLASSES = 5
_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BASE_LR_FPN = 5e-4
_BASE_LR_DECODER = 1e-3
_EMA_DECAY = 0.998
_NUM_QUERIES = 300


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


def _linear_lr_with_warmup(epoch: int, warmup: int, total: int) -> float:
    """Linear decay from 1.0 to 0.0 after warmup."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 1.0 - (epoch - warmup) / max(1, total - warmup)


def _trapezoidal_lr_with_warmup(epoch: int, warmup: int, total: int,
                                 cooldown_frac: float = 0.4) -> float:
    """Constant LR then linear cooldown.

    Warmup → constant at 1.0 → linear cooldown to 0 over last cooldown_frac of training.
    """
    if epoch < warmup:
        return (epoch + 1) / warmup
    cooldown_start = int(total * (1 - cooldown_frac))
    if epoch < cooldown_start:
        return 1.0
    progress = (epoch - cooldown_start) / max(1, total - cooldown_start)
    return 1.0 - progress


LR_SCHEDULE_FNS = {
    "cosine": _cosine_lr_with_warmup,
    "linear": _linear_lr_with_warmup,
    "trapezoidal": _trapezoidal_lr_with_warmup,
}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _grad_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {"fpn": 0.0, "decoder": 0.0, "backbone": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.data.norm(2).item()
        lo = name.lower()
        if "fpn" in lo:
            norms["fpn"] = max(norms["fpn"], n)
        elif "backbone" in lo:
            norms["backbone"] = max(norms["backbone"], n)
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
            # Upsample to 256x256 and binarise
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
    if "pred_masks" in raw:
        out["pred_masks"] = raw["pred_masks"]
    elif "mask_logits" in raw:
        out["pred_masks"] = raw["mask_logits"]

    # EoMT: no boxes — derive from masks
    if raw.get("pred_boxes") is None and raw.get("box_coords") is None:
        from mhc_path.models.eomt_decoder import masks_to_boxes
        mask_key = "pred_masks" if "pred_masks" in out else "mask_logits"
        if mask_key in out:
            out["pred_boxes"] = masks_to_boxes(out[mask_key])
        else:
            out["pred_boxes"] = torch.zeros(
                out["pred_logits"].shape[0], out["pred_logits"].shape[1], 4,
                device=out["pred_logits"].device)
        # EoMT uses softmax (num_classes+1), but detection metrics expect
        # sigmoid-style logits (num_classes). Strip the "no object" class.
        if out["pred_logits"] is not None:
            n_cls = out["pred_logits"].shape[-1]
            if n_cls == _NUM_CLASSES + 1:
                # Convert softmax logits to sigmoid-compatible:
                # take the real class logits, subtract the no-object logit
                no_obj = out["pred_logits"][..., -1:]
                out["pred_logits"] = out["pred_logits"][..., :-1] - no_obj
    else:
        out["pred_boxes"] = raw.get("pred_boxes", raw.get("box_coords"))

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
    # Unfiltered for mAP (needs full P-R curve)
    det_metrics = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    # Filtered for F1d / P / R (padding queries destroy precision)
    det_metrics_filt = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    pq_metric = PanopticQuality(num_classes=_NUM_CLASSES, iou_threshold=0.5)
    tissue_pq: Optional[TissuePanopticQuality] = None
    if tissue_ids is not None:
        tissue_pq = TissuePanopticQuality(num_classes=_NUM_CLASSES)
    seg_metric = SegmentationMetrics(num_classes=_NUM_CLASSES)

    running_loss: dict[str, float] = {}
    count = 0
    all_scores: list[torch.Tensor] = []
    all_tp_flags: list[torch.Tensor] = []
    all_pred_stats: dict[str, float] = {}
    n_stats = 0
    img_cursor = 0  # tracks position in dataset for tissue_ids lookup

    for batch in val_loader:
        images = batch["images"].to(device)
        raw = model(images)
        outputs = _normalize_output(raw)
        targets = _batch_to_targets(batch, device)

        losses = criterion(outputs, targets)

        # Track per-aux-layer losses
        loss_dict: dict[str, float] = {}
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            loss_dict[k] = val
            running_loss[k] = running_loss.get(k, 0.0) + val

        if "aux_outputs" in outputs:
            matcher = criterion.matcher
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_pred = {
                    "pred_logits": aux_out["class_logits"],
                    "pred_boxes": aux_out["box_coords"],
                }
                aux_indices = matcher.forward(aux_pred, targets)
                from mhc_path.training.losses import sigmoid_focal_loss_onehot
                # Track individual aux losses
                n_boxes = max(sum(len(t.labels) for t in targets), 1)
                aux_logits = aux_pred["pred_logits"]
                B_aux, Q_aux, C_aux = aux_logits.shape
                tgt_cls = torch.full((B_aux, Q_aux), _NUM_CLASSES, dtype=torch.int64, device=device)
                for b_i, (pi, ti) in enumerate(aux_indices):
                    if len(pi) > 0:
                        tgt_cls[b_i, pi] = targets[b_i].labels[ti].to(device)
                import torch.nn.functional as F
                tgt_oh = F.one_hot(tgt_cls, _NUM_CLASSES + 1)[..., :_NUM_CLASSES].float()
                aux_cls_loss = sigmoid_focal_loss_onehot(
                    aux_logits.reshape(-1, _NUM_CLASSES),
                    tgt_oh.reshape(-1, _NUM_CLASSES),
                    num_boxes=n_boxes,
                ).item()
                loss_dict[f"cls_aux{i}"] = aux_cls_loss

        count += 1

        # Unfiltered preds for mAP (needs full P-R curve) and ECE — no masks needed
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

        # Segmentation metrics: accumulate matched mask pairs
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
                        # Sort preds by score descending
                        order = pred_d["scores"].argsort(descending=True)
                        for pi in order:
                            pc = pl[pi].item()
                            # Find best unmatched GT with same class
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

        # ECE accumulators: collect scores and TP flags at IoU=0.3
        for b in range(outputs["pred_logits"].shape[0]):
            sc = preds_all[b]["scores"]
            all_scores.append(sc.cpu())

        # Prediction stats (unfiltered — diagnostic)
        ps = _prediction_stats(outputs)
        for k2, v2 in ps.items():
            all_pred_stats[k2] = all_pred_stats.get(k2, 0.0) + v2
        n_stats += 1

    # Aggregate
    results: dict[str, Any] = {}

    # Losses
    for k, v in running_loss.items():
        results[f"val_{k}"] = v / max(count, 1)

    # mAP from unfiltered preds (full P-R curve)
    det_results = det_metrics.compute()
    for k, v in det_results.items():
        if k.startswith("mAP") or k.startswith("AP_class"):
            results[k] = v

    # F1d, P, R from filtered preds (score-thresholded)
    det_results_filt = det_metrics_filt.compute()
    for k, v in det_results_filt.items():
        if not (k.startswith("mAP") or k.startswith("AP_class")):
            results[k] = v

    # PQ (uses mask IoU when masks available)
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

    # ECE — collect TP flags from det_metrics at threshold 0.3
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

    # Prediction stats (averaged)
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
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": {k: v for k, v in metrics.items()
                    if not isinstance(v, (list, dict))},
    }
    torch.save(ckpt, str(path))


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

    # Load best checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # --- num_queries sweep ---
    # Always include the training query count plus smaller ablations
    trained_nq = model_cfg.num_queries
    sweep_nqs = sorted(set([50, 100, trained_nq]))
    for nq in sweep_nqs:
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
            mask_upsample_factor=model_cfg.mask_upsample_factor,
            with_pixel_decoder=model_cfg.with_pixel_decoder,
            large_kernel=model_cfg.large_kernel,
            large_kernel_size=model_cfg.large_kernel_size,
        )
        sweep_model = MHCPath(cfg).to(device)
        # Load weights (skip query embed size mismatch)
        state = ckpt["model_state_dict"]
        filtered = {k: v for k, v in state.items()
                    if k in sweep_model.state_dict()
                    and v.shape == sweep_model.state_dict()[k].shape}
        sweep_model.load_state_dict(filtered, strict=False)
        sweep_model.eval()

        criterion = DetectionLoss(
            num_classes=_NUM_CLASSES,
            matcher=HungarianMatcher(),
            box_loss_type="giou",
        )
        sweep_results = full_evaluate(sweep_model, val_loader, criterion, device,
                                      score_threshold=0.3)
        results[f"queries_{nq}"] = {
            "mAP@30": sweep_results.get("mAP@30", 0.0),
            "mAP@50": sweep_results.get("mAP@50", 0.0),
            "PQ": sweep_results.get("PQ", 0.0),
            "DQ": sweep_results.get("DQ", 0.0),
            "SQ": sweep_results.get("SQ", 0.0),
            "mPQ": sweep_results.get("mPQ", 0.0),
            "bPQ": sweep_results.get("bPQ", 0.0),
            "F1d@30": sweep_results.get("F1d@30", 0.0),
            "F1d@50": sweep_results.get("F1d@50", 0.0),
            "mIoU": sweep_results.get("mIoU", 0.0),
        }
        # Per-class PQ breakdown
        for c in range(_NUM_CLASSES):
            results[f"queries_{nq}"][f"PQ_class{c}"] = sweep_results.get(f"PQ_class{c}", 0.0)
            results[f"queries_{nq}"][f"DQ_class{c}"] = sweep_results.get(f"DQ_class{c}", 0.0)
            results[f"queries_{nq}"][f"SQ_class{c}"] = sweep_results.get(f"SQ_class{c}", 0.0)
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
    results["backbone_stats"] = {
        k: v for k, v in bb_stats.items()
    }

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

    # Backbone
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vitl16", "dinov3_vitb16", "dinov3_vitg14", "phikon_v2"],
                        help="Backbone model variant (default: dinov3_vitl16)")

    # Common
    parser.add_argument("--out_dir", type=str, default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=_N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--num_queries", type=int, default=_NUM_QUERIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Min score for PQ/F1d predictions (default: 0.3)")
    parser.add_argument("--skip_post_training", action="store_true")
    parser.add_argument("--mask_loss_resolution", type=int, default=128,
                        help="Resolution for mask loss computation (default: 128)")
    parser.add_argument("--mask_upsample_factor", type=int, default=4,
                        help="Upsample pixel features before dot-product mask head (default: 4)")

    # Backbone unfreezing + per-group LR
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                        help="Unfreeze last N backbone transformer blocks (default: 0 = frozen)")
    parser.add_argument("--lr_backbone", type=float, default=0.0,
                        help="Base LR for unfrozen backbone params (default: 0.0)")
    parser.add_argument("--lr_fpn", type=float, default=_BASE_LR_FPN,
                        help=f"Base LR for FPN params (default: {_BASE_LR_FPN})")
    parser.add_argument("--lr_decoder", type=float, default=_BASE_LR_DECODER,
                        help=f"Base LR for decoder params (default: {_BASE_LR_DECODER})")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 1e-4, following RF-DETR)")
    parser.add_argument("--lr_scaling", type=str, default="linear",
                        choices=["linear", "sqrt", "none"],
                        help="LR scaling rule when batch_size differs from ref (default: linear)")
    parser.add_argument("--ref_batch_size", type=int, default=32,
                        help="Reference batch size for LR scaling (default: 32)")

    # Backbone freeze schedule
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Keep backbone frozen for this many epochs before unfreezing (default: 0 = unfreeze immediately)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience in epochs (0 = disabled, default: 0)")

    # Boundary loss
    parser.add_argument("--mask_boundary_weight", type=float, default=1.0,
                        help="Weight for mask boundary (Sobel gradient) loss (default: 1.0)")

    # Pixel decoder with ViT skip connections
    parser.add_argument("--with_pixel_decoder", action="store_true",
                        help="Use multi-scale pixel decoder with ViT skip connections")

    # Large-kernel refinement
    parser.add_argument("--large_kernel", action="store_true",
                        help="Use RepLKBlock large-kernel depthwise convs in pixel decoder")
    parser.add_argument("--large_kernel_size", type=int, default=13,
                        help="Kernel size for large-kernel blocks (default: 13)")

    # Group DETR
    parser.add_argument("--group_detr", type=int, default=3,
                        help="Number of query groups for Group DETR (1=off, 11=RF-DETR default)")

    # Decoder architecture
    parser.add_argument("--decoder", type=str, default="rfdetr",
                        choices=["rfdetr", "eomt"],
                        help="Decoder architecture (default: rfdetr)")

    # Denoising training
    parser.add_argument("--use_denoising", action="store_true",
                        help="Enable DN-DETR-style denoising training (RF-DETR decoder only)")
    parser.add_argument("--dn_groups", type=int, default=3,
                        help="Number of denoising query groups per GT (default: 3, DN-DETR uses 5)")
    parser.add_argument("--dn_label_noise", type=float, default=0.2,
                        help="Label noise ratio for denoising (default: 0.2, DN-DETR original)")
    parser.add_argument("--dn_box_noise", type=float, default=0.4,
                        help="Box noise scale for denoising (default: 0.4)")

    # LR schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "trapezoidal"],
                        help="LR schedule type (default: cosine). "
                             "linear = linear decay to 0. "
                             "trapezoidal = constant then linear cooldown.")

    # Resume from checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g. out_dir/checkpoints/best_pq.pt)")
    parser.add_argument("--schedule_free_restart", action="store_true",
                        help="On resume, restart cosine LR schedule for the remaining epochs "
                             "(warm restart). Without this, the schedule spans 0..epochs, "
                             "which causes an LR jump when extending a completed run.")

    return parser.parse_args()


def _apply_lr_scaling(args: argparse.Namespace) -> None:
    """Scale LRs when batch_size differs from ref_batch_size.

    Linear rule (Goyal et al. 2017): lr *= bs / ref_bs.
    Sqrt rule (Hoffer et al. 2017): lr *= sqrt(bs / ref_bs).
    Only scales lr_fpn and lr_decoder; lr_backbone is left untouched
    (backbone is typically frozen or at a very low, hand-tuned LR).
    """
    if args.lr_scaling == "none" or args.batch_size == args.ref_batch_size:
        return
    ratio = args.batch_size / args.ref_batch_size
    if args.lr_scaling == "sqrt":
        ratio = math.sqrt(ratio)
    args.lr_fpn *= ratio
    args.lr_decoder *= ratio
    print(f"LR scaling ({args.lr_scaling}): bs {args.ref_batch_size}->{args.batch_size}, "
          f"lr_decoder={args.lr_decoder:.2e}, lr_fpn={args.lr_fpn:.2e}")


def main() -> None:
    args = parse_args()
    _apply_lr_scaling(args)
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
        # Single-fold: internal train/val split, no separate test set
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
        # Multi-fold: merge training folds, hold out test fold
        data_root = Path(args.data_root)

        merged_coco = _merge_fold_annotations(data_root, args.train_folds)

        split_dir = out_dir / "splits"
        train_json, val_json = _split_train_val(
            merged_coco, args.val_fraction, args.seed, split_dir)

        # file_names include relative path from data_root
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
    print(f"Config: {args.backbone} + FPN + RF-DETR, AdamW, StainAug+geometric")
    print(f"  GIoU box loss, num_queries={args.num_queries}")
    print(f"  {n_epochs} epochs, batch_size={args.batch_size}")
    print(f"  lr_backbone={args.lr_backbone}, lr_fpn={args.lr_fpn}, lr_decoder={args.lr_decoder}")
    if args.unfreeze_last_n > 0:
        print(f"  unfreeze_last_n={args.unfreeze_last_n}, freeze_epochs={args.freeze_epochs}")
    print(f"  score_threshold={args.score_threshold} (for PQ/F1d)")
    print(f"  mask_loss_resolution={args.mask_loss_resolution}, mask_upsample_factor={args.mask_upsample_factor}")
    if args.mask_boundary_weight != 1.0:
        print(f"  mask_boundary_weight={args.mask_boundary_weight}")
    if args.with_pixel_decoder:
        print(f"  Pixel decoder ON (ViT skip connections)")
    if args.large_kernel:
        print(f"  Large-kernel refinement ON (k={args.large_kernel_size})")
    if args.group_detr > 1:
        print(f"  Group DETR: {args.group_detr} groups")
    if args.patience > 0:
        print(f"  Early stopping: patience={args.patience}")
    print(f"  warmup={_WARMUP_EPOCHS} epochs, EMA decay={_EMA_DECAY}")
    print(f"Device: {_DEVICE}")
    print("=" * 70)

    # --- Model ---
    # Pixel decoder handles its own upsampling; override upsample_factor
    _mask_upsample = 1 if args.with_pixel_decoder else args.mask_upsample_factor
    model_cfg = MHCPathConfig(
        backbone=args.backbone, backbone_frozen=True,
        fpn_dim=256, num_queries=args.num_queries, num_classes=_NUM_CLASSES,
        num_decoder_layers=6, with_segmentation=True,
        output_layers=(6, 12, 18, 24), fpn_levels=4,
        mask_upsample_factor=_mask_upsample,
        with_pixel_decoder=args.with_pixel_decoder,
        large_kernel=args.large_kernel,
        large_kernel_size=args.large_kernel_size,
        group_detr=args.group_detr,
    )
    if args.decoder == "eomt":
        model = MHCPathEoMT(model_cfg).to(_DEVICE)
        print(f"  Using EoMT decoder (queries injected into final 4 ViT blocks)")
    else:
        model = MHCPath(model_cfg).to(_DEVICE)

    # --- Denoising training setup ---
    dn_generator = None
    if args.use_denoising and args.decoder == "rfdetr":
        from mhc_path.training.denoising import DenoisingGenerator
        dn_generator = DenoisingGenerator(
            d_model=256, num_classes=_NUM_CLASSES,
            num_dn_groups=args.dn_groups,
            box_noise_scale=args.dn_box_noise,
            label_noise_ratio=args.dn_label_noise,
        ).to(_DEVICE)
        print(f"  Denoising training: {args.dn_groups} groups, "
              f"box_noise={args.dn_box_noise}, label_noise={args.dn_label_noise}")

    # --- Selective backbone unfreezing ---
    _backbone_unfrozen = False
    if args.unfreeze_last_n > 0 and args.freeze_epochs <= 0:
        n_unfrozen = model.backbone.unfreeze_last_n(args.unfreeze_last_n)
        print(f"  Unfroze last {args.unfreeze_last_n} blocks: {n_unfrozen:,} params")
        _backbone_unfrozen = True
    elif args.unfreeze_last_n > 0 and args.freeze_epochs > 0:
        print(f"  Backbone frozen for first {args.freeze_epochs} epochs, "
              f"then unfreeze last {args.unfreeze_last_n} blocks")

    # --- Augmentation: StainAug ON, geometric ON, HistoRotate OFF ---
    gpu_aug = GPUPathologyAugPipeline(
        target_size=256, histo_rotate=False, stain_aug=True,
        geometric=True, mode="detection",
    ).to(_DEVICE)

    # --- Loss ---
    matcher = HungarianMatcher()
    loss_weights = None
    if args.mask_boundary_weight != 1.0:
        loss_weights = {
            "cls": 2.0, "bbox": 5.0, "mask": 5.0, "dice": 2.0,
            "mask_boundary": args.mask_boundary_weight,
        }
    if args.decoder == "eomt":
        from mhc_path.training.eomt_loss import EoMTLoss
        criterion = EoMTLoss(num_classes=_NUM_CLASSES)
    else:
        criterion = DetectionLoss(
            num_classes=_NUM_CLASSES, matcher=matcher, box_loss_type="giou",
            mask_loss_resolution=args.mask_loss_resolution,
            loss_weights=loss_weights,
            group_detr=args.group_detr,
        )

    # --- Engine ---
    n_batches = (n_train_images + args.batch_size - 1) // args.batch_size
    det_cfg = DetectionConfig(
        epochs=n_epochs, batch_size=args.batch_size,
        lr_backbone=args.lr_backbone, lr_fpn=args.lr_fpn, lr_decoder=args.lr_decoder,
        weight_decay=args.weight_decay, clip_grad_norm=1.0, ema_decay=_EMA_DECAY,
        use_amp=True,
        total_train_steps=n_epochs * n_batches,
        output_dir=str(out_dir / "checkpoints"),
        num_workers=args.num_workers, log_interval=1000,
    )
    det_metrics = DetectionMetrics(num_classes=_NUM_CLASSES, iou_thresholds=_IOU_THRESHOLDS)
    engine = DetectionEngine(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        gpu_aug=gpu_aug, config=det_cfg, criterion=criterion, metrics=det_metrics,
    )

    # --- Denoising training ---
    if dn_generator is not None:
        engine.dn_generator = dn_generator

    # Stash initial LR on each param group for per-group cosine scheduling
    if engine.optimizer is not None:
        for pg in engine.optimizer.param_groups:
            pg["_base_lr"] = pg["lr"]
    # --- Logging setup ---
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=_DEVICE, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "ema_state_dict" in resume_ckpt:
            engine.ema.load_state_dict(resume_ckpt["ema_state_dict"])
        if "optimizer_state_dict" in resume_ckpt and engine.optimizer is not None:
            try:
                engine.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Could not restore optimizer state: {e}")
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"\nResumed from {args.resume}, starting at epoch {start_epoch}")
        del resume_ckpt
        if _DEVICE == "cuda":
            torch.cuda.empty_cache()

    train_log = open(out_dir / "train_log.jsonl", "a" if args.resume else "w")
    val_log = open(out_dir / "val_log.jsonl", "a" if args.resume else "w")

    best_map30 = -1.0
    best_pq = -1.0
    best_map30_epoch = 0
    best_pq_epoch = 0

    # On resume, recover best metrics from existing val_log to avoid
    # overwriting best_pq.pt / best_map30.pt with worse checkpoints
    if args.resume:
        val_log_path = out_dir / "val_log.jsonl"
        if val_log_path.exists():
            with open(val_log_path) as _vl:
                for _line in _vl:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _d = json.loads(_line)
                    except Exception:
                        continue
                    if _d.get("PQ", 0) > best_pq:
                        best_pq = _d["PQ"]
                        best_pq_epoch = _d.get("epoch", 0)
                    if _d.get("mAP@30", 0) > best_map30:
                        best_map30 = _d["mAP@30"]
                        best_map30_epoch = _d.get("epoch", 0)
            if best_pq > 0:
                print(f"  Recovered best PQ={best_pq:.4f} (epoch {best_pq_epoch}), "
                      f"best mAP@30={best_map30:.4f} (epoch {best_map30_epoch})")

    # Full config for checkpoint metadata
    full_config: dict[str, Any] = {
        "model": {
            "backbone": model_cfg.backbone,
            "backbone_frozen": model_cfg.backbone_frozen,
            "fpn_dim": model_cfg.fpn_dim,
            "num_queries": model_cfg.num_queries,
            "num_classes": model_cfg.num_classes,
            "num_decoder_layers": model_cfg.num_decoder_layers,
            "mask_upsample_factor": model_cfg.mask_upsample_factor,
            "group_detr": model_cfg.group_detr,
        },
        "training": {
            "epochs": n_epochs,
            "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone,
            "lr_fpn": args.lr_fpn,
            "lr_decoder": args.lr_decoder,
            "unfreeze_last_n": args.unfreeze_last_n,
            "warmup_epochs": _WARMUP_EPOCHS,
            "lr_schedule": args.lr_schedule,
            "ema_decay": _EMA_DECAY,
            "lr_scaling": args.lr_scaling,
            "ref_batch_size": args.ref_batch_size,
            "clip_grad_norm": 1.0,
            "box_loss": "giou",
            "mask_loss_resolution": args.mask_loss_resolution,
            "patience": args.patience,
        },
        "augmentation": {
            "stain_aug": True,
            "geometric": True,
            "histo_rotate": False,
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
    for epoch in range(start_epoch, n_epochs):
        # --- Deferred backbone unfreeze ---
        if (args.unfreeze_last_n > 0 and args.freeze_epochs > 0
                and epoch == args.freeze_epochs and not _backbone_unfrozen):
            n_unfrozen = model.backbone.unfreeze_last_n(args.unfreeze_last_n)
            _backbone_unfrozen = True
            print(f"\n[Epoch {epoch}] Unfroze last {args.unfreeze_last_n} "
                  f"backbone blocks: {n_unfrozen:,} params")
            # Add unfrozen backbone params to optimizer
            if engine.optimizer is not None:
                bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
                if bb_params:
                    engine.optimizer.add_param_group({
                        "params": bb_params,
                        "lr": args.lr_backbone,
                        "_base_lr": args.lr_backbone,
                    })
                    print(f"  Added {len(bb_params)} backbone param tensors to AdamW "
                          f"(lr={args.lr_backbone})")

        # --- LR schedule (per-group) ---
        schedule_fn = LR_SCHEDULE_FNS[args.lr_schedule]
        # Warm restart: remap epoch so the cycle covers start_epoch..n_epochs
        if args.schedule_free_restart and start_epoch > 0:
            remaining = n_epochs - start_epoch
            warmup_restart = min(5, remaining // 10)  # short warmup for restart
            lr_mult = schedule_fn(
                epoch - start_epoch, warmup_restart, remaining)
        else:
            lr_mult = schedule_fn(epoch, _WARMUP_EPOCHS, n_epochs)
        cur_lr_fpn = args.lr_fpn * lr_mult
        cur_lr_decoder = args.lr_decoder * lr_mult
        cur_lr_backbone = args.lr_backbone * lr_mult
        if engine.optimizer is not None:
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
            "grad_norm_backbone": gnorms["backbone"],
            "grad_norm_fpn": gnorms["fpn"],
            "grad_norm_decoder": gnorms["decoder"],
            "lr_backbone": cur_lr_backbone,
            "lr_fpn": cur_lr_fpn,
            "lr_decoder": cur_lr_decoder,
            "epoch_time_seconds": ep_time,
            "peak_gpu_memory_gb": peak_mem,
        }
        train_log.write(json.dumps(train_rec) + "\n")
        train_log.flush()

        # --- Validate ---
        val_epochs = {1, 5} | set(range(9, n_epochs, 10))
        if epoch in val_epochs or epoch == n_epochs - 1:
            if _DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_results = full_evaluate(
                model, engine.val_loader, criterion, torch.device(_DEVICE),
                score_threshold=args.score_threshold)

            # Serialise: separate scalars from non-scalar
            val_rec: dict[str, Any] = {"epoch": epoch}
            for k, v in val_results.items():
                if isinstance(v, (int, float)):
                    val_rec[k] = v
                elif isinstance(v, list):
                    val_rec[k] = v  # confusion matrix

            # Add per-class AP with class names
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

            # --- Periodic checkpoint (every 50 epochs) for resume safety ---
            if (epoch + 1) % 50 == 0:
                _save_checkpoint(
                    ckpt_dir / "latest.pt",
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

        # --- Early stopping ---
        if args.patience > 0 and epoch > best_pq_epoch + args.patience:
            print(f"Early stopping at epoch {epoch} (no PQ improvement since epoch {best_pq_epoch})")
            break

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

        # Reload best checkpoint for test evaluation (prefer PQ, fall back to mAP@30)
        best_ckpt = ckpt_dir / "best_pq.pt"
        if not best_ckpt.exists():
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
            # Build ordered list matching the test dataset
            test_tissue_ids = []
            for img_meta in test_ds._images:
                fname = img_meta["file_name"]
                test_tissue_ids.append(tissue_map.get(fname, 0))
            print(f"Loaded tissue types for {len(test_tissue_ids)} test images")

        test_results = full_evaluate(
            model, test_loader, criterion, torch.device(_DEVICE),
            score_threshold=args.score_threshold,
            tissue_ids=test_tissue_ids)

        # Save test results
        serialisable = {
            k: v for k, v in test_results.items()
            if isinstance(v, (int, float, list))
        }
        serialisable["checkpoint"] = best_ckpt.name
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

        best_ckpt = ckpt_dir / "best_pq.pt"
        if not best_ckpt.exists():
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
