"""COCO-style detection mAP, PQ, F1d, and segmentation IoU metrics.

Pure-torch implementation (no pycocotools dependency) that follows the
accumulate / compute pattern for use across multiple evaluation batches.

Depends on:
    - mhc_path.models.box_utils  (pairwise box IoU)
    - mhc_path.training.losses   (DetectionTarget dataclass)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from mhc_path.models.box_utils import box_iou, cxcywh_to_xyxy
from mhc_path.training.losses import DetectionTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_iou_matrix(
    pred_masks: torch.Tensor, gt_masks: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU between two sets of binary masks.

    Args:
        pred_masks: (N, H, W) predicted binary masks.
        gt_masks: (M, H, W) ground-truth binary masks.

    Returns:
        (N, M) IoU matrix.
    """
    pred_flat = pred_masks.flatten(1).float()  # (N, H*W)
    gt_flat = gt_masks.flatten(1).float()      # (M, H*W)
    intersection = pred_flat @ gt_flat.T       # (N, M)
    pred_area = pred_flat.sum(dim=1, keepdim=True)  # (N, 1)
    gt_area = gt_flat.sum(dim=1, keepdim=True)      # (M, 1)
    union = pred_area + gt_area.T - intersection    # (N, M)
    return intersection / union.clamp(min=1e-6)

_RECALL_POINTS = torch.linspace(0.0, 1.0, 101)  # 101-point interpolation


def _compute_ap(scores: torch.Tensor, tp: torch.Tensor, num_gt: int) -> float:
    """Compute Average Precision from sorted TP flags via 101-point interpolation.

    Args:
        scores: (K,) confidence scores, already sorted descending.
        tp: (K,) boolean tensor indicating true-positive at each rank.
        num_gt: total number of ground-truth instances for this class.

    Returns:
        AP value in [0, 1].
    """
    if num_gt == 0 or len(scores) == 0:
        return 0.0

    tp_cum = tp.float().cumsum(dim=0)
    fp_cum = (~tp).float().cumsum(dim=0)

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / num_gt

    # 101-point interpolation: for each recall threshold r, take the maximum
    # precision at recall >= r.
    recall_pts = _RECALL_POINTS.to(recall.device)
    ap = 0.0
    for r in recall_pts:
        mask = recall >= r
        if mask.any():
            ap += precision[mask].max().item()
    ap /= len(recall_pts)
    return ap


# ---------------------------------------------------------------------------
# Detection Metrics
# ---------------------------------------------------------------------------

class DetectionMetrics:
    """Accumulate detection predictions across batches and compute COCO-style mAP.

    Predictions are expected as post-NMS dicts (or direct DETR output):
        {"boxes": Tensor(N, 4) xyxy, "scores": Tensor(N,), "labels": Tensor(N,)}

    Targets use the project-standard ``DetectionTarget`` dataclass with
    ``boxes`` in cxcywh normalised format and ``labels`` as int64.
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    ) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds

        # Per-class accumulators: list of (score, is_tp) per IoU threshold.
        # Structure: _scores[cls] = Tensor list, _tp[iou_idx][cls] = Tensor list
        self._scores: list[list[torch.Tensor]] = [
            [] for _ in range(num_classes)
        ]
        self._tp: list[list[list[torch.Tensor]]] = [
            [[] for _ in range(num_classes)] for _ in iou_thresholds
        ]
        self._num_gt: list[int] = [0] * num_classes

        # Confusion matrix: (pred_class, true_class) for matched pairs at each threshold
        self._confusion: list[np.ndarray] = [
            np.zeros((num_classes, num_classes), dtype=np.int64)
            for _ in iou_thresholds
        ]

    # ---- accumulate -------------------------------------------------------

    def update(
        self,
        predictions: list[dict],
        targets: list[DetectionTarget],
    ) -> None:
        """Accumulate one batch of predictions and targets."""
        for pred, target in zip(predictions, targets):
            self._update_single(pred, target)

    def _update_single(self, pred: dict, target: DetectionTarget) -> None:
        pred_boxes = pred["boxes"]       # (P, 4) xyxy
        pred_scores = pred["scores"]     # (P,)
        pred_labels = pred["labels"]     # (P,)

        gt_boxes_xyxy = cxcywh_to_xyxy(target.boxes)  # (G, 4) xyxy
        gt_labels = target.labels                       # (G,)

        # Count ground truths per class.
        for lbl in gt_labels:
            cls = lbl.item()
            if 0 <= cls < self.num_classes:
                self._num_gt[cls] += 1

        if len(pred_boxes) == 0 or len(gt_boxes_xyxy) == 0:
            # Still record predictions as FP if there are predictions but no GT.
            if len(pred_boxes) > 0:
                for cls_id in range(self.num_classes):
                    cls_mask = pred_labels == cls_id
                    if cls_mask.any():
                        self._scores[cls_id].append(pred_scores[cls_mask])
                        for t_idx in range(len(self.iou_thresholds)):
                            self._tp[t_idx][cls_id].append(
                                torch.zeros(cls_mask.sum(), dtype=torch.bool,
                                            device=pred_scores.device)
                            )
            return

        # Pairwise IoU: (P, G) — class-agnostic for confusion matrix
        iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)

        # Class-agnostic matching for confusion matrix at each threshold
        for t_idx, iou_thresh in enumerate(self.iou_thresholds):
            sorted_all = pred_scores.argsort(descending=True)
            matched_gt_global = torch.zeros(len(gt_boxes_xyxy), dtype=torch.bool,
                                            device=pred_scores.device)
            for rank in sorted_all:
                ious = iou_matrix[rank].clone()
                ious[matched_gt_global] = -1.0
                best_gt = ious.argmax()
                if ious[best_gt] >= iou_thresh:
                    matched_gt_global[best_gt] = True
                    pc = pred_labels[rank].item()
                    gc = gt_labels[best_gt].item()
                    if 0 <= pc < self.num_classes and 0 <= gc < self.num_classes:
                        self._confusion[t_idx][pc, gc] += 1

        for cls_id in range(self.num_classes):
            pred_cls_mask = pred_labels == cls_id
            gt_cls_mask = gt_labels == cls_id

            if not pred_cls_mask.any():
                continue

            cls_scores = pred_scores[pred_cls_mask]
            self._scores[cls_id].append(cls_scores)

            if not gt_cls_mask.any():
                # All predictions for this class are FP.
                for t_idx in range(len(self.iou_thresholds)):
                    self._tp[t_idx][cls_id].append(
                        torch.zeros(pred_cls_mask.sum(), dtype=torch.bool,
                                    device=pred_scores.device)
                    )
                continue

            # Sub-matrix for this class.
            pred_indices = torch.where(pred_cls_mask)[0]
            gt_indices = torch.where(gt_cls_mask)[0]
            cls_iou = iou_matrix[pred_indices][:, gt_indices]  # (Pc, Gc)

            # Sort predictions by descending score.
            sorted_order = cls_scores.argsort(descending=True)

            for t_idx, iou_thresh in enumerate(self.iou_thresholds):
                tp_flags = torch.zeros(len(pred_indices), dtype=torch.bool,
                                       device=pred_scores.device)
                matched_gt = torch.zeros(len(gt_indices), dtype=torch.bool,
                                         device=pred_scores.device)

                for rank in sorted_order:
                    ious_for_pred = cls_iou[rank]
                    # Find best unmatched GT.
                    available = ious_for_pred.clone()
                    available[matched_gt] = -1.0
                    best_gt = available.argmax()
                    if available[best_gt] >= iou_thresh:
                        tp_flags[rank] = True
                        matched_gt[best_gt] = True

                self._tp[t_idx][cls_id].append(tp_flags)

    # ---- compute ----------------------------------------------------------

    def compute(self) -> dict[str, float]:
        """Compute mAP, per-class P/R/F1, and F1d from accumulated data.

        Returns dict with keys:
            - "mAP@{t}" for each threshold in ``iou_thresholds``
            - "mAP@50:95" (COCO primary metric, IoU 0.50:0.05:0.95)
            - "AP_class{c}@{t}" per-class AP at each threshold
            - "precision_class{c}@{t}", "recall_class{c}@{t}", "F1_class{c}@{t}"
            - "F1d@{t}" aggregate detection F1
        """
        results: dict[str, float] = {}

        for t_idx, iou_thresh in enumerate(self.iou_thresholds):
            aps: list[float] = []
            f1ds: list[float] = []
            thresh_label = f"{int(iou_thresh * 100)}"

            for cls_id in range(self.num_classes):
                ap = self._compute_class_ap(t_idx, cls_id)
                # Backward-compatible key format: AP_class0@50%
                results[f"AP_class{cls_id}@{iou_thresh:.0%}"] = ap
                aps.append(ap)

                # Per-class TP/FP/FN counts for P/R/F1
                tp_count, fp_count, fn_count = self._class_tp_fp_fn(t_idx, cls_id)
                prec = tp_count / max(tp_count + fp_count, 1)
                rec = tp_count / max(tp_count + fn_count, 1)
                f1 = 2 * tp_count / max(2 * tp_count + fp_count + fn_count, 1)
                results[f"precision_class{cls_id}@{thresh_label}"] = prec
                results[f"recall_class{cls_id}@{thresh_label}"] = rec
                results[f"F1_class{cls_id}@{thresh_label}"] = f1
                f1ds.append(f1)

            mean_ap = sum(aps) / max(len(aps), 1)
            results[f"mAP@{thresh_label}"] = mean_ap
            results[f"F1d@{thresh_label}"] = sum(f1ds) / max(len(f1ds), 1)

        # COCO-style mAP@50:95 — average over 10 thresholds.
        coco_thresholds = torch.arange(0.5, 1.0, 0.05).tolist()
        coco_aps: list[float] = []
        for iou_thresh in coco_thresholds:
            class_aps: list[float] = []
            for cls_id in range(self.num_classes):
                t_idx = self._find_threshold_index(iou_thresh)
                if t_idx is not None:
                    class_aps.append(self._compute_class_ap(t_idx, cls_id))
                else:
                    class_aps.append(0.0)
            coco_aps.append(sum(class_aps) / max(len(class_aps), 1))
        results["mAP@50:95"] = sum(coco_aps) / max(len(coco_aps), 1)

        return results

    def _class_tp_fp_fn(self, threshold_idx: int, cls_id: int) -> tuple[int, int, int]:
        """Count TP, FP, FN for a class at a given IoU threshold."""
        if not self._scores[cls_id]:
            return 0, 0, self._num_gt[cls_id]
        tp_flags = torch.cat(self._tp[threshold_idx][cls_id])
        tp = int(tp_flags.sum().item())
        fp = int((~tp_flags).sum().item())
        fn = self._num_gt[cls_id] - tp
        return tp, fp, fn

    def confusion_matrix(self, iou_threshold: float = 0.5) -> np.ndarray:
        """Return the (C, C) confusion matrix at the given IoU threshold.

        Entry [i, j] counts matched pairs where pred_class=i and gt_class=j.
        """
        t_idx = self._find_threshold_index(iou_threshold)
        if t_idx is None:
            return np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        return self._confusion[t_idx].copy()

    def _find_threshold_index(self, iou_thresh: float) -> Optional[int]:
        for i, t in enumerate(self.iou_thresholds):
            if abs(t - iou_thresh) < 1e-6:
                return i
        return None

    def _compute_class_ap(self, threshold_idx: int, cls_id: int) -> float:
        if not self._scores[cls_id]:
            return 0.0

        scores = torch.cat(self._scores[cls_id])
        tp_flags = torch.cat(self._tp[threshold_idx][cls_id])
        num_gt = self._num_gt[cls_id]

        if num_gt == 0:
            return 0.0

        # Sort by descending score.
        sorted_indices = scores.argsort(descending=True)
        tp_sorted = tp_flags[sorted_indices]
        scores_sorted = scores[sorted_indices]

        return _compute_ap(scores_sorted, tp_sorted, num_gt)

    # ---- reset ------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._scores = [[] for _ in range(self.num_classes)]
        self._tp = [
            [[] for _ in range(self.num_classes)]
            for _ in self.iou_thresholds
        ]
        self._num_gt = [0] * self.num_classes
        self._confusion = [
            np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for _ in self.iou_thresholds
        ]


# ---------------------------------------------------------------------------
# Panoptic Quality
# ---------------------------------------------------------------------------

class PanopticQuality:
    """Panoptic Quality metric (PQ = DQ x SQ) for instance detection.

    Accumulates per-class TP/FP/FN counts and sum of matched IoUs across
    batches, then computes PQ, DQ (detection quality), and SQ (segmentation
    quality, here measured as box IoU of matched pairs).

    Operates at a single IoU threshold (conventionally 0.5).
    """

    def __init__(self, num_classes: int, iou_threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self._tp = np.zeros(num_classes, dtype=np.int64)
        self._fp = np.zeros(num_classes, dtype=np.int64)
        self._fn = np.zeros(num_classes, dtype=np.int64)
        self._iou_sum = np.zeros(num_classes, dtype=np.float64)

    def update(
        self,
        predictions: list[dict],
        targets: list[DetectionTarget],
    ) -> None:
        """Accumulate one batch."""
        for pred, target in zip(predictions, targets):
            self._update_single(pred, target)

    def _update_single(self, pred: dict, target: DetectionTarget) -> None:
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        pred_masks = pred.get("masks")  # (P, H, W) binary or None
        gt_boxes_xyxy = cxcywh_to_xyxy(target.boxes)
        gt_labels = target.labels
        gt_masks = target.masks  # (G, H, W) binary or None

        # Use mask IoU when both sides have masks
        use_masks = (
            pred_masks is not None and gt_masks is not None
            and len(pred_masks) > 0 and len(gt_masks) > 0
        )

        if len(pred_boxes) == 0:
            for lbl in gt_labels:
                cls = lbl.item()
                if 0 <= cls < self.num_classes:
                    self._fn[cls] += 1
            return

        if len(gt_boxes_xyxy) == 0:
            for lbl in pred_labels:
                cls = lbl.item()
                if 0 <= cls < self.num_classes:
                    self._fp[cls] += 1
            return

        if use_masks:
            iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
        else:
            iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)

        for cls_id in range(self.num_classes):
            pred_mask = pred_labels == cls_id
            gt_mask = gt_labels == cls_id
            n_gt_cls = int(gt_mask.sum().item())

            if not pred_mask.any():
                self._fn[cls_id] += n_gt_cls
                continue
            if not gt_mask.any():
                self._fp[cls_id] += int(pred_mask.sum().item())
                continue

            p_idx = torch.where(pred_mask)[0]
            g_idx = torch.where(gt_mask)[0]
            cls_iou = iou_matrix[p_idx][:, g_idx]
            cls_scores = pred_scores[pred_mask]

            sorted_order = cls_scores.argsort(descending=True)
            matched_gt = torch.zeros(len(g_idx), dtype=torch.bool, device=pred_boxes.device)
            matched_pred = torch.zeros(len(p_idx), dtype=torch.bool, device=pred_boxes.device)

            for rank in sorted_order:
                avail = cls_iou[rank].clone()
                avail[matched_gt] = -1.0
                best = avail.argmax()
                if avail[best] >= self.iou_threshold:
                    matched_pred[rank] = True
                    matched_gt[best] = True
                    self._tp[cls_id] += 1
                    self._iou_sum[cls_id] += avail[best].item()

            self._fp[cls_id] += int((~matched_pred).sum().item())
            self._fn[cls_id] += int((~matched_gt).sum().item())

    def compute(self) -> dict[str, float]:
        """Compute PQ, DQ, SQ per-class and aggregate.

        Returns dict with keys:
            - "PQ", "DQ", "SQ" (macro-averaged)
            - "PQ_class{c}", "DQ_class{c}", "SQ_class{c}"
        """
        results: dict[str, float] = {}
        pqs, dqs, sqs = [], [], []

        for cls_id in range(self.num_classes):
            tp = self._tp[cls_id]
            fp = self._fp[cls_id]
            fn = self._fn[cls_id]

            dq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-6)
            sq = self._iou_sum[cls_id] / max(tp, 1)
            pq = dq * sq

            results[f"PQ_class{cls_id}"] = float(pq)
            results[f"DQ_class{cls_id}"] = float(dq)
            results[f"SQ_class{cls_id}"] = float(sq)
            pqs.append(pq)
            dqs.append(dq)
            sqs.append(sq)

        results["PQ"] = float(np.mean(pqs))
        results["DQ"] = float(np.mean(dqs))
        results["SQ"] = float(np.mean(sqs))
        return results

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._tp[:] = 0
        self._fp[:] = 0
        self._fn[:] = 0
        self._iou_sum[:] = 0.0


# ---------------------------------------------------------------------------
# Tissue-stratified PQ (mPQ / bPQ) — official PanNuke protocol
# ---------------------------------------------------------------------------

class TissuePanopticQuality:
    """Official PanNuke mPQ and bPQ: PQ computed per tissue type then averaged.

    mPQ: per-class PQ averaged within each tissue, then across tissues.
    bPQ: binary PQ (all classes as one) per tissue, then across tissues.

    Maintains one set of TP/FP/FN/IoU accumulators per (tissue, class) pair.
    """

    def __init__(
        self,
        num_classes: int,
        num_tissues: int = 19,
        iou_threshold: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.num_tissues = num_tissues
        self.iou_threshold = iou_threshold
        # Per (tissue, class) accumulators
        self._tp = np.zeros((num_tissues, num_classes), dtype=np.int64)
        self._fp = np.zeros((num_tissues, num_classes), dtype=np.int64)
        self._fn = np.zeros((num_tissues, num_classes), dtype=np.int64)
        self._iou_sum = np.zeros((num_tissues, num_classes), dtype=np.float64)
        # Binary (tissue-only, all classes as one)
        self._tp_bin = np.zeros(num_tissues, dtype=np.int64)
        self._fp_bin = np.zeros(num_tissues, dtype=np.int64)
        self._fn_bin = np.zeros(num_tissues, dtype=np.int64)
        self._iou_sum_bin = np.zeros(num_tissues, dtype=np.float64)

    def update(
        self,
        predictions: list[dict],
        targets: list[DetectionTarget],
        tissue_ids: list[int],
    ) -> None:
        """Accumulate one batch with tissue type annotations."""
        for pred, target, tid in zip(predictions, targets, tissue_ids):
            self._update_single(pred, target, tid)

    def _update_single(
        self, pred: dict, target: DetectionTarget, tissue_id: int,
    ) -> None:
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        pred_masks = pred.get("masks")
        gt_boxes_xyxy = cxcywh_to_xyxy(target.boxes)
        gt_labels = target.labels
        gt_masks = target.masks

        use_masks = (
            pred_masks is not None and gt_masks is not None
            and len(pred_masks) > 0 and len(gt_masks) > 0
        )

        t = tissue_id

        if len(pred_boxes) == 0:
            for lbl in gt_labels:
                cls = lbl.item()
                if 0 <= cls < self.num_classes:
                    self._fn[t, cls] += 1
            self._fn_bin[t] += len(gt_labels)
            return

        if len(gt_boxes_xyxy) == 0:
            for lbl in pred_labels:
                cls = lbl.item()
                if 0 <= cls < self.num_classes:
                    self._fp[t, cls] += 1
            self._fp_bin[t] += len(pred_labels)
            return

        if use_masks:
            iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
        else:
            iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)

        # --- Per-class matching (for mPQ) ---
        for cls_id in range(self.num_classes):
            pred_mask = pred_labels == cls_id
            gt_mask = gt_labels == cls_id
            n_gt_cls = int(gt_mask.sum().item())

            if not pred_mask.any():
                self._fn[t, cls_id] += n_gt_cls
                continue
            if not gt_mask.any():
                self._fp[t, cls_id] += int(pred_mask.sum().item())
                continue

            p_idx = torch.where(pred_mask)[0]
            g_idx = torch.where(gt_mask)[0]
            cls_iou = iou_matrix[p_idx][:, g_idx]
            cls_scores = pred_scores[pred_mask]

            sorted_order = cls_scores.argsort(descending=True)
            matched_gt = torch.zeros(len(g_idx), dtype=torch.bool,
                                     device=pred_boxes.device)
            matched_pred = torch.zeros(len(p_idx), dtype=torch.bool,
                                       device=pred_boxes.device)

            for rank in sorted_order:
                avail = cls_iou[rank].clone()
                avail[matched_gt] = -1.0
                best = avail.argmax()
                if avail[best] >= self.iou_threshold:
                    matched_pred[rank] = True
                    matched_gt[best] = True
                    self._tp[t, cls_id] += 1
                    self._iou_sum[t, cls_id] += avail[best].item()

            self._fp[t, cls_id] += int((~matched_pred).sum().item())
            self._fn[t, cls_id] += int((~matched_gt).sum().item())

        # --- Binary matching (for bPQ, class-agnostic) ---
        sorted_all = pred_scores.argsort(descending=True)
        matched_gt_bin = torch.zeros(len(gt_boxes_xyxy), dtype=torch.bool,
                                     device=pred_boxes.device)
        matched_pred_bin = torch.zeros(len(pred_boxes), dtype=torch.bool,
                                       device=pred_boxes.device)

        for rank in sorted_all:
            avail = iou_matrix[rank].clone()
            avail[matched_gt_bin] = -1.0
            best = avail.argmax()
            if avail[best] >= self.iou_threshold:
                matched_pred_bin[rank] = True
                matched_gt_bin[best] = True
                self._tp_bin[t] += 1
                self._iou_sum_bin[t] += avail[best].item()

        self._fp_bin[t] += int((~matched_pred_bin).sum().item())
        self._fn_bin[t] += int((~matched_gt_bin).sum().item())

    def compute(self) -> dict[str, float]:
        """Compute mPQ, bPQ, and per-class/tissue breakdowns.

        Returns dict with keys:
            - "mPQ": official multi-class PQ (primary ranking metric)
            - "bPQ": binary PQ
            - "mPQ_tissue{t}": per-tissue mPQ
            - "bPQ_tissue{t}": per-tissue bPQ
            - "PQ_class{c}": per-class PQ (macro over tissues)
        """
        results: dict[str, float] = {}

        # Per-tissue mPQ: average of per-class PQ within each tissue
        tissue_mpqs: list[float] = []
        tissue_bpqs: list[float] = []

        for t in range(self.num_tissues):
            # Per-class PQ for this tissue
            class_pqs: list[float] = []
            for c in range(self.num_classes):
                tp = self._tp[t, c]
                fp = self._fp[t, c]
                fn = self._fn[t, c]
                if tp + fp + fn == 0:
                    continue  # class not present in this tissue
                dq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-6)
                sq = self._iou_sum[t, c] / max(tp, 1)
                class_pqs.append(dq * sq)

            tissue_mpq = sum(class_pqs) / max(len(class_pqs), 1)
            tissue_mpqs.append(tissue_mpq)
            results[f"mPQ_tissue{t}"] = tissue_mpq

            # Binary PQ for this tissue
            tp_b = self._tp_bin[t]
            fp_b = self._fp_bin[t]
            fn_b = self._fn_bin[t]
            if tp_b + fp_b + fn_b == 0:
                tissue_bpqs.append(0.0)
            else:
                dq_b = tp_b / max(tp_b + 0.5 * fp_b + 0.5 * fn_b, 1e-6)
                sq_b = self._iou_sum_bin[t] / max(tp_b, 1)
                tissue_bpqs.append(dq_b * sq_b)
            results[f"bPQ_tissue{t}"] = tissue_bpqs[-1]

        results["mPQ"] = float(np.mean(tissue_mpqs))
        results["bPQ"] = float(np.mean(tissue_bpqs))

        # Per-class PQ (macro across tissues with the class present)
        for c in range(self.num_classes):
            class_pqs_across_tissues: list[float] = []
            for t in range(self.num_tissues):
                tp = self._tp[t, c]
                fp = self._fp[t, c]
                fn = self._fn[t, c]
                if tp + fp + fn == 0:
                    continue
                dq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-6)
                sq = self._iou_sum[t, c] / max(tp, 1)
                class_pqs_across_tissues.append(dq * sq)
            results[f"PQ_class{c}"] = (
                float(np.mean(class_pqs_across_tissues))
                if class_pqs_across_tissues else 0.0
            )

        return results

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._tp[:] = 0
        self._fp[:] = 0
        self._fn[:] = 0
        self._iou_sum[:] = 0.0
        self._tp_bin[:] = 0
        self._fp_bin[:] = 0
        self._fn_bin[:] = 0
        self._iou_sum_bin[:] = 0.0


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    scores: torch.Tensor,
    tp_flags: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error for detection confidence scores.

    Args:
        scores: (N,) predicted confidence scores.
        tp_flags: (N,) boolean indicating true-positive.
        n_bins: number of equal-width bins.

    Returns:
        ECE value in [0, 1].
    """
    if len(scores) == 0:
        return 0.0

    scores = scores.detach().float()
    tp = tp_flags.detach().float()
    n = len(scores)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=scores.device)
    ece = 0.0

    for i in range(n_bins):
        mask = (scores > bin_boundaries[i]) & (scores <= bin_boundaries[i + 1])
        if i == 0:
            mask = mask | (scores == bin_boundaries[i])
        count = mask.sum().item()
        if count == 0:
            continue
        avg_conf = scores[mask].mean().item()
        avg_acc = tp[mask].mean().item()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return ece


# ---------------------------------------------------------------------------
# Segmentation Metrics
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Per-class and mean IoU for binary segmentation masks.

    Accumulates intersection and union counts across batches for each class.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._intersection = torch.zeros(num_classes, dtype=torch.float64)
        self._union = torch.zeros(num_classes, dtype=torch.float64)

    def update(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> None:
        """Accumulate mask IoU for matched prediction/target pairs.

        Args:
            pred_masks: (N, H, W) predicted binary masks (or logits > 0).
            target_masks: (N, H, W) ground-truth binary masks.
            pred_labels: (N,) class labels for predicted masks.
            target_labels: (N,) class labels for target masks.

        Pairs are assumed pre-matched (e.g. via Hungarian matching), so
        pred_masks[i] corresponds to target_masks[i].
        """
        if len(pred_masks) == 0:
            return

        pred_bin = (pred_masks > 0.5).bool()
        target_bin = (target_masks > 0.5).bool()

        for i in range(len(pred_bin)):
            cls = target_labels[i].item()
            if not (0 <= cls < self.num_classes):
                continue

            intersection = (pred_bin[i] & target_bin[i]).sum().item()
            union = (pred_bin[i] | target_bin[i]).sum().item()

            self._intersection[cls] += intersection
            self._union[cls] += union

    def compute(self) -> dict[str, float]:
        """Compute per-class IoU and mean IoU.

        Returns dict with keys:
            - "IoU_class{c}" for each class
            - "mIoU" (mean over classes with > 0 union)
        """
        results: dict[str, float] = {}
        valid_ious: list[float] = []

        for cls in range(self.num_classes):
            if self._union[cls] > 0:
                iou = (self._intersection[cls] / self._union[cls]).item()
            else:
                iou = 0.0
            results[f"IoU_class{cls}"] = iou
            if self._union[cls] > 0:
                valid_ious.append(iou)

        results["mIoU"] = sum(valid_ious) / max(len(valid_ious), 1)
        return results

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._intersection.zero_()
        self._union.zero_()
