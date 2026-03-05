"""Detection Training Engine.

Training loop for detection + segmentation with Hungarian matching.
AdamW optimizer with separate FPN and decoder parameter groups.
GPU augmentation pipeline runs between DataLoader and forward pass.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mhc_path.data.dataset import detection_collate_fn
from mhc_path.training.losses import DetectionLoss, DetectionTarget, HungarianMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """Configuration for detection training."""
    epochs: int = 50
    batch_size: int = 8
    lr_fpn: float = 1e-4
    lr_decoder: float = 1e-4
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.9999
    use_amp: bool = True
    output_dir: str = "checkpoints/detection"
    num_workers: int = 0
    log_interval: int = 10

# ---------------------------------------------------------------------------
# Logger protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class _Logger(Protocol):
    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None: ...
    def flush(self) -> None: ...

class _NullLogger:
    """No-op logger when none provided."""
    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None: pass
    def flush(self) -> None: pass

# ---------------------------------------------------------------------------
# EMA model helper
# ---------------------------------------------------------------------------

class _EMAModel:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(model_p.data, alpha=1.0 - d)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.module.load_state_dict(sd)

# ---------------------------------------------------------------------------
# Parameter group builder
# ---------------------------------------------------------------------------

def _build_param_groups(
    model: nn.Module, config: DetectionConfig,
) -> list[dict[str, Any]]:
    """Split trainable params into FPN and decoder groups."""
    fpn: list[nn.Parameter] = []
    decoder: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fpn" in name.lower():
            fpn.append(p)
        else:
            decoder.append(p)

    groups: list[dict[str, Any]] = []
    if fpn:
        groups.append({"params": fpn, "lr": config.lr_fpn,
                       "weight_decay": config.weight_decay})
    if decoder:
        groups.append({"params": decoder, "lr": config.lr_decoder,
                       "weight_decay": config.weight_decay})
    return groups

# ---------------------------------------------------------------------------
# Detection Training Engine
# ---------------------------------------------------------------------------

class DetectionEngine:
    """Training loop for detection + segmentation with Hungarian matching."""

    def __init__(self, model: nn.Module, train_dataset: Dataset,
                 val_dataset: Dataset, gpu_aug: Optional[nn.Module],
                 config: DetectionConfig,
                 criterion: Optional[DetectionLoss] = None,
                 metrics: Optional[Any] = None,
                 log: Optional[Any] = None) -> None:
        self.model = model
        self.gpu_aug = gpu_aug
        self.config = config
        self._logger: Any = log or _NullLogger()
        self.device = next(model.parameters()).device

        if criterion is not None:
            self.criterion = criterion
        else:
            nc = self._infer_num_classes(model)
            self.criterion = DetectionLoss(num_classes=nc, matcher=HungarianMatcher())

        self.metrics = metrics
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
        self.ema = _EMAModel(model, decay=config.ema_decay)

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, collate_fn=detection_collate_fn,
            drop_last=False)
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=detection_collate_fn,
            drop_last=False)
        self.grad_norms: list[float] = []

    # -- helpers --

    @staticmethod
    def _infer_num_classes(model: nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_classes"):
            return model.config.num_classes
        if hasattr(model, "num_classes"):
            return model.num_classes
        return 5

    def _build_optimizer(self) -> torch.optim.AdamW:
        groups = _build_param_groups(self.model, self.config)
        return torch.optim.AdamW(
            groups, lr=self.config.lr_decoder,
            weight_decay=self.config.weight_decay)

    def _clip_gradients(self) -> float:
        """Clip gradients by norm; records post-clipping norm."""
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(params, self.config.clip_grad_norm)
        post_norm = torch.cat([p.grad.flatten() for p in params]).norm().item()
        self.grad_norms.append(post_norm)
        return post_norm

    def _optimizer_step(self) -> None:
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _batch_to_targets(self, batch: dict[str, Any]) -> list[DetectionTarget]:
        boxes, labels = batch["boxes"], batch["labels"]
        num_obj = batch["num_objects"]
        masks = batch.get("masks")
        targets: list[DetectionTarget] = []

        if isinstance(boxes, (list, tuple)):
            for i, box_t in enumerate(boxes):
                n = num_obj[i].item()
                lbl = labels[i, :n] if not isinstance(labels, list) else labels[i][:n]
                m = None
                if masks is not None:
                    m = masks[i, :n] if not isinstance(masks, list) else masks[i][:n]
                targets.append(DetectionTarget(
                    boxes=box_t[:n].to(self.device),
                    labels=lbl.to(self.device),
                    masks=m.to(self.device) if m is not None else None,
                ))
        else:
            for i in range(boxes.shape[0]):
                n = num_obj[i].item()
                targets.append(DetectionTarget(
                    boxes=boxes[i, :n].to(self.device),
                    labels=labels[i, :n].to(self.device),
                    masks=masks[i, :n].to(self.device) if masks is not None else None,
                ))
        return targets

    def _normalize_output(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalise model output keys for the loss module."""
        out: dict[str, torch.Tensor] = {}
        out["pred_logits"] = raw["pred_logits"] if "pred_logits" in raw else raw["class_logits"]
        out["pred_boxes"] = raw["pred_boxes"] if "pred_boxes" in raw else raw["box_coords"]
        if "pred_masks" in raw:
            out["pred_masks"] = raw["pred_masks"]
        elif "mask_logits" in raw:
            out["pred_masks"] = raw["mask_logits"]
        if "aux_outputs" in raw:
            out["aux_outputs"] = raw["aux_outputs"]
        return out

    # -- training --

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns mean loss components."""
        self.model.train()
        running: dict[str, float] = {}
        count = 0

        for batch in self.train_loader:
            batch["images"] = batch["images"].to(self.device)
            if self.gpu_aug is not None:
                batch = self.gpu_aug(batch)

            targets = self._batch_to_targets(batch)
            self.optimizer.zero_grad(set_to_none=True)

            amp_dev = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_dev, enabled=self.config.use_amp):
                outputs = self._normalize_output(self.model(batch["images"]))
                losses = self.criterion(outputs, targets)

            (self.scaler.scale(losses["total"]).backward() if self.config.use_amp
             else losses["total"].backward())
            self._clip_gradients()
            self._optimizer_step()
            self.ema.update(self.model)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
            count += 1

        metrics = {k: v / max(count, 1) for k, v in running.items()}
        metrics["epoch"] = float(epoch)
        self._logger.log_scalars(metrics, step=epoch, prefix="train")
        return metrics

    # -- evaluation --

    @torch.no_grad()
    def evaluate(self, use_ema: bool = False) -> dict[str, float]:
        """Evaluate on the validation set.

        Args:
            use_ema: If True, evaluate the EMA model. If False (default),
                evaluate the raw student model.
        """
        eval_model = self.ema.module if use_ema else self.model
        eval_model.eval()
        running: dict[str, float] = {}
        count = 0
        if self.metrics is not None:
            self.metrics.reset()

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            outputs = self._normalize_output(eval_model(images))
            targets = self._batch_to_targets(batch)
            losses = self.criterion(outputs, targets)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
            count += 1
            if self.metrics is not None:
                self.metrics.update(self._outputs_to_predictions(outputs), targets)

        metrics = {f"val_{k}": v / max(count, 1) for k, v in running.items()}
        if self.metrics is not None:
            metrics.update(self.metrics.compute())
        return metrics

    @staticmethod
    def _outputs_to_predictions(outputs: dict[str, torch.Tensor]) -> list[dict]:
        from mhc_path.models.box_utils import cxcywh_to_xyxy
        preds: list[dict] = []
        for b in range(outputs["pred_logits"].shape[0]):
            scores, labels = outputs["pred_logits"][b].sigmoid().max(dim=-1)
            preds.append({"boxes": cxcywh_to_xyxy(outputs["pred_boxes"][b]),
                          "scores": scores, "labels": labels})
        return preds

    # -- full loop --

    def train(self) -> None:
        """Run the full training loop for config.epochs."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        best = -float("inf")
        for epoch in range(self.config.epochs):
            tm = self.train_epoch(epoch)
            vm = self.evaluate()
            logger.info("Epoch %d | train=%.4f | val=%.4f",
                        epoch, tm.get("total", 0.0), vm.get("val_total", 0.0))
            self._logger.log_scalars(vm, step=epoch, prefix="val")
            primary = vm.get("mAP@50", -vm.get("val_total", 0.0))
            if primary > best:
                best = primary
                self._save_checkpoint(epoch, "best.pt")
        self._save_checkpoint(self.config.epochs - 1, "final.pt")

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        path = Path(self.config.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, str(path))
        logger.info("Saved checkpoint to %s", path)
