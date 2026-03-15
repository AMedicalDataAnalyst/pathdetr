"""Step 14: Logging & Monitoring for mHC-Path SSL training."""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Logger(Protocol):
    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None: ...
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None: ...
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: ...
    def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# JSONLinesLogger
# ---------------------------------------------------------------------------

class JSONLinesLogger:
    """Minimal logger for CI and headless environments. One JSON object per line."""

    def __init__(self, filepath: str) -> None:
        self._path = Path(filepath)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        record: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            record[key] = v
        self._write(record)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        # Images cannot be serialised to JSONL; log metadata only.
        self._write({"step": step, "image_tag": tag, "shape": list(image.shape)})

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._write({"step": step, "histogram_tag": tag, "numel": values.numel()})

    def flush(self) -> None:
        pass  # Each write already flushes.

    # -- internals --

    def _write(self, record: dict) -> None:
        with self._lock:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------

class WandbLogger:
    """Wraps wandb API. Falls back to no-op if wandb is not installed."""

    def __init__(self, project: str, config: dict, run_name: Optional[str] = None) -> None:
        self._wandb: Any = None
        self._lock = threading.Lock()
        try:
            import wandb  # type: ignore[import-untyped]
            self._wandb = wandb
            wandb.init(project=project, config=config, name=run_name)
        except ImportError:
            warnings.warn(
                "wandb is not installed. WandbLogger will behave as a no-op logger.",
                stacklevel=2,
            )

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        if self._wandb is None:
            return
        payload = {(f"{prefix}/{k}" if prefix else k): v for k, v in metrics.items()}
        with self._lock:
            self._wandb.log(payload, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self._wandb is None:
            return
        with self._lock:
            self._wandb.log({tag: self._wandb.Image(image.cpu().numpy())}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self._wandb is None:
            return
        with self._lock:
            self._wandb.log({tag: self._wandb.Histogram(values.cpu().numpy())}, step=step)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------

class TensorBoardLogger:
    """Wraps tensorboard SummaryWriter. No-op if tensorboard is not installed."""

    def __init__(self, log_dir: str) -> None:
        self._writer: Any = None
        self._lock = threading.Lock()
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            warnings.warn(
                "tensorboard is not installed. TensorBoardLogger will behave as a no-op.",
                stacklevel=2,
            )

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        if self._writer is None:
            return
        with self._lock:
            for k, v in metrics.items():
                tag = f"{prefix}/{k}" if prefix else k
                self._writer.add_scalar(tag, v, global_step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self._writer is None:
            return
        with self._lock:
            self._writer.add_image(tag, image, global_step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self._writer is None:
            return
        with self._lock:
            self._writer.add_histogram(tag, values, global_step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()


# ---------------------------------------------------------------------------
# CompositeLogger
# ---------------------------------------------------------------------------

class CompositeLogger:
    """Fans out to multiple loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = list(loggers)

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        for lg in self._loggers:
            lg.log_scalars(metrics, step, prefix)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        for lg in self._loggers:
            lg.log_image(tag, image, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        for lg in self._loggers:
            lg.log_histogram(tag, values, step)

    def flush(self) -> None:
        for lg in self._loggers:
            lg.flush()


# ---------------------------------------------------------------------------
# SSLHealthMonitor
# ---------------------------------------------------------------------------

class SSLHealthMonitor:
    """Computes and logs SSL-specific health metrics every iteration.

    Tracks per-component loss, teacher-student CLS cosine similarity,
    gradient norms, and throughput.
    """

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._last_time: Optional[float] = None

    @torch.no_grad()
    def on_iteration(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        loss_components: dict[str, float],
        model: nn.Module,
        step: int,
    ) -> None:
        metrics: dict[str, float] = {}

        # Per-component losses
        for name, val in loss_components.items():
            metrics[f"loss/{name}"] = float(val)

        # CLS cosine similarity between student and teacher
        s_cls = student_out[:, 0] if student_out.dim() == 3 else student_out
        t_cls = teacher_out[:, 0] if teacher_out.dim() == 3 else teacher_out
        cos = nn.functional.cosine_similarity(s_cls, t_cls, dim=-1).mean()
        metrics["ssl/cls_cosine_sim"] = cos.item()

        # Gradient norms per parameter group (first + last named param as proxy)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        metrics["ssl/grad_norm"] = total_norm ** 0.5

        # Throughput
        now = time.monotonic()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                metrics["ssl/iter_per_sec"] = 1.0 / elapsed
        self._last_time = now

        self._logger.log_scalars(metrics, step, prefix="health")


# ---------------------------------------------------------------------------
# SSLQualityProbe
# ---------------------------------------------------------------------------

class SSLQualityProbe:
    """Periodic representation quality evaluation for SSL."""

    def __init__(
        self,
        backbone: nn.Module,
        val_dataset: Dataset,
        logger: Logger,
        eval_every: int = 5,
        num_classes: int = 5,
    ) -> None:
        self._backbone = backbone
        self._val_dataset = val_dataset
        self._logger = logger
        self._eval_every = eval_every
        self._num_classes = num_classes

    # -- feature extraction (shared) --

    @torch.no_grad()
    def _extract_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._backbone.eval()
        loader = DataLoader(self._val_dataset, batch_size=64, shuffle=False)
        feats_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for batch in loader:
            x, y = batch
            out = self._backbone(x)
            # Handle sequence output (take CLS) or flat feature vector
            if out.dim() == 3:
                out = out[:, 0]
            feats_list.append(out.cpu())
            labels_list.append(y.cpu())
        return torch.cat(feats_list), torch.cat(labels_list)

    # -- k-NN evaluation --

    def evaluate_knn(self, epoch: int, k: int = 5) -> float:
        feats, labels = self._extract_features()
        n = feats.size(0)
        # Brute-force k-NN via cdist
        dists = torch.cdist(feats, feats)  # (N, N)
        # Exclude self-distance by setting diagonal to inf
        dists.fill_diagonal_(float("inf"))
        _, indices = dists.topk(k, largest=False)  # (N, k)
        neighbour_labels = labels[indices]  # (N, k)
        # Majority vote
        preds = torch.zeros(n, dtype=labels.dtype)
        for i in range(n):
            counts = torch.bincount(neighbour_labels[i], minlength=self._num_classes)
            preds[i] = counts.argmax()
        acc = (preds == labels).float().mean().item()
        self._logger.log_scalars({"knn_accuracy": acc}, step=epoch, prefix="probe")
        return acc

    # -- linear probe evaluation --

    def evaluate_linear_probe(self, epoch: int) -> float:
        feats, labels = self._extract_features()
        dim = feats.size(1)
        head = nn.Linear(dim, self._num_classes)
        optimizer = torch.optim.SGD(head.parameters(), lr=1e-1)
        loss_fn = nn.CrossEntropyLoss()

        # 50-step training loop on frozen features
        for _ in range(50):
            optimizer.zero_grad()
            logits = head(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = head(feats).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        self._logger.log_scalars({"linear_probe_accuracy": acc}, step=epoch, prefix="probe")
        return acc

    # -- PCA visualisation --

    @torch.no_grad()
    def visualize_pca(self, epoch: int) -> None:
        feats, _ = self._extract_features()
        # Centre and compute top-3 PCA components
        feats = feats - feats.mean(dim=0)
        _, _, V = torch.linalg.svd(feats, full_matrices=False)
        projected = feats @ V[:3].T  # (N, 3)
        self._logger.log_histogram("probe/pca_components", projected, step=epoch)
