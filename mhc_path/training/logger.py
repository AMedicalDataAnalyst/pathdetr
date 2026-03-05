"""Logging & Monitoring for mHC-Path training."""

from __future__ import annotations

import json
import logging
import threading
import warnings
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch

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
        self._write({"step": step, "image_tag": tag, "shape": list(image.shape)})

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._write({"step": step, "histogram_tag": tag, "numel": values.numel()})

    def flush(self) -> None:
        pass

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
