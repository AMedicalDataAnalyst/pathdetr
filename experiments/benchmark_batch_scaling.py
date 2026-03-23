"""Benchmark GPU memory and throughput across batch sizes with linear LR scaling.

Runs a few training steps at each batch size to measure:
  - Peak GPU memory (allocated + reserved)
  - Forward/backward throughput (images/sec)
  - Gradient norms (to verify LR scaling stability)
  - Memory breakdown: model params, activations, optimizer state

Linear LR scaling rule: when batch size doubles, LR doubles. The reference
point is bs=32 / lr_decoder=1e-3 / lr_fpn=5e-4 (our validated config).

Usage:
    python -m experiments.benchmark_batch_scaling [--batch_sizes 32 64 128 256]
    python -m experiments.benchmark_batch_scaling --backbone phikon_v2
    python -m experiments.benchmark_batch_scaling --num_queries 300 --group_detr 3
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from mhc_path.models.mhc_path import MHCPath, MHCPathConfig
from mhc_path.training.losses import DetectionLoss, DetectionTarget, HungarianMatcher

_NUM_CLASSES = 5
# Reference config: bs=32 was our validated baseline
_REF_BATCH_SIZE = 32
_REF_LR_DECODER = 1e-3
_REF_LR_FPN = 5e-4


def _gpu_mem_gb() -> dict[str, float]:
    """Current GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / (1024**3),
        "reserved": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated": torch.cuda.max_memory_allocated() / (1024**3),
    }


def _synthetic_batch(
    batch_size: int, num_objects: int = 15, image_size: int = 256,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, list[DetectionTarget]]:
    """Generate a synthetic batch without touching disk or DataLoader."""
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = []
    for _ in range(batch_size):
        n = num_objects
        # Random boxes in cxcywh format, normalised to [0,1]
        cx = torch.rand(n, device=device) * 0.8 + 0.1
        cy = torch.rand(n, device=device) * 0.8 + 0.1
        w = torch.rand(n, device=device) * 0.1 + 0.02
        h = torch.rand(n, device=device) * 0.1 + 0.02
        boxes = torch.stack([cx, cy, w, h], dim=-1)
        labels = torch.randint(0, _NUM_CLASSES, (n,), device=device)
        # Masks at native mask resolution
        masks = torch.randint(0, 2, (n, image_size, image_size),
                              device=device, dtype=torch.float32)
        targets.append(DetectionTarget(boxes=boxes, labels=labels, masks=masks))
    return images, targets


def _normalize_output(raw: dict) -> dict:
    out = {}
    out["pred_logits"] = raw.get("pred_logits", raw.get("class_logits"))
    out["pred_boxes"] = raw.get("pred_boxes", raw.get("box_coords"))
    if "pred_masks" in raw:
        out["pred_masks"] = raw["pred_masks"]
    elif "mask_logits" in raw:
        out["pred_masks"] = raw["mask_logits"]
    if "aux_outputs" in raw:
        out["aux_outputs"] = raw["aux_outputs"]
    return out


def _param_mem_gb(model: nn.Module) -> float:
    """Total parameter memory in GB (includes frozen params)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024**3)


def _trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_optimizer(
    model: nn.Module, lr_fpn: float, lr_decoder: float, weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW with per-group LRs matching train_pannuke.py."""
    backbone, fpn, decoder = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lo = name.lower()
        if "backbone" in lo:
            backbone.append(p)
        elif "fpn" in lo:
            fpn.append(p)
        else:
            decoder.append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": 0.0, "weight_decay": weight_decay})
    if fpn:
        groups.append({"params": fpn, "lr": lr_fpn, "weight_decay": weight_decay})
    if decoder:
        groups.append({"params": decoder, "lr": lr_decoder, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, lr=lr_decoder, weight_decay=weight_decay)


def benchmark_batch_size(
    batch_size: int,
    model_cfg: MHCPathConfig,
    mask_loss_resolution: int,
    weight_decay: float,
    n_warmup: int = 2,
    n_measure: int = 5,
) -> dict:
    """Run n_warmup + n_measure train steps and collect stats."""
    device = torch.device("cuda")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Model ---
    model = MHCPath(model_cfg).to(device)
    model.train()
    model_mem = _param_mem_gb(model)
    trainable = _trainable_param_count(model)
    after_model = _gpu_mem_gb()

    # --- Linear LR scaling ---
    lr_scale = batch_size / _REF_BATCH_SIZE
    lr_decoder = _REF_LR_DECODER * lr_scale
    lr_fpn = _REF_LR_FPN * lr_scale

    optimizer = _build_optimizer(model, lr_fpn, lr_decoder, weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    criterion = DetectionLoss(
        num_classes=_NUM_CLASSES, matcher=HungarianMatcher(),
        box_loss_type="giou", mask_loss_resolution=mask_loss_resolution,
        group_detr=model_cfg.group_detr,
    )

    after_optim = _gpu_mem_gb()

    # --- Warmup (let CUDA caching allocator settle) ---
    for _ in range(n_warmup):
        images, targets = _synthetic_batch(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = _normalize_output(model(images))
            losses = criterion(outputs, targets)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        del images, targets, outputs, losses

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # --- Measure ---
    step_times = []
    grad_norms = {"fpn": [], "decoder": [], "backbone": []}
    losses_acc = []

    for step in range(n_measure):
        images, targets = _synthetic_batch(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda"):
            outputs = _normalize_output(model(images))
            losses = criterion(outputs, targets)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Collect grad norms per group
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            gn = p.grad.data.norm(2).item()
            lo = name.lower()
            if "fpn" in lo:
                grad_norms["fpn"].append(gn)
            elif "backbone" in lo:
                grad_norms["backbone"].append(gn)
            else:
                grad_norms["decoder"].append(gn)

        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses_acc.append(losses["total"].item())

        del images, targets, outputs, losses

    peak = _gpu_mem_gb()

    # Compute optimizer state memory (after first step, buffers are allocated)
    optim_state_bytes = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                optim_state_bytes += v.numel() * v.element_size()
    optim_state_gb = optim_state_bytes / (1024**3)

    # Activation memory = peak - model - optimizer state
    activation_mem = peak["peak_allocated"] - after_optim["allocated"]

    mean_time = sum(step_times) / len(step_times)
    throughput = batch_size / mean_time

    result = {
        "batch_size": batch_size,
        "lr_decoder": lr_decoder,
        "lr_fpn": lr_fpn,
        "lr_scale": lr_scale,
        "model_params_gb": round(model_mem, 3),
        "trainable_params": trainable,
        "optimizer_state_gb": round(optim_state_gb, 3),
        "activation_mem_gb": round(max(0, activation_mem), 3),
        "peak_allocated_gb": round(peak["peak_allocated"], 3),
        "peak_reserved_gb": round(peak["reserved"], 3),
        "mean_step_time_s": round(mean_time, 4),
        "throughput_img_per_s": round(throughput, 1),
        "mean_loss": round(sum(losses_acc) / len(losses_acc), 4),
        "mean_grad_norm_fpn": round(
            max(grad_norms["fpn"]) if grad_norms["fpn"] else 0.0, 4),
        "mean_grad_norm_decoder": round(
            max(grad_norms["decoder"]) if grad_norms["decoder"] else 0.0, 4),
    }

    # Cleanup
    del model, optimizer, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU memory and throughput across batch sizes")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Batch sizes to test (default: 32 64 128 256)")
    parser.add_argument("--backbone", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vitl16", "dinov3_vitb16", "dinov3_vitg14", "phikon_v2"])
    parser.add_argument("--num_queries", type=int, default=300)
    parser.add_argument("--group_detr", type=int, default=3)
    parser.add_argument("--mask_upsample_factor", type=int, default=4)
    parser.add_argument("--mask_loss_resolution", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_measure", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="experiments/batch_scaling")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    device_name = torch.cuda.get_device_name(0)
    device_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"GPU: {device_name} ({device_mem_gb:.1f} GB)")
    print(f"Backbone: {args.backbone}")
    print(f"Queries: {args.num_queries}, Group DETR: {args.group_detr}")
    print(f"Mask: upsample={args.mask_upsample_factor}, loss_res={args.mask_loss_resolution}")
    print(f"LR scaling rule: linear from bs={_REF_BATCH_SIZE} "
          f"(lr_dec={_REF_LR_DECODER}, lr_fpn={_REF_LR_FPN})")
    print()

    model_cfg = MHCPathConfig(
        backbone=args.backbone, backbone_frozen=True, lora_rank=None,
        fpn_dim=256, num_queries=args.num_queries, num_classes=_NUM_CLASSES,
        num_decoder_layers=6, with_segmentation=True, use_mhc=False,
        output_layers=(6, 12, 18, 24), fpn_levels=4,
        mask_upsample_factor=args.mask_upsample_factor,
        group_detr=args.group_detr,
    )

    results = []
    for bs in args.batch_sizes:
        print(f"--- Batch size {bs} (lr_dec={_REF_LR_DECODER * bs / _REF_BATCH_SIZE:.1e}, "
              f"lr_fpn={_REF_LR_FPN * bs / _REF_BATCH_SIZE:.1e}) ---")
        try:
            r = benchmark_batch_size(
                bs, model_cfg, args.mask_loss_resolution,
                args.weight_decay, args.n_warmup, args.n_measure,
            )
            results.append(r)
            print(f"  Peak mem:     {r['peak_allocated_gb']:.2f} GB allocated, "
                  f"{r['peak_reserved_gb']:.2f} GB reserved")
            print(f"  Breakdown:    model={r['model_params_gb']:.2f} GB, "
                  f"optim={r['optimizer_state_gb']:.2f} GB, "
                  f"activations={r['activation_mem_gb']:.2f} GB")
            print(f"  Throughput:   {r['throughput_img_per_s']:.1f} img/s "
                  f"({r['mean_step_time_s']:.3f} s/step)")
            print(f"  Grad norms:   fpn={r['mean_grad_norm_fpn']:.4f}, "
                  f"decoder={r['mean_grad_norm_decoder']:.4f}")
            print(f"  Loss:         {r['mean_loss']:.4f}")
            print()
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}! Max feasible batch size is {results[-1]['batch_size'] if results else '<32'}.")
            gc.collect()
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            break

    if not results:
        print("No successful runs.")
        return

    # --- Summary table ---
    print("=" * 90)
    print(f"{'BS':>4}  {'LR_dec':>9}  {'LR_fpn':>9}  {'Peak GB':>8}  {'Act GB':>7}  "
          f"{'img/s':>7}  {'s/step':>7}  {'GNorm_d':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['batch_size']:>4}  {r['lr_decoder']:>9.1e}  {r['lr_fpn']:>9.1e}  "
              f"{r['peak_allocated_gb']:>8.2f}  {r['activation_mem_gb']:>7.2f}  "
              f"{r['throughput_img_per_s']:>7.1f}  {r['mean_step_time_s']:>7.3f}  "
              f"{r['mean_grad_norm_decoder']:>8.4f}")
    print("=" * 90)

    # --- Scaling analysis ---
    if len(results) >= 2:
        base = results[0]
        print("\nScaling efficiency (relative to bs={})".format(base["batch_size"]))
        for r in results[1:]:
            bs_ratio = r["batch_size"] / base["batch_size"]
            throughput_ratio = r["throughput_img_per_s"] / base["throughput_img_per_s"]
            mem_ratio = r["peak_allocated_gb"] / base["peak_allocated_gb"]
            efficiency = throughput_ratio / bs_ratio * 100
            print(f"  bs={r['batch_size']}: "
                  f"{throughput_ratio:.2f}x throughput, "
                  f"{mem_ratio:.2f}x memory, "
                  f"{efficiency:.0f}% scaling efficiency")
        headroom = device_mem_gb - results[-1]["peak_allocated_gb"]
        print(f"\nGPU headroom: {headroom:.1f} GB remaining "
              f"({headroom / device_mem_gb * 100:.0f}% of {device_mem_gb:.0f} GB)")

    # --- Save ---
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": device_name,
        "gpu_memory_gb": round(device_mem_gb, 1),
        "config": {
            "backbone": args.backbone,
            "num_queries": args.num_queries,
            "group_detr": args.group_detr,
            "mask_upsample_factor": args.mask_upsample_factor,
            "mask_loss_resolution": args.mask_loss_resolution,
        },
        "lr_scaling": {
            "rule": "linear",
            "ref_batch_size": _REF_BATCH_SIZE,
            "ref_lr_decoder": _REF_LR_DECODER,
            "ref_lr_fpn": _REF_LR_FPN,
        },
        "results": results,
    }
    out_file = out_path / "batch_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
