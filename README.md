# mHC-Path

## Overview

Histopathology cell detection and instance segmentation pipeline. Detects and segments cells in H&E stained tissue images. Evaluates on PanNuke dataset using official 3-fold cross-validation protocol.

## Architecture

- **Backbone**: DINOv2-L (frozen) or PhiKon v2 (pathology-adapted, frozen)
- **FPN**: 4-level feature pyramid from intermediate ViT layers
- **Decoder**: RF-DETR (deformable attention + learnable queries) with segmentation head
- **Alternative**: EoMT decoder (end-of-multi-task)
- Training features: Group DETR (3 groups), backbone unfreezing with LLRD

## Supported Backbones

| Backbone | Source | Notes |
|----------|--------|-------|
| dinov3_vitl16 | DINOv2-Large | Default, general-purpose |
| phikon_v2 | Owkin | Pathology-adapted, +0.007 test PQ |
| dinov3_vitb16 | DINOv2-Base | Lighter alternative |
| dinov3_vitg14 | DINOv2-Giant | Heavier, untested |

## Setup

```
pip install torch torchvision timm transformers kornia scipy wandb
```

## Data Preparation

### Download PanNuke

Download the three PanNuke folds from the official source (https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/) and extract so you have:

```
data/pannuke_raw/
  fold1/
    images.npy   # (N, 256, 256, 3)
    masks.npy    # (N, 256, 256, 6) — channels 0-4 are cell classes, 5 is background
    types.npy    # (N,) tissue type indices
  fold2/
    ...
  fold3/
    ...
```

### Convert to COCO format

```bash
python -m mhc_path.data.converters --raw_dir data/pannuke_raw --output_dir data --validate
```

This converts raw `.npy` files into COCO-format annotations with per-instance masks and bounding boxes, applying the correct PanNuke channel-to-class mapping (raw channels 2-4 are connective/dead/epithelial, not epithelial/connective/dead). It also extracts per-image tissue type labels needed for mPQ/bPQ evaluation.

After conversion you should have:

```
data/
  pannuke_fold1/
    annotations.json   # COCO-format with boxes + RLE masks
    images/            # 256x256 PNG tiles
    tissue_types.json  # per-image tissue type mapping for mPQ
  pannuke_fold2/
    ...
  pannuke_fold3/
    ...
```

## Getting Started

Before running any training, **run the batch scaling benchmark** to find the optimal batch size and verify memory headroom for your GPU:

```bash
python -m experiments.benchmark_batch_scaling --batch_sizes 32 64 128 256
```

This profiles peak memory, throughput, and gradient norms at each batch size with automatic linear LR scaling. Use the results to choose your `--batch_size` for all subsequent experiments — the largest batch size with good scaling efficiency and enough memory headroom for validation. The script saves results to `experiments/batch_scaling/batch_scaling_results.json`.

If you are using a different backbone, test that too:

```bash
python -m experiments.benchmark_batch_scaling --backbone phikon_v2 --batch_sizes 32 64 128 256
```

Once you know your target batch size, LR scaling is applied automatically in training (`--lr_scaling linear` is the default). All commands below assume you have picked a batch size based on the benchmark.

## Training

### Single fold

```bash
python -m experiments.train_pannuke --train_folds 1 2 --test_fold 3 --out_dir experiments/fold3
```

### 3-fold cross-validation (recommended)

```bash
python -m experiments.run_3fold_cv --data_root data --out_dir experiments/pannuke_3fold
```

### Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --backbone | dinov3_vitl16 | Backbone model |
| --num_queries | 300 | Number of object queries |
| --group_detr | 3 | Query groups (1=off) |
| --batch_size | 64 | Training batch size |
| --epochs | 200 | Training epochs |
| --lr_decoder | 1e-3 | Decoder learning rate |
| --lr_fpn | 5e-4 | FPN learning rate |
| --weight_decay | 1e-4 | Weight decay |
| --mask_upsample_factor | 4 | Mask head upsampling |
| --mask_loss_resolution | 128 | Mask loss resolution |
| --unfreeze_last_n | 0 | Backbone blocks to unfreeze |
| --lr_scaling | linear | LR scaling rule (linear/sqrt/none) |
| --ref_batch_size | 32 | Reference batch size for LR scaling |
| --score_threshold | 0.3 | Min confidence for PQ/F1d eval |

## Experiment Plan

Run the benchmark first (see Getting Started above), then launch these experiments in order. All commands use `--batch_size 64` as placeholder — **replace with your benchmark result**. LR scaling is automatic.

### 1. Baseline (RF-DETR + DINOv2-L, 3-fold CV)

Establishes the baseline on the new hardware. Compare against prior results to verify reproducibility.

```bash
python -m experiments.run_3fold_cv \
    --data_root data \
    --out_dir experiments/baseline \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

### 2. PhiKon v2 backbone (3-fold CV)

Pathology-adapted backbone. Previously showed +0.007 test PQ over DINOv2-L.

```bash
python -m experiments.run_3fold_cv \
    --data_root data \
    --out_dir experiments/phikon_v2 \
    --backbone phikon_v2 \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

### 3. EoMT decoder (3-fold CV)

Alternative decoder that injects queries directly into the last 4 ViT blocks. No FPN needed — trades FPN parameters for backbone compute.

```bash
python -m experiments.run_3fold_cv \
    --data_root data \
    --out_dir experiments/eomt \
    --decoder eomt \
    --batch_size 64 \
    --num_queries 300 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

### 4. Backbone unfreezing (3-fold CV)

Unfreeze last 6 ViT blocks with low LR after 20 frozen warmup epochs. Uses ~1.5 GB extra memory — verify headroom with benchmark first.

```bash
python -m experiments.run_3fold_cv \
    --data_root data \
    --out_dir experiments/unfreeze6 \
    --backbone phikon_v2 \
    --unfreeze_last_n 6 \
    --lr_backbone 1e-5 \
    --freeze_epochs 20 \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

### 5. Higher mask resolution (single fold, ablation)

Test mask_loss_resolution=256 and mask_upsample_factor=8 to see if higher resolution masks improve PQ. More memory-intensive — may need smaller batch size.

```bash
python -m experiments.train_pannuke \
    --train_folds 1 2 --test_fold 3 \
    --out_dir experiments/mask256_up8 \
    --backbone phikon_v2 \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 8 \
    --mask_loss_resolution 256 \
    --weight_decay 1e-4
```

### 6. Denoising training (single fold, ablation)

Denoising adds noised GT queries during training to speed convergence. Zero inference cost.

```bash
python -m experiments.train_pannuke \
    --train_folds 1 2 --test_fold 3 \
    --out_dir experiments/denoising \
    --backbone phikon_v2 \
    --use_denoising \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

### 7. SAHI evaluation (post-training, on any checkpoint)

Slicing Aided Hyper Inference: runs the model on overlapping crops of each image plus the full image, then merges predictions with NMS. Inference-only — no retraining needed. Tests multiple slice sizes to find the optimal configuration.

```bash
python -m experiments.eval_sahi \
    --checkpoint experiments/phikon_v2/fold3/checkpoints/best_pq.pt \
    --data_root data --test_fold 3 \
    --backbone phikon_v2 \
    --slice_sizes 96 128 192 \
    --overlap_ratios 0.25 \
    --out_dir experiments/sahi_eval
```

### 8. Combined best (3-fold CV)

After ablations 5-7, combine the winners with the best backbone and decoder for final evaluation.

```bash
python -m experiments.run_3fold_cv \
    --data_root data \
    --out_dir experiments/final \
    --backbone phikon_v2 \
    --use_denoising \
    --unfreeze_last_n 6 \
    --lr_backbone 1e-5 \
    --freeze_epochs 20 \
    --batch_size 64 \
    --num_queries 300 \
    --group_detr 3 \
    --mask_upsample_factor 4 \
    --mask_loss_resolution 128 \
    --weight_decay 1e-4
```

## Key Findings

- Cosine LR schedule is optimal (tested vs linear, trapezoidal)
- lr_decoder=1e-3 is right for the decoder
- weight_decay=1e-4 outperforms 0.01 (RF-DETR default)
- PhiKon v2 improves test metrics (+0.007 PQ)
- mask_upsample_factor=4 + mask_loss_resolution=128 is best mask config
- EMA with high decay (>=0.998) is unreliable; evaluate raw model

## Evaluation Metrics

- **mPQ**: Primary ranking metric (tissue-stratified panoptic quality)
- **PQ**: Panoptic quality (macro-averaged per-class)
- **mAP@50**: Mean average precision
- **F1d@30**: Detection F1 at IoU=0.3
- **mIoU**: Mean segmentation IoU (matched instances)

## Project Structure

```
mhc_path/
├── config/          # Class maps, reproducibility
├── data/            # Dataset, augmentation
├── models/          # Backbone, FPN, decoder, assembly
├── training/        # Engine, losses
├── evaluation/      # Metrics, diagnostics
experiments/
├── train_pannuke.py           # Main training script
├── run_3fold_cv.py            # 3-fold CV runner
├── benchmark_batch_scaling.py # GPU memory/throughput profiler
├── eval_sahi.py               # SAHI evaluation comparison
tests/               # Unit tests per component
```
