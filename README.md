# mHC-Path

Histopathology cell detection and instance segmentation pipeline. Frozen ViT backbone (DINOv2/Phikon-v2) with FPN multi-scale fusion and RF-DETR decoder with deformable attention. Trained and evaluated on PanNuke with official 3-fold cross-validation.

## Architecture

```
Input (256x256 H&E tile)
  → Frozen ViT-L backbone (DINOv2 or Phikon-v2, multi-scale features at layers 6/12/18/24)
  → FPN (4-level feature pyramid, 256-dim)
  → RF-DETR decoder (6 deformable attention layers, learnable queries)
  → Detection head (per-query class logits + 4D box refinement)
  → Mask head (dot-product segmentation with optional pixel decoder upsampling)
```

**Key features:**
- **Pixel decoder**: Multi-scale ViT skip connections for high-res mask prediction (+0.02 PQ)
- **Group DETR**: Duplicate query groups during training for better supervision (zero inference cost)
- **Backbone unfreezing**: Optionally unfreeze last N ViT blocks with separate LR
- **Boundary loss**: Sobel-gradient mask boundary supervision

## Installation

```bash
pip install -e ".[train,dev]"
```

## Data Preparation

Download PanNuke and convert to detection format:

```bash
# Convert raw PanNuke .npy files to COCO-style annotations
python -m mhc_path.data.converters --dataset pannuke --input_dir /path/to/pannuke --output_dir data/
```

Expected directory structure:
```
data/
  pannuke_fold1/
    images/          # 256x256 PNG tiles
    annotations.json # COCO-format with boxes + masks
    tissue_types.json
  pannuke_fold2/
    ...
  pannuke_fold3/
    ...
```

## Training

### Basic (single fold)

```bash
python -m experiments.train_pannuke \
    --train_folds 1 2 --test_fold 3 \
    --epochs 200 --batch_size 32
```

### Recommended configuration

```bash
python -m experiments.train_pannuke \
    --train_folds 1 2 --test_fold 3 \
    --epochs 400 --batch_size 32 \
    --backbone phikon_v2 \
    --with_pixel_decoder \
    --num_queries 300 \
    --group_detr 2 \
    --mask_loss_resolution 128 \
    --mask_upsample_factor 4 \
    --mask_boundary_weight 1.0 \
    --out_dir experiments/exp_combined
```

### 3-fold cross-validation (official PanNuke protocol)

```bash
# Fold 1 as test
python -m experiments.train_pannuke --train_folds 2 3 --test_fold 1 --out_dir experiments/fold1

# Fold 2 as test
python -m experiments.train_pannuke --train_folds 1 3 --test_fold 2 --out_dir experiments/fold2

# Fold 3 as test
python -m experiments.train_pannuke --train_folds 1 2 --test_fold 3 --out_dir experiments/fold3
```

### Resume from checkpoint

```bash
python -m experiments.train_pannuke \
    --resume experiments/exp_combined/checkpoints/best_pq.pt \
    --epochs 400 \
    ... # same args as original run
```

### Backbone unfreezing

```bash
python -m experiments.train_pannuke \
    --unfreeze_last_n 6 \
    --lr_backbone 1e-4 \
    --freeze_epochs 25 \
    ... # unfreeze last 6 ViT blocks after 25 epochs
```

## Post-training threshold calibration

Optimize per-class score thresholds on mPQ:

```bash
python -m experiments.calibrate_thresholds \
    --checkpoint experiments/exp_combined/checkpoints/best_pq.pt \
    --data_root data --test_fold 3 \
    --lo 0.1 --hi 0.75 --step 0.05
```

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `dinov3_vitl16` | Backbone: `dinov3_vitl16`, `dinov3_vitb16`, `phikon_v2` |
| `--num_queries` | 100 | Number of detection queries |
| `--group_detr` | 1 | Query groups for training (1=off, 2-3 recommended) |
| `--with_pixel_decoder` | off | Multi-scale pixel decoder for masks |
| `--mask_upsample_factor` | 1 | Upsample mask features (4 = native 64x64) |
| `--mask_loss_resolution` | 64 | Resolution for mask loss computation |
| `--mask_boundary_weight` | 1.0 | Boundary (Sobel) loss weight |
| `--unfreeze_last_n` | 0 | Unfreeze last N backbone blocks |
| `--lr_backbone` | 0.0 | LR for unfrozen backbone params |
| `--freeze_epochs` | 0 | Keep backbone frozen for N epochs first |
| `--large_kernel` | off | Large-kernel (RepLK) depthwise convs in pixel decoder |
| `--patience` | 0 | Early stopping patience (0=disabled) |
| `--score_threshold` | 0.3 | Min confidence for PQ/F1d evaluation |
| `--resume` | None | Checkpoint path to resume from |

## 5-Class Taxonomy

| Index | Class | Color |
|-------|-------|-------|
| 0 | neoplastic | red |
| 1 | inflammatory | green |
| 2 | epithelial | blue |
| 3 | connective | yellow |
| 4 | dead | cyan |

## Metrics

- **mPQ** (primary): Per-class PQ averaged within each of 19 tissue types, then across tissues
- **PQ**: Panoptic Quality = DQ x SQ (macro-averaged per class)
- **DQ**: Detection Quality (F1-like, penalizes FP and FN)
- **SQ**: Segmentation Quality (mean IoU of matched pairs)
- **bPQ**: Binary PQ (class-agnostic, tissue-stratified)
- **mAP@50**: COCO-style detection AP
- **F1d**: Detection F1 score

## Experiment Results

| Config | mPQ | PQ | DQ | SQ |
|--------|:---:|:---:|:---:|:---:|
| Frozen DINOv3, upsample=4 | 0.384 | 0.425 | 0.557 | 0.796 |
| + Pixel decoder | 0.404 | 0.443 | — | — |
| Phikon-v2 backbone | 0.415 | 0.436 | 0.560 | 0.770 |

## Project Structure

```
mhc_path/
  config/         # Class taxonomy, reproducibility
  data/           # PanNuke dataset, stain & GPU augmentation
  models/         # Backbone adapter, FPN, RF-DETR decoder, full model
  training/       # Hungarian matching, composite loss, training engine
  evaluation/     # PQ, mPQ, mAP, F1-det, calibration, diagnostics
experiments/
  train_pannuke.py          # Main training script
  calibrate_thresholds.py   # Per-class threshold optimization
tests/                      # Unit tests
```

## Tests

```bash
pytest tests/
```

## Known Issues

- `pin_memory=True` in DataLoader can cause OOM when GPU holds model; we use `pin_memory=False`
- EMA with decay >= 0.998 barely moves from init over 100 epochs; evaluation uses raw model
- Batch size 32 is optimal; bs=64 is slower due to GPU compute scaling
