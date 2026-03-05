# mHC-Path

Histopathology detection and segmentation pipeline: frozen DINOv3-L backbone, FPN multi-scale fusion, RF-DETR decoder with deformable attention. Trained and evaluated on PanNuke with 3-fold cross-validation.

## Architecture

- **Backbone**: DINOv3-L (frozen, ~300M params, zero trainable)
- **FPN**: 4-level feature pyramid, 256-dim
- **Decoder**: RF-DETR with 6 deformable attention layers, 100 queries
- **Segmentation**: Dot-product mask head with 4x transposed-conv upsampling (64x64 native masks)
- **Loss**: Focal + CIoU + Dice with Hungarian matching

## Installation

```bash
pip install -e ".[train,dev]"
```

## Quick Start

```bash
# 3-fold cross-validation (recommended)
python -m experiments.train_pannuke --train_folds 1 2 --test_fold 3
python -m experiments.train_pannuke --train_folds 1 3 --test_fold 2
python -m experiments.train_pannuke --train_folds 2 3 --test_fold 1
```

## 5-Class Taxonomy

| Index | Class | Color |
|-------|-------|-------|
| 0 | neoplastic | red |
| 1 | inflammatory | green |
| 2 | epithelial | blue |
| 3 | connective | yellow |
| 4 | dead | cyan |

## Training Defaults

| Parameter | Value |
|-----------|-------|
| Backbone | DINOv3-L (frozen) |
| mask_upsample_factor | 4 (native 64x64 masks) |
| mask_loss_resolution | 128 |
| num_queries | 100 |
| decoder_layers | 6 |
| lr_fpn | 5e-4 |
| lr_decoder | 1e-3 |
| warmup_epochs | 10 |
| weight_decay | 1e-2 |
| batch_size | 32 |
| optimizer | AdamW |
| score_threshold | 0.3 |
| box_loss | CIoU |
| EMA decay | 0.998 |

## Project Structure

```
mhc_path/
  config/         # Class taxonomy, reproducibility
  data/           # PanNuke dataset, stain & GPU augmentation
  models/         # Backbone, FPN, decoder, full model
  training/       # Loss, engine, logger
  evaluation/     # PQ, mAP, F1-det, diagnostics
experiments/      # Training scripts
tests/            # Paired tests for all modules
```

## Tests

```bash
pytest tests/
```
