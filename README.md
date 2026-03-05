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

## Tests

```bash
pytest tests/
```
