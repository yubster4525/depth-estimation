# SYNS Dataset Prediction Guide

This guide explains how to generate depth predictions for the SYNS-Patches dataset using the models in this repository.

## Setup

1. First, download the SYNS-Patches dataset ZIP file and place it in the repository root or specify its path when running the scripts.

2. Ensure all models are downloaded:
   ```
   python model_downloader.py
   ```

## Generate Predictions

Use the `generate_syns_predictions.py` script to create depth predictions for all or specific models:

```bash
# Generate predictions for all models on validation set
python generate_syns_predictions.py --syns_zip /path/to/syns_patches.zip --split val

# Generate predictions for specific models on test set
python generate_syns_predictions.py --syns_zip /path/to/syns_patches.zip --split test --models midas_small dpt_large

# Skip visualization to save space (only generate NPZ files)
python generate_syns_predictions.py --syns_zip /path/to/syns_patches.zip --no_vis
```

This will:
- Read images from the SYNS-Patches dataset
- Run inference with each model
- Save raw depth predictions as NPZ files in `submissions/npz/`
- Generate visualizations in `submissions/vis/[model_id]/`

## Normalize Predictions

Before submission, you need to normalize the predictions. Use the `normalize_predictions.py` script:

```bash
# Normalize all available predictions using affine-invariant normalization (recommended)
python normalize_predictions.py

# Normalize specific models with global min-max normalization
python normalize_predictions.py --models dpt_large monodepth2 --normalization global

# Normalize with per-image normalization
python normalize_predictions.py --normalization per_image
```

This will:
- Load the raw predictions from `submissions/npz/`
- Apply the selected normalization method
- Save normalized predictions in `submissions/normalized/`
- Generate visualizations in `submissions/vis_normalized/[model_id]/`

## Available Normalization Methods

1. `affine_invariant` (default): Uses 5th and 95th percentiles of valid depths for robust scaling, recommended for monocular depth estimation

2. `global`: Simple min-max normalization across all images

3. `per_image`: Normalize each image independently to 0-1 range

## Directory Structure

```
submissions/
├── npz/                   # Raw prediction NPZ files
│   ├── midas_small.npz
│   ├── dpt_hybrid.npz
│   └── ...
├── normalized/            # Normalized prediction NPZ files for submission
│   ├── midas_small_affine_invariant.npz
│   ├── dpt_hybrid_global.npz
│   └── ...
├── vis/                   # Visualizations of raw predictions
│   ├── midas_small/
│   ├── dpt_hybrid/
│   └── ...
└── vis_normalized/        # Visualizations of normalized predictions
    ├── midas_small/
    ├── dpt_hybrid/
    └── ...
```