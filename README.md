# HSI_global_clustering

This repository implements an end-to-end, label-free hyperspectral image (HSI) clustering framework that produces globally aligned cluster IDs across scenes. It embeds a differentiable mean-shift layer within a spectral–spatial encoder, and uses an exponential moving-average (EMA) centroid dictionary with entropy balancing for stable, automatic cluster convergence.

## Features

- **Spectral–Spatial Encoder**: Stacked 1D spectral CNN and 2D spatial CNN with ℓ²-normalized pixel embeddings.
- **Unrolled Mean-Shift Module**: Differentiable mean-shift attractor with learned bandwidth and optional approximations for scalability.
- **Global Centroid Dictionary**: EMA updates of cluster centroids, with automatic low-mass pruning.
- **Self-Supervised Losses**: Compactness, orthogonality, balance, and crop-consistency terms.
- **Data Augmentation**: Overlapping two-crop strategy, random affine, and wavelength shifts.
- **GPU-friendly Metrics**: IoU, Dice, Area RMSE, entropy, NMI, VI—all implemented in PyTorch.
- **Single-GPU Trainer**: Custom `Trainer` class with mixed-precision, gradient clipping, checkpointing, TensorBoard logging, early stopping, and flexible inference.

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Place your hyperspectral `.mat` files (each containing a `cube` array of shape `(C,H,W)`) in a directory, along with optional LabelMe-style JSON annotations:

```
/data/hsi_mat/
  scene01.mat
  scene02.mat
  ...
/data/hsi_json/
  scene01.json
  scene02.json
  ...
```

## Quick Start

Use the `run_clustering.py` script to train and evaluate. You can specify embedding dimension, number of clusters, and mean-shift iterations:

```bash
python run_clustering.py \
  --mat_dir /data/hsi_mat \
  --json_dir /data/hsi_json \
  --out_dir /outputs \
  --bands 301 \
  --crop_h 64 --crop_w 64 \
  --embed_dim 32 \
  --n_clusters 64 \
  --num_iters 3 \
  --epochs 50 \
  --batch 4 \
  --lr 1e-3 \
  --wd 1e-4 \
  --device cuda
```

This will:

1. Create output folders (`logs/`, `checkpoints/`) under `/outputs`.
2. Train the model with overlapping two-crop augmentation and mixed-precision.
3. Log losses and metrics to TensorBoard (`/outputs/logs`).
4. Save checkpoint files to `/outputs/checkpoints`.
5. Run inference on the training set and save predictions to `/outputs/predictions.pt`.

## Code Structure

- `hsi_global_clustering/datasets.py` — `JSONMATDataset` for streaming `.mat` + JSON loading.
- `hsi_global_clustering/hsi_processing.py` — GPU-friendly augmentation pipeline.
- `hsi_global_clustering/hsi_clustering.py` — `HyperspectralClusteringModel` combining encoder + mean-shift.
- `hsi_global_clustering/trainer.py` — `Trainer` class for single-GPU training and evaluation.
- `hsi_global_clustering/eval.py` — PyTorch implementations of IoU, Dice, RMSE, entropy, NMI, VI.
- `run_clustering.py` — Example script tying everything together.

## TensorBoard Monitoring

Start TensorBoard to visualize training and validation metrics:

```bash
tensorboard --logdir /outputs/logs
```

## Citation

If you use this code in your research, please cite:

> **[Paper Title]**  
> Author1, Author2, ...  
> Conference/Journal, Year.

## License

This project is licensed under the MIT License. See `LICENSE` for details.


