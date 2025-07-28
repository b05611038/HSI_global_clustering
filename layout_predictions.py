import os
import sys
import copy
import argparse

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from hsi_global_clustering import (JSONMATDataset,
                                   HyperspectralClusteringModel)

from hsi_global_clustering.hsi_processing import normalize_cube

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the HSI clustering model and layout images."
    )
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='The saved weight of the clustering model.'
    )
    parser.add_argument(
        '--mat_dir', type=str, required=True,
        help='Directory containing .mat HSI cubes'
    )
    parser.add_argument(
        '--out_dir', type=str, required=True,
        help='Directory where outputs (logs, checkpoints) will be saved'
    )
    parser.add_argument(
        '--img_dir', type=str, default='',
        help='Directory where output images will be saved'
    )
    parser.add_argument(
        '--layout_pth', action='store_true',
        help='To save layout tensors.'
    )
    parser.add_argument(
        '--layout_image', action='store_true',
        help='To save layout clustered images.'
    )
    parser.add_argument(
        '--full_load_ds', action='store_true',
        help='Use the dataset with full_load flag (large RAM use but no I/O bond).'
    )
    parser.add_argument(
        '--ds_n_workers', type=int, default=8,
        help='Number of workers in Pytorch dataloader.'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Compute device, e.g. "cuda" or "cpu"'
    )
    return parser.parse_args()

def collect_results(model, eval_ds, num_workers=0, device=torch.device('cpu'), interval=10):
    files = copy.deepcopy(eval_ds.files)
    for f_idx in range(len(files)):
        files[f_idx] = os.path.split(files[f_idx])[-1].replace('.mat', '')

    results = {}
    dataloader = DataLoader(eval_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, cube in enumerate(dataloader):
            cube = cube.to(device)
            pred_cluster = model.inference(cube)[0]
            results[files[idx]] = pred_cluster.cpu()

        if (idx + 1) % interval == 0:
            print(f"Inference progress: {idx + 1}/{len(eval_ds)}")

    return results

def save_cluster_map(
    cluster_map: torch.Tensor,
    save_dir: str,
    filename: str,
    swap_axes: bool = False,
    flip_ud: bool = False,
    flip_lr: bool = False
) -> None:
    """
    Save HSI clustering result as a color-coded PNG image, with optional orientation adjustments.

    Args:
        cluster_map (torch.Tensor): 2D tensor of shape (H, W) with integer cluster indices.
        save_dir (str): Directory to save the image.
        filename (str): Filename (without extension) for the saved image.
        swap_axes (bool): If True, transpose the cluster_map before coloring.
        flip_ud (bool): If True, flip the map upside-down.
        flip_lr (bool): If True, flip the map left-to-right.

    Raises:
        AssertionError: If cluster_map is not 2D or indices fall outside [0, 7].
    """
    # Define up to 8 distinct colors (RGB)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
    ]

    # Validate input tensor
    assert isinstance(cluster_map, torch.Tensor), \
        f"cluster_map must be a torch.Tensor, got {type(cluster_map)}"
    assert cluster_map.ndim == 2, \
        f"cluster_map must be 2D, got shape {tuple(cluster_map.shape)}"

    # Convert to int and adjust orientation
    idx_map = cluster_map.cpu().to(torch.int64)
    if swap_axes:
        idx_map = idx_map.t()
    if flip_ud or flip_lr:
        dims = []
        if flip_ud:
            dims.append(0)
        if flip_lr:
            dims.append(1)
        idx_map = torch.flip(idx_map, dims)

    idx_map = idx_map.numpy()
    H, W = idx_map.shape

    # Check cluster index range
    min_idx, max_idx = int(idx_map.min()), int(idx_map.max())
    assert min_idx >= 0, \
        f"Cluster indices must be non-negative, got min {min_idx}"
    assert max_idx < len(colors), \
        f"Cluster index {max_idx} exceeds available colors ({len(colors)-1})"

    # Create RGB image array
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in range(max_idx + 1):
        mask = (idx_map == cid)
        rgb[mask] = colors[cid]

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")

    # Save image
    img = Image.fromarray(rgb)
    img.save(save_path)

    return None

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    full_load, num_workers = False, args.ds_n_workers
    if args.full_load_ds:
        full_load = True
        num_workers = 0

    layout_image, layout_pth = False, False
    if args.layout_image:
        layout_image = True
        if len(args.img_dir) == 0:
            img_dir = args.out_dir
        else:
            img_dir = args.img_dir

    if args.layout_pth:
        layout_pth = True

    if (not layout_image) and (not layout_pth):
        sys.exit('--layout_image and --layout_pth need to be set in the program.')

    device = torch.device(args.device)
    model = HyperspectralClusteringModel.load(args.checkpoint_path)

    eval_ds = JSONMATDataset(
        mat_dir=args.mat_dir,
        data_key='cube',
        transform=None,
        normalize=normalize_cube,
        to_tensor=True,
        full_loading=full_load
    )

    results = collect_results(model, eval_ds, num_workers=num_workers, device=device)
    for filename in results:
        pred_clusters = results[filename]
        if layout_pth:
            save_path = os.path.join(args.out_dir, f"{filename}.pth")
            torch.save(pred_clusters, save_path)

        if layout_image:
            save_cluster_map(pred_clusters, img_dir, filename) 

    return None

if __name__ == '__main__':
    main()


