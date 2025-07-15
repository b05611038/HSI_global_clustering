import os
import copy
import ast
import json
import argparse

import torch
from torch.utils.data import DataLoader

from hsi_global_clustering import (JSONMATDataset,
                                   HyperspectralClusteringModel,
                                   HSIClusteringTrainer)

from hsi_global_clustering.hsi_processing import normalize_cube

def parse_mapping(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # allow Python‐style dicts too
        return ast.literal_eval(s)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference + metric eval with optional manual cluster→label mapping"
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
        '--json_dir', type=str, default=None,
        help='Directory containing LabelMe JSON annotations'
    )
    parser.add_argument(
        '--label_dir', type=str, default=None,
        help='Directory containing LabelMe JSON annotations'
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="If set, use Hungarian auto‐alignment (ignored if manual mapping is given)"
    )
    parser.add_argument(
        "--manual-mapping",
        type=parse_mapping,
        default=None,
        help=(
            "JSON mapping of labels to clusters, e.g. "
            "e.g. '{0:[0],1:[1,2,3]}' or '{\"0\":[0],\"1\":[1,2,3]}'"
        )
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

def print_metrics(metrics: dict):
    """
    Nicely print a metrics dict that contains:
      - 'mean_IoU', 'mean_Dice', 'mean_RMSE'
      - 'IoU_class_{i}', 'Dice_class_{i}' for each class i
    """
    # 1) Print global (mean) metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Mean IoU : {metrics.get('mean_IoU', 0):.4f}")
    print(f"Mean Dice: {metrics.get('mean_Dice', 0):.4f}")
    print(f"Mean RMSE: {metrics.get('mean_RMSE', 0):.4f}")

    # 2) Gather class IDs
    class_ids = sorted(
        int(k.split('_')[-1])
        for k in metrics.keys()
        if k.startswith("IoU_class_")
    )
    if not class_ids:
        return  # no per-class metrics to show

    # 3) Header for per-class table
    print("\nPer-class scores:")
    print(f"{'Class':<6} {'IoU':>8} {'Dice':>8}")
    print("-" * 24)

    # 4) Print each class
    for cid in class_ids:
        iou  = metrics.get(f"IoU_class_{cid}",   float("nan"))
        dice = metrics.get(f"Dice_class_{cid}",  float("nan"))
        print(f"{cid:<6} {iou:>8.4f} {dice:>8.4f}")

    print()  # blank line at end
    return None

def main():
    args = parse_args()
    full_load, num_workers = False, args.ds_n_workers
    if args.full_load_ds:
        full_load = True
        num_workers = 0

    # build manual_mapping dict if provided
    if args.manual_mapping:
        manual_mapping = {int(k): tuple(v) for k, v in args.manual_mapping.items()}
        auto_align = False
    else:
        manual_mapping = None
        auto_align = args.auto_align

    device = torch.device(args.device)
    model = HyperspectralClusteringModel.load(args.checkpoint_path)

    # instantiate your Trainer without training data
    trainer = HSIClusteringTrainer(
        train_dataset=None,
        val_dataset=None,
        # … plus any required args, e.g. model, optimizer args, etc. …
        device=device,
    )
    trainer.model = model

    # build your eval dataset however you normally do
    eval_ds = JSONMATDataset(
        mat_dir=args.mat_dir,
        json_dir=args.json_dir,
        label_dir=args.label_dir,
        data_key='cube',
        transform=None,
        normalize=normalize_cube,
        to_tensor=True,
        full_loading=full_load
    )

    # run inference + metrics
    aligned_preds, metrics = trainer.inference(
        dataset=eval_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        auto_align=auto_align,
        manual_mapping=manual_mapping,
    )

    # print or save results
    print_metrics(metrics)
    return None

if __name__ == '__main__':
    main()


