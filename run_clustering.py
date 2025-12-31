"""
run_clustering.py

Simple entrypoint script to train and evaluate the hyperspectral clustering model.
"""
import os
import argparse

import torch

from hsi_global_clustering import (JSONMATDataset,
                                   AugmentationPipeline,
                                   HyperspectralClusteringModel,
                                   HSIClusteringTrainer)

from hsi_global_clustering.default_argument import (
    DEFAULT_MODEL_KWARGS,
    DEFAULT_OPTIMIZER_KWARGS,
    DEFAULT_LOSS_WEIGHT_SCHEDULING,
    DEFAULT_EMA_DECAY,
    DEFAULT_EMA_KICK,
    DEFAULT_EMA_KICK_SCHEDULING,
)

from hsi_global_clustering.hsi_processing import normalize_cube

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HSI clustering model"
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
        '--out_dir', type=str, required=True,
        help='Directory where outputs (logs, checkpoints) will be saved'
    )
    parser.add_argument(
        '--full_load_ds', action='store_true',
        help='Use the dataset with full_load flag (large RAM use but no I/O bond).'
    )
    parser.add_argument(
        '--save_prediction', action='store_true',
        help='To save the result of the prediction.'
    )
    parser.add_argument(
        '--ds_n_workers', type=int, default=8,
        help='Number of workers in Pytorch dataloader.'
    )
    parser.add_argument(
        '--bands', type=int, default=DEFAULT_MODEL_KWARGS['num_bands'],
        help='Number of spectral bands in the HSI data'
    )
    parser.add_argument(
        '--crop_h', type=int, default=64,
        help='Height of spatial crop'
    )
    parser.add_argument(
        '--crop_w', type=int, default=64,
        help='Width of spatial crop'
    )
    parser.add_argument(
        '--embed_dim', type=int, default=DEFAULT_MODEL_KWARGS['encoder_kwargs']['embed_dim'],
        help='Dimension of latent embedding from the encoder'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=DEFAULT_MODEL_KWARGS['mean_shift_kwargs']['n_clusters'],
        help='Number of clusters (centroids) in the mean-shift module'
    )
    parser.add_argument(
        '--num_iters', type=int, default=DEFAULT_MODEL_KWARGS['mean_shift_kwargs']['num_iters'],
        help='Number of mean-shift iterations (unrolled steps)'
    )
    parser.add_argument(
        '--ema_decay', type=float, default=DEFAULT_EMA_DECAY,
        help='EMA decay of cluster centroid moving.'
    )
    parser.add_argument(
        '--ema_kick', type=float, default=DEFAULT_EMA_KICK,
        help='EMA random kick of non-acitivated centroid.'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Total number of training epochs'
    )
    parser.add_argument(
        '--save_interval', type=int, default=2,
        help='Interval of saving cluster model'
    )
    parser.add_argument(
        '--batch', type=int, default=4,
        help='Batch size (number of samples per batch)'
    )
    parser.add_argument(
        '--reuse_iter', type=int, default=5,
        help='Repeated use in a single HSI acquistion.'
    )
    parser.add_argument(
        '--coef_orth', type=float, default=DEFAULT_MODEL_KWARGS['loss_weights']['orth'],
        help='Coefficient of orthogonal centroid panelization in loss fuction.'
    )
    parser.add_argument(
        '--coef_bal', type=float, default=DEFAULT_MODEL_KWARGS['loss_weights']['bal'],
        help='Coefficient of balanced cluster usage in loss function.',
    )
    parser.add_argument(
        '--coef_unif', type=float, default=DEFAULT_MODEL_KWARGS['loss_weights']['unif'],
        help='Coefficient of uniformly assign cluster in loss function.',
    )
    parser.add_argument(
        '--coef_cons', type=float, default=DEFAULT_MODEL_KWARGS['loss_weights']['cons'],
        help='Coefficient of panelizing inconsistent of two cropped HSI in loss function.',
    )
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_OPTIMIZER_KWARGS['lr'],
        help='Initial learning rate for the optimizer'
    )
    parser.add_argument(
        '--beta1', type=float, default=DEFAULT_OPTIMIZER_KWARGS['betas'][0],
        help='Beta1 of AdamW.'
    )
    parser.add_argument(
        '--beta2', type=float, default=DEFAULT_OPTIMIZER_KWARGS['betas'][1],
        help='Beta1 of AdamW.'
    )
    parser.add_argument(
        '--wd', type=float, default=DEFAULT_OPTIMIZER_KWARGS['weight_decay'],
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Compute device, e.g. "cuda" or "cpu"'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    full_load, num_workers = False, args.ds_n_workers
    if args.full_load_ds:
        full_load = True
        num_workers = 0

    save_prediction = False
    if args.save_prediction:
        save_prediction = True

    # Build training dataset
    train_ds = JSONMATDataset(
        mat_dir=args.mat_dir,
        json_dir=args.json_dir,
        data_key='cube',
        label_key='label' if args.json_dir else None,
        class_to_index=None if not args.json_dir else {'class0': 0, 'class1': 1},
        transform=None,
        normalize=normalize_cube,
        to_tensor=True,
        full_loading=full_load
    )

    # (Optional) build validation dataset by splitting or separate directory
    val_ds = None

    # Instantiate Trainer
    trainer = HSIClusteringTrainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        augmentor=AugmentationPipeline((args.crop_h, args.crop_w)),
        model_kwargs={
            'num_bands': args.bands,
            'encoder_kwargs': {
                'n_spectral_layers': 3,
                'spectral_kernel_size': 9,
                'embed_dim': args.embed_dim,
                'bias': False,
            },
            'mean_shift_kwargs': {
                'embed_dim': args.embed_dim,
                'n_clusters': args.n_clusters,
                'num_iters': args.num_iters,
            },
            'loss_weights': {
                'orth': args.coef_orth,
                'bal': args.coef_bal,
                'unif': args.coef_unif,
                'cons': args.coef_cons,
            },
        },
        device=torch.device(args.device),
        optimizer_kwargs={'lr': args.lr, 'betas': (args.beta1, args.beta2), 'weight_decay': args.wd},
        loss_weight_scheduling={'lambda_unif': False, 
                                'lambda_orth': False, 
                                'lambda_bal': False, 
                                'lambda_cons': False},

        ema_decay=args.ema_decay,
        ema_kick=args.ema_kick,
        ema_kick_scheduling=True,
        num_workers=num_workers,
        num_epochs=args.epochs,
        reuse_iter=args.reuse_iter,
        batch_size=args.batch,
        log_dir=os.path.join(args.out_dir, 'logs'),
        ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
        save_interval = args.save_interval,
    )

    # Train the model
    trainer.train()

    # Inference on training set (example)
    if save_prediction:
        # still has some bug now
        print("Running inference on training set...")
        preds = trainer.inference(train_ds)
        save_path = os.path.join(args.out_dir, 'predictions.pth')
        torch.save(preds, save_path)
        print(f"Saved predictions to {save_path}")

    return None


if __name__ == '__main__':
    main()


