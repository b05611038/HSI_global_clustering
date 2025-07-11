import os
import time
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from typing import Optional, Callable, Dict, Tuple, Union

from .hsi_clustering import HyperspectralClusteringModel
from .eval import (
    iou_score, dice_score, area_rmse,
    cluster_entropy, normalized_mutual_information, variation_of_information
)


__all__ = ['HSIClusteringTrainer']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def pad_and_stack(batch: list):
    """
    Pad a batch of cubes (and optional labels) to the same H_max × W_max, then stack.

    Args:
        batch: list of items, each Tensor(C,H,W) or tuple (Tensor(C,H,W), Tensor(H,W)).
    Returns:
        cubes: Tensor(B,C,H_max,W_max)
        labels: Tensor(B,H_max,W_max) if labels present else None
    """
    cubes, labels = [], []
    has_labels = isinstance(batch[0], tuple)
    for item in batch:
        if has_labels:
            c, l = item
            cubes.append(c)
            labels.append(l)
        else:
            cubes.append(item)

        # compute max dims
        _, hs, ws = zip(*[c.shape for c in cubes])
        H_max, W_max = max(hs), max(ws)
        # pad cubes
        padded_cubes = []
        for c in cubes:
            C, h, w = c.shape
            pad = (0, W_max - w, 0, H_max - h)  # (left,right,top,bottom)
            padded_cubes.append(F.pad(c, pad, mode='reflect'))

        batch_cubes = torch.stack(padded_cubes, dim=0)
        # pad labels if present
        if has_labels:
            padded_labels = []
            for l in labels:
                h, w = l.shape
                pad = (0, W_max - w, 0, H_max - h)
                padded_labels.append(F.pad(l.unsqueeze(0), pad, mode='constant', value=0).squeeze(0))

            batch_labels = torch.stack(padded_labels, dim=0)
            return batch_cubes, batch_labels

        return batch_cubes, None

def print_epoch_summary(
    epoch: int,
    train_loss: float,
    sup_metrics: Optional[dict] = None,
    unsup_metrics: Optional[dict] = None,
    total_epochs: Optional[int] = None,
):
    """
    Print a concise terminal summary:
      • epoch / total_epochs (if given)
      • train loss
      • IoU, Dice, RMSE (if sup_metrics)
      • Entropy, NMI, VI (if unsup_metrics)
    """
    header = f"Epoch {epoch}"
    if total_epochs:
        header += f"/{total_epochs}"
    parts = [header, f"TrainLoss={train_loss:.4f}"]

    if sup_metrics is not None:
        parts += [
            f"IoU={sup_metrics['IoU']:.3f}",
            f"Dice={sup_metrics['Dice']:.3f}",
            f"RMSE={sup_metrics['RMSE']:.4f}",
        ]

    if unsup_metrics is not None:
        parts += [
            f"Ent={unsup_metrics['entropy']:.3f}",
            f"NMI={unsup_metrics['NMI']:.3f}",
            f"VI={unsup_metrics['VI']:.3f}",
        ]

    print(" | ".join(parts), flush=True)


class HSIClusteringTrainer:
    """
    Single-GPU Trainer for hyperspectral clustering.

    Args:
        train_dataset: Dataset yielding raw cubes (C, H, W) or (cube, label).
        val_dataset:   Optional dataset for evaluation.
        augmentor:     Callable to generate two-crop augmentations from a cube.
        model:         Optional pre-defined model; if None, built from model_kwargs.
        model_kwargs:  Keywords to initialize HyperspectralClusteringModel.
        device:        torch.device for computation.
        optimizer_cls: Optimizer class, e.g. torch.optim.AdamW.
        optimizer_kwargs: Dict of optimizer parameters.
        scheduler_cls: Optional learning-rate scheduler class.
        scheduler_kwargs: Dict of scheduler parameters.
        ema_decay:     EMA decay of cluster centriod. 
        batch_size:    Number of samples per batch.
        num_workers:   DataLoader num_workers.
        num_epochs:    Total training epochs.
        grad_clip:     Gradient clipping norm.
        precision:     'fp32' or 'bf16' for mixed precision.
        seed:          Random seed.
        log_dir:       Directory for TensorBoard logs.
        ckpt_dir:      Directory for saving checkpoints.
        save_interval: Epoch interval for checkpoints.
        eval_interval: Epoch interval for validation.
        log_interval:  Step interval for logging losses.
        early_stopping: Whether to use early stopping.
        early_stopping_patience: Number of epochs with no improvement.
        early_stopping_metric: Callable that returns a metric for stopping.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        augmentor: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        model: Optional[nn.Module] = None,
        model_kwargs: dict = {},
        device: torch.device = torch.device('cpu'),
        optimizer_cls: Callable = optim.AdamW,
        optimizer_kwargs: dict = {'lr': 1e-4, 'weight_decay': 1e-2},
        scheduler_cls: Optional[Callable] = None,
        scheduler_kwargs: dict = {},
        ema_decay: float = 0.95,
        batch_size: int = 4,
        num_workers: int = 8,
        num_epochs: int = 100,
        grad_clip: float = 1.,
        precision: str = 'bf16',
        seed: int = 42,
        log_dir: str = './runs',
        ckpt_dir: str = './checkpoints',
        save_interval: int = 1,
        eval_interval: int = 1,
        log_interval: int = 50,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_metric: Optional[Callable[[Dict[str, float]], float]] = None,
    ):

        set_seed(seed)
        self.device = device
        
        # Model initialization
        if model is None:
            self.model = HyperspectralClusteringModel(**model_kwargs).to(device)
        else:
            self.model = model.to(device)
        self.precision = precision
        self.augmentor = augmentor
        
        # Training DataLoader with custom collate for augmentation
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_and_stack,
        )

        # Validation DataLoader (batch_size=1)
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=pad_and_stack,
            )
        else:
            self.val_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_and_stack,
        )
        
        # Optimizer and optional scheduler
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs) if scheduler_cls else None
        
        # Training settings
        self.num_epochs = num_epochs
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        
        # Logging and checkpointing
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, timestamp))
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval

        self.prev_epoch_preds = None
        
        # Early stopping setup
        self.early_stopping = early_stopping
        self.patience = early_stopping_patience
        self.es_metric = early_stopping_metric
        self.best_metric = None
        self.no_improve = 0

    def train(self):
        """
        Main training loop: logs losses, checkpoints, and runs evaluation.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step, batch in enumerate(self.train_loader, start=1):
                cubes, _ = batch

                cubes = cubes.to(self.device, non_blocking=True)        # (B, C, H, W)
                crops = self.augmentor(cubes)                           # (B, 2, C, h, w)

                c0 = crops[:, 0]
                c1 = crops[:, 1]

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.precision != 'fp32')):
                    loss, loss_dict, ema_dict = self.model.train_step(c0, c1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                z1, p1 = ema_dict['z1'], ema_dict['p1']
                z2, p2 = ema_dict['z2'], ema_dict['p2']

                self._ema_update_centroids(z1, p1, z2, p2, ema_decay=self.ema_decay)

                if self.scheduler:
                    self.scheduler.step()

                running_loss += loss.item()
                if step % self.log_interval == 0:
                    for name, val in loss_dict.items():
                        self.writer.add_scalar(f'train/{name}', val.item(), epoch * len(self.train_loader) + step)

            # epoch end
            avg_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('train/total_loss', avg_loss, epoch)

            # Save checkpoint
            if self.save_interval > 0:
                if epoch % self.save_interval == 0:
                    path = os.path.join(self.ckpt_dir, f'epoch_{epoch}')
                    self.model.save(path)
                    self.model.to(self.device)

            # Run evaluation
            if self.val_loader and epoch % self.eval_interval == 0:
                sup_metrics, unsup_metrics = self._evaluate(epoch)
                print_epoch_summary(epoch, 
                                    train_loss=avg_loss,
                                    sup_metrics=sup_metrics,
                                    unsup_metrics=unsup_metrics,
                                    total_epochs=self.num_epochs)

            # Early stopping check
            if self.early_stopping and self.es_metric:
                if self.no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        path = os.path.join(self.ckpt_dir, 'final')
        self.model.save(path)

        self.writer.close()

        print('HSIClustering training done !!')

        return None

    @torch.no_grad()
    def _ema_update_centroids(self, z1, p1, z2, p2, ema_decay, mass_thresh=1e-3, kick_scale=0.01):
        B, D, h, w = z1.shape
        N = B * h * w
        z1_flat = z1.reshape(N, D)
        p1_flat = p1.reshape(N, self.model.cluster.n_clusters)
        z2_flat = z2.reshape(N, D)
        p2_flat = p2.reshape(N, self.model.cluster.n_clusters)
    
        # concatenate both views
        z_cat = torch.cat([z1_flat, z2_flat], dim=0)  # (2N, D)
        p_cat = torch.cat([p1_flat, p2_flat], dim=0)  # (2N, K)
    
        # weighted mean per centroid
        mass = p_cat.sum(dim=0) + 1e-8
        valid = mass > mass_thresh
        new_centers = (p_cat.t() @ z_cat) / (mass.unsqueeze(1) + 1e-8)  # (K,D)
    
        # EMA blend
        cc_data = self.model.cluster.cluster_centers.data        # nn.Parameter(K, D)
        cc_data[valid] = (
            cc_data[valid] * ema_decay +
            new_centers[valid] * (1 - ema_decay)
        )

        # kick dead centroids with small noise
        dead_mask  = ~valid
        noise = torch.randn_like(cc_data, device=cc_data.device) * kick_scale
        cc_data[dead_mask] = F.normalize(cc_data[dead_mask] + noise[dead_mask], dim=1)

        self.model.cluster.cluster_centers.data = F.normalize(cc_data, p=2, dim=1)

        return None

    def _evaluate(self, epoch: int):
        """
        Evaluate on the validation set: compute supervised and unsupervised metrics.
        """
        self.model.eval()
        curr_epoch_preds = []
        sup_iou = sup_dice = sup_rmse = 0.
        n_sup = 0

        with torch.no_grad():
            for batch in self.val_loader:
                cubes, labels = batch
                cubes = cubes.to(self.device)
                preds = self.model.inference(cubes)[0]
                curr_epoch_preds.append(preds.cpu())

                if labels is not None:
                    labels = labels.to(self.device)
                    sup_iou  += iou_score(preds, labels).item()
                    sup_dice += dice_score(preds, labels).item()
                    sup_rmse += area_rmse(preds, labels).item()
                    n_sup   += 1

        # Log supervised metrics
        if labels is not None:
            if n_sup > 0:
                self.writer.add_scalar('val/IoU',  sup_iou / n_sup, epoch)
                self.writer.add_scalar('val/Dice', sup_dice / n_sup, epoch)
                self.writer.add_scalar('val/RMSE', sup_rmse / n_sup, epoch)

            sup_metrics = {'IoU': sup_iou / n_sup, 
                           'Dice': sup_dice / n_sup,
                           'RMSE': sup_rmse / n_sup}
        else:
            sup_metrics = None

        # Log unsupervised stability
        if self.prev_epoch_preds is not None:
            ent_sum = nmi_sum = vi_sum = count = 0
            for prev, curr in zip(self.prev_epoch_preds, curr_epoch_preds):
                ent_sum += cluster_entropy(curr).item()
                nmi_sum += normalized_mutual_information(prev, curr).item()
                vi_sum  += variation_of_information(prev, curr).item()
                count  += 1

            self.writer.add_scalar('val/entropy', ent_sum/count, epoch)
            self.writer.add_scalar('val/NMI',     nmi_sum/count, epoch)
            self.writer.add_scalar('val/VI',      vi_sum/count, epoch)

            # Early stop metric update
            if self.es_metric:
                metric_val = self.es_metric({
                    'NMI': nmi_sum / n_unsup,
                    'VI': vi_sum   / n_unsup,
                    'entropy': ent_sum / n_unsup
                })
                if self.best_metric is None or metric_val > self.best_metric:
                    self.best_metric = metric_val
                    self.no_improve  = 0
                else:
                    self.no_improve += 1

            unsup_metrics = {'entropy': ent_sum/count,
                             'NMI': nmi_sum/count,
                             'VI': vi_sum/count}
        else:
            unsup_metrics = None

        self.prev_epoch_preds = curr_epoch_preds

        return sup_metrics, unsup_metrics

    def inference(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        auto_align: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Run inference on arbitrary-size cubes.

        Returns:
            preds: Tensor of shape (N, H, W)
            If dataset provides labels and auto_align=True: also returns a
            metrics dict {{'IoU', 'Dice', 'RMSE'}}.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size or 1,
            shuffle=False,
            num_workers=num_workers or 0,
            pin_memory=pin_memory,
            collate_fn=pad_and_stack
        )
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                cubes, labels = batch
                cubes = cubes.to(self.device)
                preds = self.model.inference(cubes)
                all_preds.append(preds.cpu())
                if labels is not None:
                    all_labels.append(label)

        preds_tensor = torch.stack(all_preds, dim=0)
        if not all_labels:
            return preds_tensor

        labels_tensor = torch.stack(all_labels, dim=0)

        # Auto-align clusters to labels via Hungarian
        if auto_align:
            K = int(preds_tensor.max()) + 1
            C = int(labels_tensor.max()) + 1
            if K == C:
                flat_p = preds_tensor.view(-1)
                flat_l = labels_tensor.view(-1)
                conf = torch.zeros(C, K, dtype=torch.int64)
                for c in range(C):
                    mask = flat_l == c
                    conf[c] = torch.bincount(flat_p[mask], minlength=K)
                row, col = linear_sum_assignment(-conf.numpy())
                mapping = torch.arange(K)
                for r, c in zip(row, col):
                    mapping[c] = r
                aligned = mapping[preds_tensor]
            else:
                aligned = preds_tensor
        else:
            aligned = preds_tensor

        metrics = {
            'IoU':  iou_score(aligned, labels_tensor).item(),
            'Dice': dice_score(aligned, labels_tensor).item(),
            'RMSE': area_rmse(aligned, labels_tensor).item()
        }
    
        return aligned, metrics



