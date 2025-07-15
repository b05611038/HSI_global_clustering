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

def crop_and_stack(batch: list):
    """
    Crop a batch of cubes (and optional labels) to the same H_min x W_min, then stack.

    Args:
      batch: list of items, each either
             - Tensor(C, H, W)
             - or (Tensor(C, H, W), Tensor(H, W))
    Returns:
      cubes:  Tensor(B, C, H_min, W_min)
      labels: Tensor(B, H_min, W_min) if labels present else None
    """
    has_labels = isinstance(batch[0], tuple)
    cubes = []
    labels = [] if has_labels else None
    for item in batch:
        if has_labels:
            c, l = item
            cubes.append(c)
            labels.append(l)
        else:
            cubes.append(item)

    # compute minimum dims
    hs = [c.shape[1] for c in cubes]
    ws = [c.shape[2] for c in cubes]
    H_min, W_min = min(hs), min(ws)

    # center‐crop each cube
    cropped_cubes = []
    for c in cubes:
        C, h, w = c.shape
        top  = (h - H_min) // 2
        left = (w - W_min) // 2
        cropped = c[:, top : top + H_min, left : left + W_min]
        cropped_cubes.append(cropped)
    batch_cubes = torch.stack(cropped_cubes, dim=0)

    # center‐crop labels if present
    if has_labels:
        cropped_labels = []
        for l in labels:  # each l is (H, W)
            h, w = l.shape
            top  = (h - H_min) // 2
            left = (w - W_min) // 2
            cropped_l = l[top : top + H_min, left : left + W_min]
            cropped_labels.append(cropped_l)
        batch_labels = torch.stack(cropped_labels, dim=0)
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
        reuse_iter:    Resample times of single HSI cube acqusition.
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
        optimizer_kwargs: dict = {'lr': 1e-4},
        scheduler_cls: Optional[Callable] = None,
        scheduler_kwargs: dict = {},
        ema_decay: float = 0.95,
        batch_size: int = 4,
        num_workers: int = 8,
        num_epochs: int = 10,
        reuse_iter: int = 10,
        grad_clip: float = 1.0,
        precision: str = 'bf16',
        seed: int = 42,
        log_dir: str = './runs',
        ckpt_dir: str = './checkpoints',
        save_interval: int = 1,
        eval_interval: int = 1,
        log_interval: int = 10,
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
        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=pad_and_stack,
            )
        else:
            self.train_loader = None

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
            if train_dataset is not None:
                self.val_loader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=crop_and_stack,
                )
            else:
                self.val_loader = None

        # Optimizer and optional scheduler
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs) if scheduler_cls else None
        
        # Training settings
        self.num_epochs = num_epochs
        self.reuse_iter = reuse_iter
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
        if self.train_loader is None:
            # nothing to train on; just return
            print("⚠️  No train_dataset provided—skipping train().")
            return None

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step, batch in enumerate(self.train_loader, start=1):
                cubes, _ = batch
                cubes = cubes.to(self.device, non_blocking=True)        # (B, C, H, W)
                for reuse_step in range(1, self.reuse_iter + 1):
                    crops = self.augmentor(cubes)                       # (B, 2, C, h, w)

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
                            record_iter = epoch * len(self.train_loader) * self.reuse_iter + \
                                    (step - 1) * self.reuse_iter + reuse_step
                            self.writer.add_scalar(f'train/{name}', val.item(), record_iter) 

            # epoch end
            avg_loss = running_loss / (len(self.train_loader) * self.reuse_iter)
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
    def _ema_update_centroids(self, z1, p1, z2, p2, ema_decay, 
                              mass_thresh=1e-3, kick_scale=0.01, eps=1e-6):
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
        new_centers = (p_cat.t() @ z_cat) / (mass.unsqueeze(1) + eps)  # (K,D)
    
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

    @torch.no_grad()
    def _evaluate(self, epoch: int):
        """
        Evaluate on the validation set: compute supervised and unsupervised metrics.
        """
        self.model.eval()
        curr_epoch_preds = []
        sup_iou = sup_dice = sup_rmse = 0.
        n_sup = 0

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

    @torch.no_grad()
    def inference(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        auto_align: bool = True,
        manual_mapping: Optional[Dict[int, Tuple[int, ...]]] = None,  # ← new
        progress_interval: Optional[int] = None,
    ):
        """
        Run inference on arbitrary-size cubes.

        Args:

            auto_align:  If True, do Hungarian-based alignment (as before).
            manual_mapping: If provided (and auto_align=False), should be a dict
                mapping each label index to a tuple of cluster indices. E.g.
                    { 0: (0,1),   # clusters 0 & 1 → label 0
                      1: (2,) }  # cluster 2   → label 1
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size or 1,
            shuffle=False,
            num_workers=num_workers or 0,
            pin_memory=pin_memory,
            collate_fn=pad_and_stack
        )

        self.model.eval()
        self.model.to(self.device)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                cubes, labels = batch
                cubes = cubes.to(self.device)
                preds = self.model.inference(cubes)
                all_preds.append(preds.long().cpu())
                if labels is not None:
                    all_labels.append(labels.long())

                if progress_interval is not None:
                    if (step + 1) % progress_interval == 0:
                        print(f"Inference progress: {step+1}/{len(loader)}")

        if len(all_labels) == 0:
            return all_preds

        preds_list  = all_preds   # list of Tensors [Hᵢ×Wᵢ]
        labels_list = all_labels  # list of Tensors [Hᵢ×Wᵢ]

        # ----- build mapping_arr depending on auto_align or manual_mapping -----
        if auto_align:
            # infer number of clusters (K) and labels (C)
            K = int(max(p.max().item() for p in preds_list)) + 1
            C = int(max(l.max().item() for l in labels_list)) + 1
            if K == C:
                # build confusion matrix of shape (K, C)
                device = preds_list[0].device
                conf = torch.zeros(K, C, device=device, dtype=torch.long)
                for p, l in zip(preds_list, labels_list):
                    fp = p.view(-1)
                    fl = l.view(-1)
                    idx = torch.stack([fp, fl], dim=0)
                    conf.index_put_(tuple(idx), torch.ones_like(fp), accumulate=True)

                # Hungarian on negative counts → maximize agreement
                row, col = linear_sum_assignment((-conf).cpu().numpy())
                mapping_arr = torch.arange(K, device=device)
                for r, c in zip(row, col):
                    mapping_arr[c] = r
            else:
                # mismatch: skip alignment
                mapping_arr = None

        elif manual_mapping is not None:
            K = int(max(p.max().item() for p in preds_list)) + 1
            device = preds_list[0].device
            mapping_arr = torch.full((K,), -1, dtype=torch.long, device=device)
            for label_id, cluster_idxs in manual_mapping.items():
                for c_idx in cluster_idxs:
                    mapping_arr[c_idx] = label_id
            missing = (mapping_arr < 0).nonzero(as_tuple=False).view(-1).tolist()
            if missing:
                raise ValueError(f"manual_mapping missing clusters: {missing}")

        else:
            mapping_arr = None  # identity

        # ----- apply mapping_arr (if any) to each sample -----
        if mapping_arr is not None:
            aligned_list = [mapping_arr[p] for p in preds_list]
        else:
            aligned_list = preds_list

        # ----- compute per-sample metrics and average them -----
        iou_vals  = [iou_score(a, l).item() for a, l in zip(aligned_list, labels_list)]
        dice_vals = [dice_score(a, l).item() for a, l in zip(aligned_list, labels_list)]
        rmse_vals = [area_rmse(a, l).item() for a, l in zip(aligned_list, labels_list)]

        metrics = {
            'mean_IoU':  sum(iou_vals)  / len(iou_vals),
            'mean_Dice': sum(dice_vals) / len(dice_vals),
            'mean_RMSE': sum(rmse_vals) / len(rmse_vals),
        }

        # 4) **Per‐class** IoU / Dice
        # Flatten all images into one vector each
        pred_flat  = torch.cat([p.view(-1) for p in aligned_list])
        label_flat = torch.cat([l.view(-1) for l in all_labels])
        C = int(max(pred_flat.max(), label_flat.max())) + 1

        # Build confusion matrix (rows=true, cols=pred)
        conf = torch.zeros(C, C, device=pred_flat.device, dtype=torch.long)
        idx  = torch.stack([label_flat, pred_flat], dim=0)
        conf.index_put_(tuple(idx), torch.ones_like(label_flat), accumulate=True)

        inter   = conf.diag().float()
        sum_row = conf.sum(dim=1).float()
        sum_col = conf.sum(dim=0).float()
        union   = sum_row + sum_col - inter

        class_iou  = inter / union
        class_dice = 2 * inter / (sum_row + sum_col)

        # merge into metrics
        metrics.update({
            **{f'IoU_class_{i}':  class_iou[i].item()  for i in range(C)},
            **{f'Dice_class_{i}': class_dice[i].item() for i in range(C)},
        })

        return aligned_list, metrics


