import os
import math
from typing import Optional, Callable

from torch.utils.data import Dataset
import torch.nn as nn
import torch

from .trainer import HSIClusteringTrainer, pad_and_stack, print_epoch_summary
from .data_server import DataServer

__all__ = ["AsyncHSIClusteringTrainer"]

class AsyncHSIClusteringTrainer(HSIClusteringTrainer):
    """Trainer variant that pulls batches asynchronously from a ``DataServer``."""

    def __init__(
        self,
        data_server: DataServer,
        val_dataset: Optional[Dataset] = None,
        steps_per_epoch: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(train_dataset=None, val_dataset=val_dataset, reuse_iter=1, *args, **kwargs)

        self.data_server = data_server
        self.steps_per_epoch = steps_per_epoch or math.ceil(len(self.data_server.dataset) / self.batch_size)

    def train(self):
        if self.data_server is None:
            print("⚠️  No data_server provided—skipping train().")
            return None

        self.data_server.start()

        loss_weight_kwargs = {}
        for loss_term in self.loss_weight_scheduler:
            if self.loss_weight_scheduler[loss_term]:
                loss_weight_kwargs[loss_term] = self.loss_weight_scheduler[loss_term]()
            else:
                loss_weight_kwargs[loss_term] = None

        ema_kick_scale = self.ema_kick

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step in range(1, self.steps_per_epoch + 1):
                cubes, _ = self.data_server.get_batch(self.batch_size)
                if cubes.device != self.device:
                    cubes = cubes.to(self.device, non_blocking=True)

                crops = self.augmentor(cubes)
                c0, c1 = crops[:, 0], crops[:, 1]

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.precision != 'fp32')):
                    loss, loss_dict, ema_dict = self.model.train_step(c0, c1, **loss_weight_kwargs)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                z1, p1 = ema_dict['z1'], ema_dict['p1']
                z2, p2 = ema_dict['z2'], ema_dict['p2']
                self._ema_update_centroids(z1, p1, z2, p2, self.ema_decay, ema_kick_scale)

                running_loss += loss.item()
                if step % self.log_interval == 0:
                    for name, val in loss_dict.items():
                        record_iter = epoch * self.steps_per_epoch + step
                        self.writer.add_scalar(f'train/{name}', val.item(), record_iter)

            avg_loss = running_loss / self.steps_per_epoch
            self.writer.add_scalar('train/total_loss', avg_loss, epoch)

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                path = os.path.join(self.ckpt_dir, f'epoch_{epoch}')
                self.model.save(path)
                self.model.to(self.device)

            if self.val_loader and epoch % self.eval_interval == 0:
                sup_metrics, unsup_metrics = self._evaluate(epoch)
                print_epoch_summary(epoch,
                                    train_loss=avg_loss, 
                                    sup_metrics=sup_metrics, 
                                    unsup_metrics=unsup_metrics, 
                                    total_epochs=self.num_epochs)

            if self.optim_scheduler:
                self.optim_scheduler.step()

            if self.ema_kick_scheduler:
                ema_kick_scale = self.ema_kick_scheduler.step()

            for loss_term in self.loss_weight_scheduler:
                if self.loss_weight_scheduler[loss_term]:
                    loss_weight_kwargs[loss_term] = self.loss_weight_scheduler[loss_term].step()

            if self.early_stopping and self.es_metric:
                if self.no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        path = os.path.join(self.ckpt_dir, 'final')
        self.model.save(path)
        self.writer.close()
        self.data_server.stop()
        print('HSIClustering training done !!')
        return None


