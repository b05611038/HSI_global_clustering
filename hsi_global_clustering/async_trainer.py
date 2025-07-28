import os
import math
from typing import Optional, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import threading

from torch.utils.data import Dataset

from .trainer import HSIClusteringTrainer, pad_and_stack, print_epoch_summary

__all__ = ["AsyncHSIClusteringTrainer"]


class DataServerRpc:
    """Remote data server that prefetches batches via RPC."""

    def __init__(
        self,
        mat_paths: List[str],
        batch_size: int,
        loader_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.mat_paths = mat_paths
        self.batch_size = batch_size
        self.loader_fn = loader_fn
        self.buf_a: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.buf_b: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.current = self.buf_a
        self.next = self.buf_b
        self.idx = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._prefetch_loop, daemon=True).start()

    def _prefetch_loop(self) -> None:
        N = len(self.mat_paths)
        while True:
            while len(self.next) < self.batch_size:
                path = self.mat_paths[self.idx]
                cube, label = self.loader_fn(path)
                cube = cube.pin_memory()
                label = label.pin_memory() if label is not None else None
                self.next.append((cube, label))
                self.idx = (self.idx + 1) % N

            while True:
                with self.lock:
                    if len(self.current) == 0:
                        break
                pass

            with self.lock:
                self.current, self.next = self.next, self.current

    @rpc.functions.async_execution
    def get_batch(self) -> RRef:
        with self.lock:
            batch = [self.current.pop(0) for _ in range(self.batch_size)]

        cubes, labels = pad_and_stack(batch)

        fut: torch.futures.Future = torch.futures.Future()
        fut.set_result(RRef((cubes, labels)))
        return fut


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

class AsyncHSIClusteringTrainer(HSIClusteringTrainer):
    """Trainer variant that pulls batches asynchronously from ``DataServerRpc``."""

    def __init__(
        self,
        mat_paths: List[str],
        loader_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
        server_name: str = "server",
        trainer_name: str = "trainer",
        rpc_init_method: str = "tcp://localhost:29500",
        val_dataset: Optional[Dataset] = None,
        steps_per_epoch: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(train_dataset=None, val_dataset=val_dataset, reuse_iter=1, *args, **kwargs)

        rpc.init_rpc(
            name=trainer_name,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method),
        )

        self.server_name = server_name
        self.server_rref = rpc.remote(
            to=server_name,
            func=DataServerRpc,
            args=(mat_paths, self.batch_size, loader_fn),
        )

        self.steps_per_epoch = steps_per_epoch or math.ceil(len(mat_paths) / self.batch_size)

    def get_batch(self, timeout=None):
        batch_rref: RRef = rpc.rpc_sync(
            to=self.server_name,
            func=_call_method,
            args=(DataServerRpc.get_batch, self.server_rref),
        )
        cubes, labels = batch_rref.to_here()
        cubes = cubes.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True) if labels is not None else None
        return cubes, labels

    def train(self):
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
                cubes, _ = self.get_batch()
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
        rpc.shutdown()
        print('HSIClustering training done !!')
        return None


