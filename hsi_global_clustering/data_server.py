import random
import torch.multiprocessing as mp
import torch
from typing import Union

from torch.utils.data import Dataset

from .trainer import pad_and_stack

__all__ = ["DataServer"]

class DataServer:
    """Asynchronous loader that prefetches data from a ``Dataset`` into a queue.

    Parameters
    ----------
    dataset : Dataset
        Dataset to sample from.
    queue_size : int, optional
        Maximum number of prefetched items.
    shuffle : bool, optional
        Whether to shuffle indices each epoch.
    seed : int, optional
        Random seed for shuffling.
    device : str or torch.device, optional
        Device to move tensors to before enqueueing. Defaults to ``"cuda"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        queue_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
        device: Union[str, torch.device] = "cuda",
    ):
        self.dataset = dataset
        self.queue_size = queue_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = torch.device(device)
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=queue_size)
        self._proc = None

    def start(self):
        """Launch the background worker."""
        if self._proc is None:
            self._proc = self._ctx.Process(target=self._worker, daemon=True)
            self._proc.start()

    def _worker(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        idxs = list(range(len(self.dataset)))
        rng = random.Random(self.seed)
        while True:
            if self.shuffle:
                rng.shuffle(idxs)
            for idx in idxs:
                item = self.dataset[idx]
                if isinstance(item, tuple):
                    cube, label = item
                    cube = cube.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)
                    self._queue.put((cube, label))
                else:
                    cube = item.to(self.device, non_blocking=True)
                    self._queue.put(cube)

    def get_batch(self, batch_size: int):
        batch = [self._queue.get() for _ in range(batch_size)]
        return pad_and_stack(batch)

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join()
            self._proc = None
