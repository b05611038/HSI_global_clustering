import random
import asyncio
import torch.multiprocessing as mp
import torch
from typing import Union

from torch.utils.data import Dataset

from .trainer import pad_and_stack

__all__ = ["DataServer"]

class DataServer:
    """Shared-memory data server with async I/O and double buffering.

    This class continuously preloads items from a ``Dataset`` on a background
    process. Two memory buffers are cycled so that while one buffer is being
    consumed by the main process, the other is asynchronously filled with the
    next item. Loaded tensors are moved to ``device`` and placed in shared
    memory before being transferred, minimising inter-process communication
    overhead.

    Parameters
    ----------
    dataset : Dataset
        Dataset to sample from.
    queue_size : int, optional
        Number of preloaded items (default: 2).
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
        queue_size: int = 2,
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
        self._ready_queue = self._ctx.Queue(maxsize=queue_size)
        self._free_queue = self._ctx.Queue(maxsize=queue_size)

        for i in range(queue_size):
            self._free_queue.put(i)

        self._buffers = [None] * queue_size
        self._proc = None

    def start(self):
        """Launch the background worker."""
        if self._proc is None:
            self._proc = self._ctx.Process(target=self._worker, daemon=True)
            self._proc.start()

    def _worker(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_loop())

    async def _async_loop(self):
        idxs = list(range(len(self.dataset)))
        rng = random.Random(self.seed)
        while True:
            if self.shuffle:
                rng.shuffle(idxs)
            for idx in idxs:
                buf_idx = await asyncio.to_thread(self._free_queue.get)
                item = await asyncio.to_thread(self.dataset.__getitem__, idx)

                if isinstance(item, tuple):
                    cube, label = item
                else:
                    cube, label = item, None

                cube = cube.to(self.device, non_blocking=True)
                cube.share_memory_()
                if label is not None:
                    label = label.to(self.device, non_blocking=True)
                    label.share_memory_()

                self._buffers[buf_idx] = (cube, label)
                await asyncio.to_thread(self._ready_queue.put, buf_idx)

    def get_batch(self, batch_size: int):
        """Retrieve ``batch_size`` items from the server.

        Each returned tensor resides in shared memory and is moved back to the
        free queue once stacked into a batch.
        """
        batch = []
        buf_idxs = []
        for _ in range(batch_size):
            buf_idx = self._ready_queue.get()
            batch.append(self._buffers[buf_idx])
            buf_idxs.append(buf_idx)

        cubes, labels = pad_and_stack(batch)

        for idx in buf_idxs:
            self._free_queue.put(idx)

        return cubes, labels

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join()
            self._proc = None
