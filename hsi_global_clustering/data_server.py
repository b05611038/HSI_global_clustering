import random
import asyncio
import torch.multiprocessing as mp
import torch
import queue
from torch.utils.data import Dataset

import sys

from .trainer import pad_and_stack

__all__ = ["DataServer"]

class DataServer:
    """Shared-memory data server with async I/O and double buffering.

    Preloads items from a `Dataset` on a background process. Async I/O is
    handled in `_async_loop`, while `get_batch` returns a flag indicating
    whether new data was retrieved. If no new data is available, get_batch
    returns (False, None, None), allowing the main loop to retry.
    """

    def __init__(
        self,
        dataset: Dataset,
        queue_size: int = 2,
        shuffle: bool = True,
    ):

        dataset_size = len(dataset)
        self.queue_size = min(queue_size, dataset_size)
        self.dataset = dataset
        self.shuffle = shuffle

        # shared queues
        self._ctx = mp.get_context("spawn")
        self._ready_queue = self._ctx.Queue(maxsize=queue_size)
        self._free_queue = self._ctx.Queue(maxsize=queue_size)
        for _ in range(queue_size):
            self._free_queue.put(None)

        # Event to signal first available item
        self._start_event = self._ctx.Event()
        self._proc = None

    def start(self):
        """Launch the background worker."""
        if self._proc is None:
            self._proc = self._ctx.Process(target=self._worker, daemon=True)
            self._proc.start()
            # block until at least one item has been loaded
            self._start_event.wait()

    def _worker(self):
        """Worker process to fill buffers asynchronously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_loop())

    async def _async_loop(self):
        """Asynchronous loop to load data into buffers."""
        idxs = list(range(len(self.dataset)))
        rng = random.Random()

        # Initial fill: load exactly queue_size samples
        initial = True
        while True:
            # Refill and shuffle indices when exhausted
            if not idxs:
                idxs = list(range(len(self.dataset)))
                if self.shuffle:
                    rng.shuffle(idxs)

            idx = idxs.pop(0)

            await asyncio.to_thread(self._free_queue.get)
            item = await asyncio.to_thread(self.dataset.__getitem__, idx)
            if isinstance(item, tuple) and len(item) == 2:
                cube, label = item
            else:
                cube, label = item, None

            cube.share_memory_()
            if label is not None:
                label.share_memory_()
                packed = (cube, label)
            else:
                packed = cube

            await asyncio.to_thread(self._ready_queue.put, packed)
            if initial:
                try:
                    if self._ready_queue.qsize() >= self.queue_size:
                        self._start_event.set()
                        initial = False
                except Exception:
                    self._start_event.set()
                    initial = False

    def get_batch(self, batch_size: int):
        """Retrieve a batch of items.

        Returns:
            has_new (bool): True if new data was loaded, False otherwise.
            cubes (Tensor) or None: batched data if has_new is True.
            labels (Tensor) or None: batched labels if has_new is True.
        """
        # Check if enough items are ready
        try:
            ready = self._ready_queue.qsize()
        except Exception:
            # Fallback: try peeking
            try:
                item = self._ready_queue.get_nowait()
                self._ready_queue.put(item)
                ready = 1
            except Exception:
                ready = 0

        if ready < batch_size:
            return False, None, None

        # Retrieve exactly batch_size samples
        samples = [self._ready_queue.get() for _ in range(batch_size)]

        # Pad and stack samples
        cubes, labels = pad_and_stack(samples)

        # Release free-slot tokens for the consumed slots
        for _ in range(batch_size):
            self._free_queue.put(None)

        return True, cubes, labels

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join()
            self._proc = None
            self._start_event.clear()


