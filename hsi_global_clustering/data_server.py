import random
import multiprocessing as mp

from torch.utils.data import Dataset

from .trainer import pad_and_stack

__all__ = ["DataServer"]

class DataServer:
    """Asynchronous loader that prefetches data from a Dataset into a queue."""

    def __init__(self, dataset: Dataset, queue_size: int = 8, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.queue_size = queue_size
        self.shuffle = shuffle
        self.seed = seed
        self._queue = mp.Queue(maxsize=queue_size)
        self._proc = None

    def start(self):
        if self._proc is None:
            self._proc = mp.Process(target=self._worker, daemon=True)
            self._proc.start()

    def _worker(self):
        idxs = list(range(len(self.dataset)))
        rng = random.Random(self.seed)
        while True:
            if self.shuffle:
                rng.shuffle(idxs)
            for idx in idxs:
                self._queue.put(self.dataset[idx])

    def get_batch(self, batch_size: int):
        batch = [self._queue.get() for _ in range(batch_size)]
        return pad_and_stack(batch)

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join()
            self._proc = None
