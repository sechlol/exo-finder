import math

import numpy as np
from torch.utils.data import Sampler


class BlockBatchSampler(Sampler[list[int]]):
    """Shuffle batches, not rows; within each batch, rows are contiguous."""

    def __init__(self, n: int, batch_size: int, drop_last: bool = False) -> None:
        super().__init__()
        self.n = n
        self.bs = batch_size
        self.drop_last = drop_last
        self.n_batches = (n // batch_size) if drop_last else math.ceil(n / batch_size)
        self.blocks = list(range(self.n_batches))

    def __iter__(self):
        np.random.shuffle(self.blocks)
        for b in self.blocks:
            start = b * self.bs
            end = min((b + 1) * self.bs, self.n)
            if end - start < self.bs and self.drop_last:
                continue
            yield list(range(start, end))

    def __len__(self) -> int:
        return self.n_batches if not self.drop_last else (self.n // self.bs)
