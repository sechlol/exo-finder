from typing import Optional, Sequence, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from exo_finder.utils.hdf5_wrapper import H5Reader


class H5TorchDataset(Dataset):
    """
    Thin PyTorch Dataset over H5Reader.
    - Optional 'fields' to restrict what you fetch (default: all).
    - Optional 'indices' to create a split.
    - Returns a dict[str, torch.Tensor].
    """

    def __init__(
        self,
        path: str,
        *,
        fields: Optional[Sequence[str]] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        self.reader = H5Reader(path)
        self.fields = tuple(fields) if fields is not None else self.reader.fields
        self.indices = np.asarray(indices) if indices is not None else None
        self._length = len(self.indices) if self.indices is not None else self.reader.length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ridx = int(self.indices[i]) if self.indices is not None else i
        batch = self.reader.get(ridx, fields=self.fields)
        return {k: torch.from_numpy(v) for k, v in batch.items()}


class LocalityBatchSampler(Sampler[Sequence[int]]):
    """
    Shuffle once per epoch, then sort indices *within* each batch
    to reduce chunk thrash while keeping batches random.
    """

    def __init__(self, size: int, batch_size: int, drop_last: bool = True):
        self.size = size
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

    def __iter__(self) -> Iterable[Sequence[int]]:
        perm = np.random.permutation(self.size)
        B = self.batch_size
        n_full = self.size // B
        for b in range(n_full):
            yield np.sort(perm[b * B : (b + 1) * B]).tolist()
        if not self.drop_last and self.size % B:
            yield np.sort(perm[n_full * B :]).tolist()

    def __len__(self) -> int:
        return self.size // self.batch_size if self.drop_last else (self.size + self.batch_size - 1) // self.batch_size
