from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from exo_finder.default_datasets import get_train_dataset_h5


class TorchLightcurveDataset(Dataset):
    """
    Thin PyTorch Dataset over H5Reader.
    - Optional 'fields' to restrict what you fetch (default: all).
    - Optional 'indices' to create a split.
    - Returns a dict[str, torch.Tensor].
    """

    def __init__(
        self,
        keys_to_fetch: Sequence[str],
        columns: Optional[Sequence[int]] = None,
    ):
        self._reader = get_train_dataset_h5()
        self._keys = tuple(keys_to_fetch)
        self._columns = np.array(columns, dtype=np.uint32) if columns is not None else None

        all_lengths = np.array([self._reader.get_shape(k)[0] for k in keys_to_fetch])
        assert np.all(all_lengths == all_lengths[0]), "All data must have same length"

        self._length = all_lengths[0].item()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int | Sequence[int]) -> dict[str, torch.Tensor]:
        if isinstance(idx, int):
            return {k: torch.from_numpy(self._reader.read_one(k, row=idx, cols=self._columns)) for k in self._keys}

        # helps h5py for faster fetching
        idx = np.asarray(idx, dtype=np.int64)
        return {k: torch.from_numpy(self._reader.read(k, rows=idx, cols=self._columns)) for k in self._keys}
