from typing import Optional, Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, BatchSampler, SequentialSampler
from typing_extensions import Self

import exo_finder.constants as c
from exo_finder.default_datasets import get_train_dataset_h5
from exo_finder.training.base.block_batch_sampler import BlockBatchSampler
from exo_finder.utils.hdf5_wrapper import H5Wrapper


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
        self._reader: Optional[H5Wrapper] = None
        self._keys = tuple(keys_to_fetch)
        self._columns = np.array(columns, dtype=np.uint32) if columns is not None else None

        tmp_reader = get_train_dataset_h5()
        all_lengths = np.array([tmp_reader.get_shape(k)[0] for k in keys_to_fetch])
        assert np.all(all_lengths == all_lengths[0]), "All data must have same length"

        self._length = all_lengths[0].item()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int | Sequence[int]) -> dict[str, torch.Tensor]:
        # Lazy initialization of
        if self._reader is None:
            self._reader = get_train_dataset_h5()

        # Benchmark training speed without i/o overhead
        # return {k: torch.zeros(2**12) for k in self._keys}

        if isinstance(idx, int):
            return {k: torch.from_numpy(self._reader.read_one(k, row=idx, cols=self._columns)) for k in self._keys}

        # helps h5py for faster fetching
        idx = np.asarray(idx, dtype=np.int64)
        return {k: torch.from_numpy(self._reader.read(k, rows=idx, cols=self._columns)) for k in self._keys}

    def __del__(self) -> None:
        # Close when the worker is torn down
        if self._reader is not None:
            self._reader.close()


class LcDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> Self:
        full = TorchLightcurveDataset(keys_to_fetch=[c.HDF5_KEY_SYNTHETIC_DATA])
        self.train_set, self.val_set, self.test_set = random_split(full, [0.80, 0.1, 0.1])
        return self

    def train_dataloader(self) -> DataLoader:
        sampler = BlockBatchSampler(len(self.train_set), batch_size=self.batch_size, drop_last=False)
        return DataLoader(
            self.train_set,
            batch_sampler=sampler,
            persistent_workers=True,
            num_workers=6,
            prefetch_factor=3,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = BatchSampler(SequentialSampler(self.val_set), batch_size=self.batch_size, drop_last=False)

        return DataLoader(
            self.val_set,
            batch_sampler=sampler,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = BatchSampler(SequentialSampler(self.test_set), batch_size=self.batch_size, drop_last=False)
        return DataLoader(
            self.test_set,
            batch_sampler=sampler,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def predict_dataloader(self) -> DataLoader:
        sampler = BatchSampler(SequentialSampler(self.test_set), batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.test_set, batch_sampler=sampler, num_workers=0)
