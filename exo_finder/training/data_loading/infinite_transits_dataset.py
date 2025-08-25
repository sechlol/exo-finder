from typing import Optional
import lightning as L
import astropy.units as u
import numpy as np
import pandas as pd
import torch
from numpy.random import Generator
from torch.utils.data import DataLoader, IterableDataset

from exo_finder.data_pipeline.generation.planetary_parameters import PlanetaryParameters
from exo_finder.data_pipeline.generation.time_generation import generate_time_days_of_length
from exo_finder.data_pipeline.generation.transit_generation import (
    generate_transit_parameters,
    generate_transits_from_params,
)
from exo_finder.data_pipeline.p5_synthetic_set_creation import define_data_balance
from exo_finder.default_datasets import gaia_dataset


def _get_gaia_star_parameters() -> pd.DataFrame:
    gaia_fields = ["gaia_id", "radius", "mass_flame", "teff_mean"]
    return gaia_dataset.load_gaia_parameters_dataset().view[gaia_fields].to_pandas().dropna()


class InfiniteTransitDataset(IterableDataset):
    """ """

    def __init__(self, seed: int, length: Optional[int] = None):
        super(InfiniteTransitDataset).__init__()
        self._seed = seed
        self._parameters = None
        self._probs = None
        self._time_x = None
        self._gaia_df: pd.DataFrame = None
        self._random_generator: Generator = None
        self._length = length

    def _init_process_dependent_data(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print(f"init worker {worker_info.id}, seed: {worker_info.seed}, num_workers: {worker_info.num_workers}")
            self._seed += worker_info.seed
            if self._length is not None:
                self._length = self._length // worker_info.num_workers

        print("Loading Gaia DB...")
        self._gaia_df = _get_gaia_star_parameters()
        print(f"Loaded Gaia DB {len(self._gaia_df)}")

        self._parameters = define_data_balance()
        self._random_generator = np.random.default_rng(seed=self._seed)
        self._probs = self._parameters.get_probabilities()
        self._time_x = generate_time_days_of_length(self._parameters.lightcurve_length_points)

    def __iter__(self):
        if self._random_generator is None:
            self._init_process_dependent_data()

        if self._length is None:
            while True:
                yield self._generate_one_transit()
        else:
            for _ in range(self._length):
                yield self._generate_one_transit()

    def _generate_one_transit(self):
        gaia_row = self._gaia_df.sample(n=1, random_state=self._random_generator.integers(low=0, high=2e9))

        i = self._random_generator.integers(low=0, high=self._parameters.count_distributions())
        p, generation_specs = self._parameters.transits_distribution[i]
        if generation_specs is not None:
            transit_parameters = generate_transit_parameters(
                planet_type=generation_specs.planet_type,
                orbital_period_interval=generation_specs.transit_period_range,
                transit_midpoint_range=generation_specs.transit_midpoint_range,
                star_radius=gaia_row["radius"].item() * u.solRad,
                star_mass=gaia_row["mass_flame"].item() * u.solMass,
                star_t_eff=gaia_row["teff_mean"].item() * u.K,
                rnd_generator=self._random_generator,
            )
            generated_flux = generate_transits_from_params(
                params=transit_parameters,
                time_x=self._time_x,
                median_flux=0,
            )
            parameters_array = transit_parameters.to_numpy()
        else:
            parameters_array = np.zeros(PlanetaryParameters.parameter_count(), dtype=np.float32)
            generated_flux = np.zeros(self._parameters.lightcurve_length_points, dtype=np.float32)

        return {"syn_lc_data": generated_flux, "params": parameters_array}


class InfiniteLcDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            InfiniteTransitDataset(seed=1),
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=6,
            prefetch_factor=3,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            InfiniteTransitDataset(seed=2, length=2**12),
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            InfiniteTransitDataset(seed=3, length=2**12),
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            InfiniteTransitDataset(seed=4, length=2**12),
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=3,
        )
