from typing import Optional, Generator, Sequence

import numpy as np
import pandas as pd
import pydantic
import astropy.units as u
from tqdm import tqdm

from exo_finder.data_pipeline.generation.time_generation import generate_time_days_of_length
from exo_finder.data_pipeline.generation.transit_generation import (
    PlanetType,
    generate_transit_parameters,
    generate_transits_from_params,
)
from exo_finder.default_datasets import gaia_dataset


class TransitProfile(pydantic.BaseModel):
    planet_type: PlanetType
    transit_period_range: tuple[float, float]
    transit_midpoint_range: tuple[float, float]
    weight: float = 1


class SyntheticTransitDatasetGenerationParameters(pydantic.BaseModel):
    dataset_length: int
    lightcurve_length_points: int
    transits_distribution: list[tuple[float, Optional[TransitProfile]]]


def _get_gaia_star_parameters() -> pd.DataFrame:
    gaia_fields = ["gaia_id", "radius", "mass_flame", "teff_mean"]
    return gaia_dataset.load_gaia_parameters_dataset().view[gaia_fields].to_pandas().dropna()


def generate_synthetic_transits(params: SyntheticTransitDatasetGenerationParameters, seed: int) -> Generator:
    transit_probabilities = [x[0] for x in params.transits_distribution]
    transit_profiles = [x[1] for x in params.transits_distribution]
    if sum(transit_probabilities) != 1.0:
        raise ValueError("Transit probabilities must sum to 1.0")

    np.random.seed(seed)

    gaia_data = _get_gaia_star_parameters()
    gaia_data = gaia_data.sample(
        n=params.dataset_length,
        replace=params.dataset_length > len(gaia_data),
        random_state=seed,
    )

    random_profiles: Sequence[TransitProfile | None] = np.random.choice(
        transit_profiles,
        size=params.dataset_length,
        p=transit_probabilities,
        replace=True,
    )

    time_in_days = generate_time_days_of_length(length=params.lightcurve_length_points)
    flat_transits = np.zeros_like(time_in_days)

    for (gaia_i, gaia_row), transit_profile in tqdm(zip(gaia_data.iterrows(), random_profiles)):
        if transit_profile is None:
            yield flat_transits
            continue

        generated_prams = generate_transit_parameters(
            planet_type=transit_profile.planet_type,
            orbital_period_interval=transit_profile.transit_period_range,
            transit_midpoint_range=transit_profile.transit_midpoint_range,
            star_radius=gaia_row["radius"] * u.solRad,
            star_mass=gaia_row["mass_flame"] * u.solMass,
            star_t_eff=gaia_row["teff_mean"] * u.K,
        )

        yield generate_transits_from_params(params=generated_prams, time_x=time_in_days)


if __name__ == "__main__":
    p = SyntheticTransitDatasetGenerationParameters(
        dataset_length=1000,
        lightcurve_length_points=2**14,
        transits_distribution=[
            (0.25, None),
            (
                0.75,
                TransitProfile(
                    planet_type=PlanetType.JUPITER,
                    transit_period_range=(2, 5),
                    transit_midpoint_range=(0, 5),
                ),
            ),
        ],
    )
    generator = generate_synthetic_transits(p, seed=8)
    lel = list(generator)
    blel = np.array([(x != 0).sum() for x in lel])
    blyat = (blel == 0).sum() / len(lel)
    print(blyat)
