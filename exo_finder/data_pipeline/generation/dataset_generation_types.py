from typing import Optional

import pydantic

from exo_finder.data_pipeline.generation.transit_generation import PlanetType


class TransitProfile(pydantic.BaseModel):
    planet_type: PlanetType
    transit_period_range: tuple[float, float]
    transit_midpoint_range: tuple[float, float]
    weight: float = 1


class SyntheticTransitGenerationParameters(pydantic.BaseModel):
    dataset_length: int
    lightcurve_length_points: int
    transits_distribution: list[tuple[float, Optional[TransitProfile]]]


class SyntheticLcDatasetMetadata(pydantic.BaseModel):
    lc_data_shape: tuple[int, int]
    params_data_shape: tuple[int, int]
    lc_window_size: int
    generation_parameters: SyntheticTransitGenerationParameters
