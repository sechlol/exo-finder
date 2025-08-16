from dataclasses import dataclass
from typing import Optional

from exo_finder.data_pipeline.generation.transit_generation import PlanetType, PeriodFrequency


@dataclass
class DataGenerationParams:
    dataset_size: int | None
    planet_type: PlanetType
    transits_probabilities: tuple[float, float]
    fixed_lc_length: Optional[int] = None
    period_minmax: Optional[tuple[float, float]] = None
    days_num_minmax: Optional[tuple[int, int]] = None
    midpoint_minmax: Optional[tuple[float, float]] = None
    star_radius_minmax: tuple[float, float] = (0.7, 1.3)
    planet_period: Optional[PeriodFrequency] = None
    patch_length: int = 1
