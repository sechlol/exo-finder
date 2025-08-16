from typing import Any, NamedTuple

import batman
import numpy as np

from .time_generation import time_of_first_transit_midpoint


class PlanetaryParameters(NamedTuple):
    period_d: float
    transit_midpoint_d: float
    planet_radius_solrad: float
    star_radius_solrad: float
    semi_major_axis_solrad: float
    planet_mass_solmass: float
    inclination_deg: float
    eccentricity: float
    argument_of_periastron_deg: float
    limb_darkening_c1: float
    limb_darkening_c2: float

    @property
    def first_transit_midpoint_d(self) -> float:
        return time_of_first_transit_midpoint(t0=0, transit_midpoint=self.transit_midpoint_d, period=self.period_d)

    @property
    def planet_to_star_radius(self) -> float:
        return self.planet_radius_solrad / self.star_radius_solrad

    @property
    def semimajor_axis_to_star_radius(self) -> float:
        return self.semi_major_axis_solrad / self.star_radius_solrad

    def to_batman(self) -> batman.TransitParams:
        params = batman.TransitParams()
        params.t0 = self.transit_midpoint_d
        params.rp = self.planet_to_star_radius
        params.per = self.period_d
        params.a = self.semimajor_axis_to_star_radius
        params.inc = self.inclination_deg
        params.ecc = self.eccentricity
        params.w = self.argument_of_periastron_deg
        params.u = (self.limb_darkening_c1, self.limb_darkening_c2)
        params.limb_dark = "quadratic"
        return params

    def to_dict(self) -> dict[str, Any]:
        return self._asdict()

    def to_numpy(self) -> np.ndarray:
        return np.array(list(self))

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(self)

    def to_numpy_for_training(self, data_time_min: float, data_time_max: float) -> np.ndarray:
        """
        Adjust the values of the planetary parameters for training.
        """
        midpoint = self.transit_midpoint_d
        period = self.period_d

        # ensure that the transit midpoint is the first to appear in the data
        if midpoint > 0:
            midpoint = time_of_first_transit_midpoint(t0=data_time_min, transit_midpoint=midpoint, period=period)

        # if the data contains only one transit, it's impossible to determine the period.
        # Therefore, set the period to 0 in this case
        if midpoint + period > data_time_max:
            period = 0

        return np.array([midpoint, period], dtype=np.float32)
