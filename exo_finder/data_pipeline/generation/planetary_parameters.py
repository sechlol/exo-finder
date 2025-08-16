from typing import Any

import batman
import numpy as np

from .time_generation import time_of_first_transit_midpoint


class PlanetaryParameters(batman.TransitParams):
    def __repr__(self):
        fields = ", ".join(f"{key}={value}" for key, value in vars(self).items())
        return f"{self.__class__.__name__}({fields})"

    @property
    def period(self) -> float | None:
        """Get the orbital period in days"""
        return self.per or 0

    @period.setter
    def period(self, value: float):
        """Set the orbital period in days"""
        self.per = value

    @property
    def transit_midpoint(self) -> float | None:
        """
        Get the midpoint of the first recorded transit in days
        """
        return self.t0 or 0

    @transit_midpoint.setter
    def transit_midpoint(self, value: float):
        """Set the midpoint of the first recorded transit in days"""
        self.t0 = value

    @property
    def limb_darkening_coefficient(self) -> list[float] | None:
        """Get the limb darkening coefficients"""
        return self.u

    @limb_darkening_coefficient.setter
    def limb_darkening_coefficient(self, value):
        """Set the limb darkening coefficients"""
        self.u = value

    @property
    def limb_darkening_model(self):
        """Get the limb darkening model"""
        return self.limb_dark

    @limb_darkening_model.setter
    def limb_darkening_model(self, value: str):
        """Set the limb darkening model (e.g., 'quadratic', 'nonlinear')"""
        self.limb_dark = value

    @property
    def planet_to_star_radius(self) -> float | None:
        """Get the planet radius relative to the star radius"""
        return self.rp

    @planet_to_star_radius.setter
    def planet_to_star_radius(self, value: float):
        """Set the planet radius relative to the star radius"""
        self.rp = value

    @property
    def semimajor_axis_to_star_radius(self) -> float | None:
        """Get the semi-major axis relative to the star radius"""
        return self.a

    @semimajor_axis_to_star_radius.setter
    def semimajor_axis_to_star_radius(self, value: float):
        """Set the semi-major axis relative to the star radius"""
        self.a = value

    @property
    def inclination(self) -> float | None:
        """Get the orbital inclination in degrees"""
        return self.inc

    @inclination.setter
    def inclination(self, value: float):
        """Set the orbital inclination in degrees"""
        self.inc = value

    @property
    def eccentricity(self) -> float | None:
        """Get the orbital eccentricity"""
        return self.ecc

    @eccentricity.setter
    def eccentricity(self, value: float):
        """Set the orbital eccentricity"""
        self.ecc = value

    @property
    def argument_of_periastron(self) -> float | None:
        """Get the argument of periastron in degrees"""
        return self.w

    @argument_of_periastron.setter
    def argument_of_periastron(self, value: float):
        """Set the argument of periastron in degrees"""
        self.w = value

    @property
    def planet_flux(self):
        """Get the planet-to-star flux ratio"""
        return self.fp

    @planet_flux.setter
    def planet_flux(self, value: float):
        """Set the planet-to-star flux ratio"""
        self.fp = value

    @property
    def secondary_eclipse_time(self) -> float | None:
        """Get the time of secondary eclipse in days"""
        return self.t_secondary

    @secondary_eclipse_time.setter
    def secondary_eclipse_time(self, value: float):
        """Set the time of secondary eclipse in days"""
        self.t_secondary = value

    def to_json(self) -> dict[str, Any]:
        """
        Convert the planetary parameters to a JSON-serializable dictionary.

        Returns:
            dict: A dictionary containing all planetary parameters with descriptive keys
        """
        return {
            "transit_midpoint": self.transit_midpoint,
            "period": self.period,
            "planet_to_star_radius": self.planet_to_star_radius,
            "semimajor_axis_to_star_radius": self.semimajor_axis_to_star_radius,
            "inclination": self.inclination,
            "eccentricity": self.eccentricity,
            "argument_of_periastron": self.argument_of_periastron,
            "limb_darkening_coefficient": self.limb_darkening_coefficient,
            "limb_darkening_model": self.limb_darkening_model,
            "planet_flux": self.planet_flux,
            "secondary_eclipse_time": self.secondary_eclipse_time,
        }

    def to_numpy(self) -> np.ndarray:
        """
        Convert the essential planetary parameters to a numpy array.

        Returns:
            np.ndarray: Array containing the key planetary parameters in a standardized order
        """
        return np.array(
            [
                self.transit_midpoint,
                self.period,
                self.planet_to_star_radius,
                self.semimajor_axis_to_star_radius,
                self.inclination,
                self.eccentricity,
                self.argument_of_periastron,
                self.planet_flux,
                self.secondary_eclipse_time,
                *self.limb_darkening_coefficient,
            ],
            dtype=np.float32,
        )

    def to_numpy_for_training(self, data_time_min: float, data_time_max: float) -> np.ndarray:
        """
        Adjust the values of the planetary parameters for training.
        """
        midpoint = self.transit_midpoint
        period = self.period

        # ensure that the transit midpoint is the first to appear in the data
        if midpoint > 0:
            midpoint = time_of_first_transit_midpoint(t0=data_time_min, transit_midpoint=midpoint, period=period)

        # if the data contains only one transit, it's impossible to determine the period.
        # Therefore, set the period to 0 in this case
        if midpoint + period > data_time_max:
            period = 0

        return np.array([midpoint, period], dtype=np.float32)
