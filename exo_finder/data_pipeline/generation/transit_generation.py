import enum
import batman
import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from scipy import stats
from typing import Optional

from .limb_darkening import estimate_limb_darkening_coefficients_clamped, estimate_teff_from_mass
from .planetary_parameters import PlanetaryParameters


class Density(enum.Enum):
    ROCKY = enum.auto()
    GAS = enum.auto()


class PlanetType(enum.Enum):
    EARTH = enum.auto()
    SUPER_EARTH = enum.auto()
    MINI_NEPTUNE = enum.auto()
    NEPTUNE = enum.auto()
    JUPITER = enum.auto()
    BROWN_DWARF = enum.auto()


# Planet orbital periods
class PeriodFrequency:
    SUB_DAY = (0.15, 1.0)
    ONE_TO_THREE_DAYS = (1.0, 3.0)
    THREE_TO_TEN_DAYS = (3.0, 10.0)
    TEN_DAYS_TO_MONTH = (10.0, 30.0)
    MONTH_TO_YEAR = (30.0, 365.0)


_PLANET_PARAMS_LOOKUP_TABLE = {
    # densities as (mean, std, min_value)
    Density.ROCKY: (5.5, 0.5, 3.0),
    Density.GAS: (1.3, 0.3, 0.3),
    # planet radii in Earth radii and density category
    PlanetType.EARTH: ((0.7, 1.2), Density.ROCKY),
    PlanetType.SUPER_EARTH: ((1.2, 2.5), Density.ROCKY),
    PlanetType.MINI_NEPTUNE: ((2.5, 5.0), Density.GAS),
    PlanetType.NEPTUNE: ((5, 7), Density.GAS),
    PlanetType.JUPITER: ((7, 13), Density.GAS),
    PlanetType.BROWN_DWARF: ((13, 18), Density.GAS),
}


def generate_transits_from_params(params: PlanetaryParameters, time_x: np.ndarray) -> np.ndarray:
    return batman.TransitModel(params, time_x).light_curve(params) - 1


def generate_transit_parameters(
    planet_type: PlanetType,
    orbital_period_interval: tuple[float, float],
    star_radius: Quantity,
    star_t_eff: Optional[Quantity] = None,
    star_mass: Optional[Quantity] = None,
    transit_midpoint_range: Optional[tuple[float, float]] = None,
    rnd_generator: Optional[np.random.Generator] = None,
) -> PlanetaryParameters:
    rnd = rnd_generator or np.random.default_rng()

    # Planet size & density parameters from lookup
    size_params, density_category = _PLANET_PARAMS_LOOKUP_TABLE[planet_type]
    density_params = _PLANET_PARAMS_LOOKUP_TABLE[density_category]

    # Stellar parameters
    star_radius = star_radius.to(u.solRad)

    # empirical M-R relation for calculating sar mass given radius
    if star_mass is None:
        star_mass = star_radius.value**1.25 * u.solMass

    if star_t_eff is None:
        star_t_eff = estimate_teff_from_mass(star_mass)

    # Sample planet radius
    pl_radius = rnd.uniform(*size_params) * u.R_earth
    planet_mass = _calculate_mass_capped(pl_radius, density_params, rnd)

    # Sample orbital period
    p_min, p_max = orbital_period_interval
    period = 10 ** rnd.uniform(np.log10(p_min), np.log10(p_max)) * u.day

    # Compute semi-major axis and a/R*
    semi_major_axis = _calculate_semimajor_axis(period, star_mass, planet_mass)
    semimajor_axis_to_stellar_ratio = semi_major_axis.to(u.solRad) / star_radius

    eccentricity = _calculate_eccentricity(period, seed=int(rnd.integers(0, 1e9)))

    # Limb darkening
    limb_darkening_coeff = estimate_limb_darkening_coefficients_clamped(star_t_eff)

    # Transit midpoint
    if transit_midpoint_range is None:
        transit_midpoint_range = rnd.uniform(0, period.value)
    else:
        transit_midpoint_range = rnd.uniform(*transit_midpoint_range)

    # Assemble planetary parameters
    params = PlanetaryParameters()
    params.t0 = transit_midpoint_range
    params.rp = (pl_radius.to(u.solRad) / star_radius).value
    params.per = period.value
    params.a = semimajor_axis_to_stellar_ratio.value
    params.inc = 90
    params.ecc = eccentricity
    params.w = 90
    params.u = limb_darkening_coeff
    params.limb_dark = "quadratic"
    return params


def _calculate_eccentricity(period: Quantity, seed: int) -> float:
    # Eccentricity: short vs long split at 10 days
    if period.value < 10:
        ecc = stats.beta.rvs(0.867, 3.03, random_state=seed)
    else:
        ecc = stats.beta.rvs(1.12, 3.09, random_state=seed)

    # If eccentricity is too high, the planet will crash into the star, and it will also crash batman.
    # It usually happens for values around 0.99 Here, set an upper limit
    return min(ecc, 0.8)


def _calculate_mass(radius: Quantity, density_params: tuple[float, ...], rnd: np.random.Generator) -> Quantity:
    density = rnd.normal(*density_params) * u.g / u.cm**3
    rad_km = radius.to(u.km)
    mass = density * 4 / 3 * np.pi * rad_km**3
    return mass.to(u.solMass)


def _calculate_mass_capped(
    radius: Quantity, density_params: tuple[float, float, float], rnd: np.random.Generator
) -> Quantity:
    mean_density, std_density, min_density = density_params
    density_val = max(rnd.normal(mean_density, std_density), min_density) * u.g / u.cm**3
    rad_km = radius.to(u.km)
    mass = density_val * 4 / 3 * np.pi * rad_km**3
    return mass.to(u.solMass)


def _calculate_orbital_period(semi_major_axis, star_mass, planet_mass) -> float:
    T = np.sqrt((4 * np.pi**2 * semi_major_axis.to(u.m) ** 3) / (const.G * (star_mass + planet_mass).to(u.kg)))
    return T.to(u.day).value


def _calculate_semimajor_axis(orbital_period: Quantity, star_mass: Quantity, planet_mass: Quantity) -> Quantity:
    total_mass = (star_mass + planet_mass).to(u.kg)
    axis = ((const.G * total_mass * orbital_period.to(u.s) ** 2) / (2 * np.pi**2)) ** (1 / 3)
    return axis.to(u.au)
