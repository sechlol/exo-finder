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


class PlanetType(enum.StrEnum):
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


def generate_transits_from_params(
    params: PlanetaryParameters,
    time_x: np.ndarray,
    median_flux: float = 0,
) -> np.ndarray:
    batman_params = params.to_batman()
    # TODO: optimize performance by pre-computing the step size factor if if limb darkening method is != quadratic.
    # See https://lkreidberg.github.io/batman/docs/html/trouble.html#help-batman-is-running-really-slowly-why-is-this
    return batman.TransitModel(batman_params, time_x).light_curve(batman_params).astype(np.float32) + (median_flux - 1)


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
    pl_radius = (rnd.uniform(*size_params) * u.R_earth).to(u.solRad)
    planet_mass = _calculate_mass_capped(pl_radius, density_params, rnd)

    # Sample orbital period
    p_min, p_max = orbital_period_interval
    # period = 10 ** rnd.uniform(np.log10(p_min), np.log10(p_max)) * u.day
    period = rnd.uniform(p_min, p_max) * u.day

    # Compute semi-major axis and a/R*
    semi_major_axis = _calculate_semimajor_axis(period, star_mass, planet_mass).to(u.solRad)
    eccentricity = _calculate_eccentricity(period, seed=int(rnd.integers(0, 1e9)))

    # Limb darkening
    limb_darkening_coeff = estimate_limb_darkening_coefficients_clamped(star_t_eff)

    # Transit midpoint
    if transit_midpoint_range is None:
        transit_midpoint = rnd.uniform(0, period.value)
    else:
        transit_midpoint = rnd.uniform(*transit_midpoint_range)

    # Orbital inclination and periastron
    a_over_rs = (semi_major_axis / star_radius).value
    rp_over_rs = (pl_radius / star_radius).value

    w_deg = _generate_w(rnd)
    i_deg = _generate_inclination_deg(
        a_over_rs=a_over_rs, rp_over_rs=rp_over_rs, e=eccentricity, w_deg=w_deg, rnd=rnd, allow_grazing=False
    )

    return PlanetaryParameters(
        period_d=period.value,
        transit_midpoint_d=transit_midpoint,
        planet_radius_solrad=pl_radius.value,
        star_radius_solrad=star_radius.value,
        semi_major_axis_solrad=semi_major_axis.value,
        planet_mass_solmass=planet_mass.value,
        inclination_deg=i_deg,  # Always edge-on: the planet crosses the stellar disk in the middle
        eccentricity=eccentricity,
        argument_of_periastron_deg=w_deg,
        limb_darkening_c1=limb_darkening_coeff[0],
        limb_darkening_c2=limb_darkening_coeff[1],
    )


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


def _generate_w(rnd: np.random.Generator) -> float:
    """Argument of periastron ω in DEGREES, uniform on [0, 360)."""
    return rnd.uniform(0.0, 360.0)


def _generate_inclination_deg(
    a_over_rs: float,
    rp_over_rs: float,
    e: float,
    w_deg: float,
    rnd: np.random.Generator,
    allow_grazing: bool = False,
) -> float:
    """
    Generate a random orbital inclination (in degrees) that guarantees
    a valid transit given stellar, orbital, and planetary parameters.

    The inclination is drawn indirectly by first sampling a random
    **impact parameter** b (the projected distance of the planet’s
    trajectory from the stellar center in units of stellar radii).
    For bound orbits, b and inclination i are related through

        b = (a / R*) * cos(i) * (1 - e^2) / (1 + e * sin(ω)),

    where:
        - a / R* is the scaled semi-major axis,
        - e is the eccentricity,
        - ω is the argument of periastron.

    This function samples b uniformly within the range required to
    guarantee a visible transit:
        - Non-grazing (default):  0 ≤ b ≤ 1 - Rp/R*,
          ensuring the planet’s disk is fully contained within the
          stellar disk at mid-transit.
        - Grazing (if allow_grazing=True): 0 ≤ b ≤ 1 + Rp/R*,
          allowing partial overlaps.

    It then converts b back to the corresponding inclination angle i.

    Parameters
    ----------
    a_over_Rs : float
        Semi-major axis divided by stellar radius (a / R*),
        dimensionless.
    Rp_over_Rs : float
        Planet-to-star radius ratio (Rp / R*), dimensionless.
    e : float
        Orbital eccentricity (0 ≤ e < 1 for bound orbits).
    w_deg : float
        Argument of periastron in **degrees**. Used to compute the
        impact parameter conversion factor.
    rnd : np.random.Generator
        Random number generator for reproducibility.
    allow_grazing : bool, optional
        If False (default), inclination is sampled such that the
        transit is always fully inside the stellar disk.
        If True, grazing transits are allowed.

    Returns
    -------
    i_deg : float
        Orbital inclination in degrees, within [0, 180].

    Notes
    -----
    - For circular orbits (e=0), the conversion reduces to
      b = (a / R*) * cos(i).
    - Only inclinations close to 90° yield transits; this function
      guarantees that condition by constraining the sampled impact
      parameter.
    - Numerical clipping is applied to avoid floating-point domain
      errors when computing arccos.

    """
    # Geometry factor K = (R*/a) * (1 + e sin ω) / (1 - e^2)
    w_rad = np.deg2rad(w_deg)
    K = (1.0 / a_over_rs) * (1.0 + e * np.sin(w_rad)) / (1.0 - e**2)

    # Max b permitted by cos i ≤ 1: K*b ≤ 1 -> b ≤ 1/K (if K>0). K is positive for e<1.
    # Numerical safety: handle tiny/negative K (shouldn't occur for bound orbits).
    eps = 1e-12
    if K <= eps:
        # Fall back to central transit
        return 90.0

    # Transit constraint
    b_geom_max = (1.0 - rp_over_rs) if not allow_grazing else (1.0 + rp_over_rs)
    # Also respect the cos i ≤ 1 bound
    b_max = min(b_geom_max, 1.0 / K)
    b_max = max(b_max, 0.0)  # guard against pathological inputs

    # Sample b and convert to inclination
    b = rnd.uniform(0.0, b_max)
    cos_i = K * b
    cos_i = np.clip(cos_i, -1.0, 1.0)  # numerical safety
    i_deg = np.degrees(np.arccos(cos_i))
    return float(i_deg)
