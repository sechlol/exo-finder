import numpy as np
import warnings

from astropy.units import Quantity


def estimate_teff_from_mass(mass: Quantity) -> Quantity:
    """
    Estimate effective temperature (K) for main sequence stars from mass.

    Parameters:
    -----------
    mass : float or array-like
        Stellar mass in solar units (M_sun)

    Returns:
    --------
    teff : float or ndarray
        Effective temperature (K)

    Reference: Eker et al. (2018), arXiv:1501.06585
    """
    mass = np.asarray(mass, dtype=float)
    teff = np.zeros_like(mass)

    # Empirical relations
    # Coefficients for different mass ranges (Eker+ 2018, MNRAS 479, 5491)
    ranges = [
        (0.179, 0.45, 3.556, 0.487),
        (0.45, 0.72, 3.611, 0.413),
        (0.72, 1.05, 3.623, 0.415),
        (1.05, 2.40, 3.698, 0.282),
        (2.40, 7.00, 3.821, 0.145),
        (7.00, 32.00, 4.079, 0.034),
    ]

    for m_min, m_max, a, b in ranges:
        mask = (mass >= m_min) & (mass < m_max)
        teff[mask] = 10 ** (a + b * np.log10(mass[mask]))

    return teff if teff.size > 1 else teff.item()


def estimate_limb_darkening_coefficients_clamped(t_eff: Quantity) -> tuple | np.ndarray:
    """
    Estimate quadratic limb darkening coefficients [a, b] for main sequence stars.

    Based on polynomial fits to CoRoT and Kepler limb darkening data from
    Claret & Bloemen (2011, A&A 529, A75) for solar metallicity and log(g) = 4.5.

    Parameters:
    -----------
    t_eff : float or array-like
        Effective temperature in Kelvin. Recommended range: 4000-7500 K

    Returns:
    --------
    list or numpy.ndarray
        [a, b] coefficients for quadratic limb darkening law
    """

    a_coeffs = [-6.158116e00, 3.793995e-03, -6.857610e-07, 3.922755e-11]
    b_coeffs = [5.597328e00, -3.039616e-03, 5.455844e-07, -3.121324e-11]

    t_eff = np.asarray(t_eff, dtype=float)
    scalar_input = t_eff.ndim == 0
    if scalar_input:
        t_eff = t_eff[np.newaxis]

    if np.any(t_eff <= 0):
        raise ValueError("Temperature must be positive")
    if np.any(t_eff > 50000):
        raise ValueError("Temperature > 50000 K is unreasonably high")

    # Clamp to model range to avoid nonsense
    t_eff_clamped = np.clip(t_eff, 3500, 7500)
    a = np.polyval(a_coeffs[::-1], t_eff_clamped)
    b = np.polyval(b_coeffs[::-1], t_eff_clamped)

    if scalar_input:
        return float(a[0]), float(b[0])
    else:
        return np.column_stack([a, b])
