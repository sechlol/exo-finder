import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries

# Cadence of light curve measurement, in seconds. Don't change this or everything breaks
CADENCE_S = 120

# 120 seconds time interval in JD format, representing one tic in the typical lightcurve
TIME_STEP_JD_120s = 0.001388888806

# Conversion factor from days to 120s
TIME_STEP_DAY_120s = 120 / (24 * 3600)


def data_count_to_days(data_points_count: int) -> float:
    return (data_points_count * CADENCE_S) / (24 * 60 * 60)


def generate_time_days(days: float) -> np.ndarray:
    total_seconds = days * 24 * 3600
    number_of_points = round(total_seconds / CADENCE_S)
    return np.linspace(0, days, number_of_points)


def generate_time_days_of_length(length: int) -> np.ndarray:
    return np.arange(length) * TIME_STEP_DAY_120s


def time_of_first_transit_midpoint(t0: float, transit_midpoint: float, period: float) -> float:
    # Calculate the offset from the first transit midpoint before t0
    offset = (t0 - transit_midpoint) % period

    # Determine the first transit midpoint that occurs at or after t0
    first_midpoint = t0 - offset
    if offset > 0:
        first_midpoint += period

    return first_midpoint


def generate_time_jd(days: float, start_at_0: bool = False) -> np.ndarray:
    """
    NOTE: the output array must be float64 or batman will halt for some reason
    """
    # Center to current time
    t0 = Time.now() - (days * u.day / 2)
    samples = int(days * 24 * 3600 / CADENCE_S)
    x = TimeSeries(time_start=t0, time_delta=CADENCE_S * u.s, n_samples=samples).time.jd
    if start_at_0:
        return x - x[0]
    return x


def generate_time_jd_of_length(length: int) -> np.ndarray:
    """
    NOTE: the output array must be float64 or batman will halt for some reason
    """
    return np.arange(length) * TIME_STEP_JD_120s
