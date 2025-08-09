"""
Module containing the LightCurveStats NamedTuple definition.
This is in a separate module to avoid pickling issues when using multiprocessing.
"""

from typing import NamedTuple

import numpy as np


class LightCurveStats(NamedTuple):
    """Named tuple for storing light curve statistics."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile: float
    length: int


def gap_ratio(time: np.ndarray) -> float:
    """
    Calculates the ratio between the duration of the gaps in the data over the whole observation time
    """
    time_differences = time[1:] - time[:-1]
    time_threshold = np.median(time_differences) * 10
    gaps = time_differences[time_differences > time_threshold]
    return gaps.sum() / (time[-1] - time[0])
