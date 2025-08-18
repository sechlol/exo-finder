import contextlib
import io
import sys
from typing import Optional

import numpy as np
import wotan


def fit_with_wotan(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: float = 0.5,
    verbose: bool = False,
    return_trend: bool = True,
    mask: Optional[np.ndarray] = None,
    method: str = "biweight",
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Fit lightcurve with Wotan library, optionally providing a mask to ignore planet transits
    Returns:
        A tuple with
        [0]: flat_flux: Flattened flux.
        [1]: trend_flux: Trend in the flux (only returned if return_trend=True)
    """
    output_stream = sys.stdout if verbose else io.StringIO()

    with contextlib.redirect_stdout(output_stream):
        return wotan.flatten(
            time,  # Array of time values
            flux,  # Array of flux values
            method=method,
            robust=True,  # True: more effective at ignoring the planet transit dips
            window_length=window_length,  # The length of the filter window in units of ``time``
            edge_cutoff=0.5,  # length (in units of time) to be cut off each edge.
            break_tolerance=0.5,  # Split into segments at breaks longer than that
            return_trend=return_trend,  # Return trend and flattened light curve
            cval=5.0,  # Tuning parameter for the robust estimators
            mask=mask,
        )
