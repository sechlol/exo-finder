import os
import re
from typing import Optional, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from exo_finder.compute.lc_utils import arg_split_array_in_contiguous_chunks, gap_ratio
from exo_finder.constants import LC_WINDOW_SIZE, LC_WINDOW_MIN_SIZE
from exo_finder.compute.parallel_execution import parallel_execution, TaskDistribution, TaskProfile
from exotools import LightcurveDB
from paths import LIGHTCURVES_PATH, LC_STATS_RESULT_FILE

# Use to match tic_id and obs_id from the lightcurve file path
LC_ID_PATTERN = re.compile(r"(\d+)[/\\](\d+).fits")


def find_lightcurve_paths() -> list[str]:
    paths = []
    for current_path, dirs, files in LIGHTCURVES_PATH.walk():
        for f in files:
            if f.endswith(".fits"):
                paths.append(str(current_path / f))
    return paths


class LightCurveStats(NamedTuple):
    tic_id: int
    obs_id: int
    length: int
    gap_ratio: float
    split_ratio: float
    split_count: int
    mad: float
    normalized_std: float
    min_f: float
    max_f: float
    median_f: float


def load_and_calculate_lightcurve_statistics(lc_path: str) -> Optional[LightCurveStats]:
    if not os.path.exists(lc_path) or not os.path.isfile(lc_path):
        return None

    try:
        lightcurve = LightcurveDB.load_lightcurve_plus(lc_path)
    except Exception as e:
        print(f"Failed to load {lc_path}: {repr(e)}")
        return None

    lc_no_outliers = lightcurve.remove_outliers().remove_nans()
    median = np.median(lc_no_outliers.flux_y).item()

    # If median is < 0, probably the lightcurve has corrupted data.
    if median < 0:
        return None

    match = LC_ID_PATTERN.search(lc_path)
    if match:
        tic_id = int(match.group(1))
        obs_id = int(match.group(2))
    else:
        raise ValueError(f"Invalid lightcurve path: {lc_path}")

    contiguous_splits = arg_split_array_in_contiguous_chunks(
        array=lc_no_outliers.time_x,
        chunk_size=LC_WINDOW_SIZE,
        tolerate_if_len_at_least=LC_WINDOW_MIN_SIZE,
    )

    # Note: values need to be float(), otherwise the serialization to json will fail
    return LightCurveStats(
        tic_id=tic_id,
        obs_id=obs_id,
        median_f=median,
        min_f=lc_no_outliers.flux_y.min().item(),
        max_f=lc_no_outliers.flux_y.max().item(),
        length=len(lightcurve),
        gap_ratio=gap_ratio(lightcurve.time.value),
        mad=median_abs_deviation(lightcurve.flux_y, nan_policy="omit"),
        normalized_std=lc_no_outliers.normalize().flux_y.std().item(),
        split_count=len(contiguous_splits),
        split_ratio=sum(i2 - i1 for i1, i2 in contiguous_splits) / len(lightcurve),
    )


def load_and_calculate_lightcurve_statistics_batched(lc_paths: list[str]):
    return [load_and_calculate_lightcurve_statistics(lc_path) for lc_path in lc_paths]


def analyze_lightcurves():
    # Find all the downloaded lightcurves in the dataset
    all_lc_paths = find_lightcurve_paths()
    print(f"Found {len(all_lc_paths)} lightcurves in the dataset")

    LC_STATS_RESULT_FILE.parent.mkdir(exist_ok=True, parents=True)
    results = parallel_execution(
        func=load_and_calculate_lightcurve_statistics_batched,
        params=all_lc_paths,
        task_distribution=TaskDistribution.STREAMED_BATCHES,
        batch_size=50,
        sort_result=False,
        task_profile=TaskProfile.CPU_BOUND,
    )

    df = pd.DataFrame(data=[s for s in results if s is not None], columns=LightCurveStats._fields)
    print(f"Saving {len(df)} results...")
    df.to_feather(LC_STATS_RESULT_FILE)


if __name__ == "__main__":
    analyze_lightcurves()
