import re
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from exotools.utils import get_contiguous_interval_indices

# Use to match tic_id and obs_id from the lightcurve file path
LC_ID_PATTERN = re.compile(r"(\d+)[/\\](\d+).fits")


def standardize_array(a: np.ndarray):
    return (a - a.mean()) / a.std()


def normalize_flux(a: np.ndarray):
    return (a / np.median(a)) - 1


def split_array_in_contiguous_chunks_generator(array: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    for i_start, i_end in arg_split_array_in_contiguous_chunks(array=array, chunk_size=chunk_size):
        yield array[i_start : i_end + 1]


def arg_split_array_in_contiguous_chunks(
    array: np.ndarray,
    chunk_size: int,
    tolerate_if_len_at_least: Optional[int] = None,
) -> list[tuple[int, int]]:
    threshold = tolerate_if_len_at_least or chunk_size
    contiguous_intervals = list(
        filter(
            lambda interval: interval[1] - interval[0] + 1 >= threshold,
            get_contiguous_interval_indices(x=array, greater_than_median=25),
        )
    )
    contiguous_chunks = []
    for i_start, i_end in contiguous_intervals:
        interval_len = i_end - i_start + 1
        n_chunks = interval_len // chunk_size
        for i_chunk in range(n_chunks):
            contiguous_chunks.append((i_start + i_chunk * chunk_size, i_start + (i_chunk + 1) * chunk_size - 1))

        if tolerate_if_len_at_least and interval_len % chunk_size >= tolerate_if_len_at_least:
            contiguous_chunks.append((i_start + n_chunks * chunk_size, i_end))

    return contiguous_chunks


def split_array_in_contiguous_chunks(
    array: np.ndarray,
    chunk_size: int,
    tolerate_if_len_at_least: Optional[int] = None,
) -> list[tuple[float, float]]:
    return [
        (array[i_start].item(), array[i_end].item())
        for i_start, i_end in arg_split_array_in_contiguous_chunks(
            array=array, chunk_size=chunk_size, tolerate_if_len_at_least=tolerate_if_len_at_least
        )
    ]


def gap_ratio(time: np.ndarray) -> float:
    """
    Calculates the ratio between the duration of the gaps in the data over the whole observation time
    """
    time_differences = time[1:] - time[:-1]
    time_threshold = np.median(time_differences) * 10
    gaps = time_differences[time_differences > time_threshold]
    return gaps.sum() / (time[-1] - time[0])


def parse_tic_id_obs_id_from_lc_path(lc_path: str | Path) -> tuple[int, int]:
    match = LC_ID_PATTERN.search(str(lc_path))
    if match:
        tic_id = int(match.group(1))
        obs_id = int(match.group(2))
        return tic_id, obs_id
    else:
        raise ValueError(f"Invalid lightcurve path: {lc_path}")
