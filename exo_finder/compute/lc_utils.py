from typing import Generator, Optional
import numpy as np

from exo_finder.constants import LC_WINDOW_SIZE, LC_WINDOW_MIN_SIZE
from exotools.utils import get_contiguous_interval_indices


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


if __name__ == "__main__":
    arr1 = np.arange(3276)
    arr2 = np.arange(4757)
    for i_start, i_end in arg_split_array_in_contiguous_chunks(
        array=arr1, chunk_size=LC_WINDOW_SIZE, tolerate_if_len_at_least=LC_WINDOW_MIN_SIZE
    ):
        pass


def gap_ratio(time: np.ndarray) -> float:
    """
    Calculates the ratio between the duration of the gaps in the data over the whole observation time
    """
    time_differences = time[1:] - time[:-1]
    time_threshold = np.median(time_differences) * 10
    gaps = time_differences[time_differences > time_threshold]
    return gaps.sum() / (time[-1] - time[0])
