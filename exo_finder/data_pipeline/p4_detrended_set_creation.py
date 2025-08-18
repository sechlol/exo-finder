import numpy as np
import pandas as pd
import pydantic

import exo_finder.constants as consts
from exo_finder.compute.detrending import fit_with_wotan
from exo_finder.compute.lc_utils import arg_split_array_in_contiguous_chunks, parse_tic_id_obs_id_from_lc_path
from exo_finder.compute.parallel_execution import TaskDistribution, parallel_execution
from exo_finder.default_datasets import candidate_dataset, exo_dataset, sunlike_lightcurves_ds, train_dataset_h5
from exotools import LightcurveDB
from paths import LC_STATS_RESULT_FILE

_MANDATORY_FIELDS = ["pl_rade", "pl_trandur", "pl_tranmid", "pl_orbsmax"]


class LcDatasetMetadata(pydantic.BaseModel):
    data_shape: tuple[int, int]
    unique_tic_ids_count: int
    unique_obs_ids_count: int
    lc_window_size: int
    lc_window_min_size: int


def _select_subset_obs_ids(combined_analysis: pd.DataFrame) -> np.ndarray:
    """Define rules for selecting "well-behaving" lightcurves, and then filters the given list to a subset"""

    mad_threshold = np.percentile(combined_analysis["mad"], 70)
    std_threshold = np.percentile(combined_analysis["normalized_std"], 70)

    mask = (
        (combined_analysis.split_count > 0)
        & (combined_analysis.mad <= mad_threshold)
        & (combined_analysis.normalized_std <= std_threshold)
    )

    return combined_analysis.loc[mask, "obs_id"].values


def _select_well_behaving_lightcurve_subset() -> LightcurveDB:
    exo_db = exo_dataset.load_known_exoplanets_dataset(with_gaia_star_data=True)
    candidate_db = candidate_dataset.load_candidate_exoplanets_dataset()
    lc_db = sunlike_lightcurves_ds.load_lightcurve_dataset()
    lc_analysis = pd.read_feather(LC_STATS_RESULT_FILE)

    # exclude invalid lightcurves
    lc_analysis = lc_analysis[lc_analysis["normalized_std"] > 0]

    # exclude lightcurves with known or candidate planets
    planets_tic_ids = list(set(exo_db.unique_tic_ids) | set(candidate_db.unique_tic_ids))
    lc_analysis = lc_analysis[~lc_analysis["tic_id"].isin(planets_tic_ids)]

    # Selects only the well behaving lightcurves
    subset_obs_ids = _select_subset_obs_ids(lc_analysis)

    # return selected lightcurves
    return lc_db.where(obs_id=subset_obs_ids)


def _pad_and_concatenate(time: np.ndarray, flux: np.ndarray, trend: np.ndarray) -> np.ndarray:
    if len(time) < consts.LC_WINDOW_SIZE:
        padding_size = consts.LC_WINDOW_SIZE - len(time)

        # Pad with nans
        time = np.pad(time, (0, padding_size), constant_values=np.nan)
        flux = np.pad(flux, (0, padding_size), constant_values=np.nan)
        trend = np.pad(trend, (0, padding_size), constant_values=np.nan)

    return np.concatenate((time, flux, trend))


def _detrend_lightcurves_batch(paths: list[str]) -> dict[str, np.ndarray]:
    data_batch = []
    data_size = []
    data_tic = []
    data_obs = []

    for p in paths:
        lc_plus = LightcurveDB.load_lightcurve_plus(fits_file_path=p).to_bjd_time()
        tic_id, obs_id = parse_tic_id_obs_id_from_lc_path(p)

        time = lc_plus.time_x
        flux = lc_plus.flux_y
        splits = arg_split_array_in_contiguous_chunks(
            array=time,
            chunk_size=consts.LC_WINDOW_SIZE,
            tolerate_if_len_at_least=consts.LC_WINDOW_MIN_SIZE,
        )

        for i_start, i_end in splits:
            ti = time[i_start : i_end + 1]
            fi = flux[i_start : i_end + 1]
            _, trend = fit_with_wotan(
                time=ti,
                flux=fi,
                return_trend=True,
                window_length=0.25,
                method="biweight",
            )
            # Append time, flux and trend on the same row
            concatenated = _pad_and_concatenate(ti, fi, trend)
            data_batch.append(concatenated)
            data_size.append(len(concatenated))
            data_tic.append(tic_id)
            data_obs.append(obs_id)

    # Stack all lightcurves
    return {
        "data": np.vstack(data_batch, dtype=np.float32),
        "size": np.array(data_size, dtype=np.uint16)[:, np.newaxis],
        "tic_id": np.array(data_tic, dtype=np.uint64)[:, np.newaxis],
        "obs_id": np.array(data_tic, dtype=np.uint64)[:, np.newaxis],
    }


def create_lightcurve_training_set():
    print("Selecting well behaving lightcurves...")
    selected_lc_db = _select_well_behaving_lightcurve_subset()

    total_rows = 0
    cols = 0

    print(f"Detrending {len(selected_lc_db.unique_obs_ids)} lightcurves...")
    for batch_data in parallel_execution(
        func=_detrend_lightcurves_batch,
        params=selected_lc_db.all_paths,
        batch_size=25,
        description="Generating Lightcurve Dataset",
        sort_result=False,
        task_distribution=TaskDistribution.STREAMED_BATCHES,
    ):
        train_dataset_h5.append(consts.HDF5_KEY_LC_DATA, batch_data["data"])
        train_dataset_h5.append(consts.HDF5_KEY_LC_SIZES, batch_data["size"])
        train_dataset_h5.append(consts.HDF5_KEY_LC_TIC_IDS, batch_data["tic_id"])

        length, width = batch_data["data"].shape
        total_rows += length
        cols = width

    meta = LcDatasetMetadata(
        data_shape=(total_rows, cols),
        unique_tic_ids_count=len(selected_lc_db.unique_tic_ids),
        unique_obs_ids_count=len(selected_lc_db.unique_obs_ids),
        lc_window_size=consts.LC_WINDOW_SIZE,
        lc_window_min_size=consts.LC_WINDOW_MIN_SIZE,
    )

    train_dataset_h5.write_json(json_key=consts.HDF5_KEY_LC_JSON_META, data=meta.model_dump())
    train_dataset_h5.flush()

    print(f"All done! Dataset written to {train_dataset_h5.file_path}")
    print(meta.model_dump_json(indent=4))


if __name__ == "__main__":
    create_lightcurve_training_set()
