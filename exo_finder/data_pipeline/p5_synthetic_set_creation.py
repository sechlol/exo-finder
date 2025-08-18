import os
from multiprocessing import Manager
from typing import Any, Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import softmax

import exo_finder.constants as consts
from exo_finder.compute.parallel_execution import (
    ShareMode,
    TaskDistribution,
    TaskProfile,
    get_worker_context,
    parallel_execution,
)
from exo_finder.data_pipeline.generation.dataset_generation_types import (
    SyntheticLcDatasetMetadata,
    SyntheticTransitGenerationParameters,
    TransitProfile,
)
from exo_finder.data_pipeline.generation.time_generation import data_count_to_days, generate_time_days_of_length
from exo_finder.data_pipeline.generation.transit_generation import (
    PlanetType,
    generate_transit_parameters,
    generate_transits_from_params,
)
from exo_finder.default_datasets import gaia_dataset, train_dataset_h5


def _get_gaia_star_parameters() -> pd.DataFrame:
    gaia_fields = ["gaia_id", "radius", "mass_flame", "teff_mean"]
    return gaia_dataset.load_gaia_parameters_dataset().view[gaia_fields].to_pandas().dropna()


def _init_globals(extra_arguments: dict[str, Any]) -> dict[str, Any]:
    gen_params = SyntheticTransitGenerationParameters.model_validate(extra_arguments["generation_parameters"])
    pid = extra_arguments["id_queue"].get()

    return {
        "pid": pid,
        "random_generator": np.random.default_rng(seed=pid),
        "gaia_df": _get_gaia_star_parameters(),
        "time_x": generate_time_days_of_length(gen_params.lightcurve_length_points),
        "gen_parameters": gen_params,
    }


def _define_data_balance() -> list[TransitProfile]:
    lightcurve_max_d = data_count_to_days(data_points_count=consts.SYNTHETIC_DATASET_LENGTH)
    default_midpoint_range = (0, lightcurve_max_d)

    # Uniform class labels
    return [
        TransitProfile(
            planet_type=PlanetType.EARTH,
            transit_period_range=(0.4, lightcurve_max_d),
            transit_midpoint_range=default_midpoint_range,
            weight=1,
        ),
        TransitProfile(
            planet_type=PlanetType.SUPER_EARTH,
            transit_period_range=(0.5, lightcurve_max_d),
            transit_midpoint_range=default_midpoint_range,
            weight=1,
        ),
        TransitProfile(
            planet_type=PlanetType.MINI_NEPTUNE,
            transit_period_range=(0.7, lightcurve_max_d),
            transit_midpoint_range=default_midpoint_range,
            weight=1,
        ),
        TransitProfile(
            planet_type=PlanetType.NEPTUNE,
            transit_period_range=(0.85, lightcurve_max_d),
            transit_midpoint_range=default_midpoint_range,
            weight=1,
        ),
        TransitProfile(
            planet_type=PlanetType.JUPITER,
            transit_period_range=(0.95, lightcurve_max_d),
            transit_midpoint_range=default_midpoint_range,
            weight=1,
        ),
    ]


def generate_synthetic_transits():
    data_balance = _define_data_balance()
    probabilities = softmax([x.weight for x in data_balance])

    generation_parameters = SyntheticTransitGenerationParameters(
        dataset_length=consts.SYNTHETIC_DATASET_LENGTH,
        lightcurve_length_points=consts.LC_WINDOW_SIZE,
        transits_distribution=list(zip(probabilities, data_balance)),
    )

    n_jobs = os.cpu_count() - 2
    batch_size = 256
    manager = Manager()

    # Create a queue of IDs used to seed the random number generator for each process
    id_queue = manager.Queue()
    for i in range(n_jobs):
        id_queue.put(i)

    # Create the shared parameters
    job_params = {
        "generation_parameters": generation_parameters.model_dump(),
        "id_queue": id_queue,
    }

    random_indices = np.random.choice(
        list(range(len(data_balance))),
        size=consts.SYNTHETIC_DATASET_LENGTH,
        replace=True,
        p=probabilities,
    )

    total_rows = 0
    transits_col = 0
    params_col = 0
    for transits, generating_params in parallel_execution(
        func=_generate_transits_batch,
        params=random_indices,
        task_distribution=TaskDistribution.STREAMED_BATCHES,
        description="Creating simulated transits",
        batch_size=batch_size,
        n_jobs=n_jobs,
        sort_result=False,
        task_profile=TaskProfile.CPU_BOUND,
        extra_arguments=job_params,
        share_mode=ShareMode.GLOBAL,
        init_process=_init_globals,
    ):
        train_dataset_h5.append(consts.HDF5_KEY_SYNTHETIC_DATA, transits)
        train_dataset_h5.append(consts.HDF5_KEY_SYNTHETIC_PARAMS, generating_params)
        total_rows += transits.shape[0]
        transits_col = transits.shape[1]
        params_col = generating_params.shape[1]

    meta = SyntheticLcDatasetMetadata(
        lc_data_shape=(total_rows, transits_col),
        params_data_shape=(total_rows, params_col),
        lc_window_size=consts.LC_WINDOW_SIZE,
        generation_parameters=generation_parameters,
    )

    train_dataset_h5.write_json(json_key=consts.HDF5_KEY_SYNTHETIC_JSON_META, data=meta.model_dump())

    print(f"All done! Dataset written to {train_dataset_h5.file_path}")
    print(meta.model_dump_json(indent=4))


def _generate_transits_batch(indices: Sequence[int]) -> tuple[npt.NDArray, npt.NDArray]:
    worker_context = get_worker_context()

    gaia_df = worker_context.get("gaia_df")
    global_time_x = worker_context.get("time_x")
    rng = worker_context.get("random_generator")
    generation_parameters = worker_context.get("gen_parameters")

    gaia_subset = gaia_df.sample(len(indices), random_state=rng.integers(low=0, high=2e9))
    all_generated_transits = []
    all_generated_params = []

    for i, (_, gaia_row) in zip(indices, gaia_subset.iterrows()):
        p, generation_specs = generation_parameters.transits_distribution[i]
        transit_parameters = generate_transit_parameters(
            planet_type=generation_specs.planet_type,
            orbital_period_interval=generation_specs.transit_period_range,
            transit_midpoint_range=generation_specs.transit_midpoint_range,
            star_radius=gaia_row["radius"] * u.solRad,
            star_mass=gaia_row["mass_flame"] * u.solMass,
            star_t_eff=gaia_row["teff_mean"] * u.K,
            rnd_generator=rng,
        )
        generated_transits = generate_transits_from_params(params=transit_parameters, time_x=global_time_x)

        all_generated_params.append(transit_parameters.to_numpy())
        all_generated_transits.append(generated_transits)

    return np.vstack(all_generated_transits), np.vstack(all_generated_params)


if __name__ == "__main__":
    generate_synthetic_transits()
