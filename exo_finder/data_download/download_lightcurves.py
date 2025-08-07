from typing import Optional

import numpy as np

from exo_finder.default_datasets import (
    candidate_dataset,
    exo_dataset,
    tic_observations,
    TRANSITING_LIGHTCURVES_DS,
    SUNLIKE_LIGHTCURVES_DS,
)
from exo_finder.utils.logger import Logger
from exotools import ExoDB, TicObsDB

logger = Logger(__name__)


def download_known_planets_lightcurves():
    exo_db: ExoDB = exo_dataset.load_known_exoplanets_dataset()
    observation_db: TicObsDB = tic_observations.load_observation_metadata()

    # Select known transiting exoplanets from Tess and Kepler that have data products
    transiting_exoplanets = exo_db.get_transiting_planets(kepler_or_tess_only=True).with_valid_ids()
    transiting_planets_meta = observation_db.select_by_tic_id(transiting_exoplanets.unique_tic_ids)
    logger.info(
        f"Selected {len(transiting_planets_meta)} observations for {len(transiting_exoplanets.unique_tic_ids)} stars"
    )

    # # Download lightcurves in the selected metadata
    logger.info(f"Downloading {len(transiting_planets_meta)} lightcurves")
    lc_db = TRANSITING_LIGHTCURVES_DS.download_lightcurves_from_tic_db(tic_obs_db=transiting_planets_meta)
    logger.success(f"Downloaded {len(lc_db)} lightcurves")


def download_sunlike_stars_lightcurves(
    unique_stars_limit: Optional[int] = None, limit_observations: Optional[int] = None
):
    # Load data
    exo_db = exo_dataset.load_known_exoplanets_dataset()
    candidate_db = candidate_dataset.load_candidate_exoplanets_dataset()
    observation_db: TicObsDB = tic_observations.load_observation_metadata()
    logger.info(f"Observations: {len(observation_db)}")

    # Remove stars that might have transiting exoplanets
    stars_to_exclude = np.hstack((exo_db.unique_tic_ids, candidate_db.unique_tic_ids))
    tic_ids_without_transiting = ~np.isin(observation_db.tic_ids, stars_to_exclude)
    observation_db = observation_db.where_true(tic_ids_without_transiting)
    logger.info(f"Observations after removing transiting exoplanets: {len(observation_db)}")

    # Select the 20% most recent observations
    t_obs_release = observation_db.view["t_obs_release"]
    observation_db = observation_db.where_true(t_obs_release > np.percentile(t_obs_release, 80))
    logger.info(f"Observations after filtering for recent observations: {len(observation_db)}")

    # Select all the observations from a sample of random stars
    if unique_stars_limit:
        sampled_stars_id = np.random.choice(observation_db.unique_tic_ids, size=unique_stars_limit, replace=False)
        observation_db = observation_db.where(tic_id=sampled_stars_id)
        logger.info(f"Observations after sampling {unique_stars_limit} stars: {len(observation_db)}")

    # Take a random subset of observations
    if limit_observations:
        sampled_observations = np.random.choice(observation_db.obs_id, size=limit_observations, replace=False)
        observation_db = observation_db.select_by_obs_id(sampled_observations)

    # Download lightcurves
    logger.info(f"Downloading {len(observation_db)} observations from {len(observation_db.unique_tic_ids)} stars")
    lc_db = SUNLIKE_LIGHTCURVES_DS.download_lightcurves_from_tic_db(tic_obs_db=observation_db)
    logger.success(f"Downloaded {len(lc_db)} lightcurves")


if __name__ == "__main__":
    download_known_planets_lightcurves()
    download_sunlike_stars_lightcurves(limit_observations=10000)
