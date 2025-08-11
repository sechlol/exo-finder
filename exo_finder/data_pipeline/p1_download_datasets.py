from itertools import chain

from exo_finder.default_datasets import (
    exo_dataset,
    candidate_dataset,
    candidate_tic_catalog,
    sunlike_tic_catalog,
    gaia_dataset,
    tic_observations,
)
from exo_finder.default_storage import DEFAULT_STORAGE
from exo_finder.utils.logger import Logger
from exotools import (
    KnownExoplanetsDataset,
    ExoDB,
    CandidateExoplanetsDataset,
    CandidateDB,
)

logger = Logger(__name__)


def download_datasets():
    exo_db: ExoDB = exo_dataset.load_known_exoplanets_dataset(with_gaia_star_data=True)
    if not exo_db:
        logger.info("***\nQuery Nasa Exoplanet Archive for Known Exoplanets (full catalog)")
        logger.info("Should take between 5-10 minutes.")
        exo_db = KnownExoplanetsDataset(storage=DEFAULT_STORAGE).download_known_exoplanets(
            with_gaia_star_data=True, store=True
        )
    gaia_exo_db = exo_dataset.load_gaia_dataset_of_known_exoplanets()
    logger.success(f"Downloaded {len(exo_db)} known exoplanets records and {len(gaia_exo_db)} gaia records")

    toi_db: CandidateDB = candidate_dataset.load_candidate_exoplanets_dataset()
    if not toi_db:
        logger.info("***\nQuery Nasa Exoplanet Archive for Candidate Exoplanets (full catalog)")
        logger.info("Should take around 5 minutes.")
        toi_db = CandidateExoplanetsDataset(storage=DEFAULT_STORAGE).download_candidate_exoplanets(store=True)
    logger.success(f"Downloaded {len(toi_db)} candidate exoplanets records")

    # The GAIA_ID is not included in the TOI table, so we need to download it from the TIC database
    toi_tic_ids = toi_db.view["tic_id"].value
    toi_tic_db = candidate_tic_catalog.load_tic_target_dataset()
    if not toi_tic_db:
        logger.info("Downloading gaia_id for TOI exoplanets")
        toi_tic_db = candidate_tic_catalog.download_tic_targets_by_ids(tic_ids=toi_tic_ids, store=True)
    logger.success(f"Downloaded {len(toi_tic_db)} gaia_id for TOI exoplanets")

    sunlike_star_tic_db = sunlike_tic_catalog.load_tic_target_dataset()
    if not sunlike_star_tic_db:
        logger.info("\n***\nQuery MAST TIC database for stars close to the Sun's mass")
        logger.info("Should take around 1 minute.")
        sunlike_star_tic_db = sunlike_tic_catalog.download_tic_targets(
            star_mass_range=(0.7, 1.3),
            priority_threshold=0.001,
            store=True,
        )
    logger.success(f"Downloaded {len(sunlike_star_tic_db)} stars close to the Sun's mass")

    # Combine all the gaia_id from the collected stars, and fetch their astrophysical parameters
    gaia_ids = set(chain.from_iterable([toi_tic_db.view["gaia_id"], sunlike_star_tic_db.view["gaia_id"]]))

    # "-1", represents a null value
    if -1 in gaia_ids:
        gaia_ids.remove(-1)

    gaia_db = gaia_dataset.load_gaia_parameters_dataset()
    if not gaia_db:
        print("***\nQuery GAIA stars astrophysical parameters for all the collected stars")
        gaia_db = gaia_dataset.download_gaia_parameters(gaia_ids=list(gaia_ids), store=True)
    logger.success(f"Downloaded {len(gaia_db)} Gaia star records")

    unique_tic_ids = set(chain.from_iterable((exo_db.tic_ids, toi_tic_ids, sunlike_star_tic_db.tic_ids)))
    if -1 in unique_tic_ids:
        unique_tic_ids.remove(-1)

    tess_metadata = tic_observations.load_observation_metadata()
    if not tess_metadata:
        logger.info("\n***\nQuery MAST TIC database for observations metadata")
        tess_metadata = tic_observations.download_observation_metadata(targets_tic_id=list(unique_tic_ids), store=True)
    logger.success(f"Downloaded {len(tess_metadata)} observations metadata")

    logger.success("***\nALL DATASETS ARE NOW DOWNLOADED!")
    logger.info(f"\t* exo_dataset: {len(exo_db)} rows")
    logger.info(f"\t* toi_dataset: {len(toi_db)} rows")
    logger.info(f"\t* toi_tic_dataset: {len(toi_tic_db)} rows")
    logger.info(f"\t* tic_sunlike_dataset: {len(sunlike_star_tic_db)} rows")
    logger.info(f"\t* gaia_dataset: {len(gaia_db) + len(gaia_exo_db)} rows")
    logger.info(f"\t* meta_dataset: {len(tess_metadata)} rows")


if __name__ == "__main__":
    download_datasets()
