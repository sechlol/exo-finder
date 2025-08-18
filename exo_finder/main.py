import numpy as np

from exo_finder.data_pipeline.p1_download_datasets import download_datasets
from exo_finder.data_pipeline.p2_download_lightcurves import (
    download_known_planets_lightcurves,
    download_sunlike_stars_lightcurves,
)
from exo_finder.data_pipeline.p3_analyze_lightcurves import analyze_lightcurves
from exo_finder.data_pipeline.p4_detrended_set_creation import create_lightcurve_training_set
from exo_finder.data_pipeline.p5_synthetic_set_creation import generate_synthetic_transits
from exo_finder.utils.secrets import get_secret
from exotools import GaiaParametersDataset, TicCatalogDataset, TicObservationsDataset


def login_to_services():
    # Authenticate with services using credentials
    TicObservationsDataset.authenticate_mast(mast_token=get_secret("MAST_TOKEN"))
    TicCatalogDataset.authenticate_casjobs(
        username=get_secret("CASJOB_USER"),
        password=get_secret("CASJOB_PASSWORD"),
    )
    GaiaParametersDataset.authenticate(
        username=get_secret("GAIA_USER"),
        password=get_secret("GAIA_PASSWORD"),
    )


def main():
    np.random.seed(666)
    login_to_services()

    # Download astronomical datasets
    download_datasets()
    download_known_planets_lightcurves()
    download_sunlike_stars_lightcurves(limit_observations=10000)

    # Create dataset for training the DL models
    analyze_lightcurves()
    create_lightcurve_training_set()
    generate_synthetic_transits()


if __name__ == "__main__":
    main()
