import numpy as np

from exo_finder.data_download.download_datasets import download_datasets
from exo_finder.data_download.download_lightcurves import (
    download_known_planets_lightcurves,
    download_sunlike_stars_lightcurves,
)
from exo_finder.utils.secrets import get_secret
from exotools import TicObservationsDataset, TicCatalogDataset, GaiaParametersDataset


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

    download_datasets()
    download_known_planets_lightcurves()
    download_sunlike_stars_lightcurves(limit_observations=10000)


if __name__ == "__main__":
    main()
