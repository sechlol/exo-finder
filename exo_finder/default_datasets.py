from exo_finder.default_storage import DEFAULT_STORAGE
from exo_finder.utils.hdf5_wrapper import H5Wrapper
from exotools import (
    CandidateExoplanetsDataset,
    KnownExoplanetsDataset,
    LightcurveDataset,
    LightcurveDB,
    TicCatalogDataset,
)
from exotools.datasets import GaiaParametersDataset
from exotools.datasets.tic_observations import TicObservationsDataset
from paths import LIGHTCURVES_PATH, TRAINING_DATASET_FILE

exo_dataset = KnownExoplanetsDataset(storage=DEFAULT_STORAGE)
candidate_dataset = CandidateExoplanetsDataset(storage=DEFAULT_STORAGE)
gaia_dataset = GaiaParametersDataset(storage=DEFAULT_STORAGE)
tic_observations = TicObservationsDataset(storage=DEFAULT_STORAGE)
candidate_tic_catalog = TicCatalogDataset(storage=DEFAULT_STORAGE, dataset_tag="candidates")
sunlike_tic_catalog = TicCatalogDataset(storage=DEFAULT_STORAGE, dataset_tag="sunlike_stars")

transiting_lightcurves_ds = LightcurveDataset(lc_storage_path=LIGHTCURVES_PATH, dataset_tag="transiting_exoplanets")
sunlike_lightcurves_ds = LightcurveDataset(lc_storage_path=LIGHTCURVES_PATH, dataset_tag="sunlike_stars")


def get_combined_lightcurve_db() -> LightcurveDB:
    lc_db1 = transiting_lightcurves_ds.load_lightcurve_dataset()
    lc_db2 = sunlike_lightcurves_ds.load_lightcurve_dataset()

    return lc_db1.append(lc_db2)


train_dataset_h5 = H5Wrapper(file_path=TRAINING_DATASET_FILE)
