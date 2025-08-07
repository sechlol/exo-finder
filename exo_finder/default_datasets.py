from exo_finder.default_storage import DEFAULT_STORAGE
from exotools import KnownExoplanetsDataset, CandidateExoplanetsDataset, TicCatalogDataset, LightcurveDataset
from exotools.datasets import GaiaParametersDataset
from exotools.datasets.tic_observations import TicObservationsDataset
from paths import LIGHTCURVES_PATH

exo_dataset = KnownExoplanetsDataset(storage=DEFAULT_STORAGE)
candidate_dataset = CandidateExoplanetsDataset(storage=DEFAULT_STORAGE)
gaia_dataset = GaiaParametersDataset(storage=DEFAULT_STORAGE)
tic_observations = TicObservationsDataset(storage=DEFAULT_STORAGE)
candidate_tic_catalog = TicCatalogDataset(storage=DEFAULT_STORAGE, dataset_tag="candidates")
sunlike_tic_catalog = TicCatalogDataset(storage=DEFAULT_STORAGE, dataset_tag="sunlike_stars")

TRANSITING_LIGHTCURVES_DS = LightcurveDataset(lc_storage_path=LIGHTCURVES_PATH, dataset_tag="transiting_exoplanets")
SUNLIKE_LIGHTCURVES_DS = LightcurveDataset(lc_storage_path=LIGHTCURVES_PATH, dataset_tag="sunlike_stars")
