import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

# Dataset paths
DATA_PATH = PROJECT_ROOT / "data"
DATASET_PATH = DATA_PATH / "datasets"
LIGHTCURVES_PATH = DATA_PATH / "lightcurves"
TRAINING_DATASET_PATH = DATA_PATH / "training"


# Analysis artifacts paths
ANALYSIS_PATH = DATA_PATH / "analysis"
LC_STATS_RESULT_FILE = ANALYSIS_PATH / "lightcurve_stats.feather"
TRAINING_DATASET_FILE = TRAINING_DATASET_PATH / "training_dataset.hdf5"

# Model data and checkpoints
MODEL_DATA_PATH = PROJECT_ROOT / "models"
