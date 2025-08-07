import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

# Dataset paths
DATA_PATH = PROJECT_ROOT / "data"
DATASET_PATH = DATA_PATH / "datasets"
LIGHTCURVES_PATH = DATA_PATH / "lightcurves"
