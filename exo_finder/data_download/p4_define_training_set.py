import numpy as np
import pandas as pd

from exo_finder.default_datasets import exo_dataset, candidate_dataset
from paths import LC_STATS_RESULT_FILE

_mandatory_fields = ["pl_rade", "pl_trandur", "pl_tranmid", "pl_orbsmax"]


def select_subset(combined_analysis: pd.DataFrame) -> pd.DataFrame:
    """Define rules for selecting "well-behaving" lightcurves, and then filters the given list to a subset"""
    gap_ratio_threshold = 0.15
    mad_threshold = np.percentile(combined_analysis["mad"], 70)
    std_threshold = np.percentile(combined_analysis["normalized_std"], 70)

    subset = combined_analysis[
        (combined_analysis.gap_ratio <= gap_ratio_threshold)
        & (combined_analysis.mad <= mad_threshold)
        & (combined_analysis.normalized_std <= std_threshold)
    ]

    return subset


def define_training_set() -> pd.DataFrame:
    exo_db = exo_dataset.load_known_exoplanets_dataset(with_gaia_star_data=True)
    candidate_db = candidate_dataset.load_candidate_exoplanets_dataset()
    lc_analysis = pd.read_feather(LC_STATS_RESULT_FILE)

    # exclude invalid lightcurves
    lc_analysis = lc_analysis[lc_analysis["normalized_std"] > 0]

    # exclude lightcurves with known or candidate planets
    planets_tic_ids = list(set(exo_db.unique_tic_ids) | set(candidate_db.unique_tic_ids))
    lc_analysis = lc_analysis[~lc_analysis["tic_id"].isin(planets_tic_ids)]

    # Selects only the well behaving lightcurves
    return select_subset(lc_analysis)


if __name__ == "__main__":
    define_training_set()
