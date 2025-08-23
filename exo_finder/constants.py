# ----------
# Constants
# ----------
LC_WINDOW_SIZE = 2**12  # 4096 points, or about 5.68 days per light curve
LC_WINDOW_MIN_SIZE = int(LC_WINDOW_SIZE * 0.8)
SYNTHETIC_DATASET_LENGTH = 2**16  # 65_536 examples

# LC_WINDOW_SIZE = 2**12  # 4096 points, or about 5.68 days per light curve
# LC_WINDOW_SIZE = 2**11  # 2048 points, or about 2.84 days per light curve
# DATASET_LENGTH = 2**17  # 131_072 examples

# ----------
# HDF5 keys
# ----------
HDF5_KEY_LC_DATA = "lc_data"
HDF5_KEY_LC_SIZES = "lc_sizes"
HDF5_KEY_LC_TIC_IDS = "lc_tic_ids"
HDF5_KEY_SYNTHETIC_DATA = "syn_lc_data"
HDF5_KEY_SYNTHETIC_PARAMS = "syn_lc_params"

HDF5_KEY_LC_JSON_META = "lc_meta"
HDF5_KEY_SYNTHETIC_JSON_META = "syn_meta"
