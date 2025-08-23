from pathlib import Path

from exo_finder.default_storage import DEFAULT_STORAGE
from exo_finder.utils.hdf5_wrapper import H5Wrapper
from exotools import (
    CandidateExoplanetsDataset,
    PlanetarySystemsDataset,
    LightcurveDataset,
    LightcurveDB,
    TicCatalogDataset,
)
from exotools.datasets import GaiaParametersDataset
from exotools.datasets.tic_observations import TicObservationsDataset
from paths import LIGHTCURVES_PATH, TRAINING_DATASET_FILE

exo_dataset = PlanetarySystemsDataset(storage=DEFAULT_STORAGE)
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


def get_train_dataset_h5(file_path: Path | str = TRAINING_DATASET_FILE) -> H5Wrapper:
    """
    lazily returns a handle for the training dataset, that can be used in multiprocessing context and in
    pytorch datasets.

    chunk_rows=128
        For batch_size=128 it’s perfectly aligned → most batches hit ~1 chunk per row-block.
        For batch_size=256, each batch touches ~2 chunks (still efficient).
        For batch_size=64 and 32, each chunk serves multiple batches, so the cache is effective.

    compression=None
        On local NVMe, compression adds CPU and increases small random-read latency. Since you’re doing random,
        row-oriented reads, uncompressed is consistently faster. (If you later train from a NAS, switch to
        compression="lzf" and keep chunk_rows the same.)

    rdcc_slots=1_000_003
        Number of hash slots for cached chunks.
        HDF5 looks up chunks in a hash table. If the number of slots is too small, different chunks collide in the same
            slot, causing cache misses even if there’s enough memory.
        Rule of thumb: pick a large prime number, ideally bigger than the expected number of chunks in the cache.
        Value 1_000_003 is just a “big prime ≈ 1e6”, large enough to keep collisions rare for typical dataset sizes.

    rdcc_w0=0.75
        Total byte size of the chunk cache.
        Each dataset access will try to keep chunks up to this size in memory.
        If the dataset chunks are small (e.g., 64×width), you’ll want many of them cached to avoid re-reading from
            disk when sampling random rows.
        You set this to 256 * 1024 * 1024 (256 MB) so each open file handle gets ~256 MB of chunk cache.
            On a machine with multiple PyTorch workers, each process gets its own cache.
    """
    return H5Wrapper(
        file_path=file_path,
        compression=None,  # fastest for local disk + random reads
        chunk_rows=128,  # sweet spot across 32–256 batch size reads
        rdcc_bytes=256 * 2**20,  # (256 MB) per worker
        shuffle=False,  # irrelevant without compression
        rdcc_slots=1_000_003,
        rdcc_w0=0.75,
        libver="latest",
    )
