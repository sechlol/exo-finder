import json
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
import numpy.typing as npt


class H5Wrapper:
    """
    Single-Writer / Multiple-Reader HDF5 wrapper.

    - Array datasets live at the file root, each as 2D (N, width).
    - JSON metadata lives under the "json/" group (keys are prefixed automatically).
    - Datasets are created lazily on first append; width & dtype are fixed then.
    - Appends grow along axis 0 (rows). Width (columns) is immutable.
    - Readers open the file in SWMR read mode with large raw-chunk caches.

    Concurrency:
      * One writer process at a time.
      * Many independent reader processes are fine (SWMR).
      * Do NOT share a single h5py.File handle across processes.
    """

    # -------------------- lifecycle --------------------

    def __init__(
        self,
        file_path: Path,
        compression: Literal["gzip", "lzf"] = "lzf",
        *,
        # Chunking / compression
        chunk_rows: Optional[int] = None,  # fixed rows per chunk; if None, use target_chunk_mib heuristic
        target_chunk_mib: float = 2.0,  # used if chunk_rows is None
        compression_opts: Optional[int] = None,  # gzip level or None
        shuffle: Optional[bool] = None,  # default: True for gzip, False for lzf
        # Read-side raw data chunk cache (per reader handle)
        rdcc_bytes: int = 256 * 1024 * 1024,
        rdcc_slots: int = 1_000_003,
        rdcc_w0: float = 0.75,
        # HDF5 lib options
        libver: Literal["latest", "earliest"] = "latest",
        enable_file_locking: Optional[bool] = None,  # set HDF5_USE_FILE_LOCKING; None = don't touch env
    ):
        self._file_path = Path(file_path)
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_rows = chunk_rows
        self.target_chunk_mib = float(target_chunk_mib)
        self.shuffle = (compression == "gzip") if shuffle is None else bool(shuffle)

        self._rdcc = dict(rdcc_nbytes=int(rdcc_bytes), rdcc_nslots=int(rdcc_slots), rdcc_w0=float(rdcc_w0))
        self.libver = libver

        # Optional env flag (rarely needed; only on read-only NFS quirks)
        if enable_file_locking is not None:
            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE" if enable_file_locking else "FALSE"

        # Lazy handles: one for writing, one for reading
        self._wf: Optional[h5py.File] = None
        self._rf: Optional[h5py.File] = None

    @property
    def file_path(self) -> Path:
        return self._file_path

    def _open_writer(self) -> h5py.File:
        if self._wf is None:
            os.makedirs(self._file_path.parent, exist_ok=True)
            # "a" = create or read/write
            self._wf = h5py.File(self._file_path, "a", libver=self.libver)
            if "libver" not in self._wf.attrs:
                self._wf.attrs["libver"] = self.libver
        return self._wf

    def _open_reader(self) -> h5py.File:
        if self._rf is None:
            # SWMR read with tuned raw chunk cache
            self._rf = h5py.File(self._file_path, "r", libver=self.libver, swmr=True, **self._rdcc)
        return self._rf

    @staticmethod
    def _json_path(key: str) -> str:
        # store JSON values as datasets of variable-length UTF-8 strings
        return f"json/{key}"

    @staticmethod
    def _bytes_per_row(width: int, dtype: np.dtype) -> int:
        return int(width) * np.dtype(dtype).itemsize

    def _choose_chunk_rows(self, width: int, dtype: np.dtype) -> int:
        if self.chunk_rows is not None:
            return int(self.chunk_rows)
        target = int(self.target_chunk_mib * 1024 * 1024)
        bpr = self._bytes_per_row(width, dtype)
        rows = max(1, target // max(1, bpr))
        # clamp/snap to friendly range
        rows = max(32, min(1024, rows))
        gran = 16
        return (rows + gran - 1) // gran * gran

    # -------------------- JSON API --------------------

    def write_json(self, json_key: str, data: dict[str, Any]):
        """
        Lazy initialize JSON group; throws if same JSON key already exists.
        """
        f = self._open_writer()
        path = self._json_path(json_key)
        if "json" not in f:
            f.create_group("json")
        if path in f:
            raise KeyError(f"JSON key already exists: {json_key}")
        # store as a scalar UTF-8 string dataset
        ds = f.create_dataset(path, shape=(), dtype=h5py.string_dtype("utf-8"))
        ds[()] = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    def read_json(self, json_key: str) -> dict[str, Any]:
        """
        Read JSON object; raises if missing.
        """
        f = self._open_reader()
        path = self._json_path(json_key)
        if path not in f:
            raise KeyError(f"JSON key not found: {json_key}")
        raw = f[path][()].decode("utf-8") if isinstance(f[path][()], (bytes, bytearray)) else f[path][()]
        return json.loads(raw)

    # -------------------- Array append API --------------------

    def _ensure_dataset(self, key: str, width: int, dtype: np.dtype) -> h5py.Dataset:
        f = self._open_writer()
        if key in f:
            d = f[key]
            if not isinstance(d, h5py.Dataset) or d.ndim != 2:
                raise ValueError(f"Existing object at '{key}' is not a 2D dataset.")
            # Enforce fixed width/dtype
            fixed_width = int(d.attrs.get("width", d.shape[1]))
            fixed_dtype = np.dtype(d.attrs.get("dtype", d.dtype))
            if fixed_width != int(width):
                raise ValueError(f"Width mismatch for '{key}': existing {fixed_width}, new {width}.")
            if np.dtype(fixed_dtype) != np.dtype(dtype):
                raise ValueError(f"dtype mismatch for '{key}': existing {fixed_dtype}, new {np.dtype(dtype)}.")
            return d

        # Create new dataset
        chunk_rows = self._choose_chunk_rows(width, dtype)
        d = f.create_dataset(
            key,
            shape=(0, int(width)),
            maxshape=(None, int(width)),
            dtype=np.dtype(dtype),
            chunks=(chunk_rows, int(width)),
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=bool(self.shuffle),
        )
        d.attrs["width"] = int(width)
        d.attrs["dtype"] = str(np.dtype(dtype))
        d.attrs["length"] = 0
        return d

    @staticmethod
    def _normalize_append_array(arr: npt.NDArray) -> Tuple[np.ndarray, int, int]:
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"append expects 1D (W,) or 2D (B,W); got shape {arr.shape}.")
        B, W = arr.shape
        if W <= 0:
            raise ValueError("Width must be > 0.")
        return arr, int(B), int(W)

    def append(self, dataset_key: str, data: npt.NDArray):
        """
        Lazy initialize dataset if not exists. Data can be 1D or 2D:
          (W,), (1, W), or (B, W).
        Fixes dtype & width at first insert; length can grow.
        Appends along axis 0 (vstack).
        """
        arr, B, W = self._normalize_append_array(np.asarray(data))
        d = self._ensure_dataset(dataset_key, width=W, dtype=arr.dtype)
        i0 = int(d.attrs.get("length", d.shape[0]))
        i1 = i0 + B
        d.resize((i1, W))
        d[i0:i1] = arr
        d.attrs["length"] = i1

    # -------------------- Read API --------------------

    def _read_dataset(self, key: str) -> h5py.Dataset:
        f = self._open_reader()
        if key not in f:
            raise KeyError(f"Dataset not found: {key}")
        d = f[key]
        if not isinstance(d, h5py.Dataset) or d.ndim != 2:
            raise ValueError(f"Object at '{key}' is not a 2D dataset.")
        return d

    @staticmethod
    def _sorted_take(dset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        """
        Fancy-take along axis 0 with good locality and duplicate safety:
          - sort indices,
          - make strictly increasing via np.unique (h5py requirement),
          - read once,
          - re-expand duplicates,
          - restore original caller order.
        """
        if indices.size == 0:
            return np.empty((0, dset.shape[1]), dtype=dset.dtype)

        # Sort (stable keeps equal elements' relative order)
        order = np.argsort(indices, kind="stable")
        sorted_idx = indices[order]

        # Strictly increasing for h5py; keep inverse to rebuild duplicates
        uniq_idx, uniq_inverse = np.unique(sorted_idx, return_inverse=True)

        # Single HDF5 read
        uniq_rows = dset[uniq_idx.tolist()]  # strictly increasing list

        # Reconstruct the sorted output (with duplicates)
        out_sorted = uniq_rows[uniq_inverse]

        # Unscramble back to the original order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return np.asarray(out_sorted)[inv]

    @staticmethod
    def _normalize_index(idx: Optional[int | Sequence[int] | slice], N: int) -> Tuple[Optional[np.ndarray], bool]:
        """
        Normalize an index spec into (indices | None, is_single):
          - None  -> (None, False)   meaning "all"
          - slice -> (np.arange(...), False)
          - int   -> (array([i]), True)
          - seq   -> (np.array(seq), len==1)
        """
        if idx is None:
            return None, False

        if isinstance(idx, slice):
            start, stop, step = idx.indices(N)
            return np.arange(start, stop, step, dtype=np.int64), False

        if isinstance(idx, (int, np.integer)):
            return np.asarray([int(idx)], dtype=np.int64), True

        arr = np.asarray(idx, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("Indices must be 1D.")
        return arr, (arr.size == 1)

    def read(
        self,
        dataset_key: str,
        rows: Optional[int | Sequence[int] | slice] = None,
        cols: Optional[int | Sequence[int] | slice] = None,
    ) -> np.ndarray:
        """
        - Raises if dataset does not exist.
        - If rows is None and cols is None: returns the full dataset.
        - Sorts indices internally for locality; returns in the same order given.
        - Does NOT boundary-check: out-of-range errors propagate from h5py.
        - Shape policy:
            * single row XOR single col  -> return 1D
            * single row AND single col  -> return scalar
            * otherwise                  -> return 2D
        """
        d = self._read_dataset(dataset_key)
        N, W = d.shape

        # Normalize selections (helpers return (indices|None, is_single_bool))
        row_idx, single_row = self._normalize_index(rows, N)
        col_idx, single_col = self._normalize_index(cols, W)

        # Full dataset fast-path
        if row_idx is None and col_idx is None:
            return np.asarray(d[...])

        # If rows are fancy (list/ndarray), always take rows first using a duplicate-safe,
        # strictly-increasing read; then slice columns in NumPy.
        if isinstance(row_idx, np.ndarray):
            taken = self._sorted_take(d, row_idx)  # (B, W)
            result = taken if col_idx is None else taken[:, col_idx]

        # Else if columns are fancy, take columns via transpose with the same trick,
        # then slice rows in NumPy.
        elif isinstance(col_idx, np.ndarray):
            taken_T = self._sorted_take(d.T, col_idx)  # (C, N) over transposed
            taken = taken_T.T  # (N, C)
            result = taken if row_idx is None else taken[row_idx, :]

        # Else (no fancy lists): direct indexing with ints/slices only.
        else:
            rsel = slice(None) if row_idx is None else row_idx
            csel = slice(None) if col_idx is None else col_idx
            result = np.asarray(d[rsel, csel])

        # Shape policy
        if single_row ^ single_col:
            return np.asarray(result).reshape(-1)  # 1D
        if single_row and single_col:
            return np.asarray(result).reshape(())  # scalar
        return result

    # -------------------- Introspection --------------------

    def list_dataset_keys(self) -> List[str]:
        """
        Returns array dataset keys (excludes JSON namespace).
        """
        f = self._open_reader()
        keys: List[str] = []
        for k, obj in f.items():
            if k == "json":
                continue
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                keys.append(k)
        return keys

    def list_json_keys(self) -> List[str]:
        f = self._open_reader()
        if "json" not in f:
            return []
        return [k.split("/", 1)[1] for k in f["json"].keys()]

    def shape(self, dataset_key: str) -> Tuple[int, int]:
        d = self._read_dataset(dataset_key)
        return tuple(d.shape)  # (N, width)

    def exists(self, dataset_key: str) -> bool:
        f = self._open_reader()
        return dataset_key in f and isinstance(f[dataset_key], h5py.Dataset)

    # -------------------- Lifecycle controls --------------------

    def flush(self):
        """
        Flush writer-side metadata & data to disk.
        """
        if self._wf is not None:
            self._wf.flush()

    def finalize(self):
        """
        Enable SWMR on the writer handle so other processes can open in read (swmr=True).
        Call this after you're done mutating.
        """
        if self._wf is None:
            # nothing to do if never opened for write
            return
        self._wf.flush()
        # Flip SWMR mode for the writer; readers should open with swmr=True
        self._wf.swmr_mode = True

    def close(self):
        """
        Close any open handles (idempotent).
        """
        if self._wf is not None:
            try:
                self._wf.close()
            finally:
                self._wf = None
        if self._rf is not None:
            try:
                self._rf.close()
            finally:
                self._rf = None
