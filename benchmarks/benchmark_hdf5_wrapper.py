"""
Benchmark H5Wrapper performance for JSON and dataset operations.

- JSON write/read throughput.
- Dataset append throughput across batch sizes and dtypes.
- Dataset sequential-read and random minibatch-read throughput across batch sizes and dtypes.

Defaults:
  width = 2**13 = 8192
  dtypes = [float32, int8]
  batch_sizes = [1, 32, 128, 512, 2048, 8192]
  samples = 50_000  # total rows per dtype (tunable via CLI)

Usage:
  python bench_h5wrapper.py --file bench.h5 --samples 50000 --compression gzip --gzip-level 1
"""

import platform
from tqdm import tqdm
import statistics as stats
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np

from exo_finder.utils.hdf5_wrapper import H5Wrapper


def mib(bytes_count: int) -> float:
    return bytes_count / (1024.0 * 1024.0)


def fmt(n: float) -> str:
    return f"{n:,.2f}"


def generate_batch(bsz: int, width: int, dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    if np.dtype(dtype) == np.float32:
        return rng.standard_normal((bsz, width), dtype=np.float32)
    elif np.dtype(dtype) == np.int8:
        return rng.integers(low=-128, high=127, size=(bsz, width), dtype=np.int8)
    else:
        raise ValueError(f"Unsupported dtype for benchmark: {dtype}")


def bench_json(h5: H5Wrapper, num_entries: int = 100) -> Tuple[float, float]:
    """
    Returns: (writes_per_sec, reads_per_sec)
    """
    # Write
    t0 = time.perf_counter()
    for i in range(num_entries):
        h5.set_json(f"entry_{i:04d}", {"i": i, "payload": "x" * 64})
    h5.flush()
    t1 = time.perf_counter()
    # Read
    t2 = time.perf_counter()
    for i in range(num_entries):
        _ = h5.get_json(f"entry_{i:04d}")
    t3 = time.perf_counter()

    writes_per_sec = num_entries / (t1 - t0)
    reads_per_sec = num_entries / (t3 - t2)
    return writes_per_sec, reads_per_sec


def bytes_per_row(width: int, dtype: np.dtype) -> int:
    return int(width) * np.dtype(dtype).itemsize


def plan_batches(total_rows: int, batch_size: int) -> List[int]:
    """Return a list of batch sizes that sums to total_rows."""
    n_full = total_rows // batch_size
    batches = [batch_size] * n_full
    rem = total_rows - n_full * batch_size
    if rem > 0:
        batches.append(rem)
    return batches


def bench_write(
    file_path: Path,
    key: str,
    total_rows: int,
    batch_size: int,
    width: int,
    dtype: np.dtype,
    compression: str,
    gzip_level: int,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    # New file per write test to avoid mixing results across configs
    if file_path.exists():
        file_path.unlink()
    h5 = H5Wrapper(
        file_path,
        compression=compression,  # "gzip" or "lzf"
        compression_opts=(gzip_level if compression == "gzip" else None),
        target_chunk_mib=2.0,
    )
    batches = plan_batches(total_rows, batch_size)
    bpr = bytes_per_row(width, dtype)

    latencies = []
    rows_written = 0
    t0 = time.perf_counter()
    for bsz in batches:
        data = generate_batch(bsz, width, dtype, rng)
        t1 = time.perf_counter()
        h5.append(key, data)
        t2 = time.perf_counter()
        latencies.append(t2 - t1)
        rows_written += bsz
    h5.finalize()
    h5.close()
    t3 = time.perf_counter()

    total_bytes = rows_written * bpr
    wall_s = t3 - t0
    median_batch_ms = (stats.median(latencies) * 1000.0) if latencies else float("nan")
    return {
        "rows": rows_written,
        "bytes_mib": mib(total_bytes),
        "wall_s": wall_s,
        "throughput_mibs": mib(total_bytes) / wall_s,
        "median_batch_ms": median_batch_ms,
    }


def bench_read_random(
    file_path: Path,
    key: str,
    total_rows: int,
    batch_size: int,
    width: int,
    dtype: np.dtype,
    repeats: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    # Random batches
    latencies = []
    bytes_total = 0
    bpr = bytes_per_row(width, dtype)
    # Use a fresh handle in read mode
    h5 = H5Wrapper(file_path)
    for _ in range(repeats):
        idx = rng.integers(low=0, high=total_rows, size=(batch_size,), dtype=np.int64)
        t1 = time.perf_counter()
        _ = h5.read(key, rows=idx)
        t2 = time.perf_counter()
        latencies.append(t2 - t1)
        bytes_total += bpr * batch_size
    h5.close()
    total_s = sum(latencies)
    return {
        "reads": repeats,
        "bytes_mib": mib(bytes_total),
        "avg_latency_ms": (total_s / max(1, repeats)) * 1000.0,
        "throughput_mibs": mib(bytes_total) / total_s if total_s > 0 else float("inf"),
    }


def bench_read_sequential(
    file_path: Path, key: str, total_rows: int, batch_size: int, width: int, dtype: np.dtype, seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)  # not used; kept for symmetry
    bpr = bytes_per_row(width, dtype)
    batches = plan_batches(total_rows, batch_size)
    h5 = H5Wrapper(file_path)
    latencies = []
    row_start = 0
    for bsz in batches:
        idx = np.arange(row_start, row_start + bsz, dtype=np.int64)
        t1 = time.perf_counter()
        _ = h5.read(key, rows=idx)
        t2 = time.perf_counter()
        latencies.append(t2 - t1)
        row_start += bsz
    h5.close()
    total_bytes = sum(batches) * bpr
    total_s = sum(latencies)
    return {
        "reads": len(batches),
        "bytes_mib": mib(total_bytes),
        "avg_latency_ms": (total_s / max(1, len(batches))) * 1000.0,
        "throughput_mibs": mib(total_bytes) / total_s if total_s > 0 else float("inf"),
    }


def main(
    file: Path = Path("bench.h5"),
    samples: int = 2**12,
    width: int = 2**12,
    compression: str = "lzf",
    gzip_level: int = 1,
    seed: int = 0,
    json_entries: int = 200,
    random_read_repeats: int = 50,
    batches: list[int] = [2048, 512, 128, 32, 4],
):
    # ---------------- Environment ----------------

    print("=== Environment ===")
    print(f"h5py: {h5py.version.info}")
    print(f"NumPy: {np.__version__}")
    print(f"Python: {platform.python_version()}  |  Platform: {platform.platform()}")
    print()

    results: dict[str, Any] = {
        "params": dict(
            samples=samples,
            width=width,
            compression=compression,
            gzip_level=(gzip_level if compression == "gzip" else None),
            batches=batches,
            random_read_repeats=random_read_repeats,
        ),
        "json": {},
        "datasets": [],
    }

    # ---------------- JSON benchmark (progress + collect) ----------------
    json_path = file.with_suffix(".jsonbench.h5")
    if json_path.exists():
        json_path.unlink()
    h5j = H5Wrapper(
        json_path,
        compression=compression,
        compression_opts=(gzip_level if compression == "gzip" else None),
    )
    t0 = time.perf_counter()
    for i in tqdm(range(json_entries), desc="JSON: writing", unit="item"):
        h5j.set_json(f"entry_{i:04d}", {"i": i, "payload": "x" * 64})
    h5j.flush()
    t1 = time.perf_counter()
    for i in tqdm(range(json_entries), desc="JSON: reading", unit="item"):
        _ = h5j.get_json(f"entry_{i:04d}")
    t2 = time.perf_counter()
    h5j.finalize()
    h5j.close()

    results["json"] = {
        "entries": json_entries,
        "write_items_per_s": json_entries / (t1 - t0) if (t1 - t0) > 0 else float("inf"),
        "read_items_per_s": json_entries / (t2 - t1) if (t2 - t1) > 0 else float("inf"),
        "wall_s": (t2 - t0),
    }

    # ---------------- Dataset benchmarks (progress + collect) ----------------
    dtypes = [np.float32, np.int8]
    print("=== Dataset benchmarks (collecting, progress shown) ===")
    print(f"Total rows per dtype: {samples}")
    print(f"Width: {width} elements")
    print(f"Compression: {compression} (level={gzip_level if compression == 'gzip' else 'n/a'})")
    print(f"Batch sizes: {batches}")
    print()

    for dtype in dtypes:
        for bsz in batches:
            key = f"data_{np.dtype(dtype).name}_bs{bsz}"
            path = file.with_suffix(f".{np.dtype(dtype).name}.bs{bsz}.h5")
            if path.exists():
                path.unlink()

            rng = np.random.default_rng(seed)
            # ---- WRITE with progress ----
            h5w = H5Wrapper(
                path,
                compression=compression,
                compression_opts=(gzip_level if compression == "gzip" else None),
                target_chunk_mib=2.0,
            )
            batch_plan = plan_batches(samples, bsz)
            write_latencies: list[float] = []
            t0w = time.perf_counter()
            with tqdm(total=samples, desc=f"WRITE {np.dtype(dtype).name} bs={bsz}", unit="row") as pbar:
                for b in batch_plan:
                    data = generate_batch(b, width, dtype, rng)
                    t1w = time.perf_counter()
                    h5w.append(key, data)
                    t2w = time.perf_counter()
                    write_latencies.append(t2w - t1w)
                    pbar.update(b)
            h5w.finalize()
            h5w.close()
            t3w = time.perf_counter()

            total_bytes = samples * bytes_per_row(width, dtype)
            write_wall = t3w - t0w
            write_throughput = mib(total_bytes) / write_wall if write_wall > 0 else float("inf")
            write_median_batch_ms = (np.median(write_latencies) * 1000.0) if write_latencies else float("nan")

            # ---- SEQ READ with progress ----
            seq_batches = plan_batches(samples, bsz)
            seq_latencies: list[float] = []
            h5r_seq = H5Wrapper(path)  # reader handle
            row_start = 0
            with tqdm(total=samples, desc=f"SEQ  {np.dtype(dtype).name} bs={bsz}", unit="row") as pbar:
                for b in seq_batches:
                    idx = np.arange(row_start, row_start + b, dtype=np.int64)
                    t1 = time.perf_counter()
                    _ = h5r_seq.read(key, rows=idx)
                    t2 = time.perf_counter()
                    seq_latencies.append(t2 - t1)
                    pbar.update(b)
                    row_start += b
            h5r_seq.close()
            seq_total_s = sum(seq_latencies)
            seq_throughput = mib(total_bytes) / seq_total_s if seq_total_s > 0 else float("inf")
            seq_avg_ms = (seq_total_s / max(1, len(seq_latencies))) * 1000.0 if seq_latencies else float("nan")

            # ---- RAND READ with progress ----
            rand_latencies: list[float] = []
            h5r_rand = H5Wrapper(path)
            with tqdm(total=random_read_repeats, desc=f"RAND {np.dtype(dtype).name} bs={bsz}", unit="batch") as pbar:
                for _ in range(random_read_repeats):
                    idx = np.random.default_rng(seed).integers(0, samples, size=(bsz,), dtype=np.int64)
                    t1 = time.perf_counter()
                    _ = h5r_rand.read(key, rows=idx)
                    t2 = time.perf_counter()
                    rand_latencies.append(t2 - t1)
                    pbar.update(1)
            h5r_rand.close()
            rand_bytes = bytes_per_row(width, dtype) * bsz * random_read_repeats
            rand_total_s = sum(rand_latencies)
            rand_throughput = mib(rand_bytes) / rand_total_s if rand_total_s > 0 else float("inf")
            rand_avg_ms = (
                (rand_total_s / max(1, random_read_repeats)) * 1000.0 if random_read_repeats > 0 else float("nan")
            )

            # Collect row
            results["datasets"].append(
                {
                    "dtype": np.dtype(dtype).name,
                    "batch_size": bsz,
                    "write": {
                        "throughput_mibs": write_throughput,
                        "wall_s": write_wall,
                        "median_batch_ms": write_median_batch_ms,
                    },
                    "seq_read": {
                        "throughput_mibs": seq_throughput,
                        "avg_batch_ms": seq_avg_ms,
                    },
                    "rand_read": {
                        "throughput_mibs": rand_throughput,
                        "avg_batch_ms": rand_avg_ms,
                    },
                }
            )

    # ---------------- Final consolidated report ----------------
    print("\n=== RESULTS SUMMARY ===")
    print(
        f"Params: samples={results['params']['samples']}, width={results['params']['width']}, "
        f"compression={results['params']['compression']}, "
        f"level={results['params']['gzip_level']}, batches={results['params']['batches']}, "
        f"rand_repeats={results['params']['random_read_repeats']}"
    )
    print("\nJSON:")
    j = results["json"]
    print(
        f"  entries={j['entries']}, write={fmt(j['write_items_per_s'])} items/s, "
        f"read={fmt(j['read_items_per_s'])} items/s, wall={fmt(j['wall_s'])} s"
    )

    # Tabular dataset summary
    header = (
        "dtype".ljust(8)
        + " | "
        + "bs".rjust(6)
        + " | "
        + "write MB/s".rjust(12)
        + " | "
        + "write med ms".rjust(12)
        + " | "
        + "seq MB/s".rjust(10)
        + " | "
        + "seq avg ms".rjust(10)
        + " | "
        + "rand MB/s".rjust(11)
        + " | "
        + "rand avg ms".rjust(12)
    )
    print("\nDATASETS:")
    print(header)
    print("-" * len(header))
    for row in results["datasets"]:
        print(
            f"{row['dtype']:<8} | "
            f"{row['batch_size']:>6d} | "
            f"{fmt(row['write']['throughput_mibs']):>12} | "
            f"{fmt(row['write']['median_batch_ms']):>12} | "
            f"{fmt(row['seq_read']['throughput_mibs']):>10} | "
            f"{fmt(row['seq_read']['avg_batch_ms']):>10} | "
            f"{fmt(row['rand_read']['throughput_mibs']):>11} | "
            f"{fmt(row['rand_read']['avg_batch_ms']):>12}"
        )

    # Return results for plotting/analysis in IDE
    return results


if __name__ == "__main__":
    main()
