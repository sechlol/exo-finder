import math
import threading
import time
from multiprocessing import Manager
from typing import Callable, Iterable, Any, Optional, Literal, List, Sequence

from joblib import Parallel, delayed
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn


# --- Worker helpers (must be top-level for pickling) ---


def _run_and_inc(arg, func, counter, lock, use_manager: bool):
    """Run single task and increment shared counter (combined progress)."""
    res = func(arg)
    if use_manager:
        with lock:
            counter["count"] += 1
    else:
        with lock:
            counter[0] += 1
    return res


def _process_chunk(chunk, func, idx: int, counters, locks, use_manager: bool):
    """Process a chunk of tasks and increment per-worker counter."""
    out = []
    for arg in chunk:
        out.append(func(arg))
        if use_manager:
            with locks[idx]:
                counters[idx] += 1
        else:
            with locks[idx]:
                counters[idx][0] += 1
    return out


# --- Public API ---


def parallel_execution(
    func: Callable[[Any], Any] | Callable[[Sequence[Any]], list[Any]],
    params: Iterable[Any],
    n_jobs: int,
    description: str,
    total: Optional[int] = None,
    batch: bool = False,
    backend: Literal["multiprocessing", "threading"] = "multiprocessing",
    combined_progress: bool = False,
):
    params = list(params)
    if total is None:
        total = len(params)
    if total < 0:
        raise ValueError("total must be non-negative or None")

    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    n_jobs = min(n_jobs, len(params)) if len(params) else 1

    use_multiproc = backend == "multiprocessing"

    # Prepare shared counters & locks (manager proxies for multiprocessing)
    if use_multiproc:
        manager = Manager()
        if combined_progress:
            counter = manager.dict({"count": 0})
            lock = manager.Lock()
            counters = None
            locks = None
        else:
            # chunk the params first to know number of workers
            chunk_size = math.ceil(len(params) / n_jobs) if len(params) else 1
            chunks = [params[i : i + chunk_size] for i in range(0, len(params), chunk_size)]
            n_workers = len(chunks)
            counters = manager.list([0 for _ in range(n_workers)])  # int proxies
            locks = [manager.Lock() for _ in range(n_workers)]
            counter = None
            lock = None
    else:
        # threading backend: use in-process structures
        if combined_progress:
            counter = [0]  # mutable container
            lock = threading.Lock()
            counters = None
            locks = None
        else:
            chunk_size = math.ceil(len(params) / n_jobs) if len(params) else 1
            chunks = [params[i : i + chunk_size] for i in range(0, len(params), chunk_size)]
            n_workers = len(chunks)
            counters = [[0] for _ in range(n_workers)]  # list of one-element lists as mutable ints
            locks = [threading.Lock() for _ in range(n_workers)]
            counter = None
            lock = None

    progress = Progress(
        TextColumn("[bold blue]{task.fields[desc]}", justify="right"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}% ({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
    )

    results = []
    with progress:
        stop_updater = threading.Event()

        # Updater thread reads counters and advances rich tasks
        def _updater(task_ids: List[int], combined_mode: bool, last_snapshot: List[int]):
            while not stop_updater.is_set():
                if combined_mode:
                    # combined counter may be manager.dict or local list
                    if use_multiproc:
                        curr = counter["count"]
                    else:
                        curr = counter[0]
                    delta = curr - last_snapshot[0]
                    if delta > 0:
                        progress.advance(task_ids[0], delta)
                        last_snapshot[0] = curr
                else:
                    # per-worker counters
                    for i, tid in enumerate(task_ids):
                        if use_multiproc:
                            curr = counters[i]
                        else:
                            curr = counters[i][0]
                        delta = curr - last_snapshot[i]
                        if delta > 0:
                            progress.advance(tid, delta)
                            last_snapshot[i] = curr
                time.sleep(0.05)

        if combined_progress:
            task_id = progress.add_task("", total=total, desc=description)
            # Start updater
            last = [0]
            updater = threading.Thread(target=_updater, args=([task_id], True, last), daemon=True)
            updater.start()

            # Launch parallel jobs; pass only picklable shared proxies (not Progress)
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_run_and_inc)(p, func, counter, lock, use_multiproc) for p in params
            )

            # finish: ensure UI shows full progress
            stop_updater.set()
            updater.join()
            # final catch-up (in case)
            if use_multiproc:
                final = counter["count"]
            else:
                final = counter[0]
            progress.update(task_id, completed=final)

        else:
            # per-worker: chunk params, create tasks per chunk
            # (chunks already computed for multiprocessing; compute if not)
            if not use_multiproc and "chunks" not in locals():
                chunk_size = math.ceil(len(params) / n_jobs) if len(params) else 1
                chunks = [params[i : i + chunk_size] for i in range(0, len(params), chunk_size)]
            task_ids = []
            for i, chunk in enumerate(chunks):
                task_ids.append(progress.add_task("", total=len(chunk), desc=f"{description} [W{i}]"))

            last_snapshot = [0 for _ in range(len(chunks))]
            updater = threading.Thread(target=_updater, args=(task_ids, False, last_snapshot), daemon=True)
            updater.start()

            results_nested = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_process_chunk)(chunk, func, i, counters, locks, use_multiproc)
                for i, chunk in enumerate(chunks)
            )

            stop_updater.set()
            updater.join()

            # catch-up and flatten
            for i, tid in enumerate(task_ids):
                if use_multiproc:
                    final = counters[i]
                else:
                    final = counters[i][0]
                progress.update(tid, completed=final)

            # flatten results preserving chunk order
            results = [item for sub in results_nested for item in sub]

    return results
