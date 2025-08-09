# parallel_execution.py
from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)
from enum import Enum
from typing import Any, Callable, Generator, Optional

import cloudpickle  # pip install cloudpickle
from rich.progress import Progress, TaskID, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

logger = logging.getLogger(__name__)


class TaskProfile(Enum):
    CPU_BOUND = "cpu"
    IO_BOUND = "io"


def _is_sequence(obj: object) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def _chunked_from_sequence(seq: Sequence[Any], batch_size: int) -> list[tuple[int, list[Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = len(seq)
    if batch_size >= n:
        return [(0, list(seq))]
    batches = []
    for start in range(0, n, batch_size):
        batches.append((start, list(seq[start : start + batch_size])))
    return batches


def _chunked_from_iterable(it: Iterable[Any], batch_size: int) -> Generator[tuple[int, list[Any]], None, None]:
    it = iter(it)
    idx = 0
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            break
        yield idx, chunk
        idx += len(chunk)


# --------------------
# Helpers for invoking user func on batches with fallback to per-item mapping
# --------------------
def _apply_func_to_batch_with_fallback(func: Callable[..., Any], batch: Sequence[Any]) -> list[Any]:
    """
    Preferred behavior: call func(batch) and expect list-like result with same length.
    If that fails (raises, or returns something with wrong length/type), fall back to mapping func(item).
    """
    # first try the batch-aware call
    try:
        out = func(batch)
        # accept if it's a sequence of same length
        if isinstance(out, Sequence) and len(out) == len(batch):
            return list(out)
        # if it's an iterator of correct length, try to convert (rare)
        try:
            out_list = list(out)
            if len(out_list) == len(batch):
                return out_list
        except Exception:
            pass
        # fallback to per-item mapping
    except TypeError:
        # common case: func doesn't accept a sequence
        pass
    except Exception:
        # some other runtime error — try fallback mapping, but re-raise if mapping also fails
        try:
            return [func(x) for x in batch]
        except Exception:
            raise

    # fallback mapping
    return [func(x) for x in batch]


# --------------------
# Top-level worker callables (must be top-level for picklability)
# --------------------
def _call_single(func: Callable[[Any], Any], param: Any) -> Any:
    return func(param)


def _call_batch(func: Callable[[Sequence[Any]], list[Any]] | Callable[[Any], Any], batch: Sequence[Any]) -> list[Any]:
    """Call a batch function with a batch of parameters."""
    return _apply_func_to_batch_with_fallback(func, batch)


# cloudpickle-assisted callables for ProcessPoolExecutor
def _call_single_pickled(pickled_func: bytes, param: Any) -> Any:
    func = cloudpickle.loads(pickled_func)
    return func(param)


def _call_batch_pickled(pickled_func: bytes, batch: Sequence[Any]) -> list[Any]:
    """Call a batch function with a batch of parameters, using cloudpickle to unpickle the function."""
    func = cloudpickle.loads(pickled_func)
    return _apply_func_to_batch_with_fallback(func, batch)


# --------------------
# Main API
# --------------------
def parallel_execution(
    func: Callable[[Any], Any] | Callable[[Sequence[Any]], list[Any]],
    params: Sequence[Any] | Iterable[Any],
    n_jobs: int,
    description: str,
    batch_size: Optional[int] = None,
    task_profile: Optional[TaskProfile] = None,  # CPU_BOUND or IO_BOUND -> implementation chooses backend
    show_combined_progress: bool = False,
    unordered: bool = False,  # False => results may be returned in any order; True => preserve original order
) -> Generator[Any, None, None]:
    if n_jobs <= 0:
        raise ValueError("n_jobs must be >= 1")

    if n_jobs > 16 and not show_combined_progress:
        logger.warning("n_jobs > 16: forcing show_combined_progress=True")
        show_combined_progress = True

    is_seq = _is_sequence(params)
    if not is_seq and not show_combined_progress:
        logger.warning("params is not a Sequence and per-worker progress requested -> forcing combined progress")
        show_combined_progress = True

    # choose executor type
    if task_profile is None:
        executor_cls = ProcessPoolExecutor
    else:
        executor_cls = ThreadPoolExecutor if task_profile == TaskProfile.IO_BOUND else ProcessPoolExecutor

    # prepare tasks or streaming mode
    tasks: list[tuple[int, Any]] = []
    stream_batches = False
    if batch_size is None:
        if is_seq:
            tasks = [(i, params[i]) for i in range(len(params))]
        else:
            stream_batches = False
    else:
        if is_seq:
            batch_list = _chunked_from_sequence(params, batch_size)
            tasks = [(start_idx, batch) for start_idx, batch in batch_list]
        else:
            stream_batches = True

    # compute total units for progress (None if unknown)
    if batch_size is None:
        total_units = len(params) if is_seq else None
    else:
        if is_seq:
            total_units = len(tasks)
        else:
            total_units = None

    progress = Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress_task: Optional[TaskID] = None
    per_worker_tasks: Optional[list[TaskID]] = None

    def _generator() -> Generator[Any, None, None]:
        nonlocal progress_task, per_worker_tasks

        use_per_worker = (not show_combined_progress) and is_seq and (executor_cls is ThreadPoolExecutor)

        # Per-worker mode (only for threads)
        if use_per_worker:
            seq_len = len(params)
            base, rem = divmod(seq_len, n_jobs)
            chunks: list[tuple[int, list[Any]]] = []
            idx = 0
            for i in range(n_jobs):
                size = base + (1 if i < rem else 0)
                if size == 0:
                    chunks.append((idx, []))
                    continue
                chunk = [params[j] for j in range(idx, idx + size)]
                chunks.append((idx, chunk))
                idx += size

            with progress:
                per_worker_tasks = []
                for i, (start_idx, chunk) in enumerate(chunks):
                    task_id = progress.add_task(f"{description} (worker {i + 1}/{len(chunks)})", total=len(chunk))
                    per_worker_tasks.append(task_id)
                with executor_cls(max_workers=n_jobs) as executor:
                    futures = {}
                    for i, (start_idx, chunk) in enumerate(chunks):

                        def _worker_process(chunk_local, tid):
                            res = []
                            if batch_size is None:
                                # Single item processing
                                for item in chunk_local:
                                    r = func(item)
                                    res.append(r)
                                    progress.update(tid, advance=1)
                            else:
                                # Batch processing
                                # Process items in batches of batch_size
                                for i in range(0, len(chunk_local), batch_size):
                                    batch = chunk_local[i:i+batch_size]
                                    batch_res = _apply_func_to_batch_with_fallback(func, batch)
                                    res.extend(batch_res)
                                    progress.update(tid, advance=1)
                            return res

                        futures[executor.submit(_worker_process, chunk, per_worker_tasks[i])] = (start_idx, len(chunk))

                    if not unordered:
                        for fut in as_completed(futures):
                            start_idx, _ = futures[fut]
                            batch_res = fut.result()
                            for r in batch_res:
                                yield r
                    else:
                        completed = {}
                        for fut in as_completed(futures):
                            start_idx, _ = futures[fut]
                            completed[start_idx] = fut.result()
                        for start in sorted(completed.keys()):
                            for r in completed[start]:
                                yield r
            return

        # Combined progress or process-based execution
        with progress:
            label = description
            if batch_size is not None:
                label = f"{label} (Batches)"
            total = total_units if total_units is not None else None
            progress_task = progress.add_task(label, total=total)

            # If using processes, pre-dump func with cloudpickle once
            is_process_backend = executor_cls is ProcessPoolExecutor
            pickled_func = cloudpickle.dumps(func) if is_process_backend else None

            with executor_cls(max_workers=n_jobs) as executor:
                futures_map: dict[Future, tuple[int, int]] = {}

                # Prebuilt tasks (Sequence)
                if tasks:
                    for start_idx, payload in tasks:
                        if batch_size is None:
                            if is_process_backend:
                                fut = executor.submit(_call_single_pickled, pickled_func, payload)
                            else:
                                fut = executor.submit(_call_single, func, payload)
                            futures_map[fut] = (start_idx, 1)
                        else:
                            if is_process_backend:
                                fut = executor.submit(_call_batch_pickled, pickled_func, payload)
                            else:
                                fut = executor.submit(_call_batch, func, payload)
                            futures_map[fut] = (start_idx, len(payload))
                else:
                    # streaming cases
                    if batch_size is None:
                        idx = 0
                        max_in_flight = max(16, n_jobs * 2)
                        in_flight = set()
                        it = iter(params)
                        buffer_order = {}
                        next_yield_idx = 0
                        try:
                            while True:
                                while len(in_flight) < max_in_flight:
                                    item = next(it)
                                    if is_process_backend:
                                        fut = executor.submit(_call_single_pickled, pickled_func, item)
                                    else:
                                        fut = executor.submit(_call_single, func, item)
                                    futures_map[fut] = (idx, 1)
                                    in_flight.add(fut)
                                    idx += 1
                                done_iter = as_completed(in_flight)
                                done_fut = next(done_iter)
                                in_flight.remove(done_fut)
                                start_idx, cnt = futures_map.pop(done_fut)
                                res = done_fut.result()
                                progress.update(progress_task, advance=1)
                                if not unordered:
                                    yield res
                                else:
                                    buffer_order[start_idx] = res
                                    while next_yield_idx in buffer_order:
                                        yield buffer_order.pop(next_yield_idx)
                                        next_yield_idx += 1
                        except StopIteration:
                            for fut in as_completed(list(in_flight)):
                                start_idx, cnt = futures_map.pop(fut)
                                res = fut.result()
                                progress.update(progress_task, advance=1)
                                if not unordered:
                                    yield res
                                else:
                                    buffer_order[start_idx] = res
                            for i in sorted(buffer_order.keys()):
                                yield buffer_order[i]
                            return
                    else:
                        # streaming batches
                        idx_gen = _chunked_from_iterable(params, batch_size)
                        max_in_flight = max(8, n_jobs * 2)
                        in_flight = set()
                        pending_iter = iter(idx_gen)
                        buffer_order = {}
                        next_yield_idx = 0
                        try:
                            while True:
                                while len(in_flight) < max_in_flight:
                                    start_idx, batch = next(pending_iter)
                                    if is_process_backend:
                                        fut = executor.submit(_call_batch_pickled, pickled_func, batch)
                                    else:
                                        fut = executor.submit(_call_batch, func, batch)
                                    futures_map[fut] = (start_idx, len(batch))
                                    in_flight.add(fut)
                                done = next(as_completed(in_flight))
                                in_flight.remove(done)
                                start_idx, blen = futures_map.pop(done)
                                batch_res = done.result()
                                progress.update(progress_task, advance=1)
                                if not unordered:
                                    for r in batch_res:
                                        yield r
                                else:
                                    buffer_order[start_idx] = batch_res
                                    while next_yield_idx in buffer_order:
                                        for r in buffer_order.pop(next_yield_idx):
                                            yield r
                                        # increment next_yield_idx by count of the yielded batch
                                        # careful: previous batch size unknown here; we'll set next_yield_idx to next available key
                                        # for simplicity we rely on sorted buffer keys at drain time
                        except StopIteration:
                            for fut in as_completed(list(in_flight)):
                                start_idx, blen = futures_map.pop(fut)
                                batch_res = fut.result()
                                progress.update(progress_task, advance=1)
                                if not unordered:
                                    for r in batch_res:
                                        yield r
                                else:
                                    buffer_order[start_idx] = batch_res
                            for s in sorted(buffer_order.keys()):
                                for r in buffer_order[s]:
                                    yield r
                            return

                # If prebuilt tasks were submitted (Sequence cases), collect results
                if tasks:
                    if not unordered:
                        for fut in as_completed(list(futures_map.keys())):
                            start_idx, cnt = futures_map.pop(fut)
                            result = fut.result()
                            progress.update(progress_task, advance=1)
                            if batch_size is None:
                                yield result
                            else:
                                for r in result:
                                    yield r
                    else:
                        completed = {}
                        for fut in as_completed(list(futures_map.keys())):
                            start_idx, cnt = futures_map.pop(fut)
                            completed[start_idx] = fut.result()
                            progress.update(progress_task, advance=1)
                        for start in sorted(completed.keys()):
                            res = completed[start]
                            if batch_size is None:
                                yield res
                            else:
                                for r in res:
                                    yield r

    return _generator()


import time


def task_batch(params: list):
    for i, x in enumerate(params):
        time.sleep(0.01)
    return params


def task(params: int):
    time.sleep(0.01)
    return params


def main():
    iterator = parallel_execution(
        func=task, params=range(1000), n_jobs=10, description="Test", show_combined_progress=False
    )
    print("Started")
    l = list(iterator)
    print(f"Done {len(l)} items")


def main_batch():
    iterator = parallel_execution(
        func=task_batch, params=range(1000), n_jobs=10, batch_size=10, description="Test", show_combined_progress=False
    )
    print("Started")
    l = list(iterator)
    print(f"Done {len(l)} items")


if __name__ == "__main__":
    main()
    main_batch()
