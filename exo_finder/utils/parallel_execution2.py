# parallel_execution_v3.py
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Callable, Generator, Iterable, Optional, Sequence

from joblib import Parallel, delayed
from multiprocessing import Manager
from threading import Thread
from tqdm.auto import tqdm


class TaskProfile2(Enum):
    BALANCED = "balanced"
    STREAMED_BATCHES = "streamed_batches"
    STREAMED_SINGLE = "streamed_single"


def _is_sized(obj: Iterable[Any]) -> bool:
    try:
        len(obj)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def _chunked_iterable(it: Iterable[Any], size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _split_evenly(seq: Sequence[Any], n_parts: int):
    total = len(seq)
    if n_parts <= 0:
        n_parts = 1
    base = total // n_parts
    rem = total % n_parts
    out = []
    idx = 0
    for i in range(n_parts):
        take = base + (1 if i < rem else 0)
        out.append(seq[idx : idx + take])
        idx += take
    return out


def _progress_thread(totals, queue):
    if isinstance(totals, list):
        pbars = [tqdm(total=total, desc=f"Worker {i + 1}", position=i, leave=True) for i, total in enumerate(totals)]
        splitted = True
    else:
        pbars = [tqdm(total=totals)]
        splitted = False

    while True:
        msg = queue.get()
        if msg is None or msg == "done":
            break
        if isinstance(msg, tuple) and msg[0] == "update":
            pid = msg[1]
            n = msg[2] if len(msg) > 2 else 1
            if splitted:
                pbars[pid].update(n)
            else:
                pbars[0].update(n)
    for p in pbars:
        p.close()


def _worker_call_on_bucket(pid: int, bucket: Sequence[Any], func: Callable, queue):
    """
    Try to call func(bucket). If that fails (TypeError), fall back to calling
    func(item) for each item in bucket. Always report per-item updates to `queue`.
    Return a list of results (one element per input item).
    """
    try:
        res = func(bucket)
        if isinstance(res, list):
            for _ in res:
                queue.put(("update", pid, 1))
            return res
        else:
            queue.put(("update", pid, len(bucket)))
            return [res]
    except TypeError:
        results = []
        for x in bucket:
            results.append(func(x))
            queue.put(("update", pid, 1))
        return results


def _worker_call_on_streamed_chunk(chunk, func):
    return func(chunk)


def _worker_call_single(item, func):
    return func(item)


def parallel_execution2(
    func: Callable[[Any], Any] | Callable[[Sequence[Any]], list[Any]],
    params: Sequence[Any] | Iterable[Any],
    n_jobs: int = os.cpu_count(),
    description: str = "parallel",
    batch_size: Optional[int] = None,
    task_profile: Optional[TaskProfile2] = None,
    sort_result: bool = True,
) -> Generator[Any, None, None]:
    if task_profile is None:
        raise ValueError("task_profile must be specified and be one of TaskProfile2")

    is_sized = _is_sized(params)
    params_list = list(params) if is_sized else None
    total_items: Optional[int] = len(params_list) if is_sized else None

    if task_profile == TaskProfile2.STREAMED_BATCHES:
        if not (batch_size and batch_size > 0):
            raise ValueError("TaskProfile2.STREAMED_BATCHES requires batch_size > 0 (streamed chunks).")

    if task_profile == TaskProfile2.BALANCED and batch_size is not None:
        raise ValueError(
            "TaskProfile2.BALANCED does not accept batch_size. "
            "BALANCED will split `params` evenly between workers and call `func` on each bucket."
        )

    if task_profile == TaskProfile2.STREAMED_SINGLE and (batch_size is not None and batch_size > 1):
        raise ValueError("TaskProfile2.STREAMED_SINGLE dispatches items one-by-one. Set batch_size to None or 1.")

    manager = Manager()
    queue = manager.Queue()
    progress_thread = None
    need_progress_thread = False
    pbar = None

    if task_profile == TaskProfile2.BALANCED:
        if not is_sized:
            raise ValueError("TaskProfile2.BALANCED requires a sized Sequence for `params`.")
        buckets = _split_evenly(params_list, n_jobs)
        totals = [len(b) for b in buckets]
        need_progress_thread = True
        tasks = (delayed(_worker_call_on_bucket)(pid, bucket, func, queue) for pid, bucket in enumerate(buckets))

    elif task_profile == TaskProfile2.STREAMED_BATCHES:
        if is_sized:
            chunks = (params_list[i : i + batch_size] for i in range(0, len(params_list), batch_size))
            totals = total_items
        else:
            chunks = _chunked_iterable(params, batch_size)
            totals = None
        tasks = (delayed(_worker_call_on_streamed_chunk)(chunk, func) for chunk in chunks)

    else:  # STREAMED_SINGLE
        if is_sized:
            items = (params_list[i] for i in range(len(params_list)))
            totals = total_items
        else:
            items = (x for x in params)
            totals = None
        tasks = (delayed(_worker_call_single)(item, func) for item in items)

    try:
        if need_progress_thread:
            progress_thread = Thread(target=_progress_thread, args=(totals, queue), daemon=True)
            progress_thread.start()
        else:
            pbar = tqdm(total=totals, desc=description)

        return_mode = "generator" if sort_result else "generator_unordered"
        fallback_to_list = False
        try:
            parallel = Parallel(n_jobs=n_jobs, return_as=return_mode)
        except ValueError:
            parallel = Parallel(n_jobs=n_jobs)
            fallback_to_list = True

        results_iterable = parallel(tasks)
        iterable = iter(results_iterable) if fallback_to_list else results_iterable

        for task_result in iterable:
            if isinstance(task_result, list):
                for r in task_result:
                    yield r
                if pbar is not None:
                    pbar.update(len(task_result))
            else:
                yield task_result
                if pbar is not None:
                    pbar.update(1)

    finally:
        try:
            if need_progress_thread:
                queue.put("done")
                if progress_thread is not None:
                    progress_thread.join()
            else:
                if pbar is not None:
                    pbar.close()
        except Exception:
            pass
        try:
            manager.shutdown()
        except Exception:
            pass
