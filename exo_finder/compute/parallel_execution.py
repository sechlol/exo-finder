import os
from enum import Enum
from multiprocessing import Manager as MPManager

# queue/manager selection for cross-backend progress reporting
from queue import Queue as ThreadQueue
from threading import Thread
from typing import Any, Callable, Generator, Iterable, Optional, Sequence, TypedDict, Literal

from joblib import Parallel, delayed
from tqdm.auto import tqdm


class TaskDistribution(Enum):
    BALANCED = "balanced"
    STREAMED_BATCHES = "streamed_batches"
    STREAMED_SINGLE = "streamed_single"


class TaskProfile(Enum):
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"


class ShareMode(Enum):
    ARG = "arg"
    GLOBAL = "global"
    MANAGER = "manager"


_CONTEXT_INIT = False  # per-process flag
_WORKER_CONTEXT: Any = None  # per-process context object


class _ContextSpec(TypedDict, total=False):
    mode: Literal["arg", "global", "manager", ""]
    extra: Any
    init_process: Optional[Callable[[Any], Any]]


def get_worker_context():
    """Access the per-process context (initialized via init_process)."""
    return _WORKER_CONTEXT


def _ensure_global_context(init_process: Optional[Callable[[Any], Any]], extra: Any):
    """
    Run once per worker process to initialize and cache a context object.
    Stored inside THIS module to avoid module-alias issues under spawn.
    """
    global _CONTEXT_INIT, _WORKER_CONTEXT
    if not _CONTEXT_INIT:
        _WORKER_CONTEXT = init_process(extra) if init_process else extra
        _CONTEXT_INIT = True
    return _WORKER_CONTEXT


def _call_with_context(func: Callable, payload: Any, ctx: Optional[_ContextSpec]):
    """
    Invoke `func` with the requested sharing semantics.
    - For BALANCED: `payload` is a bucket (Sequence) or item when falling back.
    - For STREAMED_*: `payload` is a chunk or single item.
    """
    # No context requested
    if not ctx or not ctx.get("mode"):
        return func(payload)

    mode = ctx["mode"]

    if mode == "arg":
        return func(payload, extra_arguments=ctx.get("extra"))
    elif mode == "global":
        _ensure_global_context(ctx.get("init_process"), ctx.get("extra"))
        return func(payload)  # user code reads global(s) inside the worker
    elif mode == "manager":
        # manager proxies are picklable, so pass as kwarg
        return func(payload, extra_arguments=ctx.get("extra"))
    else:
        # fallback to arg mode
        return func(payload, extra_arguments=ctx.get("extra"))


def _worker_call_on_bucket(pid: int, bucket: Sequence[Any], func: Callable, queue, ctx: Optional[_ContextSpec]):
    """
    Called for BALANCED profile: try func(bucket) (batch-aware), else fallback
    to per-item calls. Always emit per-item progress updates via `queue`.
    Returns list of results (one element per input item).
    """
    try:
        res = _call_with_context(func, bucket, ctx)
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
            results.append(_call_with_context(func, x, ctx))
            queue.put(("update", pid, 1))
        return results


def _worker_call_on_streamed_chunk(chunk, func, ctx: Optional[_ContextSpec]):
    """Called for STREAMED_BATCHES: chunk is a list of items."""
    return _call_with_context(func, chunk, ctx)


def _worker_call_single(item, func, ctx: Optional[_ContextSpec]):
    """Called for STREAMED_SINGLE: item is a single param."""
    return _call_with_context(func, item, ctx)


def _is_sized(obj: Iterable[Any]) -> bool:
    try:
        len(obj)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def _chunked_iterable(it: Iterable[Any], size: int):
    """Yield lists of at most `size` items from iterator `it`."""
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _split_evenly(seq: Sequence[Any], n_parts: int):
    """Split sequence into n_parts buckets as evenly as possible."""
    total = len(seq)
    n_parts = max(1, n_parts)
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
    """Update tqdm bars reading messages from `queue`."""
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
        # expected ("update", pid, n)
        if isinstance(msg, tuple) and msg and msg[0] == "update":
            pid = msg[1]
            n = msg[2] if len(msg) > 2 else 1
            if splitted:
                pbars[pid].update(n)
            else:
                pbars[0].update(n)
    for p in pbars:
        p.close()


def parallel_execution(
    func: Callable[[Any], Any] | Callable[[Sequence[Any]], list[Any]],
    params: Sequence[Any] | Iterable[Any],
    task_distribution: TaskDistribution,
    n_jobs: int = os.cpu_count(),
    description: str = "",
    batch_size: Optional[int] = None,
    sort_result: bool = True,
    task_profile: TaskProfile = TaskProfile.CPU_BOUND,
    *,
    extra_arguments: Any = None,
    share_mode: Optional[ShareMode] = None,
    init_process: Optional[Callable[[Any], Any]] = None,
) -> Generator[Any, None, None]:
    """
    Parallel execution with configurable backend (threads/processes), streaming modes,
    and progress reporting.

    - backend: if 'threading' is selected, joblib will use threads (good for I/O-bound tasks like downloads).
               other valid values: 'loky', 'multiprocessing', or None (joblib default).
    """

    backend = "loky" if task_profile == TaskProfile.CPU_BOUND else "threading"
    is_sized = _is_sized(params)
    params_list = list(params) if is_sized else None
    total_items: Optional[int] = len(params_list) if is_sized else None

    # Validate task_profile / batch_size semantics
    if task_distribution == TaskDistribution.STREAMED_BATCHES:
        if not (batch_size and batch_size > 0):
            raise ValueError("TaskDistribution.STREAMED_BATCHES requires batch_size > 0 (streamed chunks).")
    if task_distribution == TaskDistribution.BALANCED and batch_size is not None:
        raise ValueError(
            "TaskDistribution.BALANCED does not accept batch_size. "
            "BALANCED will split `params` evenly between workers and call `func` on each bucket."
        )
    if task_distribution == TaskDistribution.STREAMED_SINGLE and (batch_size is not None and batch_size > 1):
        raise ValueError("TaskDistribution.STREAMED_SINGLE dispatches items one-by-one. Set batch_size to None or 1.")

    # Choose queue implementation
    manager: Optional[MPManager] = None
    queue = None
    use_thread_queue = backend == "threading"
    try:
        if use_thread_queue:
            queue = ThreadQueue()
        else:
            manager = MPManager()
            queue = manager.Queue()

        # Manager-backed shared state, if requested
        mode_str = share_mode.value if share_mode else ""
        ctx_extra = extra_arguments

        if mode_str == "manager":
            if use_thread_queue:
                pass  # threads share memory already
            else:
                if isinstance(extra_arguments, dict):
                    md = manager.dict()
                    md.update(extra_arguments)
                    ctx_extra = md
                elif isinstance(extra_arguments, list):
                    ml = manager.list(extra_arguments)
                    ctx_extra = ml
                else:
                    ctx_extra = extra_arguments

        ctx: _ContextSpec = {"mode": mode_str, "extra": ctx_extra, "init_process": init_process}

        # Build joblib tasks based on profile
        if task_distribution == TaskDistribution.BALANCED:
            if not is_sized:
                raise ValueError("TaskDistribution.BALANCED requires a sized Sequence for `params`.")
            buckets = _split_evenly(params_list, n_jobs)
            totals = [len(b) for b in buckets]
            tasks = (
                delayed(_worker_call_on_bucket)(pid, bucket, func, queue, ctx) for pid, bucket in enumerate(buckets)
            )
        elif task_distribution == TaskDistribution.STREAMED_BATCHES:
            if is_sized:
                chunks = (params_list[i : i + batch_size] for i in range(0, len(params_list), batch_size))
                # Progress counts batches, not items
                totals = (len(params_list) + batch_size - 1) // batch_size
            else:
                chunks = _chunked_iterable(params, batch_size)
                totals = None
            tasks = (delayed(_worker_call_on_streamed_chunk)(chunk, func, ctx) for chunk in chunks)
        else:  # STREAMED_SINGLE
            if is_sized:
                items = (params_list[i] for i in range(len(params_list)))
                totals = total_items
            else:
                items = (x for x in params)
                totals = None
            tasks = (delayed(_worker_call_single)(item, func, ctx) for item in items)

        # Progress UI
        progress_thread = None
        pbar = None
        if task_distribution == TaskDistribution.BALANCED:
            progress_thread = Thread(target=_progress_thread, args=(totals, queue), daemon=True)
            progress_thread.start()
        else:
            pbar = tqdm(
                total=totals,
                desc=description,
                unit="batch" if task_distribution == TaskDistribution.STREAMED_BATCHES else "it",
            )

        # Construct Parallel
        return_mode = "generator" if sort_result else "generator_unordered"
        fallback_to_list = False
        try:
            if backend is None:
                parallel = Parallel(n_jobs=n_jobs, return_as=return_mode)
            else:
                parallel = Parallel(n_jobs=n_jobs, backend=backend, return_as=return_mode)
        except ValueError:
            if backend is None:
                parallel = Parallel(n_jobs=n_jobs)
            else:
                parallel = Parallel(n_jobs=n_jobs, backend=backend)
            fallback_to_list = True

        results_iterable = parallel(tasks)
        iterable = iter(results_iterable) if fallback_to_list else results_iterable

        # Yield results; update pbar
        for task_result in iterable:
            if isinstance(task_result, list):
                for r in task_result:
                    yield r
            else:
                yield task_result

            if pbar is not None:
                if task_distribution == TaskDistribution.STREAMED_BATCHES:
                    pbar.update(1)  # one completed batch
                else:
                    if isinstance(task_result, list):
                        pbar.update(len(task_result))
                    else:
                        pbar.update(1)

    finally:
        # Clean up progress UI and manager
        try:
            if "progress_thread" in locals() and progress_thread is not None:
                queue.put("done")
                progress_thread.join()
            elif "pbar" in locals() and pbar is not None:
                pbar.close()
        except Exception:
            pass
        try:
            if manager is not None:
                manager.shutdown()
        except Exception:
            pass
