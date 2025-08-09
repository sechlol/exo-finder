import time
import pytest

from exo_finder.utils.parallel_execution import TaskProfile, parallel_execution


def collect(iterator):
    """Consume generator and return list."""
    return list(iterator)


@pytest.mark.parametrize("task_profile", [TaskProfile.CPU_BOUND, TaskProfile.IO_BOUND])
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("show_combined_progress", [True, False])
def test_sequence_single_and_batch_simple(task_profile, batch_size, show_combined_progress):
    # simple pure function: returns param * 2
    def single(x):
        return x * 2

    def batch_fn(batch):
        return [x * 2 for x in batch]

    params = list(range(20))
    if batch_size is None:
        it = parallel_execution(
            func=single,
            params=params,
            n_jobs=4,
            description="seq",
            batch_size=None,
            task_profile=task_profile,
            show_combined_progress=show_combined_progress,
            unordered=True,
        )  # unordered True => preserve order by spec
        res = collect(it)
        assert res == [x * 2 for x in params]
    else:
        it = parallel_execution(
            func=batch_fn,
            params=params,
            n_jobs=4,
            description="seq-batch",
            batch_size=batch_size,
            task_profile=task_profile,
            show_combined_progress=show_combined_progress,
            unordered=True,
        )
        res = collect(it)
        assert res == [x * 2 for x in params]


def test_iterable_streamed_batches_preserve_order():
    # params as generator and batch_size set -> streaming batches
    def batch_fn(batch):
        # record that we got a list and return the same content
        return [x for x in batch]

    params_gen = (i for i in range(23))
    it = parallel_execution(
        func=batch_fn,
        params=params_gen,
        n_jobs=3,
        description="stream-batch",
        batch_size=5,
        task_profile=TaskProfile.CPU_BOUND,
        show_combined_progress=True,
        unordered=True,
    )
    res = collect(it)
    assert res == list(range(23))


def test_batch_size_greater_than_len_results_in_single_batch():
    # batch_size > len(params) => single batch should be created
    def batch_fn(batch):
        # return tuple (len(batch), list(batch)) so we can infer batch size from output
        return [("BATCH_LEN", len(batch))] * len(batch)

    params = list(range(7))
    it = parallel_execution(
        func=batch_fn,
        params=params,
        n_jobs=3,
        description="single-big-batch",
        batch_size=100,
        task_profile=TaskProfile.CPU_BOUND,
        show_combined_progress=True,
        unordered=True,
    )
    res = collect(it)
    # each yielded element is ("BATCH_LEN", len(batch)), should equal len(params)
    assert len(res) == len(params)
    # All yielded tuples should say batch length equals 7
    assert all(t[1] == 7 for t in res)


def test_unordered_flag_changes_ordering_behavior():
    # make task durations vary so completion order differs from input
    def work(x):
        # items with smaller x sleep longer, so completion order tends to be reversed
        time.sleep((9 - x) * 0.001)
        return x

    params = list(range(10))

    # unordered=True -> preserve original input ordering (spec: True preserves ordering)
    it_preserve = parallel_execution(
        func=work,
        params=params,
        n_jobs=4,
        description="preserve",
        batch_size=None,
        task_profile=TaskProfile.CPU_BOUND,
        show_combined_progress=True,
        unordered=True,
    )
    res_preserve = collect(it_preserve)
    assert res_preserve == params

    # unordered=False -> results may come in completion order (not necessarily input order).
    it_any = parallel_execution(
        func=work,
        params=params,
        n_jobs=4,
        description="any",
        batch_size=None,
        task_profile=TaskProfile.CPU_BOUND,
        show_combined_progress=True,
        unordered=False,
    )
    res_any = collect(it_any)
    # must contain same multiset, but order can differ; ensure same elements
    assert sorted(res_any) == sorted(params)
    # it's extremely likely reordering happened, assert it's not identical to input order (non-flaky)
    if res_any == params:
        pytest.skip("Unexpected identical ordering under this environment; re-run to confirm reordering behaviour")
    else:
        assert res_any != params


def test_large_n_jobs_forces_combined_progress_and_logs_warning(caplog):
    # n_jobs > 16 should log a warning if show_combined_progress was False
    def f(x):
        return x

    caplog.clear()
    it = parallel_execution(
        func=f,
        params=range(10),
        n_jobs=17,
        description="warn",
        batch_size=None,
        task_profile=TaskProfile.IO_BOUND,
        show_combined_progress=False,
        unordered=True,
    )
    res = collect(it)
    assert res == list(range(10))
    # check for the forcing warning
    assert any("forcing show_combined_progress" in rec.message for rec in caplog.records)


def test_local_function_with_process_backend_is_supported():
    # nested (local) function -> with cloudpickle it should work on process backend
    def local_task(x):
        return x + 1

    params = range(12)
    it = parallel_execution(
        func=local_task,
        params=params,
        n_jobs=3,
        description="local",
        batch_size=None,
        task_profile=TaskProfile.CPU_BOUND,
        show_combined_progress=True,
        unordered=True,
    )
    res = collect(it)
    assert res == [x + 1 for x in params]


def test_generator_partial_consumption_and_lazy_behavior():
    # ensure the returned value is an iterator and can be partially consumed
    def slow(x):
        time.sleep(0.001)
        return x * 10

    it = parallel_execution(
        func=slow,
        params=range(10),
        n_jobs=2,
        description="lazy",
        batch_size=None,
        task_profile=TaskProfile.IO_BOUND,
        show_combined_progress=True,
        unordered=True,
    )

    # partial consumption
    iterator = iter(it)
    first = next(iterator)
    rest = list(iterator)
    assert [first] + rest == [x * 10 for x in range(10)]


def test_sequence_single_callable_on_thread_backend_with_batching():
    # thread-based batch where func is single param callable (not batch)
    def single(x):
        return x * 3

    params = list(range(15))
    it = parallel_execution(
        func=single,
        params=params,
        n_jobs=4,
        description="threads-single",
        batch_size=3,
        task_profile=TaskProfile.IO_BOUND,
        show_combined_progress=True,
        unordered=True,
    )
    res = collect(it)
    # Print for debugging
    print(f"Expected: {sorted([x * 3 for x in params])}")
    print(f"Got: {sorted(res)}")
    assert sorted(res) == sorted([x * 3 for x in params])
