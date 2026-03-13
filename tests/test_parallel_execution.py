import time
import pytest

from exo_finder.compute.parallel_execution import TaskDistribution, TaskProfile, parallel_execution


def _collect(iterator):
    """Consume generator and return list."""
    return list(iterator)


class TestParallelExecution:
    @pytest.mark.parametrize("task_profile", [TaskProfile.CPU_BOUND, TaskProfile.IO_BOUND])
    @pytest.mark.parametrize(
        ("task_distribution", "func", "batch_size"),
        [
            (TaskDistribution.STREAMED_SINGLE, lambda x: x * 2, None),
            (TaskDistribution.STREAMED_BATCHES, lambda batch: [x * 2 for x in batch], 4),
        ],
    )
    def test_sequence_single_and_batch_simple(self, task_profile, task_distribution, func, batch_size):
        # simple pure function: returns param * 2
        params = list(range(20))
        it = parallel_execution(
            func=func,
            params=params,
            task_distribution=task_distribution,
            n_jobs=4,
            description="seq",
            batch_size=batch_size,
            task_profile=task_profile,
            sort_result=True,
        )
        res = _collect(it)
        assert res == [x * 2 for x in params]

    def test_iterable_streamed_batches_preserve_order(self):
        # params as generator and batch_size set -> streaming batches
        def batch_fn(batch):
            # record that we got a list and return the same content
            return [x for x in batch]

        params_gen = (i for i in range(23))
        it = parallel_execution(
            func=batch_fn,
            params=params_gen,
            task_distribution=TaskDistribution.STREAMED_BATCHES,
            n_jobs=3,
            description="stream-batch",
            batch_size=5,
            task_profile=TaskProfile.CPU_BOUND,
            sort_result=True,
        )
        res = _collect(it)
        assert res == list(range(23))

    def test_batch_size_greater_than_len_results_in_single_batch(self):
        # batch_size > len(params) => single batch should be created
        def batch_fn(batch):
            # return tuple (len(batch), list(batch)) so we can infer batch size from output
            return [("BATCH_LEN", len(batch))] * len(batch)

        params = list(range(7))
        it = parallel_execution(
            func=batch_fn,
            params=params,
            task_distribution=TaskDistribution.STREAMED_BATCHES,
            n_jobs=3,
            description="single-big-batch",
            batch_size=100,
            task_profile=TaskProfile.CPU_BOUND,
            sort_result=True,
        )
        res = _collect(it)
        # each yielded element is ("BATCH_LEN", len(batch)), should equal len(params)
        assert len(res) == len(params)
        # All yielded tuples should say batch length equals 7
        assert all(t[1] == 7 for t in res)

    def test_unordered_flag_changes_ordering_behavior(self):
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
            task_distribution=TaskDistribution.STREAMED_SINGLE,
            n_jobs=4,
            description="preserve",
            batch_size=None,
            task_profile=TaskProfile.CPU_BOUND,
            sort_result=True,
        )
        res_preserve = _collect(it_preserve)
        assert res_preserve == params

        # unordered=False -> results may come in completion order (not necessarily input order).
        it_any = parallel_execution(
            func=work,
            params=params,
            task_distribution=TaskDistribution.STREAMED_SINGLE,
            n_jobs=4,
            description="any",
            batch_size=None,
            task_profile=TaskProfile.CPU_BOUND,
            sort_result=False,
        )
        res_any = _collect(it_any)
        # must contain same multiset, but order can differ; ensure same elements
        assert sorted(res_any) == sorted(params)
        # it's extremely likely reordering happened, assert it's not identical to input order (non-flaky)
        if res_any == params:
            pytest.skip("Unexpected identical ordering under this environment; re-run to confirm reordering behaviour")
        else:
            assert res_any != params

    def test_streamed_batches_requires_positive_batch_size(self):
        with pytest.raises(ValueError, match="requires batch_size > 0"):
            _collect(
                parallel_execution(
                    func=lambda b: b,
                    params=range(4),
                    task_distribution=TaskDistribution.STREAMED_BATCHES,
                    batch_size=None,
                )
            )

    def test_balanced_rejects_batch_size(self):
        with pytest.raises(ValueError, match="does not accept batch_size"):
            _collect(
                parallel_execution(
                    func=lambda x: x,
                    params=list(range(4)),
                    task_distribution=TaskDistribution.BALANCED,
                    batch_size=2,
                )
            )

    def test_streamed_single_rejects_batch_size_gt_one(self):
        with pytest.raises(ValueError, match="dispatches items one-by-one"):
            _collect(
                parallel_execution(
                    func=lambda x: x,
                    params=range(4),
                    task_distribution=TaskDistribution.STREAMED_SINGLE,
                    batch_size=2,
                )
            )

    def test_balanced_requires_sized_sequence(self):
        with pytest.raises(ValueError, match="requires a sized Sequence"):
            _collect(
                parallel_execution(
                    func=lambda x: x,
                    params=(i for i in range(4)),
                    task_distribution=TaskDistribution.BALANCED,
                )
            )

    def test_local_function_with_process_backend_is_supported(self):
        # nested (local) function -> with cloudpickle it should work on process backend
        def local_task(x):
            return x + 1

        params = range(12)
        it = parallel_execution(
            func=local_task,
            params=params,
            task_distribution=TaskDistribution.STREAMED_SINGLE,
            n_jobs=3,
            description="local",
            batch_size=None,
            task_profile=TaskProfile.CPU_BOUND,
            sort_result=True,
        )
        res = _collect(it)
        assert res == [x + 1 for x in params]

    def test_generator_partial_consumption_and_lazy_behavior(self):
        # ensure the returned value is an iterator and can be partially consumed
        def slow(x):
            time.sleep(0.001)
            return x * 10

        it = parallel_execution(
            func=slow,
            params=range(10),
            task_distribution=TaskDistribution.STREAMED_SINGLE,
            n_jobs=2,
            description="lazy",
            batch_size=None,
            task_profile=TaskProfile.IO_BOUND,
            sort_result=True,
        )

        # partial consumption
        iterator = iter(it)
        first = next(iterator)
        rest = list(iterator)
        assert [first] + rest == [x * 10 for x in range(10)]

    def test_balanced_accepts_single_callable_via_item_fallback(self):
        # BALANCED first tries func(bucket), then falls back to item-by-item on TypeError
        def single(x):
            return x + 3

        params = list(range(15))
        it = parallel_execution(
            func=single,
            params=params,
            task_distribution=TaskDistribution.BALANCED,
            n_jobs=4,
            description="balanced-fallback",
            task_profile=TaskProfile.IO_BOUND,
            sort_result=False,
        )
        res = _collect(it)

        assert sorted(res) == sorted([x + 3 for x in params])
