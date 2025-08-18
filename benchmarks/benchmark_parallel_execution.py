import time
from typing import NamedTuple
import argparse
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.table import Table
import sys
import os

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exo_finder.compute.parallel_execution import parallel_execution, TaskProfile


class TestNamedTuple(NamedTuple):
    gap_ratio: float
    mad: float
    normalized_std: float
    median: float
    min: float
    max: float
    length: int


def f_tuple(param: int) -> tuple:
    time.sleep(0.01)
    return (0, 0, 0, 0, 0, 0, 0)


def f_named_tuple(param: int) -> TestNamedTuple:
    time.sleep(0.01)
    return TestNamedTuple(0, 0, 0, 0, 0, 0, 0)


def f_batched(params: list[int]) -> list[tuple]:
    time.sleep(0.01)
    return [(0, 0, 0, 0, 0, 0, 0) for _ in range(len(params))]


def f_batched_named_tuple(params: list[int]) -> list[TestNamedTuple]:
    time.sleep(0.01)
    return [TestNamedTuple(0, 0, 0, 0, 0, 0, 0) for _ in range(len(params))]


# More realistic workload functions
def cpu_work(param: int, complexity: float = 1.0) -> float:
    """CPU-bound work simulation"""
    result = 0
    iterations = int(10000 * complexity)
    for i in range(iterations):
        result += np.sin(i * 0.01) * np.cos(i * 0.01)
    return result


def cpu_work_batch(params: list[int], complexity: float = 1.0) -> list[float]:
    """CPU-bound batch work simulation"""
    results = []
    iterations = int(10000 * complexity)
    for param in params:
        result = 0
        for i in range(iterations):
            result += np.sin(i * 0.01) * np.cos(i * 0.01)
        results.append(result)
    return results


def run_parallel_execution(func, params, n_jobs, batch_size=None, task_profile=None, show_combined_progress=False):
    """Run and time parallel_execution"""
    start = timer()
    results = list(
        parallel_execution(
            func=func,
            params=params,
            n_jobs=n_jobs,
            description=f"PE-{'batch' if batch_size else 'single'}",
            batch_size=batch_size,
            task_profile=task_profile,
            show_combined_progress=show_combined_progress,
            sort_result=False,
        )
    )
    end = timer()
    return results, end - start


def run_joblib(func, params, n_jobs, batch_size=None):
    """Run and time joblib.Parallel"""
    start = timer()

    if batch_size is None:
        # Single item processing
        results = Parallel(n_jobs=n_jobs)(delayed(func)(param) for param in params)
    else:
        # Batch processing - we need to chunk the data ourselves
        batches = [params[i : i + batch_size] for i in range(0, len(params), batch_size)]
        batch_results = Parallel(n_jobs=n_jobs)(delayed(func)(batch) for batch in batches)
        # Flatten results
        results = [item for sublist in batch_results for item in sublist]

    end = timer()
    return results, end - start


def benchmark_batch_vs_non_batch(n_items=1000, n_jobs=4, n_runs=3):
    """Compare batch vs non-batch processing for both parallel_execution and joblib"""
    results = []
    batch_sizes = [None, 10, 50, 100]

    params = list(range(n_items))

    for task_type in ["CPU", "IO"]:
        task_profile = TaskProfile.CPU_BOUND if task_type == "CPU" else TaskProfile.IO_BOUND

        # Select appropriate functions based on task type
        if task_type == "CPU":
            single_func = cpu_work
            batch_func = cpu_work_batch
        else:
            single_func = f_tuple
            batch_func = f_batched

        for batch_size in batch_sizes:
            # For non-batch processing, use the single-item function
            # For batch processing, use the batch function
            pe_func = single_func if batch_size is None else batch_func
            jl_func = single_func if batch_size is None else batch_func

            # Run parallel_execution
            pe_times = []
            for _ in range(n_runs):
                _, pe_time = run_parallel_execution(
                    pe_func,
                    params,
                    n_jobs,
                    batch_size=batch_size,
                    task_profile=task_profile,
                    show_combined_progress=True,
                )
                pe_times.append(pe_time)
            pe_avg_time = sum(pe_times) / len(pe_times)

            # Run joblib
            jl_times = []
            for _ in range(n_runs):
                _, jl_time = run_joblib(jl_func, params, n_jobs, batch_size=batch_size)
                jl_times.append(jl_time)
            jl_avg_time = sum(jl_times) / len(jl_times)

            # Record results
            results.append(
                {
                    "Task Type": task_type,
                    "Batch Size": batch_size if batch_size is not None else "None",
                    "parallel_execution (s)": pe_avg_time,
                    "joblib (s)": jl_avg_time,
                    "Speedup": jl_avg_time / pe_avg_time if pe_avg_time > 0 else 0,
                }
            )

    return pd.DataFrame(results)


def benchmark_progress_bars(n_items=1000, n_jobs=4, batch_size=50, n_runs=3):
    """Compare combined vs multiple progress bars for parallel_execution"""
    results = []

    params = list(range(n_items))

    for task_type in ["CPU", "IO"]:
        task_profile = TaskProfile.CPU_BOUND if task_type == "CPU" else TaskProfile.IO_BOUND

        # Select appropriate functions based on task type
        if task_type == "CPU":
            single_func = cpu_work
            batch_func = cpu_work_batch

        else:
            single_func = f_tuple
            batch_func = f_batched

        for show_combined in [True, False]:
            # Skip non-combined progress for CPU tasks as it forces combined
            if task_type == "CPU" and not show_combined:
                continue

            # For non-batch processing, use the single-item function
            # For batch processing, use the batch function
            pe_func = single_func if batch_size is None else batch_func

            # Run parallel_execution with different progress bar settings
            pe_times = []
            for _ in range(n_runs):
                _, pe_time = run_parallel_execution(
                    pe_func,
                    params,
                    n_jobs,
                    batch_size=batch_size,
                    task_profile=task_profile,
                    show_combined_progress=show_combined,
                )
                pe_times.append(pe_time)
            pe_avg_time = sum(pe_times) / len(pe_times)

            # Record results
            results.append(
                {
                    "Task Type": task_type,
                    "Batch Size": batch_size if batch_size is not None else "None",
                    "Progress Bar": "Combined" if show_combined else "Per-worker",
                    "Time (s)": pe_avg_time,
                }
            )

    return pd.DataFrame(results)


def print_results(df, title):
    """Print benchmark results in a nice table"""
    console = Console()
    console.print(f"\n[bold]{title}[/bold]")

    if "Speedup" in df.columns:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Task Type")
        table.add_column("Batch Size")
        table.add_column("parallel_execution (s)")
        table.add_column("joblib (s)")
        table.add_column("Speedup")

        for _, row in df.iterrows():
            table.add_row(
                str(row["Task Type"]),
                str(row["Batch Size"]),
                f"{row['parallel_execution (s)']:.3f}",
                f"{row['joblib (s)']:.3f}",
                f"{row['Speedup']:.2f}x",
            )
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Task Type")
        table.add_column("Batch Size")
        table.add_column("Progress Bar")
        table.add_column("Time (s)")

        for _, row in df.iterrows():
            table.add_row(
                str(row["Task Type"]), str(row["Batch Size"]), str(row["Progress Bar"]), f"{row['Time (s)']:.3f}"
            )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel_execution vs joblib")
    parser.add_argument("--n_items", type=int, default=500, help="Number of items to process")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs for averaging")
    args = parser.parse_args()

    # Run batch vs non-batch benchmark
    print(f"Running batch vs non-batch benchmark with {args.n_items} items, {args.n_jobs} jobs, {args.n_runs} runs...")
    batch_results = benchmark_batch_vs_non_batch(args.n_items, args.n_jobs, args.n_runs)
    print_results(batch_results, "Batch vs Non-Batch Processing Benchmark")

    # Run progress bar benchmark
    print(f"\nRunning progress bar benchmark with {args.n_items} items, {args.n_jobs} jobs, {args.n_runs} runs...")
    progress_results = benchmark_progress_bars(args.n_items, args.n_jobs, 50, args.n_runs)
    print_results(progress_results, "Progress Bar Benchmark (Combined vs Per-worker)")


if __name__ == "__main__":
    main()
