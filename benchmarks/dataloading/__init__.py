"""
Benchmarking module for dataset performance comparison.

Disclaimer and credit: Unless otherwise stated, the code in this directory was originally generated
using Claude Sonnet 4 (via Github Copilot) and has been modified afterwards to fit the project.
"""

from .benchmark_utils import BenchmarkResults, BenchmarkTimer, MemoryProfiler
from .dataset_benchmark import DatasetBenchmark
from .plotting import BenchmarkPlotter

__all__ = [
    "DatasetBenchmark",
    "BenchmarkTimer",
    "MemoryProfiler",
    "BenchmarkResults",
    "BenchmarkPlotter",
]
