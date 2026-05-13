"""
Utility classes for benchmarking dataset performance.
"""

import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import psutil
import torch


@dataclass
class BenchmarkResults:
    """Store benchmark results."""

    dataset_type: str
    dataset_size: int
    batch_size: int
    num_workers: int

    init_time: float  # [seconds]
    first_batch_time: float  # [seconds]
    full_iteration_time: float  # [seconds]
    avg_batch_time: float  # [seconds]

    init_memory_peak: float  # [MB]
    iteration_memory_peak: float  # [MB]
    memory_overhead: float  # [MB]

    total_batches: int
    events_per_second: float

    creation_memory_overhead: float = 0.0
    traced_memory_peak: float = 0.0

    dataset_config: str = ""
    actual_partitions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResults":
        """Create an instance from a dictionary."""
        return cls(**data)


class BenchmarkTimer:
    """
    Timer
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            raise ValueError("Timer was not started")
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    @contextmanager
    def time_context(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            elapsed = self.stop()
            self.elapsed_time = elapsed


class MemoryProfiler:
    """
    Record the memory usage of functions
    """

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self.creation_memory: Optional[float] = None
        self.traced_memory: Optional[float] = None
        self._tracing_active: bool = False

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def set_baseline(self) -> None:
        """Set baseline memory usage with garbage collection."""
        gc.collect()
        self.baseline_memory = self.get_current_memory_mb()
        self.peak_memory = self.baseline_memory

    def start_tracing(self) -> None:
        """Memory tracing with tracemalloc."""
        if not self._tracing_active:
            tracemalloc.start()
            self._tracing_active = True

    def stop_tracing(self) -> float:
        """Stop memory tracing and return peak traced memory in MB."""
        if self._tracing_active:
            _, peak_trace = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._tracing_active = False
            self.traced_memory = peak_trace / 1024 / 1024  # Convert to MB
            return self.traced_memory
        return 0.0

    def update_peak(self) -> float:
        """Update peak memory usage and return current usage."""
        current = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        return current

    def get_memory_overhead(self) -> float:
        """Get memory overhead compared to baseline in MB."""
        if self.baseline_memory is None:
            raise ValueError("Baseline memory not set")
        return self.peak_memory - self.baseline_memory

    def get_creation_overhead(self) -> float:
        """Get memory overhead from creation phase."""
        if self.baseline_memory is None or self.creation_memory is None:
            return 0.0
        return self.creation_memory - self.baseline_memory

    def measure_creation_memory(self, creation_func, *args, **kwargs):
        """
        Measure memory usage during dataset creation

        Args:
            creation_func: Function that creates the dataset
            *args, **kwargs: Arguments to pass to creation_func

        Returns:
            Tuple of (dataset, creation_overhead_mb, traced_memory_mb)
        """
        gc.collect()
        self.set_baseline()
        self.start_tracing()
        dataset = creation_func(*args, **kwargs)
        self.clear_memory()
        self.creation_memory = self.get_current_memory_mb()

        traced_memory = self.stop_tracing()
        creation_overhead = self.get_creation_overhead()

        return dataset, creation_overhead, traced_memory

    @contextmanager
    def profile_memory(self):
        """Context manager for memory profiling."""
        if self.baseline_memory is None:
            self.set_baseline()
        try:
            yield self
        finally:
            self.update_peak()

    @classmethod
    def clear_memory(cls):
        """
        Run python garbage collection and clear potential GPU memory
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def warm_up_system(cls):
        """
        Warm up system by allocating some large tensors."""
        for _ in range(3):
            dummy = torch.randn(1000, 1000)
            del dummy

        cls.clear_memory()
        time.sleep(0.1)
