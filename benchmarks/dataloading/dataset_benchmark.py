"""
Main benchmark runner for comparing dataset performance.
"""

import logging
import time
from typing import Optional

from torch.utils.data import DataLoader

from benchmarks.dataloading.benchmark_utils import (
    BenchmarkResults,
    BenchmarkTimer,
    MemoryProfiler,
)
from needle.ml.datasets import PaddedDaskDataset, PaddedDataset, PaddedTorchDataset
from needle.utils.dask_ingestor import Ingestor
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("benchmarks")


class DatasetBenchmark:
    """
    Benchmark runner for comparing dataset performance.
    """

    results: list[BenchmarkResults] = []

    def __init__(self, verbose: bool = True):
        """
        Initialize the runner.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose

    def run(
        self,
        features: Ingestor,
        labels: Ingestor,
        dataset_type: str,
        batch_size: int = 32,
        chunk_size: Optional[int] = None,
        num_workers: int = 0,
        shuffle_chunks: bool = False,
        shuffle_events: bool = False,
        random_seed: int = 42,
    ) -> BenchmarkResults:
        """
        Benchmark a single dataset configuration.

        Args:
            features: Ingestor instance
            labels: Ingestor instance
            dataset_type: One of these
                - "PaddedEagerDataset"
                - "PaddedTorchDataset"
                - "ParticleDaskChunked"
            batch_size: Batch size for DataLoader
            num_workers: Number of DataLoader workers
            shuffle_chunks: Whether to shuffle chunks
            shuffle_events: Whether to shuffle events inside each chunk
            random_seed: Self-explanatory

        Returns:
            BenchmarkResults object with all metrics
        """
        if self.verbose:
            logging.info(f"Benchmarking {dataset_type} with batch_size={batch_size}")

        MemoryProfiler.clear_memory()
        MemoryProfiler.warm_up_system()

        memory_profiler = MemoryProfiler()
        timer = BenchmarkTimer()

        def create_dataset():
            nonlocal chunk_size
            if dataset_type == "PaddedEagerDataset":
                return PaddedDataset(features=features, labels=labels)
            elif dataset_type == "PaddedTorchDataset":
                if chunk_size is None:
                    chunk_size = 1000
                return PaddedTorchDataset(
                    features=features,
                    labels=labels,
                    shuffle_partitions=shuffle_chunks,
                    shuffle_events=shuffle_events,
                    random_seed=random_seed,
                )
            elif dataset_type == "ParticleDaskChunked":
                return PaddedDaskDataset(
                    features=features,
                    labels=labels,
                    shuffle_partitions=shuffle_chunks,
                    shuffle_events=shuffle_events,
                    random_seed=random_seed,
                )
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

        with timer.time_context():
            dataset: PaddedDataset
            dataset, creation_overhead, traced_memory = memory_profiler.measure_creation_memory(create_dataset)

        init_time = timer.elapsed_time
        init_memory_peak = creation_overhead

        if dataset_type != "PaddedTorchDataset":
            num_workers = 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=dataset.SHUFFLE_ALLOWED,
        )
        memory_profiler.set_baseline()
        timer = BenchmarkTimer()

        with timer.time_context():
            first_batch = next(iter(dataloader))
            del first_batch

        first_batch_time = timer.elapsed_time

        memory_profiler.set_baseline()
        batch_times = []

        with timer.time_context():
            with memory_profiler.profile_memory():
                for i, batch in enumerate(dataloader):
                    batch_start = time.perf_counter()
                    _ = batch[0].shape, batch[1].shape
                    batch_end = time.perf_counter()
                    batch_times.append(batch_end - batch_start)

                    memory_profiler.update_peak()
                    del batch

        full_iteration_time = timer.elapsed_time
        iteration_memory_peak = memory_profiler.get_memory_overhead()

        dataset_size = len(dataset)
        total_batches = len(batch_times)
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
        events_per_second = dataset_size / full_iteration_time if full_iteration_time > 0 else 0.0

        results = BenchmarkResults(
            dataset_type=dataset_type,
            dataset_size=dataset_size,
            batch_size=batch_size,
            num_workers=num_workers,
            init_time=init_time,
            first_batch_time=first_batch_time,
            full_iteration_time=full_iteration_time,
            avg_batch_time=avg_batch_time,
            init_memory_peak=init_memory_peak,
            iteration_memory_peak=iteration_memory_peak,
            memory_overhead=max(init_memory_peak, iteration_memory_peak),  # Use max for overall overhead
            total_batches=total_batches,
            events_per_second=events_per_second,
            creation_memory_overhead=creation_overhead,
            traced_memory_peak=traced_memory,
        )
        self.results.append(results)

        if self.verbose:
            logger.info(f"Init time: {init_time:.3f}s")
            logger.info(f"Full iteration: {full_iteration_time:.3f}s")
            logger.info(f"Events/sec: {events_per_second:.1f}")
            logger.info(f"Memory overhead: {results.memory_overhead:.1f} MB")

        del dataset, dataloader
        MemoryProfiler.clear_memory()
        return results
