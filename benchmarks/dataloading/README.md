# Benchmarking Module

## Overview

This module provides a comprehensive benchmarking framework for evaluating the performance of different dataset implementations in the NEEDLE ML pipeline. The main objective is to compare initialization time, memory usage and total IO speed for different kinds of datasets. Currently this is used to compare multi-processing methods between dask and pytorch.

## Objects

 - **DatasetBenchmark**: Main class that orchestrates the benchmarks. Handles dataset instantiation, memory profiling, and execution timing for each configuration of hyperparameters and dataset types.
 - **BenchmarkResults**: Data container for benchmark metrics. Currently used as entries in a list but could be its own object in the future.
 - **BenchmarkPlotter**: Converts the list of BenchmarkResults to a pandas DataFrame for plotting purposes

## Design Choices

### Benchmark Methodology

We use `psutil` to capture the process's memory usage. This includes

- Python object overhead
- Garbage collection. I dont think this makes a big difference but we want this to be as consistent as possible
- Shared memory usage in multiprocessing scenarios, as for the dask Queue object.

### Configuration Management

We use the same yaml style config parsing as in the main NEEDLE project. In the code this points directly to `"./ml/benchmarks/benchmark_config.yaml"`.

## Usage

### Basic Benchmarking

Due to some functions being defined upstream of this module, we have to hack the imports a bit. Therefore we need to be in the `orchestrator` directory and call

```bash
python ml/benchmarks/main.py
```

The above runs the complete benchmark suite across all dataset types, sizes, and worker configurations.


Extra CLI arguments:
 - To run a version with a reduced set of configurations `--test`
 - Add extra logging info by adding the `--verbose` flag.
 - Only execute the part of the code responsible for plotting using `--only-plot`.

## Extending the Framework

### Adding New Dataset Types

1. Implement the dataset class following the existing interface patterns
2. Add the class name to the `dataset_types` list in `main.py`
3. Ensure the dataset handles multiprocessing correctly if using `IterableDataset`

### Adding New Metrics

1. Extend `BenchmarkResults` dataclass with new metric fields
2. Update `DatasetBenchmark.run()` to collect the new measurements
3. Modify `BenchmarkPlotter` to visualize the additional metrics
