"""
Entry point for dataset benchmarks.
"""

import itertools
import logging
import sys
from pathlib import Path
from typing import Iterator

import hydra
from omegaconf import DictConfig, OmegaConf

from benchmarks.dataloading import BenchmarkPlotter, BenchmarkResults, DatasetBenchmark
from benchmarks.dataloading.config import BenchmarkConfig
from needle.utils.dask_ingestor import Ingestor
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("benchmarks")
sys.path.append(str(Path(__file__).parent.parent))  # add the ml package directory to the path


def product_dicts(**kwargs) -> Iterator[dict]:
    """
    Generate a dictionary of the cartesian product of the input lists.

    Args:
        **kwargs: Keyword arguments where each key is a variable name and the value is a list

    Returns:
        Iterator yielding a dictionary with a given combination of the input lists.

    Credit: Seth Johnson (from StackOverflow)
    Source: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()

    for prod in itertools.product(*kwargs.values()):
        yield dict(zip(keys, prod))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry function.

    Steps:
        1. Load benchmark configuration.
        2. For each combination of the benchmarking variables, load the datasets and run the benchmarking functions.
        3. Save the results to a pandas DataFrame and plot the results.

    TODO Maybe fix the relative path for 'plots' in the plotter argument
    """
    config: BenchmarkConfig = OmegaConf.structured(cfg)
    logger.setLevel(logging.DEBUG) if config.verbose else logger.setLevel(logging.INFO)
    benchmark = DatasetBenchmark(verbose=config.verbose)

    if config.only_plot:
        plotter = BenchmarkPlotter.from_csv(input_path=str(Path(__file__).parent / "plots/benchmark_summary.csv"))
        all_results = plotter.results
    else:
        all_results: list[BenchmarkResults] = []

        benchmark_variables = product_dicts(**config.variables)  # type: ignore

        for variable_dict in benchmark_variables:
            features = Ingestor(
                paths=config.data.files_to_load,
                format=config.format,
                columns=config.data.features_columns,
                max_number_events=variable_dict["number_events"],
            )
            labels = Ingestor(
                paths=config.data.files_to_load,
                format=config.format,
                columns=config.data.labels_columns,
                max_number_events=variable_dict["number_events"],
            )
            logger.info(f"Dataset size: {features.length:,} events")
            logger.info(f"Dataset number of partitions: {features.array.npartitions}")

            results = benchmark.run(
                features=features,
                labels=labels,
                dataset_type=variable_dict["dataset_types"],
                batch_size=variable_dict["batch_sizes"],
                num_workers=variable_dict["number_workers"],
                shuffle_chunks=False,
            )
            all_results.append(results)
            del features, labels  # Free up memory. Avoid GC to fake the memory overhead

        plotter = BenchmarkPlotter(all_results)

    plotter.save_all(results=all_results, output_dir="ml/benchmarks/plots")  # TODO Change to new path
    plotter.print_summary()


if __name__ == "__main__":
    main()
