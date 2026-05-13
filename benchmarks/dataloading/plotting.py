"""
Plotting utilities for visualizing benchmark results.

NOTE: I am aware that the object handling could be done better. The list of BenchmarkResults
would be more convenient if it were an instance of a dedicated class that also handles the
to and from pandas DataFrame conversion. Currently, the plotting class BenchmarkPlotter is
doing both the Dataframe handling and the plotting. @KylianSchmidt
"""

import inspect
import os
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.dataloading.benchmark_utils import BenchmarkResults


def plot(func: Callable) -> Callable:
    """Decorator to indicate that this function should be called during plotting."""
    func._marked_for_plotting = True  # type: ignore[attr-defined]
    func._plot_name = func.__name__.removeprefix("plot_")  # type: ignore[attr-defined]
    return func


class BenchmarkPlotter:
    """
    Modular plotting class for visualizing dataset benchmark results.
    """

    def __init__(
        self,
        results: List[BenchmarkResults],
        figsize: Tuple[float, float] = (6.5, 5),
    ):
        """
        Initialize the plotter with styling options.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.results = results
        self.figsize = figsize
        self.setup_style()
        self.df = self.results_to_df(results)

    def setup_style(self, style: str = "belle2") -> None:
        """Setup matplotlib styling."""
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")
            plt.rcParams.update(
                {
                    "font.size": 10,
                    "axes.titlesize": 12,
                    "axes.labelsize": 10,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 9,
                    "figure.titlesize": 14,
                }
            )

        self.dataset_colors = {
            "PaddedEagerDataset": "#1f77b4",
            "PaddedTorchDataset": "#ff7f0e",
            "ParticleDaskChunked": "#2ca02c",
        }
        self.dataset_markers = {
            "PaddedEagerDataset": "o",  # Circle
            "PaddedTorchDataset": "s",  # Square
            "ParticleDaskChunked": "^",  # Triangle
        }

    def results_to_df(self, results: List[BenchmarkResults] = None) -> pd.DataFrame:
        """
        Convert benchmark results to pandas DataFrame for easier plotting.

        Args:
            results: List of benchmark results

        Returns:
            pandas DataFrame with all results
        """
        results = results or self.results
        data = [result.to_dict() for result in results]
        df = pd.DataFrame(data)
        df = df.sort_values(["dataset_type", "dataset_size", "batch_size"])
        return df

    def results_to_csv(
        self,
        results: List[BenchmarkResults] = None,
        output_path: str = "benchmark_summary.csv",
    ) -> None:
        """Save benchmark results to CSV file."""
        results = results or self.results
        df = self.results_to_df(results)
        df.to_csv(output_path, index=False)

    @staticmethod
    def results_from_csv(
        input_path: str = "benchmark_summary.csv",
    ) -> List[BenchmarkResults]:
        """
        Load benchmark results from CSV file.

        Each row in the dataframe is an entry in the list of BenchmarkResults.
        """
        df = pd.read_csv(input_path)
        return [BenchmarkResults.from_dict(dict(row)) for index, row in df.iterrows()]

    @classmethod
    def from_csv(cls, input_path: str = "benchmark_summary.csv") -> "BenchmarkPlotter":
        """
        Create a BenchmarkPlotter instance from a CSV file.

        Args:
            input_path: Path to the CSV file containing benchmark results

        Returns:
            An instance of BenchmarkPlotter

        First load the list of BenchmarkResults then create a new instance of this class from it.
        """
        results = cls.results_from_csv(input_path)
        return cls(results=results)

    def _save_plot(self, fig: plt.Figure, save_path: str) -> None:  # type: ignore
        """
        Save plot to file with proper directory creation.

        Args:
            fig: matplotlib Figure to save
            save_path: Path to save the plot
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

    def _get_unique_dataset_type(self, df: pd.DataFrame = None) -> Iterator[Tuple[str, pd.DataFrame]]:
        df = df or self.df

        for dataset_name in df["dataset_type"].unique():
            yield (dataset_name, df[df["dataset_type"] == dataset_name])

    def print_summary(
        self,
        results: List[BenchmarkResults] = None,
    ) -> None:
        results = results or self.results
        summary = self.results_to_df(results)

        print("BENCHMARK SUMMARY")
        print("=================")
        print(summary)

        print("SUMMARY BY DATASET SIZE AND PARTITIONS")
        print("======================================")

        df_by_size = pd.DataFrame([r.to_dict() for r in results])  # TODO Could this be the 'summary' df?
        size_summary = (
            df_by_size.groupby(
                [
                    "dataset_type",
                    "dataset_size",
                ]
            )
            .agg(
                {
                    "events_per_second": "max",
                    "memory_overhead": "mean",
                    "init_time": "mean",
                }
            )
            .round(3)
        )
        print(size_summary.to_string())

    def save_all(
        self,
        results: List[BenchmarkResults] = None,
        output_dir: str = "plots",
    ) -> None:
        """
        Generate and save all standard plots.

        Args:
            results: List of benchmark results
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        results = results or self.results
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def is_plotting_function(func: Callable) -> bool:
            return inspect.ismethod(func) and getattr(func, "_marked_for_plotting", False)

        for _, plot_func in inspect.getmembers(self, predicate=is_plotting_function):
            fig = plot_func()
            save_path = os.path.join(output_dir, f"{plot_func._plot_name}.pdf")
            self._save_plot(fig, save_path)
            print(f"Plot '{plot_func._plot_name}' saved to {save_path}")

        summary = self.results_to_df(results)
        summary.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
        return None

    @plot
    def plot_iteration_time(self) -> plt.Figure:  # type: ignore
        """
        Plot the iteration time of the benchmarks.
        """
        fig = plt.figure(figsize=self.figsize)

        for dataset_name, subset in self._get_unique_dataset_type():
            summary = subset.groupby("dataset_size")["full_iteration_time"].agg(["mean", "std"])
            plt.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"],
                label=dataset_name,
                c=self.dataset_colors[dataset_name],
                marker=self.dataset_markers[dataset_name],
            )

        plt.xlabel("Dataset Size [events]")
        plt.ylabel("Iteration Time [s]")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.xlim(1e3 - 100, None)
        return fig

    @plot
    def plot_init_memory(self) -> plt.Figure:  # type: ignore
        """
        Plot the initialization memory overhead of the benchmarks.
        """
        fig = plt.figure(figsize=self.figsize)

        for dataset_name, subset in self._get_unique_dataset_type():
            summary = subset.groupby("dataset_size")["iteration_memory_peak"].agg(["mean", "std"])
            plt.errorbar(
                summary.index,
                summary["mean"],
                summary["std"],
                label=dataset_name,
                c=self.dataset_colors[dataset_name],
                marker=self.dataset_markers[dataset_name],
            )

        plt.xlabel("Dataset Size")
        plt.ylabel("Init Memory Overhead [MB]")
        plt.xscale("log")
        plt.yscale("linear")
        plt.xlim(1e3 - 100, None)
        plt.ylim(0 - 10, None)
        plt.legend()
        return fig
