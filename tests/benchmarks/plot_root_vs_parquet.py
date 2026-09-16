"""Plotting script for the ROOT vs Parquet ingestion benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd

FILE_TYPES = ["parquet", "root"]
COMPONENTS = ["Graph Building", "Column-based Iteration", "Row-based Iteration"]
TEST_METHODS = ["only_metadata", "materialize_partitions", "iterate_dataloader"]
COLORS = ["lightcoral", "lightgreen", "lightblue"]
DEFAULT_OUTPUT = Path("tests/benchmarks/plots/root_vs_parquet.png")  # TODO Not happy with default path


def load_benchmark_json(path: Union[str, Path], merge_index: bool = False) -> pd.DataFrame:
    """Load a pytest-benchmark autosave JSON file into a flat DataFrame.

    Args:
        path: Path to the `.json` file autosaved by pytest-benchmark (`--benchmark-autosave`).
        merge_index: If True, set/sort the index to
            `["file_type", "column_mode", "file_percentage", "num_events"]`.

    Returns:
        pd.DataFrame: One row per benchmarked parametrization, with `mean_time`, `median_time`,
            `min_time`, `max_time`, `stddev` and `rounds` extracted from the benchmark stats.
    """
    with open(path) as f:
        data = json.load(f)

    rows = []
    for b in data["benchmarks"]:
        params = b["params"]
        stats = b["stats"]
        rows.append(
            {
                "name": b["name"],
                "file_type": params["file_type"],
                "num_events": params["num_events"],
                "test_method": params["test_method"],
                "column_mode": params["column_mode"],
                "file_percentage": params["file_percentage"],
                "mean_time": stats["mean"],
                "median_time": stats["median"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "stddev": stats["stddev"],
                "rounds": stats["rounds"],
            }
        )

    df = pd.DataFrame(rows)

    if merge_index:
        df = df.set_index(["file_type", "column_mode", "file_percentage", "num_events"]).sort_index()

    return df


def find_latest_benchmark_json(benchmarks_dir: Union[str, Path] = ".benchmarks") -> Path:
    """Find the most recently modified pytest-benchmark autosave JSON under `benchmarks_dir`.

    Args:
        benchmarks_dir: Root directory to search recursively for `*.json` files. Defaults to the
            `--benchmark-autosave` output directory `.benchmarks/`.

    Returns:
        Path: The most recently modified benchmark JSON file found.

    Raises:
        FileNotFoundError: If no `*.json` files are found under `benchmarks_dir`.
    """
    candidates = sorted(Path(benchmarks_dir).rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark JSON files found under {benchmarks_dir!r}. Run "
            "`pytest --benchmark-only tests/benchmarks/test_root_vs_parquet.py` first "
            "(autosave is enabled by default via `--benchmark-autosave`)."
        )
    return candidates[-1]


def select_benchmarks(
    df: pd.DataFrame,
    column_mode: str = "config",
    file_percentage: float = 0.0,
    num_events: int = 1000,
) -> pd.Series:
    """Filter to a single `(column_mode, file_percentage, num_events)` slice and group by
    `(file_type, test_method)`, averaging `mean_time` over rounds.

    Args:
        df: Flat DataFrame as returned by `load_benchmark_json`.
        column_mode: Which `column_mode` parametrization to select (`"one"` or `"config"`).
        file_percentage: Which `file_percentage` parametrization to select.
        num_events: Which `num_events` parametrization to select.

    Returns:
        pd.Series: `mean_time` indexed by `(file_type, test_method)`, matching the `grouped`
            object expected by `plot_root_vs_parquet`.

    Raises:
        ValueError: If no rows match the requested filter.
    """
    mask = (
        (df["column_mode"] == column_mode)
        & (df["file_percentage"] == file_percentage)
        & (df["num_events"] == num_events)
    )
    subset = df[mask]
    if subset.empty:
        available = df[["column_mode", "file_percentage", "num_events"]].drop_duplicates()
        raise ValueError(
            f"No benchmark rows found for column_mode={column_mode!r}, "
            f"file_percentage={file_percentage!r}, num_events={num_events!r}.\n"
            f"Available combinations:\n{available.to_string(index=False)}"
        )
    return subset.groupby(["file_type", "test_method"])["mean_time"].mean()


def plot_root_vs_parquet(
    grouped: pd.Series,
    output_path: Union[str, Path],
    annotation: Optional[str] = None,
) -> plt.Figure:
    """Build the Graph Building / Column-based Iteration / Row-based Iteration comparison chart.

    Args:
        grouped: `mean_time` indexed by `(file_type, test_method)`, as returned by
            `select_benchmarks`.
        output_path: Where to save the figure. Parent directories are created if needed.
        annotation: Optional text box

    Returns:
        matplotlib.figure.Figure: The created figure (caller is responsible for `plt.close(fig)`).
    """
    times = {}
    for ft in FILE_TYPES:
        graph_building = grouped.loc[ft, "only_metadata"]
        materialization = grouped.loc[ft, "materialize_partitions"] - graph_building
        total = grouped.loc[ft, "iterate_dataloader"] - graph_building
        times[ft] = [graph_building, materialization, total]

    x = np.arange(len(FILE_TYPES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    for i, comp in enumerate(COMPONENTS):
        ax.bar(
            x + i * width,
            [times[ft][i] for ft in FILE_TYPES],
            width,
            label=comp,
            color=COLORS[i],
            alpha=1,
            zorder=2,
        )
        for j, ft in enumerate(FILE_TYPES):
            bar_x = x[j] + i * width
            bar_height = times[ft][i]
            unit = "ms" if bar_height < 1 else "s"
            value = bar_height * 1000 if unit == "ms" else bar_height
            ax.text(bar_x, bar_height + 0.01, f"{value:.1f}{unit}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("File Type")
    ax.set_ylabel("Average Time [s]")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FILE_TYPES)
    max_height = max(v for values in times.values() for v in values)
    ax.set_ylim(top=max_height * 1.35)
    ax.legend(loc="upper left")

    if annotation:
        ax.text(
            0.02,
            0.65,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    mplhep.label.exp_label(loc=0, exp="NEEDLE", ax=ax, rlabel="")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)

    return fig


def main(argv: Optional[list] = None) -> Path:
    """CLI entry point. Loads a benchmark JSON, filters it, plots it and saves the figure.

    Args:
        argv: Optional argument list (for testing). Defaults to `sys.argv[1:]`.

    Returns:
        Path: The path the figure was saved to.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a pytest-benchmark JSON file. Defaults to the latest autosave under .benchmarks/.",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Where to save the plot.")
    args = parser.parse_args(argv)

    input_path = Path(args.input) if args.input else find_latest_benchmark_json()
    df = load_benchmark_json(input_path)
    grouped = select_benchmarks(df)
    annotation = "Files: 800 columns, 130GB\n" "Read: 8 columns, 1.3M events"
    fig = plot_root_vs_parquet(grouped, args.output, annotation=annotation)
    plt.close(fig)

    print(f"Saved plot to {args.output} (source: {input_path})")
    return Path(args.output)


if __name__ == "__main__":
    main()
