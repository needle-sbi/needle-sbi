"""Plotting script for the ROOT vs Parquet ingestion benchmark.

Run after generating the benchmark JSON(s):

    pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m "not slow"
    pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m slow

    python tests/benchmarks/plot_root_vs_parquet.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

plt.style.use(hep.style.CMS)
plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

FILE_TYPES = ["parquet", "root"]
COMPONENTS = ["Graph Building", "Column-based Iteration", "Row-based Iteration"]
TEST_METHODS = ["only_metadata", "materialize_partitions", "iterate_dataloader"]
COLORS = ["lightcoral", "lightgreen", "lightblue"]
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"
DEFAULT_OUTPUT = PLOTS_DIR / "ingestion_decomposed"
INPUT_FILES = [RESULTS_DIR / "root_vs_parquet_fast.json", RESULTS_DIR / "root_vs_parquet_slow.json"]


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


def load_default_benchmarks(merge_index: bool = False) -> pd.DataFrame:
    """Load and concatenate the fast/slow benchmark JSONs written by `tests/benchmarks/conftest.py`.

    Args:
        merge_index: If True, set/sort the index to
            `["file_type", "column_mode", "file_percentage", "num_events"]`.

    Returns:
        pd.DataFrame: Rows from every existing, non-empty file in `INPUT_FILES`.

    Raises:
        FileNotFoundError: If none of `INPUT_FILES` exist yet.
    """
    frames = [load_benchmark_json(p) for p in INPUT_FILES if p.exists() and p.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(
            f"No benchmark JSON found in {[str(p) for p in INPUT_FILES]}. Run "
            '`pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m "not slow"` '
            "(and/or `-m slow`) first."
        )
    df = pd.concat(frames, ignore_index=True)
    if merge_index:
        df = df.set_index(["file_type", "column_mode", "file_percentage", "num_events"]).sort_index()
    return df


def select_benchmarks(
    df: pd.DataFrame,
    column_mode: str,
    file_percentage: float,
    num_events: int,
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
        output_path: Where to save the figure. The `.pdf` and `.png` suffixes are added
            automatically. Parent directories are created if needed.
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

    fig, ax = plt.subplots(figsize=(7, 6))
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
            ax.text(bar_x, bar_height + 10, f"{value:.1f}{unit}", ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("File Type")
    ax.set_ylabel("Average Time [s]")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FILE_TYPES)
    max_height = max(v for values in times.values() for v in values)
    ax.set_ylim(top=max_height * 1.35)
    ax.legend(loc="upper right", fontsize=12, frameon=True)

    ax.text(0.03, 0.97, r"$\bf{NEEDLE}$ $\it{Benchmark}$", transform=ax.transAxes, fontsize=20, va="top")
    if annotation:
        ax.text(0.03, 0.9, annotation, transform=ax.transAxes, fontsize=12, va="top")

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_path}.png", dpi=400, bbox_inches="tight")

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
        help=f"Path to a pytest-benchmark JSON file. Defaults to {[str(p) for p in INPUT_FILES]}.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="Where to save the plot (without extension)."
    )
    args = parser.parse_args(argv)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_benchmark_json(Path(args.input)) if args.input else load_default_benchmarks()
    grouped = select_benchmarks(df, column_mode="config", file_percentage=0.1, num_events=-1)
    annotation = "Files: 800 columns, 130GB\n" "Read: 8 columns, 1.3M events"
    fig = plot_root_vs_parquet(grouped, args.output, annotation=annotation)
    plt.close(fig)

    source = args.input if args.input else [str(p) for p in INPUT_FILES if p.exists()]
    print(f"Saved plot to {args.output}.pdf/.png (source: {source})")
    return Path(f"{args.output}.png")


if __name__ == "__main__":
    main()
