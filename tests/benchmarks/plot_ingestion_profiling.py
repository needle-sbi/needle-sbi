"""
Plot the results of `test_ingestion_profiling.py`: setup cost and total read time across four
ingestion strategies (ROOT via dask, ROOT via IterativeIngestor, parquet via dask, parquet via
IterativeParquetIngestor), on the same Delphes dataset/column sets.

Produces two plot groups:
- `ingestion_setup_cost__...`: one 2-panel figure (1 column / 14 columns, shared y-axis), the
  upfront cost of resolving length/fields before any event data is read.
- `ingestion_total_time__...`: two standalone NEEDLE-styled figures (one per column count), the
  total time until all requested data has been read (setup + read).

Run after generating the benchmark JSON(s), e.g.:

    pytest tests/benchmarks/test_ingestion_profiling.py --benchmark-only -s -m "not slow" \\
        --benchmark-json=tests/benchmarks/results/ingestion_profiling_fast.json
    pytest tests/benchmarks/test_ingestion_profiling.py --benchmark-only -s -m slow \\
        --benchmark-json=tests/benchmarks/results/ingestion_profiling_slow.json

    python tests/benchmarks/plot_ingestion_profiling.py

A simplified, notebook-friendly twin of the total-time plot lives in
`tests/benchmarks/plot_ingestion_profiling.ipynb` for ad-hoc tweaking (not committed --
`tests/**/*.ipynb` is gitignored).

Disclaimer: Part of this code was written with the help of GPT-5.
"""

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mplhep as hep
import yaml

plt.style.use(hep.style.CMS)
plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"
INPUT_FILES = [RESULTS_DIR / "ingestion_profiling_fast.json", RESULTS_DIR / "ingestion_profiling_slow.json"]
EVENTS_PER_FILE = 10_000
COLUMN_MODES = {"few": "1", "many": "14"}
# (setup benchmark, read benchmark, legend label, marker) -- one row per strategy, fixed
# order/color/marker used consistently across every plot below.
METHODS = [
    ("test_root_dask_setup", "test_root_dask_read", "ROOT uproot.dask", "o"),
    ("test_root_iterative_setup", "test_root_iterative_read", "ROOT uproot.iterate", "s"),
    ("test_parquet_dask_setup", "test_parquet_dask_read", "Parquet dak.from_parquet", "^"),
    ("test_parquet_iterative_setup", "test_parquet_iterative_read", "Parquet ak.from_parquet", "D"),
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "unknown"


def _load_benchmarks() -> List[Dict[str, Any]]:
    benchmarks: List[Dict[str, Any]] = []
    for path in INPUT_FILES:
        if not path.exists():
            continue
        with open(path) as f:
            benchmarks.extend(json.load(f)["benchmarks"])
    if not benchmarks:
        raise FileNotFoundError(f"No benchmark JSON found in {[str(p) for p in INPUT_FILES]}")
    return benchmarks


def _index_by(benchmarks: List[Dict[str, Any]], prefix: str) -> Dict[tuple, Dict[str, float]]:
    """Map (num_files, column_mode) -> {"min": ..., "median": ..., "max": ...} for one test function."""
    out: Dict[tuple, Dict[str, float]] = {}
    for b in benchmarks:
        if not b["name"].startswith(prefix + "["):
            continue
        key = (b["params"]["num_files"], b["params"]["column_mode"])
        out[key] = {"min": b["stats"]["min"], "median": b["stats"]["median"], "max": b["stats"]["max"]}
    return out


def _write_sidecar(path_stem: Path, description: str) -> None:
    meta = {
        "description": description,
        "dataset": "Delphes v1 (KIT, /ceph/epfeffer/mlpaper/delphes_v1)",
        "inputs": [str(p) for p in INPUT_FILES if p.exists()],
        "git_commit": _git_commit(),
        "generated_by": "tests/benchmarks/plot_ingestion_profiling.py",
    }
    with open(f"{path_stem}.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def plot_total_time(benchmarks: List[Dict[str, Any]], num_files_list: List[int], column_mode: str) -> Path:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 6))

    all_lo: List[float] = []
    all_hi: List[float] = []

    for color, (setup_name, read_name, label, marker) in zip(colors, METHODS):
        setup, read = _index_by(benchmarks, setup_name), _index_by(benchmarks, read_name)
        xs = [n for n in num_files_list if (n, column_mode) in setup and (n, column_mode) in read]

        if not xs:
            continue

        med = [setup[(n, column_mode)]["median"] + read[(n, column_mode)]["median"] for n in xs]
        lo = [setup[(n, column_mode)]["min"] + read[(n, column_mode)]["min"] for n in xs]
        hi = [setup[(n, column_mode)]["max"] + read[(n, column_mode)]["max"] for n in xs]
        all_lo += lo
        all_hi += hi
        ax.errorbar(
            xs,
            med,
            yerr=[[m - v for m, v in zip(med, lo)], [v - m for m, v in zip(med, hi)]],
            marker=marker,
            markersize=7,
            linestyle="-",
            color=color,
            label=label,
            capsize=3,
        )

    ax.set_ylim(min(all_lo) / 2.0, max(all_hi) * 2.0)

    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Ellapsed Time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(num_files_list)
    ax.set_xticklabels([str(n * EVENTS_PER_FILE // 1000) + "k" for n in num_files_list])
    ax.legend(loc="lower right", fontsize=12, frameon=True)

    ax.text(0.03, 0.97, r"$\bf{NEEDLE}$ $\it{Benchmark}$", transform=ax.transAxes, fontsize=20, va="top")
    info = f"Delphes Simulation\n" f"Read {COLUMN_MODES[column_mode]} column(s)"
    ax.text(0.03, 0.9, info, transform=ax.transAxes, fontsize=12, va="top")

    fig.tight_layout()

    path_stem = PLOTS_DIR / f"ingestion_total_time__delphes_v1__{column_mode}_columns__needle_style"
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path_stem


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = _load_benchmarks()
    num_files_list = sorted({b["params"]["num_files"] for b in benchmarks})

    for column_mode, n_columns in COLUMN_MODES.items():
        stem = plot_total_time(benchmarks, num_files_list, column_mode)
        _write_sidecar(
            stem,
            f"Total wall time until all requested data has been read ({n_columns} column(s)), for "
            "four ingestion strategies on the same Delphes dataset. The two dask-backed strategies "
            "(ROOT, parquet) read via PaddedDataset + DataLoader, the same path an actual training "
            "run uses; the two iterative strategies read via a full pass over .iterate(). Error bars "
            "are the sum of each step's independent min/max.",
        )

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
