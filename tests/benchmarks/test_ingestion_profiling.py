"""
Profile ROOT/parquet ingestion setup and read cost across four strategies:

- ``root_dask``: `Ingestor(format="root")` -- `uproot.dask` graph/form construction.
- ``root_iterative``: `IterativeIngestor` -- `uproot.num_entries` metadata only, no dask graph.
- ``parquet_dask``: `Ingestor(format="parquet")` -- `dak.from_parquet` graph construction.
- ``parquet_iterative``: `IterativeParquetIngestor` -- `pyarrow.parquet.ParquetFile` metadata only,
  no dask graph.

See `tests/benchmarks/plot_ingestion_profiling.py` for the plots built from this data, and
`tests/benchmarks/test_root_vs_parquet.py` for the original, broader-scope ROOT-vs-parquet
benchmark this suite complements.

Run:
    pytest tests/benchmarks/test_ingestion_profiling.py \\
        --benchmark-only -s -m "not slow" \\
        --benchmark-json=tests/benchmarks/results/ingestion_profiling_fast.json
    pytest tests/benchmarks/test_ingestion_profiling.py \\
        --benchmark-only -s -m slow \\
        --benchmark-json=tests/benchmarks/results/ingestion_profiling_slow.json

Requires the `DELPHES_DATA_ROOT` and `DELPHES_DATA_PARQUET` environment variables (see
`tests/conftest.py`). Files are large (~950 MB / 10k events / ~800 branches each for ROOT) --
keep `num_files` small unless explicitly running the `-m slow` sweep.

Disclaimer: Part of this code was written with the help of GPT-5 and Claude Sonnet 5.
"""

from typing import Callable, List

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from needle.etl.array import resolve_paths
from needle.etl.dask_ingestor import Ingestor
from needle.etl.iterative_ingestor import IterativeIngestor, IterativeParquetIngestor

pytestmark = pytest.mark.benchmark

# Columns known to be valid for the Delphes v1 dataset, see tests/conf_tests/datasets/delphes.yaml.
# ("Track.*"/"Photon.*" only -- avoids the invalid branches dropped elsewhere, e.g. "fBits".)
COLUMN_SETS: dict[str, List[str]] = {
    "few": ["Track.PT"],
    "many": [
        "Track.PID",
        "Track.Charge",
        "Track.P",
        "Track.PT",
        "Track.Eta",
        "Track.Phi",
        "Track.C",
        "Track.Mass",
        "Photon.PT",
        "Photon.Eta",
        "Photon.Phi",
        "Photon.E",
        "Photon.T",
        "Photon.EhadOverEem",
    ],
}

NUM_FILES = [
    pytest.param(1, id="files_1"),
    pytest.param(5, id="files_5", marks=pytest.mark.slow),
    pytest.param(10, id="files_10", marks=pytest.mark.slow),
    pytest.param(15, id="files_15", marks=pytest.mark.slow),
    pytest.param(20, id="files_20", marks=pytest.mark.slow),
]


def _filter_name(columns: List[str]) -> Callable[[str], bool]:
    """Build a `filter_name` predicate that keeps only the requested columns (ROOT only)."""

    def _filter(name: str) -> bool:
        return name in columns

    return _filter


def _compute_dask_array(ingestor: Ingestor) -> None:
    ingestor.array.compute()


@pytest.fixture()
def root_paths(delphes_sample_root: str) -> List[str]:
    return resolve_paths(delphes_sample_root)


@pytest.fixture()
def parquet_paths(delphes_sample_parquet: str) -> List[str]:
    return resolve_paths(delphes_sample_parquet)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_dask_setup(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Upfront cost of `Ingestor(format="root")`: `uproot.dask` graph/form construction across
    every input file, plus `eager_compute_divisions()`. No event data is read.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]

    def _setup() -> Ingestor:
        return Ingestor(
            paths=paths, format="root", columns=columns, reader_kwargs={"filter_name": _filter_name(columns)}
        )

    benchmark.pedantic(_setup, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_dask_read(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Cost of reading all requested columns from an already-built ROOT `Ingestor` -- a single
    `ingestor.array.compute()`, the one dask-graph execution a real consumer triggers.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]
    ingestor = Ingestor(
        paths=paths, format="root", columns=columns, reader_kwargs={"filter_name": _filter_name(columns)}
    )

    benchmark.pedantic(_compute_dask_array, args=(ingestor,), rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_iterative_setup(
    benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `IterativeIngestor`: `uproot.num_entries` (TTree header only, all files) plus
    a single-file field lookup. No event data is read, and no dask graph is ever built.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]

    def _setup() -> IterativeIngestor:
        return IterativeIngestor(paths=paths, columns=columns)

    benchmark.pedantic(_setup, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_iterative_read(
    benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of a full `uproot.iterate` pass over all files via `IterativeIngestor.iterate()`."""
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]
    ingestor = IterativeIngestor(paths=paths, columns=columns)

    def _read() -> None:
        for _ in ingestor.iterate():
            pass

    benchmark.pedantic(_read, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_dask_setup(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `Ingestor(format="parquet")`: `dak.from_parquet` graph construction (cheap
    `pyarrow.parquet.ParquetFile` metadata per file) plus `eager_compute_divisions()`.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]

    def _setup() -> Ingestor:
        return Ingestor(paths=paths, format="parquet", columns=columns)

    benchmark.pedantic(_setup, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_dask_read(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of reading all requested columns from an already-built parquet `Ingestor` -- a single
    `ingestor.array.compute()`, the one dask-graph execution a real consumer triggers.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]
    ingestor = Ingestor(paths=paths, format="parquet", columns=columns)

    benchmark.pedantic(_compute_dask_array, args=(ingestor,), rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_iterative_setup(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `IterativeParquetIngestor`: `pyarrow.parquet.ParquetFile` metadata per file,
    no dask graph at all.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]

    def _setup() -> IterativeParquetIngestor:
        return IterativeParquetIngestor(paths=paths, columns=columns)

    benchmark.pedantic(_setup, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_iterative_read(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of a full `ak.from_parquet`-per-file pass via `IterativeParquetIngestor.iterate()`."""
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]
    ingestor = IterativeParquetIngestor(paths=paths, columns=columns)

    def _read() -> None:
        for _ in ingestor.iterate():
            pass

    benchmark.pedantic(_read, rounds=3, iterations=1, warmup_rounds=0)
