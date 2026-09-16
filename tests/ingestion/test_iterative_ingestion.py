from pathlib import Path
from typing import Callable

import awkward as ak
import numpy as np
import pytest
import uproot

from needle.etl.array import NestedArrayIndexer
from needle.etl.dask_ingestor import Ingestor
from needle.etl.iterative_ingestor import IterableIngestor
from tests.conftest import ArrayField


def test_parquet_file_simple(make_parquet_file: Callable) -> None:
    """Mirrors `test_ingestion.test_parquet_file_simple`, but for `IterableIngestor`."""
    test_template = ArrayField(dtype=np.float32, shape=(100, 1, 1))
    parquet_file = make_parquet_file(columns={"pt": test_template}, file_name="simple")
    ingestor = IterableIngestor(paths=parquet_file, columns="pt")

    assert ingestor.fields == ["pt"]
    assert ingestor.num_classes == 1
    assert ingestor.SEPARATOR == "."
    assert ingestor.format == "parquet"
    assert ingestor.length == 100

    chunks = list(ingestor.iterate())
    assert len(chunks) == 1
    assert len(chunks[0]) == 100
    assert chunks[0].fields == ["pt"]

    with pytest.raises(ValueError):
        IterableIngestor(paths=parquet_file, columns="Non-existent.column")


@pytest.fixture
def nested_file(make_parquet_file: Callable) -> str:
    test_template = ArrayField(dtype=np.float32, shape=(100, 1, 1))
    return make_parquet_file(
        columns={
            "Lepton": {
                "pt": test_template,
                "eta": test_template,
            }
        },
        file_name="nested",
    )


def test_parquet_file_nested(nested_file: str) -> None:
    ingestor = IterableIngestor(paths=nested_file)
    assert ingestor.fields == ["Lepton.pt", "Lepton.eta"]
    assert ingestor.num_classes == 2
    assert ingestor.length == 100

    (chunk,) = list(ingestor.iterate())
    assert len(chunk) == 100


def test_parquet_max_number_events(nested_file: str) -> None:
    ingestor = IterableIngestor(paths=nested_file, max_number_events=30)
    assert ingestor.length == 30

    chunks = list(ingestor.iterate())
    assert sum(len(chunk) for chunk in chunks) == 30


@pytest.fixture
def root_file(tmp_path: Path) -> Callable[..., str]:
    def _make_root_file(num_events: int = 100, file_name: str = "simple") -> str:
        path = str(tmp_path / f"{file_name}.root")

        with uproot.recreate(path) as file:
            file.mktree("tree", {"pt": "float64", "eta": "float64"})
            file["tree"].extend(
                {
                    "pt": np.arange(num_events, dtype=np.float64),
                    "eta": np.arange(num_events, dtype=np.float64),
                }
            )
        return path

    return _make_root_file


def test_root_file_simple(root_file: Callable[..., str]) -> None:
    path = root_file(num_events=137)
    ingestor = IterableIngestor(paths=path, step_size=20)

    assert set(ingestor.fields) == {"pt", "eta"}
    assert ingestor.num_classes == 2
    assert ingestor.format == "root"
    assert ingestor.treename == "tree"
    assert ingestor.length == 137

    chunks = list(ingestor.iterate())
    assert sum(len(chunk) for chunk in chunks) == 137
    assert len(chunks) > 1  # step_size smaller than the file forces multiple chunks
    assert all(set(chunk.fields) == {"pt", "eta"} for chunk in chunks)


def test_root_file_columns_and_missing_column(root_file: Callable[..., str]) -> None:
    path = root_file(num_events=10)
    ingestor = IterableIngestor(paths=path, columns="pt")
    assert ingestor.fields == ["pt"]

    with pytest.raises(ValueError):
        IterableIngestor(paths=path, columns=["not_a_branch"])


def test_root_max_number_events(root_file: Callable[..., str]) -> None:
    path = root_file(num_events=50)
    ingestor = IterableIngestor(paths=path, step_size=7, max_number_events=22)
    assert ingestor.length == 22

    chunks = list(ingestor.iterate())
    assert sum(len(chunk) for chunk in chunks) == 22


def test_etl_lazy_exports_resolve() -> None:
    """Every name in `needle.etl.__all__` must actually be importable through the package's
    `__getattr__`. This catches typos/renames between `__all__`/`_MODULE_BY_NAME` and what the
    target module actually defines (e.g. `IngestorColumn`/`IngestorChunk` referencing names that
    did not yet exist in `needle.etl.protocols`)."""
    import needle.etl as etl

    for name in etl.__all__:
        assert getattr(etl, name) is not None


def test_parquet_multiple_files(make_parquet_file: Callable) -> None:
    """One chunk per file is yielded, in file order, and `divisions` reflects per-file sizes."""
    template_a = ArrayField(dtype=np.float32, shape=(60, 1, 1))
    template_b = ArrayField(dtype=np.float32, shape=(40, 1, 1))
    file_a = make_parquet_file(columns={"pt": template_a}, file_name="a")
    file_b = make_parquet_file(columns={"pt": template_b}, file_name="b")

    ingestor = IterableIngestor(paths=sorted([file_a, file_b]))

    assert ingestor.length == 100
    assert ingestor.divisions == (0, 60, 100)

    chunks = list(ingestor.iterate())
    assert [len(chunk) for chunk in chunks] == [60, 40]


def test_root_missing_treename_raises(tmp_path: Path) -> None:
    """Auto-detection must fail loudly when a file has more than one TTree."""
    path = str(tmp_path / "ambiguous.root")
    with uproot.recreate(path) as file:
        file.mktree("tree_one", {"pt": "float64"})
        file["tree_one"].extend({"pt": np.arange(5, dtype=np.float64)})
        file.mktree("tree_two", {"pt": "float64"})
        file["tree_two"].extend({"pt": np.arange(5, dtype=np.float64)})

    with pytest.raises(ValueError):
        IterableIngestor(paths=path)


def test_unsupported_format_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("pt\n1.0\n")

    with pytest.raises(ValueError):
        IterableIngestor(paths=str(path))


def test_dask_and_iterative_parity(nested_file: str) -> None:
    """The dask-backed and non-dask Ingestors must agree on metadata for the same input file,
    and the events streamed by `IterableIngestor.iterate()` must match the events exposed by
    `Ingestor.__getitem__(...).compute()`, field by field."""
    dask_ingestor = Ingestor(paths=nested_file)
    iterative_ingestor = IterableIngestor(paths=nested_file)

    assert dask_ingestor.fields == iterative_ingestor.fields
    assert dask_ingestor.length == iterative_ingestor.length

    (chunk,) = list(iterative_ingestor.iterate())

    for field in dask_ingestor.fields:
        dask_column = dask_ingestor[field].compute()
        iterative_column = NestedArrayIndexer.get_nested_field(chunk, field, iterative_ingestor.SEPARATOR)
        assert list(ak.ravel(dask_column)) == list(ak.ravel(iterative_column))
