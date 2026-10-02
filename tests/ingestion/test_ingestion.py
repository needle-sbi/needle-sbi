from pathlib import Path
from typing import Callable

import awkward as ak
import numpy as np
import pytest

from needle.etl.array import are_divisions_valid
from needle.etl.dask_ingestor import Ingestor
from tests.conftest import ArrayField


def test_parquet_file_simple(make_parquet_file: Callable) -> None:
    """Test reading a simple parquet file with the Ingestor class

    TODO:
        - length of ingestor array
        - check nestedness
        - check shape
    """

    test_template = ArrayField(dtype=np.float32, shape=(100, 1, 1))
    parquet_file = make_parquet_file(columns={"pt": test_template}, file_name="simple")
    ingestor = Ingestor(paths=parquet_file, columns="pt")
    assert ingestor.fields == ["pt"]
    assert ingestor.num_classes == 1
    assert ingestor.SEPARATOR == "."
    assert ingestor.length == 100

    array = ingestor.array.compute()
    assert len(array) == 100
    with pytest.raises(ValueError):
        ingestor["Lepton"]
    with pytest.raises(ValueError):
        ingestor["Lepton-pt"]
    with pytest.raises(ValueError):
        ingestor["Non-existent.column"]


@pytest.fixture
def nested_file(make_parquet_file: Callable):
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


def test_parquet_file_nested(nested_file) -> None:
    ingestor = Ingestor(paths=nested_file)
    assert ingestor.fields == ["Lepton.pt", "Lepton.eta"]
    assert ingestor.num_classes == 2
    assert ingestor.SEPARATOR == "."
    assert ingestor.length == 100


def test_parquet_single_file_split_row_groups_has_valid_divisions(tmp_path: Path) -> None:
    """Regression test: a single parquet file split into multiple dask partitions by row group
    must still produce strictly increasing, non-zero interior divisions.

    `dak.from_parquet(..., split_row_groups=True)` correctly creates one partition per row group,
    but `eager_compute_divisions()` collapses every interior/trailing boundary to zero for this
    case. `Ingestor` must detect this and reconstruct the boundaries from the file's own
    row-group metadata (see `needle.etl.array.brute_force_row_group_divisions`).
    """
    n_rows = 1000
    n_row_groups = 10
    path = str(tmp_path / "multi_row_group.parquet")
    array = ak.Array({"pt": np.arange(n_rows, dtype=np.float32)})
    ak.to_parquet(array, path, row_group_size=n_rows // n_row_groups)

    ingestor = Ingestor(paths=path, reader_kwargs={"split_row_groups": True})

    assert ingestor.array.npartitions == n_row_groups
    assert len(ingestor.array.divisions) == n_row_groups + 1
    assert are_divisions_valid(ingestor.array.divisions)
    assert ingestor.array.divisions[0] == 0
    assert ingestor.array.divisions[-1] == n_rows
    assert ingestor.length == n_rows
