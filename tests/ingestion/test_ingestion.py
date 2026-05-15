from typing import Callable

import numpy as np
import pytest

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
