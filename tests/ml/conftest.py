from typing import Callable

import pytest

from needle.etl.dask_ingestor import Ingestor
from tests.conftest import ArrayField


@pytest.fixture
def ingestor(make_parquet_file: Callable):
    template = ArrayField(dtype=float, shape=(100, 1, 1))
    file = make_parquet_file(columns={"Lepton": {"pt": template}}, file_name="simple")
    return Ingestor(file)
