import pytest

from needle.etl.dask_ingestor import Ingestor
from tests.conftest import ArrayField


@pytest.fixture
def ingestor(request: pytest.FixtureRequest):
    try:
        make_parquet_file = request.getfixturevalue("make_parquet_file")
    except pytest.FixtureLookupError:
        pytest.skip("Fixture 'make_parquet_file' from preprocessor could not be found")

    template = ArrayField(dtype=float, shape=(100, 1, 1))
    file = make_parquet_file(columns={"Lepton": {"pt": template}}, file_name="simple")
    return Ingestor(file)
