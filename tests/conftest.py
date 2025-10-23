import os
from pathlib import Path

import pytest

from ml.utils.config import MLConfig


@pytest.fixture(scope="session")
def fair_universe_sample() -> str:
    path = os.getenv("FAIR_UNIVERSE_DATA")

    if not path:
        pytest.skip(
            "Environment variable 'FAIR_UNIVERSE_DATA' not set. Should point to a parquet file: "
            "'export FAIR_UNIVERSE_DATA=../fair_universe/input_data/train/data/data.parquet'."
        )

    if not Path(path).exists():
        pytest.skip(f"Path {path} does not exist")

    return path


@pytest.fixture
def config_yaml(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> str:
    pytest.importorskip("preprocessor", reason="Could not import 'preprocessor'")
    from preprocessor.tests.conftest import ArrayField

    try:
        make_parquet_file = request.getfixturevalue("make_parquet_file")
    except pytest.FixtureLookupError:
        pytest.skip("Fixture 'make_parquet_file' from preprocessor could not be found")

    template = ArrayField(dtype=float, shape=(10000, 1, 1))
    file = make_parquet_file(
        columns={
            "Lepton": {"pt": template},
            "Jet": {"eta": template},
        },
        file_name="simple",
    )

    config = MLConfig()
    config.files_to_load = [file]
    config.total_epoch = 1
    config.features_columns = ["Lepton.pt"]
    config.labels_columns = ["Jet.eta"]
    config_path = os.path.abspath(tmp_path / "config.yaml")
    config.to_yaml(config_path)
    return config_path
