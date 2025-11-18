import os
from pathlib import Path
from typing import Callable

import hydra
import pytest
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


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
def simple_sample(request: pytest.FixtureRequest) -> str:
    pytest.importorskip("preprocessor", reason="Could not import 'preprocessor'")
    from preprocessor.tests.conftest import ArrayField

    try:
        make_parquet_file: Callable = request.getfixturevalue("make_parquet_file")
    except pytest.FixtureLookupError:
        pytest.skip("Fixture 'make_parquet_file' from preprocessor could not be found")

    template = ArrayField(dtype=float, shape=(10000, 1, 1))
    file_path: str = make_parquet_file(
        columns={
            "Lepton": {"pt": template},
            "Jet": {"eta": template},
        },
        file_name="simple",
    )
    return file_path


@pytest.fixture
def config() -> MainConfig:
    with hydra.initialize(config_path="hydra_test_conf"):
        config_dict = hydra.compose(config_name="config")
        config: MainConfig = OmegaConf.structured(config_dict)
        config.datasets.max_number_events = 1000

    return config
