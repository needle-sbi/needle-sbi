import pytest
from pathlib import Path
import os


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
