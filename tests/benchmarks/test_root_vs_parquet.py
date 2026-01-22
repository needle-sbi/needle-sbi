"""
Compare the speed up between root and parquet files on the same Delphes dataset. Requires that
the Environment variable for both the root and parquet version of the datasets are defined. Will
convert the root files to parquet if not already done so.

Disclaimer: Part of this code was written with the help of GPT-5

Run these tests using the following command:

```python3
pytest --benchmark-only -s
```

The pytest mark `benchmark` is automatically added with the pytest fixture of the same name. This
test suite requires the specific Delphes dataset from KIT. There are two environment variables to
set:

```
export DELPHES_DATA_ROOT=/path/to/*.root
export DELPHES_DATA_PARQUET=/path/to/*.parquet
```

These must be a glob pattern of all the files. The columns and other configs are read from the
dedicated test `hydra_test_conf` directory for all tests. In that config it is not mandatory to set
the paths to the datasets because they are overwritten by the two environment variables mentioned
above.
"""

from pathlib import Path
from typing import Callable

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from orchestrator.config import MainConfig

pytest.importorskip("preprocessor", reason="Could not import 'preprocessor'")
from preprocessor.ingestion.formatter import Ingestor  # noqa: E402
from preprocessor.utils.conversion import convert_root_to_parquet  # noqa: E402


def run_test(config: MainConfig) -> Callable:
    def _run_test():
        ingestor = Ingestor(
            paths=config.datasets.paths,
            format="automatic",
            columns=config.datasets.features_columns,
            max_number_events=config.datasets.max_number_events,
            reader_kwargs=config.datasets.dak_reader_kwargs,
        )
        ingestor.array

    return _run_test


def test_root_speed(
    benchmark: BenchmarkFixture,
    delphes_sample_root: str,
    config_factory,
) -> None:
    config: MainConfig = config_factory(overrides=["datasets=delphes"])

    if delphes_sample_root:
        config.datasets.paths = delphes_sample_root
    benchmark(run_test(config=config))


def test_parquet_speed(
    benchmark: BenchmarkFixture,
    delphes_sample_root: str,
    delphes_sample_parquet: str,
    config_factory,
) -> None:
    """Test the speed of reading from parquet files. Will first convert the dataset to parquet if not
    already done.
    """
    convert_root_to_parquet(
        delphes_sample_root,
        Path(delphes_sample_parquet).parent,
        drop_branches=["ref", "fName", "fSize", "fP", "fE", "fBits"],
    )
    config: MainConfig = config_factory(overrides=["datasets=delphes"])

    if delphes_sample_parquet:
        config.datasets.paths = delphes_sample_parquet

    benchmark(run_test(config=config))
