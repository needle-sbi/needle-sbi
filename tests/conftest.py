import os
import resource
from pathlib import Path
from typing import Callable, List, Protocol, cast

import awkward as ak
import hydra
import numpy as np
import pydantic
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from needle.etl.dask_ingestor import Ingestor
from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import resolve_defaults

type NestDictType = dict[str, "NestDictType | ArrayField"]


class ArrayField(pydantic.BaseModel):
    dtype: type
    shape: tuple[int, ...] = (100, 1, 1)


@pydantic.validate_call
def create_array_field(field_template: ArrayField):
    """Create a regular ak.Array with the desired shape and dtype"""
    numpy_array = np.pi * np.random.random(field_template.shape)
    numpy_array = numpy_array.astype(field_template.dtype)
    return ak.Array(numpy_array)


def create_nested_structure(columns: dict) -> ak.Array:
    """Template to create nested ak.Record structures from a nested dictionary"""
    array_dict = {}

    for key, value in columns.items():
        if isinstance(value, dict):
            array_dict[key] = create_nested_structure(value)
        elif isinstance(value, pydantic.BaseModel):
            # Use model_validate to handle the case where ArrayField is imported
            # under two different module paths (conftest vs tests.conftest).
            try:
                field = ArrayField.model_validate(value.model_dump())
            except pydantic.ValidationError:
                raise ValueError("Value must be either a dict or an ArrayField.")
            array_dict[key] = create_array_field(field)
        else:
            raise ValueError("Value must be either a dict or an ArrayField.")

    return ak.Array(array_dict)


@pytest.fixture
def make_parquet_file(tmp_path: Path) -> Callable[[NestDictType, str], str]:
    """Pytest fixture (wraps the inner function and provides the temporary path)"""

    def _make_parquet_file(
        columns: NestDictType,
        file_name: str = "test",
    ) -> str:
        """Create a parquet file with the desired structure"""
        array = create_nested_structure(columns)

        path = str(os.path.abspath(tmp_path / (file_name + ".parquet")))
        ak.to_parquet(array, path)
        return path

    return _make_parquet_file


@pytest.fixture
def ingestor(make_parquet_file: Callable) -> Ingestor:
    template = ArrayField(dtype=float, shape=(100, 1, 1))
    file = os.path.abspath(make_parquet_file(columns={"Lepton": {"pt": template}}, file_name="nested"))
    return Ingestor(paths=file)


def pytest_sessionstart(session: pytest.Session):
    """Enable the maximum amount of File Descriptors for the benchmarks

    Args:
        session (pytest.Session): Current pytest session
    """
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


@pytest.fixture
def simple_sample(make_parquet_file: Callable) -> str:
    template = ArrayField(dtype=float, shape=(10000, 1, 1))
    file_path: str = make_parquet_file(
        columns={
            "Lepton": {"pt": template},
            "Jet": {"eta": template},
        },
        file_name="simple",
    )
    return file_path


@pytest.fixture(scope="session")
def check_cli_path() -> Callable[[str], str]:
    def _check_cli_path(env_path: str, extension: str = None) -> str:
        path = os.getenv(env_path)

        if not path:
            pytest.skip(
                f"Environment variable '{env_path}' not set. Should point to {extension} files: "
                f"'export {env_path}=/path/to/files.{extension}'"
            )
        return path

    return _check_cli_path


@pytest.fixture()
def fair_universe_sample(check_cli_path) -> str:
    return check_cli_path("FAIR_UNIVERSE_DATA", "parquet")


@pytest.fixture()
def delphes_sample_root(check_cli_path) -> str:
    return check_cli_path("DELPHES_DATA_ROOT", "root")


@pytest.fixture()
def delphes_sample_parquet(check_cli_path) -> str:
    return check_cli_path("DELPHES_DATA_PARQUET", "parquet")


class MainConfigFactory(Protocol):
    def __call__(self, overrides: list[str] | None = None) -> MainConfig:
        ...


@pytest.fixture()
def config_factory() -> Callable[..., MainConfig]:
    """Create configs from the .yaml file together with the defaults from the corresponding
    dataclass.

    Returns:
        Callable[List[str] | None, MainConfig]: Factory to create new configs. Use the hydra `overrides`
            argument to replace a value from the .yaml with a new value.

    Example:
        Default config with no overrides

        ```python
        config: MainConfig = config_factory()
        ```

        Config with extra overrides

        ```python
        config: MainConfig = config_factory(overrides=["datasets=delphes"])
        ```

    Note:
        The schema of the overrides is determined by the hydra package, but follows mostly the str
        version of keyword assignment, e.g. `'<key>=<value>'` as a list of str.

    """

    def _factory(overrides: List[str] | None = None):
        with hydra.initialize(config_path="conf_tests"):
            cfg_dict = hydra.compose(config_name="config", overrides=overrides)
            cfg_defaults = OmegaConf.structured(MainConfig)
            cfg = OmegaConf.merge(cfg_defaults, cfg_dict)
            cfg_dir = Path(__file__).parent / "conf_tests"
            cfg = resolve_defaults(cfg, cfg_dir)
            return cast(MainConfig, cfg)

    return _factory


@pytest.fixture(scope="function")
def config(config_factory) -> MainConfig:
    return config_factory(overrides=None)


@pytest.fixture(scope="session")
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=2,
        memory_limit="20GB",
    )
    client = Client(cluster)

    yield client

    client.close()
    cluster.close()
