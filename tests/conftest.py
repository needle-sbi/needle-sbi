import os
import resource
from typing import Callable, List, cast

import hydra
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


def pytest_sessionstart(session: pytest.Session):
    """Enable the maximum amount of File Descriptors for the benchmarks

    Args:
        session (pytest.Session): Current pytest session
    """
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


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


@pytest.fixture(scope="session")
def check_cli_path() -> Callable[[str], str]:
    def _check_cli_path(env_path: str, extension: str = None) -> str:
        path = os.getenv(env_path)

        if not path:
            pytest.skip(
                f"Environment variable '{env_path}' not set. Should point to {extension} files: "
                f"'export {env_path}=<path/to/files.{extension}'"
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
        with hydra.initialize(config_path="hydra_test_conf"):
            cfg_dict = hydra.compose(config_name="config", overrides=overrides)
            cfg_defaults = OmegaConf.structured(MainConfig)
            cfg = OmegaConf.merge(cfg_defaults, cfg_dict)
            return cast(MainConfig, cfg)

    return _factory


@pytest.fixture(scope="function")
def config(config_factory) -> MainConfig:
    return config_factory(overrides=None)
