import os
from typing import Callable, List, cast

import hydra
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


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
def config_factory() -> Callable[[None], MainConfig]:
    """Create configs from the .yaml file together with the defaults from the corresponding
    dataclass.

    Returns:
        Callable[[None], MainConfig]: Factory to create new configs. Use the hydra `overrides`
            argument to replace a value from the .yaml with a new value.
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


@pytest.fixture(scope="session")
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        memory_limit="10GB",
    )
    client = Client(cluster)
    client.run(lambda: None)

    yield client

    client.close()
    cluster.close()
