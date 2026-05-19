import graphlib
from typing import Generator

import hydra
import pytest

from needle.utils.config_schema import EstimatorConfig, MainConfig
from needle.utils.config_utils import validate_graph


class TestValidateGraph:
    def test_no_cycles_or_missing_dependencies(self) -> None:
        cfg = MainConfig(
            estimators={
                "a": EstimatorConfig(),
                "b": EstimatorConfig(requires=["a"]),
                "c": EstimatorConfig(requires=["b"]),
            }
        )

        validate_graph(cfg)

    def test_missing_dependency_raises_value_error(self) -> None:
        cfg = MainConfig(
            estimators={
                "a": EstimatorConfig(requires=["missing"]),
            }
        )

        with pytest.raises(ValueError, match="depends on undefined estimators"):
            validate_graph(cfg)

    def test_cycle_raises_cycle_error(self) -> None:
        cfg = MainConfig(
            estimators={
                "a": EstimatorConfig(requires=["b"]),
                "b": EstimatorConfig(requires=["a"]),
            }
        )

        with pytest.raises(graphlib.CycleError):
            validate_graph(cfg)


@pytest.fixture(scope="function")
def hydra_initialize_context() -> Generator:
    with hydra.initialize(config_path="../conf_tests"):
        yield


class TestResolveDefaults:
    # TODO
    pass
