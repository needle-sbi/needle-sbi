import graphlib
from typing import Generator

import hydra
import pytest

from needle.utils.config_schema import (
    DownstreamTaskConfig,
    EstimatorConfig,
    MainConfig,
    SystematicConfig,
)
from needle.utils.config_utils import validate_graph


class TestResourcesField:
    """`resources` is a plain, unvalidated dict - just verify it round-trips through the schema."""

    def test_estimator_config_defaults_to_none(self) -> None:
        assert EstimatorConfig().resources is None

    def test_estimator_config_accepts_arbitrary_dict(self) -> None:
        cfg = EstimatorConfig(resources={"RequestMemory": 2048, "RequestCpus": 2})
        assert cfg.resources == {"RequestMemory": 2048, "RequestCpus": 2}

    def test_systematic_config_accepts_arbitrary_dict(self) -> None:
        cfg = SystematicConfig(resources={"RequestMemory": 8192})
        assert cfg.resources == {"RequestMemory": 8192}

    def test_downstream_task_config_accepts_arbitrary_dict(self) -> None:
        cfg = DownstreamTaskConfig(resources={"request_cpus": 4})
        assert cfg.resources == {"request_cpus": 4}


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
