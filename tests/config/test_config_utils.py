import graphlib
from pathlib import Path
from typing import Generator

import hydra
import pytest

from needle.utils.config_schema import (
    DownstreamTaskConfig,
    EstimatorConfig,
    MainConfig,
    SystematicConfig,
)
from needle.utils.config_utils import NeedleConfigError, initialize_hydra_config, validate_graph

CONF_TESTS_DIR = Path(__file__).parent.parent / "conf_tests"


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


class TestHydraOverrides:
    """Regression tests for GH #12: overrides on fields absent from the raw YAML, and the
    override/sub-config precedence order documented in `docs/concepts/lightning_and_hydra_integration.md`.
    """

    def test_override_on_field_absent_from_yaml(self) -> None:
        # `model_B` in conf_tests/config.yaml declares no `dataset_override` at all - only the
        # schema default provides it - so this override key is not literally present in the file.
        cfg = initialize_hydra_config(
            config_dir=str(CONF_TESTS_DIR),
            config_name="config",
            overrides=["estimators.model_B.dataset_override.paths=/tmp/demo.parquet"],
        )

        assert cfg.estimators["model_B"].dataset_override.paths == "/tmp/demo.parquet"

    def test_manual_override_in_yaml_takes_precedence_over_sub_config(self) -> None:
        # `model_A` sets `dataset_override.labels_columns` in the yaml, while the referenced
        # `fair_universe` dataset sub-config sets its own (different) `labels_columns`.
        cfg = initialize_hydra_config(config_dir=str(CONF_TESTS_DIR), config_name="config")

        assert cfg.estimators["model_A"].dataset_override.labels_columns == ["PRI_lep_eta"]

    def test_runtime_override_takes_precedence_over_manual_yaml_override(self) -> None:
        cfg = initialize_hydra_config(
            config_dir=str(CONF_TESTS_DIR),
            config_name="config",
            overrides=["estimators.model_A.dataset_override.labels_columns=[PRI_n_jets]"],
        )

        assert cfg.estimators["model_A"].dataset_override.labels_columns == ["PRI_n_jets"]

    def test_unknown_override_key_still_raises(self) -> None:
        with pytest.raises(NeedleConfigError, match="Unknown config key"):
            initialize_hydra_config(
                config_dir=str(CONF_TESTS_DIR),
                config_name="config",
                overrides=["estimators.model_A.doesnotexist=1"],
            )
