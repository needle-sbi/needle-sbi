import graphlib
from pathlib import Path
from typing import Generator

import hydra
import pytest
from omegaconf import OmegaConf

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
    """Precedence is schema defaults < referenced sub-config < YAML < runtime overrides."""

    @staticmethod
    def _load(overrides: list[str] | None = None) -> MainConfig:
        return initialize_hydra_config(str(CONF_TESTS_DIR), "config", overrides)

    def test_override_on_field_absent_from_yaml(self) -> None:
        # model_B declares no `dataset_override` in the YAML
        cfg = self._load(["estimators.model_B.dataset_override.paths=/tmp/demo.parquet"])

        assert cfg.estimators["model_B"].dataset_override.paths == "/tmp/demo.parquet"

    def test_schema_defaults_do_not_overwrite_sub_config(self) -> None:
        cfg = self._load()
        sub_cfg = OmegaConf.load(CONF_TESTS_DIR / "datasets" / "fair_universe.yaml")

        assert cfg.estimators["model_B"].dataset_override.features_columns == sub_cfg.features_columns
        assert cfg.estimators["model_B"].dataset_override.max_number_events == sub_cfg.max_number_events

    def test_yaml_takes_precedence_over_sub_config(self) -> None:
        cfg = self._load()

        assert cfg.estimators["model_A"].dataset_override.labels_columns == ["PRI_lep_eta"]

    def test_override_takes_precedence_over_yaml(self) -> None:
        cfg = self._load(["estimators.model_A.dataset_override.labels_columns=[PRI_n_jets]"])

        assert cfg.estimators["model_A"].dataset_override.labels_columns == ["PRI_n_jets"]

    def test_override_of_group_name_reloads_sub_config(self) -> None:
        cfg = self._load(["estimators.model_B.dataset=delphes"])
        sub_cfg = OmegaConf.load(CONF_TESTS_DIR / "datasets" / "delphes.yaml")

        assert cfg.estimators["model_B"].dataset_override.features_columns == sub_cfg.features_columns

    def test_hydra_add_and_delete_syntax(self) -> None:
        cfg = self._load(["+estimators.model_B.dataset_override.paths=/tmp/a.parquet", "~estimators.model_B.requires"])

        assert cfg.estimators["model_B"].dataset_override.paths == "/tmp/a.parquet"
        assert not cfg.estimators["model_B"].requires

    def test_unknown_override_key_raises(self) -> None:
        with pytest.raises(NeedleConfigError, match="doesnotexist"):
            self._load(["estimators.model_A.doesnotexist=1"])
