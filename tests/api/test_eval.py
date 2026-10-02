"""Tests for needle/api/eval.py.

`load_snapshot`/`aggregate_siblings` are exercised directly (fast, no training). The full
`Estimator` load-and-forward path is exercised end-to-end in `TestEstimatorEndToEnd`,
which trains a tiny real DAG via the b2luigi `MainTask` (in-process, `local_scheduler`)
against the bundled `fair_universe_demo_parquet` fixture, then loads the resulting
`dag_snapshot.json`/`config.yaml` back through `needle.api.eval.Estimator`.
"""

from __future__ import annotations

import json
from pathlib import Path

import omegaconf
import pytest
import torch
import torch.nn as nn

from needle.api.eval import (
    Estimator,
    _checkpoint_scores,
    _clean_state_dict,
    _Node,
    _resolve_devices,
    _VectorizedSiblings,
    load_snapshot,
)
from needle.api.run import run
from needle.utils.config_schema import AggregationSpec
from tests.conftest import MainConfigFactory


def _write_snapshot(tmp_path: Path, flat: dict) -> None:
    with open(tmp_path / "dag_snapshot.json", "w") as f:
        json.dump(flat, f)


def test_load_snapshot_unflattens_single_estimator(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        {
            "est=model_A&syst=nominal&ensem=0&fold=0": "ckpt_a_0_0.ckpt",
            "est=model_A&syst=nominal&ensem=0&fold=1": "ckpt_a_0_1.ckpt",
            "est=model_A&syst=nominal&ensem=1&fold=0": "ckpt_a_1_0.ckpt",
            "est=model_B&syst=nominal&ensem=0&fold=0": "ckpt_b_0_0.ckpt",
        },
    )

    nested = load_snapshot(tmp_path, "model_A")

    assert set(nested.keys()) == {"nominal"}
    assert nested["nominal"][0] == {0: "ckpt_a_0_0.ckpt", 1: "ckpt_a_0_1.ckpt"}
    assert nested["nominal"][1] == {0: "ckpt_a_1_0.ckpt"}


def test_load_snapshot_raises_for_unknown_estimator(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, {"est=model_A&syst=nominal&ensem=0&fold=0": "ckpt.ckpt"})

    with pytest.raises(KeyError, match="model_B"):
        load_snapshot(tmp_path, "model_B")


def test_load_snapshot_raises_if_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dag_snapshot.json"):
        load_snapshot(tmp_path, "model_A")


class TestCleanStateDict:
    def test_ignores_extra_unprefixed_keys(self) -> None:
        model = nn.Linear(2, 1)
        state = {f"model.{k}": v for k, v in model.state_dict().items()}
        state["loss.weight"] = torch.ones(1)
        cleaned = _clean_state_dict(state, model)
        assert set(cleaned) == set(model.state_dict())
        model.load_state_dict(cleaned)


class TestCheckpointScores:
    def test_reads_best_score_of_monitored_metric(self) -> None:
        checkpoint = {
            "callbacks": {
                "ModelCheckpoint{...}": {"monitor": "val_loss", "best_model_score": torch.tensor(0.25)},
                "EarlyStopping{...}": {"wait_count": 1},
            }
        }
        assert _checkpoint_scores(checkpoint) == {"val_loss": 0.25}

    def test_no_callbacks_gives_no_scores(self) -> None:
        assert _checkpoint_scores({"state_dict": {}}) == {}


class TestSiblingMetrics:
    @staticmethod
    def _nodes(*scores: dict) -> list[_Node]:
        return [_Node(torch.zeros(1), s) for s in scores]

    def test_looks_up_metric_key_per_sibling(self) -> None:
        nodes = self._nodes({"val_loss": 0.5}, {"val_loss": 0.1})
        metrics = Estimator._sibling_metrics(nodes, AggregationSpec(method="best", metric_key="val_loss"))
        assert metrics == [0.5, 0.1]

    def test_forwarded_to_any_method_when_metric_key_is_set(self) -> None:
        nodes = self._nodes({"val_loss": 0.5}, {"val_loss": 0.1})
        for method in ("mean", "my_package.aggregators.weighted_mean"):
            spec = AggregationSpec(method=method, metric_key="val_loss")
            assert Estimator._sibling_metrics(nodes, spec) == [0.5, 0.1]

    def test_not_needed_without_metric_key_or_for_a_lone_sibling(self) -> None:
        nodes = self._nodes({}, {})
        assert Estimator._sibling_metrics(nodes, AggregationSpec(method="mean")) is None
        assert Estimator._sibling_metrics(nodes, AggregationSpec(method="my_package.aggregators.f")) is None
        assert Estimator._sibling_metrics(nodes[:1], AggregationSpec(method="best")) is None

    @pytest.mark.parametrize(
        "method, metric_key", [("best", None), ("best", "missing"), ("mean", "missing"), ("pkg.custom", "missing")]
    )
    def test_missing_metric_raises_a_clear_error(self, method: str, metric_key: str | None) -> None:
        nodes = self._nodes({"val_loss": 0.5}, {"val_loss": 0.1})
        with pytest.raises(ValueError, match="metric_key"):
            Estimator._sibling_metrics(nodes, AggregationSpec(method=method, metric_key=metric_key))


class TestResolveDevices:
    def test_cpu_is_single_device(self) -> None:
        assert _resolve_devices("cpu", n_gpus=4) == [torch.device("cpu")]

    def test_default_resolves_to_a_device(self) -> None:
        devices = _resolve_devices(None, n_gpus=1)
        assert len(devices) == 1


class TestVectorizedEnsemble:
    def test_matches_individual_models_and_registers_buffers(self) -> None:
        models = [nn.Linear(3, 2).eval() for _ in range(3)]
        keys = ["a", "b", "c"]
        group = _VectorizedSiblings(keys, models, torch.device("cpu"))
        x = torch.rand(4, 3)
        out = group(x)
        for key, model in zip(keys, models):
            assert torch.allclose(out[key], model(x), atol=1e-6)
        assert len(list(group.buffers())) == 2  # weight + bias stacks follow `.to()`
        assert len(list(group.modules())) == 1  # the template model is not registered a second time

    def test_rejects_differing_buffers(self) -> None:
        models = [nn.BatchNorm1d(3).eval() for _ in range(2)]
        models[1].running_mean.fill_(5.0)
        with pytest.raises(ValueError, match="differing buffers"):
            _VectorizedSiblings(["a", "b"], models, torch.device("cpu"))


@pytest.mark.b2luigi
@pytest.mark.slow
class TestEstimatorEndToEnd:
    def test_forward_returns_mean_and_std_matching_output_shape(
        self,
        config_factory: MainConfigFactory,
        tmp_path: Path,
        fair_universe_demo_parquet: Path,
    ) -> None:
        estimator_name = "model_A"
        config = config_factory()
        dataset_config = config.estimators[estimator_name].dataset_override
        assert dataset_config
        dataset_config.paths = str(fair_universe_demo_parquet)
        # MainTask trains every estimator in the DAG, not just `estimator_name` - model_B
        # (declared in tests/conf_tests/config.yaml with `requires: ["model_A"]`) also needs a
        # real data path, otherwise its TrainingTask fails on the default empty `paths: ""`.
        other_dataset_config = config.estimators["model_B"].dataset_override
        assert other_dataset_config
        other_dataset_config.paths = str(fair_universe_demo_parquet)
        config._resolved = True
        config_file = tmp_path / "config.yaml"
        omegaconf.OmegaConf.save(config, config_file, resolve=True)

        main = run(
            task="MainTask",
            config_file=config_file,
            results_path=str(tmp_path),
        )
        assert main

        model = Estimator(tmp_path, estimator_name)
        assert isinstance(model, Estimator)
        assert len(model.models) == 4  # 2 ensembles * 2 folds, one systematic (only "up_qcd" is configured)
        assert dataset_config.features_columns

        x = torch.rand(5, len(dataset_config.features_columns))
        mean, std = model(x)

        assert mean.shape == std.shape
        assert mean.shape[0] == 5

        # All execution modes must agree with the sequential reference.
        for mode in ("parallel", "vectorized"):
            mode_mean, mode_std = model(x, execution=mode)
            assert torch.allclose(mode_mean, mean, atol=1e-5), mode
            assert torch.allclose(mode_std, std, atol=1e-5), mode

        with pytest.raises(ValueError, match="No systematics matched"):
            model(x, systematics_keys=["does_not_exist"])
