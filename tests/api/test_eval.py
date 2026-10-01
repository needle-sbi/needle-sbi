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
    _clean_state_dict,
    _resolve_devices,
    _VectorizedSiblings,
    aggregate_siblings,
    load_snapshot,
)
from needle.api.run import run
from needle.utils.config_schema import AggregationSpec
from tests.conftest import MainConfigFactory


def _write_snapshot(tmp_path: Path, flat: dict) -> None:
    with open(tmp_path / "dag_snapshot.json", "w") as f:
        json.dump(flat, f)


def _weighted_mean(
    outputs: list[torch.Tensor],
    metrics: list[float] | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Example user-defined aggregator, wired in via a dotted `AggregationSpec.method` path,
    implementing the `needle.api.eval.Aggregator` protocol. NEEDLE has no built-in "weighted_mean"
    and `AggregationSpec` carries no generic `weights` field on purpose: a custom aggregator that
    wants weights just captures them itself (a module-level constant here; a real project would
    more likely read them off its own config, or use `functools.partial`) instead of routing them
    through the framework.

    A real project would put this in its own installed package rather than a test module.
    """
    weights = [3.0, 1.0]  # arbitrary, hardcoded for this example
    stacked = torch.stack(outputs, dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()
    view_shape = (-1,) + (1,) * (stacked.dim() - 1)

    aggregated = (weight_tensor.view(view_shape) * stacked).sum(dim=0)
    std = torch.sqrt((weight_tensor.view(view_shape) * (stacked - aggregated) ** 2).sum(dim=0))
    return aggregated, std


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


class TestAggregate:
    def test_mean_is_the_default(self) -> None:
        outputs = [torch.ones(2, 1), torch.zeros(2, 1)]
        mean, std = aggregate_siblings(outputs)
        assert torch.allclose(mean, torch.full((2, 1), 0.5))
        assert torch.allclose(std, torch.stack(outputs, dim=0).std(dim=0))

    def test_sum(self) -> None:
        outputs = [torch.ones(2, 1), torch.ones(2, 1)]
        total, _ = aggregate_siblings(outputs, method="sum")
        assert torch.allclose(total, torch.full((2, 1), 2.0))

    def test_best_selects_lowest_metric(self) -> None:
        outputs = [torch.full((2, 1), 10.0), torch.full((2, 1), 20.0)]
        best, std = aggregate_siblings(outputs, method="best", metrics=[0.5, 0.1])
        assert torch.allclose(best, torch.full((2, 1), 20.0))
        assert torch.allclose(std, torch.zeros(2, 1))

    def test_best_requires_metrics(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="metrics required"):
            aggregate_siblings(outputs, method="best")

    def test_custom_aggregator_implements_weighted_mean(self) -> None:
        # NEEDLE has no built-in "weighted_mean" method; this shows how to add one yourself via a
        # dotted-path `method`, resolved by `aggregate_siblings`. See `_weighted_mean` above and
        # docs/concepts/hydra_config.md for the same example.
        outputs = [torch.zeros(2, 1), torch.full((2, 1), 4.0)]
        mean, _ = aggregate_siblings(outputs, method="tests.api.test_eval._weighted_mean")
        assert torch.allclose(mean, torch.full((2, 1), 1.0))

    def test_custom_aggregator_missing_attribute_raises_unknown_method(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_siblings(outputs, method="tests.api.test_eval._does_not_exist")

    def test_single_output_is_passthrough(self) -> None:
        output = torch.full((2, 1), 3.0)
        mean, std = aggregate_siblings([output])
        assert torch.equal(mean, output)
        assert torch.equal(std, torch.zeros_like(output))

    def test_unknown_method_raises(self) -> None:
        outputs = [torch.zeros(2, 1), torch.ones(2, 1)]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_siblings(outputs, method="median")

    def test_config_driven_spec_forwards_via_call_signature(self) -> None:
        # This is how `Estimator.forward` calls `aggregate_siblings`: reading `method`/`metric_key`
        # off an `AggregationSpec` straight from the resolved config.
        outputs = [torch.ones(2, 1), torch.zeros(2, 1)]
        spec = AggregationSpec(method="mean")
        mean, _ = aggregate_siblings(outputs, method=spec.method, metric_key=spec.metric_key)
        assert torch.allclose(mean, torch.full((2, 1), 0.5))


class TestCleanStateDict:
    def test_ignores_extra_unprefixed_keys(self) -> None:
        model = nn.Linear(2, 1)
        state = {f"model.{k}": v for k, v in model.state_dict().items()}
        state["loss.weight"] = torch.ones(1)
        cleaned = _clean_state_dict(state, model)
        assert set(cleaned) == set(model.state_dict())
        model.load_state_dict(cleaned)


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
