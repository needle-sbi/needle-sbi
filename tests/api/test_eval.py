"""Tests for needle/api/eval.py.

`load_snapshot`/`aggregate_siblings` are exercised directly (fast, no training). The full
`Estimator` load-and-forward path is exercised end-to-end in `TestEstimatorEndToEnd`,
which trains a tiny real DAG via the b2luigi `MainTask` (in-process, `local_scheduler`)
against the bundled `fair_universe_demo_parquet` fixture, then loads the resulting
`dag_snapshot.json`/`config.yaml` back through `needle.api.eval.Estimator`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

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

    def test_rejects_mismatched_parameter_shapes(self) -> None:
        models = [nn.Linear(3, 2).eval(), nn.Linear(3, 4).eval()]
        with pytest.raises((ValueError, RuntimeError), match="equal size|shape"):
            _VectorizedSiblings(["a", "b"], models, torch.device("cpu"))

    def test_stacked_weights_are_copies_of_the_siblings(self) -> None:
        models = [nn.Linear(3, 2).eval() for _ in range(2)]
        group = _VectorizedSiblings(["a", "b"], models, torch.device("cpu"))
        x = torch.rand(4, 3)
        before = group(x)["a"].clone()
        with torch.no_grad():
            models[0].weight.add_(1.0)
        assert torch.equal(group(x)["a"], before)

    def test_follows_dtype_conversion_of_the_module(self) -> None:
        models = [nn.Linear(3, 2).eval() for _ in range(2)]
        group = _VectorizedSiblings(["a", "b"], models, torch.device("cpu")).double()
        out = group(torch.rand(4, 3, dtype=torch.float64))
        assert out["a"].dtype == torch.float64


def _constant_model(value: float, n_in: int = 3, n_out: int = 1) -> nn.Module:
    """A linear model that outputs `value` for every input."""
    model = nn.Linear(n_in, n_out)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(value)
    return model.eval()


def _stub_estimator(
    values: dict[str, dict[int, dict[int, float]]],
    scores: dict[str, float] | None = None,
    systematic_aggregation: AggregationSpec | None = None,
    num_workers: int = 2,
) -> Estimator:
    """An `Estimator` built without checkpoints: `values[systematic][ensemble][fold]` is the constant that
    leaf outputs, `scores` maps a leaf key to its "val_loss". The real `forward`/aggregation code runs.
    """
    est = object.__new__(Estimator)
    nn.Module.__init__(est)
    spec = AggregationSpec()
    est.execution = "sequential"
    est.num_workers = num_workers
    est.estimator_config = SimpleNamespace(  # type: ignore[assignment]
        expands=SimpleNamespace(folds=SimpleNamespace(aggregation=spec), ensembles=SimpleNamespace(aggregation=spec)),
        systematic_aggregation=systematic_aggregation or spec,
    )
    est.snapshot = {s: {e: {f: "unused" for f in folds} for e, folds in ens.items()} for s, ens in values.items()}
    est.models = nn.ModuleDict()
    est._scores = {}
    est._vectorized_groups = nn.ModuleDict()
    for leaf in est._leaves(list(est.snapshot)):
        est.models[leaf.key] = _constant_model(values[leaf.systematic][leaf.ensemble][leaf.fold])
        est._scores[leaf.key] = {"val_loss": (scores or {}).get(leaf.key, 1.0)}
    return est


class TestEstimatorForward:
    """`Estimator.forward` on hand-built leaves (no training), so every number is checkable by hand."""

    def test_folds_ensembles_and_systematics_are_averaged_bottom_up(self) -> None:
        est = _stub_estimator(
            {"nominal": {0: {0: 1.0, 1: 3.0}, 1: {0: 5.0}}, "shifted": {0: {0: 10.0}}},
        )
        mean, _ = est(torch.rand(4, 3))
        # nominal: ensemble 0 -> mean(1, 3) = 2, ensemble 1 -> 5, so mean(2, 5) = 3.5; shifted -> 10
        assert torch.allclose(mean, torch.full((4, 1), (3.5 + 10.0) / 2))

    @pytest.mark.parametrize("execution", ["sequential", "parallel", "vectorized"])
    def test_execution_modes_agree_on_exact_values(self, execution: str) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}, 1: {0: 5.0, 1: 9.0}}, "shifted": {0: {0: 10.0}}})
        x = torch.rand(4, 3)
        reference_mean, reference_std = est(x)
        mean, std = est(x, execution=execution)
        assert torch.allclose(mean, reference_mean) and torch.allclose(std, reference_std)

    def test_best_systematic_uses_the_lowest_score(self) -> None:
        est = _stub_estimator(
            {"nominal": {0: {0: 1.0}}, "shifted": {0: {0: 10.0}}},
            scores={"syst=nominal&ensem=0&fold=0": 0.9, "syst=shifted&ensem=0&fold=0": 0.2},
            systematic_aggregation=AggregationSpec(method="best", metric_key="val_loss"),
        )
        mean, std = est(torch.rand(2, 3))
        assert torch.allclose(mean, torch.full((2, 1), 10.0))
        assert torch.equal(std, torch.zeros(2, 1))

    def test_systematics_keys_selects_exactly_the_named_systematics(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0}}, "shifted": {0: {0: 10.0}}})
        mean, _ = est(torch.rand(2, 3), systematics_keys=["shifted"])
        assert torch.allclose(mean, torch.full((2, 1), 10.0))

    def test_systematics_keys_as_a_string_is_not_a_substring_match(self) -> None:
        # `s in "up_qcd"` is a substring test, so the systematic "up" is silently selected as well.
        est = _stub_estimator({"up": {0: {0: 1.0}}, "up_qcd": {0: {0: 9.0}}})
        try:
            mean, _ = est(torch.rand(2, 3), systematics_keys="up_qcd")  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        assert torch.allclose(mean, torch.full((2, 1), 9.0)), "systematics_keys='up_qcd' also aggregated 'up'"

    def test_unknown_systematic_among_valid_ones_raises(self) -> None:
        # A typo next to a valid name is silently dropped, so the caller gets a different systematic mix.
        est = _stub_estimator({"nominal": {0: {0: 1.0}}, "shifted": {0: {0: 10.0}}})
        with pytest.raises(ValueError, match="typo"):
            est(torch.rand(2, 3), systematics_keys=["shifted", "typo"])

    def test_empty_systematics_keys_raises(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0}}})
        with pytest.raises(ValueError, match="No systematics matched"):
            est(torch.rand(2, 3), systematics_keys=[])

    def test_unknown_execution_mode_raises_at_call_time(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0}}})
        with pytest.raises(ValueError, match="Unknown execution mode"):
            est(torch.rand(2, 3), execution="threaded")  # type: ignore[arg-type]

    def test_std_reports_the_spread_between_ensemble_members(self) -> None:
        # With a single trained systematic the outer aggregation has one sibling and `aggregate_siblings`
        # passes it through with std = 0, discarding the real fold/ensemble spread. Documented as
        # "intermediate uncertainties are not propagated", but the spread of the ensemble is what a user
        # of `Estimator` expects as the uncertainty.
        est = _stub_estimator({"nominal": {0: {0: 1.0}, 1: {0: 3.0}}})
        _, std = est(torch.rand(2, 3))
        assert torch.all(std > 0), "std is identically zero although the two ensemble members disagree"

    def test_does_not_modify_input_and_returns_detached_outputs(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 2.0}}})
        x = torch.rand(3, 3, requires_grad=True)
        before = x.detach().clone()
        for execution in ("sequential", "parallel", "vectorized"):
            mean, std = est(x, execution=execution)
            assert not mean.requires_grad and not std.requires_grad, execution
        assert torch.equal(x.detach(), before)

    def test_zero_events_pass_through_every_execution_mode(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 2.0}}, "shifted": {0: {0: 3.0}}})
        for execution in ("sequential", "parallel", "vectorized"):
            mean, std = est(torch.zeros(0, 3), execution=execution)
            assert mean.shape == std.shape == (0, 1), execution

    def test_nan_input_event_does_not_contaminate_other_events(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}})
        x = torch.rand(3, 3)
        x[1, 0] = float("nan")
        mean, _ = est(x)
        assert torch.isnan(mean[1]).all()
        assert torch.allclose(mean[[0, 2]], torch.full((2, 1), 2.0))

    def test_float64_input_gives_a_clear_error_or_works(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}})
        try:
            mean, _ = est(torch.rand(2, 3, dtype=torch.float64))
        except (TypeError, ValueError, RuntimeError) as exc:
            assert "dtype" in str(exc)
        else:
            assert torch.allclose(mean.double(), torch.full((2, 1), 2.0, dtype=torch.float64))

    def test_parallel_with_zero_workers_fails_with_a_clear_error(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}}, num_workers=0)
        with pytest.raises(ValueError, match="max_workers|num_workers"):
            est(torch.rand(2, 3), execution="parallel")

    def test_train_mode_does_not_leak_into_inference(self) -> None:
        # `Estimator.train()` is the usual `nn.Module` call that ends up in user pipelines; the leaves
        # contain dropout here and must stay deterministic.
        est = _stub_estimator({"nominal": {0: {0: 1.0}}})
        est.models["syst=nominal&ensem=0&fold=0"] = nn.Sequential(nn.Dropout(0.5), nn.Linear(3, 1)).eval()
        est.train()
        x = torch.rand(64, 3)
        first, _ = est(x)
        second, _ = est(x)
        assert torch.equal(first, second), "Estimator.train() made inference stochastic"

    def test_state_dict_round_trip_restores_outputs(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}})
        other = _stub_estimator({"nominal": {0: {0: 7.0, 1: 9.0}}})
        other.load_state_dict(est.state_dict())
        x = torch.rand(2, 3)
        assert torch.equal(est(x)[0], other(x)[0])

    def test_deepcopy_with_cached_vectorized_group_is_independent(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}})
        x = torch.rand(2, 3)
        reference = est(x, execution="vectorized")[0]
        clone = copy.deepcopy(est)
        assert torch.equal(clone(x, execution="vectorized")[0], reference)
        assert clone._vectorized_groups["nominal"]._template[0] is clone.models["syst=nominal&ensem=0&fold=0"]


class TestEstimatorDevice:
    def test_device_follows_the_first_model(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0}}})
        assert est.device == torch.device("cpu")
        est.to("meta")
        assert est.device == torch.device("meta")

    def test_vectorized_group_is_moved_with_the_estimator_dtype(self) -> None:
        est = _stub_estimator({"nominal": {0: {0: 1.0, 1: 3.0}}})
        est(torch.rand(2, 3), execution="vectorized")  # builds and caches the group
        est.double()
        mean, _ = est(torch.rand(2, 3, dtype=torch.float64), execution="vectorized")
        assert mean.dtype == torch.float64


class TestResolveDevicesAttacks:
    @pytest.mark.parametrize("n_gpus", [0, -3])
    def test_non_positive_n_gpus_still_gives_one_device(self, n_gpus: int) -> None:
        assert _resolve_devices("cpu", n_gpus=n_gpus) == [torch.device("cpu")]

    def test_unavailable_accelerator_fails_with_a_clear_error(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, nothing to provoke")
        with pytest.raises((RuntimeError, ValueError), match="NVIDIA|cuda|CUDA|available"):
            _resolve_devices("cuda", n_gpus=1)


class TestCleanStateDictAttacks:
    def test_matching_keys_are_returned_unchanged(self) -> None:
        model = nn.Linear(2, 1)
        state = model.state_dict()
        assert _clean_state_dict(state, model) is state

    @pytest.mark.parametrize("prefix", ["model.", "module.", "_orig_mod."])
    def test_strips_a_single_wrapper_prefix(self, prefix: str) -> None:
        model = nn.Linear(2, 1)
        state = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
        assert set(_clean_state_dict(state, model)) == set(model.state_dict())

    def test_resolves_compiled_model_inside_lightning_module(self) -> None:
        # `torch.compile(self.model)` inside a LightningModule saves keys as `model._orig_mod.<name>`.
        wrapper = nn.Module()
        wrapper.model = nn.Linear(2, 1)
        state = {k.replace("model.", "model._orig_mod.", 1): v for k, v in wrapper.state_dict().items()}
        wrapper.load_state_dict(_clean_state_dict(state, wrapper))

    def test_unresolvable_mismatch_is_left_for_load_state_dict_to_report(self) -> None:
        model = nn.Linear(2, 1)
        state = {"foo": torch.zeros(1)}
        assert _clean_state_dict(state, model) is state
        with pytest.raises(RuntimeError, match="Missing key"):
            model.load_state_dict(_clean_state_dict(state, model))


class TestCheckpointScoresAttacks:
    def test_nan_score_is_not_reported_as_a_valid_score(self) -> None:
        checkpoint = {"callbacks": {"mc": {"monitor": "val_loss", "best_model_score": torch.tensor(float("nan"))}}}
        assert "val_loss" not in _checkpoint_scores(checkpoint), "a diverged run's NaN score would win 'best'"

    def test_monitor_without_a_best_score_is_skipped(self) -> None:
        checkpoint = {"callbacks": {"mc": {"monitor": "val_loss", "best_model_score": None}}}
        assert _checkpoint_scores(checkpoint) == {}

    def test_scores_of_different_metrics_are_kept_apart(self) -> None:
        checkpoint = {
            "callbacks": {
                "a": {"monitor": "val_loss", "best_model_score": torch.tensor(0.1)},
                "b": {"monitor": "val_auc", "best_model_score": torch.tensor(0.9)},
            }
        }
        assert _checkpoint_scores(checkpoint) == {"val_loss": pytest.approx(0.1), "val_auc": pytest.approx(0.9)}


class TestLoadSnapshotAttacks:
    def test_systematic_names_with_url_special_characters_survive(self, tmp_path: Path) -> None:
        # Keys are written as `est=<name>&syst=<name>&...` without any quoting (`snapshot_as_dict`), but read back
        # through `parse_qsl`, which turns '+' into ' ' and decodes '%xx'.
        name = "jes_up+1sigma"
        _write_snapshot(tmp_path, {f"est=model_A&syst={name}&ensem=0&fold=0": "ckpt.ckpt"})
        assert set(load_snapshot(tmp_path, "model_A")) == {name}

    def test_unknown_systematic_lookup_does_not_mutate_result(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, {"est=model_A&syst=nominal&ensem=0&fold=0": "ckpt.ckpt"})
        nested = load_snapshot(tmp_path, "model_A")
        with pytest.raises(KeyError):
            nested["typo"]
        assert set(nested) == {"nominal"}


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
