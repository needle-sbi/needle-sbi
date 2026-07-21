"""Tests for the b2luigi workflow backend.

Verifies task DAG structure, output paths, parameter handling, and downstream
branching without executing actual pipeline runs (no training, no file I/O).

All tests are marked ``@pytest.mark.b2luigi`` and excluded from the default
test run (see ``pyproject.toml`` ``addopts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlencode

import omegaconf
import pytest

from needle.b2luigi_tasks.downstream import DownstreamTask
from needle.b2luigi_tasks.ensemble import EnsembleTask
from needle.b2luigi_tasks.estimator import EstimatorTask
from needle.b2luigi_tasks.fold import FoldTask
from needle.b2luigi_tasks.main import MainTask
from needle.b2luigi_tasks.snapshot import SnapshotTask
from needle.b2luigi_tasks.systematic import SystematicTask
from needle.utils.config_schema import MainConfig
from tests.conftest import MainConfigFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(config: MainConfig, tmp_path: Path) -> Path:
    """Persist a MainConfig to a YAML file and return its path."""
    config._resolved = True
    config_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_file, resolve=True)
    return config_file


# ---------------------------------------------------------------------------
# FoldTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiFoldTask:
    def test_output_paths(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        fold = FoldTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=2,
            results_path=str(tmp_path),
        )

        out = fold.output()

        assert "ckpt" in out
        assert "model_config" in out
        assert "outputs" in out
        assert "input_models" in out

        ckpt_path = Path(out["ckpt"].path)
        assert ckpt_path.name == "model.ckpt"
        assert f"est__{estimator_name}" in str(ckpt_path)
        assert "syst__nominal" in str(ckpt_path)
        assert "ensem__0" in str(ckpt_path)
        assert "fold__2" in str(ckpt_path)

    def test_output_as_dict_is_identity(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        fold = FoldTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        out = fold.output()
        assert fold.output_as_dict(out) is out

    def test_requires_empty_without_deps(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        fold = FoldTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        assert fold.requires() == []

    def test_estimator_task_class_returns_b2luigi_estimator(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        fold = FoldTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        assert fold._estimator_task_class() is EstimatorTask


# ---------------------------------------------------------------------------
# EnsembleTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiEnsembleTask:
    def test_requires_creates_fold_tasks(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]
        config = config_factory()
        n_folds = config.estimators[estimator_name].expands.folds

        ensemble = EnsembleTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            results_path=str(tmp_path),
        )
        deps = ensemble.requires()

        assert len(deps) == n_folds
        assert all(isinstance(d, FoldTask) for d in deps)
        assert [d.fold_index for d in deps] == list(range(n_folds))

    def test_output_path(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        ensemble = EnsembleTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=1,
            results_path=str(tmp_path),
        )
        out = ensemble.output()
        assert "outputs" in out
        assert Path(out["outputs"].path).name == "ensemble_results.json"
        assert "ensem__1" in out["outputs"].path


# ---------------------------------------------------------------------------
# SystematicTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiSystematicTask:
    def test_requires_creates_ensemble_tasks(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]
        config = config_factory()
        num_ensembles = max(1, config.estimators[estimator_name].expands.ensembles.num_ensembles or 1)

        syst = SystematicTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            results_path=str(tmp_path),
        )
        deps = syst.requires()

        assert len(deps) == num_ensembles
        assert all(isinstance(d, EnsembleTask) for d in deps)

    def test_output_path(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        syst = SystematicTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            results_path=str(tmp_path),
        )
        out = syst.output()
        assert "outputs" in out
        assert Path(out["outputs"].path).name == "systematic_results.json"


# ---------------------------------------------------------------------------
# EstimatorTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiEstimatorTask:
    def test_requires_creates_systematic_tasks(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        est = EstimatorTask(
            config_file=config_file,
            estimator=estimator_name,
            results_path=str(tmp_path),
        )
        deps = est.requires()

        assert len(deps) >= 1
        assert all(isinstance(d, SystematicTask) for d in deps)

    def test_output_paths(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        est = EstimatorTask(
            config_file=config_file,
            estimator=estimator_name,
            results_path=str(tmp_path),
        )
        out = est.output()
        assert "outputs" in out
        assert "input_models" in out
        assert Path(out["outputs"].path).name == "estimator_result.json"
        assert f"est__{estimator_name}" in out["outputs"].path


# ---------------------------------------------------------------------------
# MainTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiMainTask:
    def test_requires_creates_estimator_tasks(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        config = config_factory()
        n_estimators = len(config.estimators)

        main = MainTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        deps = main.requires()

        assert len(deps) == n_estimators
        assert all(isinstance(d, EstimatorTask) for d in deps)

    def test_estimator_names_match_config(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        config = config_factory()
        expected_names = set(config.estimators.keys())

        main = MainTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        deps = main.requires()

        actual_names = {d.estimator for d in deps}
        assert actual_names == expected_names


# ---------------------------------------------------------------------------
# SnapshotTask
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiSnapshotTask:
    def test_output_path(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        snapshot = SnapshotTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        out = snapshot.output()
        assert "dag_snapshot" in out
        assert Path(out["dag_snapshot"].path).name == "dag_snapshot.json"

    def test_main_task_class_returns_b2luigi_main(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        snapshot = SnapshotTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        assert snapshot._main_task_class() is MainTask


# ---------------------------------------------------------------------------
# DownstreamTask branching
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiDownstreamTaskBranching:
    def test_branch_params_empty_means_no_expansion(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        dt = DownstreamTask(
            config_file=config_file,
            downstream="snapshot",  # any key; won't be resolved without a real config
            results_path=str(tmp_path),
            branch_params="",
        )
        assert dt.branch_parameters == {}

    def test_branch_params_roundtrip(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        params = {"alpha": "0.1", "beta": "5"}
        encoded = urlencode(params)

        dt = DownstreamTask(
            config_file=config_file,
            downstream="snapshot",
            results_path=str(tmp_path),
            branch_params=encoded,
        )
        decoded = dt.branch_parameters
        assert decoded["alpha"] == "0.1"
        assert decoded["beta"] == "5"

    def test_task_ids_differ_across_branches(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        dt_a = DownstreamTask(
            config_file=config_file,
            downstream="snapshot",
            results_path=str(tmp_path),
            branch_params=urlencode({"alpha": "0.1"}),
        )
        dt_b = DownstreamTask(
            config_file=config_file,
            downstream="snapshot",
            results_path=str(tmp_path),
            branch_params=urlencode({"alpha": "0.2"}),
        )
        assert dt_a.task_id != dt_b.task_id


# ---------------------------------------------------------------------------
# Backend isolation: b2luigi tasks must not import law at class-definition time
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestBackendIsolation:
    def test_fold_task_importable_without_law_side_effects(self) -> None:
        """Importing b2luigi FoldTask must not cause law-specific attributes to appear."""
        assert not hasattr(FoldTask, "create_branch_map"), (
            "b2luigi FoldTask should not have LAW's create_branch_map"
        )

    def test_fold_task_is_b2luigi_task(self) -> None:
        import b2luigi

        assert issubclass(FoldTask, b2luigi.Task)

    def test_fold_task_is_not_local_workflow(self) -> None:
        # b2luigi FoldTask should NOT inherit from LAW's LocalWorkflow
        try:
            import law
            assert not issubclass(FoldTask, law.LocalWorkflow), (
                "b2luigi FoldTask must not inherit from law.LocalWorkflow"
            )
        except ImportError:
            pass  # law not installed — certainly not inherited

    def test_b2luigi_output_as_dict_is_identity(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        fold = FoldTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        output = fold.output()
        assert FoldTask.output_as_dict(output) is output, (
            "b2luigi FoldTask.output_as_dict must return the dict unchanged (no TargetCollection)"
        )
