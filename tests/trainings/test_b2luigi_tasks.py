"""Tests for the b2luigi workflow backend.

Verifies task DAG structure, output paths, parameter handling, and downstream
branching without executing actual pipeline runs (no training, no file I/O).

All tests are marked ``@pytest.mark.b2luigi`` and excluded from the default
test run (see ``pyproject.toml`` ``addopts``).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import omegaconf
import pytest

from needle.tasks.b2luigi.downstream import DownstreamTask
from needle.tasks.b2luigi.ensemble import EnsembleTask
from needle.tasks.b2luigi.estimator import EstimatorTask
from needle.tasks.b2luigi.fold import FoldTask
from needle.tasks.b2luigi.main import MainTask
from needle.tasks.b2luigi.systematic import SystematicTask
from needle.tasks.b2luigi.training import TrainingTask
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
# FoldTask — marker wrapper
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiFoldTask:
    def test_output_is_done_marker(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
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
        assert Path(out.path).name == ".done"
        assert f"est__{estimator_name}" in out.path
        assert "syst__nominal" in out.path
        assert "ensem__0" in out.path
        assert "fold__2" in out.path

    def test_requires_returns_single_training_task(
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
        deps = fold.requires()
        assert len(deps) == 1
        assert isinstance(deps[0], TrainingTask)

    def test_training_task_class_returns_b2luigi_training(
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
        assert fold._training_task_class() is TrainingTask


# ---------------------------------------------------------------------------
# TrainingTask — the only b2luigi.Task in the backend
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiTrainingTask:
    def test_output_paths(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        task = TrainingTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=2,
            results_path=str(tmp_path),
        )

        out = task.output()

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

        task = TrainingTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        out = task.output()
        assert TrainingTask.output_as_dict(out) is out

    def test_requires_empty_without_deps(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        task = TrainingTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        assert task.requires() == []

    def test_estimator_task_class_returns_b2luigi_estimator(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        task = TrainingTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        assert task._estimator_task_class() is EstimatorTask


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

    def test_output_is_done_marker(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
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
        assert Path(out.path).name == ".done"
        assert "ensem__1" in out.path


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

    def test_output_is_done_marker(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        syst = SystematicTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            results_path=str(tmp_path),
        )
        out = syst.output()
        assert Path(out.path).name == ".done"
        assert "syst__nominal" in out.path


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

    def test_output_is_done_marker(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        est = EstimatorTask(
            config_file=config_file,
            estimator=estimator_name,
            results_path=str(tmp_path),
        )
        out = est.output()
        assert Path(out.path).name == ".done"
        assert f"est__{estimator_name}" in out.path


# ---------------------------------------------------------------------------
# MainTask — merged entry point + snapshot writer
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestB2LuiMainTask:
    def test_output_is_dag_snapshot(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        main = MainTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        out = main.output()
        assert "dag_snapshot" in out
        assert Path(out["dag_snapshot"].path).name == "dag_snapshot.json"

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
        actual_names = {d.estimator for d in main.requires()}
        assert actual_names == expected_names


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
            downstream="snapshot",
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
# Backend isolation: TrainingTask is the only b2luigi.Task
# ---------------------------------------------------------------------------


@pytest.mark.b2luigi
class TestBackendIsolation:
    def test_fold_task_has_no_create_branch_map(self) -> None:
        """FoldTask is a plain marker wrapper — no law workflow attributes."""
        assert not hasattr(FoldTask, "create_branch_map")

    def test_fold_task_is_not_b2luigi_task(self) -> None:
        """Only TrainingTask should be a b2luigi.Task; FoldTask is plain luigi."""
        import b2luigi

        assert not issubclass(FoldTask, b2luigi.Task)

    def test_training_task_is_b2luigi_task(self) -> None:
        import b2luigi

        assert issubclass(TrainingTask, b2luigi.Task)

    def test_training_task_is_not_local_workflow(self) -> None:
        try:
            import law

            assert not issubclass(TrainingTask, law.LocalWorkflow)
        except ImportError:
            pass

    def test_training_task_output_as_dict_is_identity(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)
        estimator_name = list(config_factory().estimators.keys())[0]

        task = TrainingTask(
            config_file=config_file,
            estimator=estimator_name,
            systematic="nominal",
            ensemble=0,
            fold_index=0,
            results_path=str(tmp_path),
        )
        output = task.output()
        assert TrainingTask.output_as_dict(output) is output
