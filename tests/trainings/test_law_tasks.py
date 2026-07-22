"""Tests for the law workflow backend.

Verifies task DAG structure, output paths, parameter handling, and downstream
branching without executing actual pipeline runs (no training, no file I/O).
Mirrors ``tests/trainings/test_b2luigi_tasks.py`` for the law backend.

All tests are marked ``@pytest.mark.law`` and excluded from the default test
run (see ``pyproject.toml`` ``addopts``).
"""

from __future__ import annotations

from pathlib import Path

import law
import omegaconf
import pytest

from needle.tasks.law.downstream import DownstreamTask
from needle.tasks.law.ensemble import EnsembleTask
from needle.tasks.law.estimator import EstimatorTask
from needle.tasks.law.fold import FoldTask
from needle.tasks.law.main import MainTask
from needle.tasks.law.systematic import SystematicTask
from needle.tasks.law.training import TrainingTask
from needle.utils.config_schema import DownstreamTaskConfig, MainConfig
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


@pytest.mark.law
class TestLawFoldTask:
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
        assert isinstance(out, law.LocalFileTarget)
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

    def test_training_task_class_returns_law_training(
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
# TrainingTask — the only law workflow leaf
# ---------------------------------------------------------------------------


@pytest.mark.law
class TestLawTrainingTask:
    # branch=0 is required: without it the task is the workflow root (branch=-1) and
    # output() returns a TargetCollection instead of the plain per-branch dict.
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
            branch=0,
        )

        out = task.output()

        assert "ckpt" in out
        assert "model_config" in out
        assert "input_models" in out

        ckpt_path = Path(out["ckpt"].path)
        assert ckpt_path.name == "model.ckpt"
        assert f"est__{estimator_name}" in str(ckpt_path)
        assert "syst__nominal" in str(ckpt_path)
        assert "ensem__0" in str(ckpt_path)
        assert "fold__2" in str(ckpt_path)

    def test_output_as_dict_unwraps_plain_dict(
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
            branch=0,
        )
        out = task.output()
        assert TrainingTask.output_as_dict(out) is out

    def test_create_branch_map_is_single_branch(
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
        assert task.create_branch_map() == {0: None}

    def test_estimator_task_class_returns_law_estimator(
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


@pytest.mark.law
class TestLawEnsembleTask:
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
        assert isinstance(out, law.LocalFileTarget)
        assert Path(out.path).name == ".done"
        assert "ensem__1" in out.path


# ---------------------------------------------------------------------------
# SystematicTask
# ---------------------------------------------------------------------------


@pytest.mark.law
class TestLawSystematicTask:
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
        assert isinstance(out, law.LocalFileTarget)
        assert Path(out.path).name == ".done"
        assert "syst__nominal" in out.path


# ---------------------------------------------------------------------------
# EstimatorTask
# ---------------------------------------------------------------------------


@pytest.mark.law
class TestLawEstimatorTask:
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
        assert isinstance(out, law.LocalFileTarget)
        assert Path(out.path).name == ".done"
        assert f"est__{estimator_name}" in out.path


# ---------------------------------------------------------------------------
# MainTask — entry point + snapshot writer
# ---------------------------------------------------------------------------


@pytest.mark.law
class TestLawMainTask:
    def test_output_is_dag_snapshot(self, config_factory: MainConfigFactory, tmp_path: Path) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        main = MainTask(
            config_file=config_file,
            results_path=str(tmp_path),
        )
        out = main.output()
        assert isinstance(out["dag_snapshot"], law.LocalFileTarget)
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


@pytest.mark.law
class TestLawDownstreamTaskBranching:
    def test_no_expansion_yields_single_default_branch(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        dt = DownstreamTask(
            config_file=config_file,
            downstream="my_task",
            results_path=str(tmp_path),
        )
        dt.config = MainConfig(
            downstream_tasks={"my_task": DownstreamTaskConfig(args={"_target_": "some.module.Task"})}
        )
        branch_map = dt.create_branch_map()
        assert list(branch_map.keys()) == [0]
        assert branch_map[0].name == "default"
        assert branch_map[0].parameters == {}

    def test_expansion_creates_one_branch_per_combination(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        dt = DownstreamTask(
            config_file=config_file,
            downstream="my_task",
            results_path=str(tmp_path),
        )
        dt.config = MainConfig(
            downstream_tasks={
                "my_task": DownstreamTaskConfig(
                    args={"_target_": "some.module.Task"},
                    expands={"num_jets": [1, 2]},
                )
            }
        )
        branch_map = dt.create_branch_map()
        assert len(branch_map) == 2
        assert {b.parameters["num_jets"] for b in branch_map.values()} == {1, 2}

    def test_requires_without_downstream_deps_needs_main_task(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        # branch=0 makes this a branch instance so `requires()` resolves to
        # DownstreamTask's own override rather than law's workflow-root branch expansion.
        dt = DownstreamTask(
            config_file=config_file,
            downstream="my_task",
            results_path=str(tmp_path),
            branch=0,
        )
        dt.config = MainConfig(
            downstream_tasks={"my_task": DownstreamTaskConfig(args={"_target_": "some.module.Task"})}
        )
        deps = dt.requires()
        assert "main" in deps
        assert isinstance(deps["main"], MainTask)

    def test_requires_with_downstream_deps_chains_to_other_downstream_tasks(
        self, config_factory: MainConfigFactory, tmp_path: Path
    ) -> None:
        config_file = _write_config(config_factory(), tmp_path)

        dt = DownstreamTask(
            config_file=config_file,
            downstream="my_task",
            results_path=str(tmp_path),
            branch=0,
        )
        dt.config = MainConfig(
            downstream_tasks={
                "my_task": DownstreamTaskConfig(args={"_target_": "some.module.Task"}, requires=["upstream_task"]),
                "upstream_task": DownstreamTaskConfig(args={"_target_": "some.module.Other"}),
            }
        )
        deps = dt.requires()
        assert "upstream_task" in deps
        assert isinstance(deps["upstream_task"], DownstreamTask)
        assert deps["upstream_task"].downstream == "upstream_task"


# ---------------------------------------------------------------------------
# Backend isolation: TrainingTask is the only law workflow task
# ---------------------------------------------------------------------------


@pytest.mark.law
class TestLawBackendIsolation:
    def test_fold_task_has_no_create_branch_map(self) -> None:
        """FoldTask is a plain marker wrapper — no law workflow attributes."""
        assert not hasattr(FoldTask, "create_branch_map")

    def test_fold_task_is_not_local_workflow(self) -> None:
        assert not issubclass(FoldTask, law.LocalWorkflow)

    def test_fold_task_is_plain_law_task(self) -> None:
        assert issubclass(FoldTask, law.Task)

    def test_training_task_is_local_workflow(self) -> None:
        assert issubclass(TrainingTask, law.LocalWorkflow)

    def test_training_task_is_not_b2luigi_task(self) -> None:
        try:
            import b2luigi

            assert not issubclass(TrainingTask, b2luigi.Task)
        except ImportError:
            pass
