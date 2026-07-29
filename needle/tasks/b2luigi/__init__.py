"""needle.tasks.b2luigi: b2luigi-based workflow backend for NEEDLE.

Task Hierarchy:
    MainTask (entry point, plain luigi.Task)
    └── EstimatorTask (plain luigi.Task)
         └── SystematicTask (plain luigi.Task)
              └── EnsembleTask (plain luigi.Task)
                   └── FoldTask (plain luigi.Task, marker wrapper)
                        └── TrainingTask (b2luigi.Task — the only batch-submitted task)

Supporting Components:
    - DownstreamTask: Runs post-training analysis tasks

MainTask both drives the training DAG and writes dag_snapshot.json on completion.

Batch dispatch is configured via ``settings.json`` or by calling
``needle.tasks.b2luigi.workflows.common.configure_b2luigi()``.

Usage (local):
    import b2luigi
    from needle.tasks.b2luigi import MainTask
    b2luigi.process(MainTask(config_file="conf/config.yaml"))

Usage (HTCondor):
    from needle.tasks.b2luigi.workflows.common import configure_b2luigi
    configure_b2luigi(results_path="runs", batch_system="htcondor")
    b2luigi.process(MainTask(config_file="conf/config.yaml"))
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.tasks.b2luigi.downstream import DownstreamTask
    from needle.tasks.b2luigi.ensemble import EnsembleTask
    from needle.tasks.b2luigi.estimator import EstimatorTask
    from needle.tasks.b2luigi.fold import FoldTask
    from needle.tasks.b2luigi.main import MainTask
    from needle.tasks.b2luigi.systematic import SystematicTask
    from needle.tasks.b2luigi.training import TrainingTask

__all__ = [
    "MainTask",
    "EstimatorTask",
    "EnsembleTask",
    "SystematicTask",
    "FoldTask",
    "TrainingTask",
    "DownstreamTask",
]

#: name -> submodule providing it. Only `TrainingTask` pulls in torch/lightning/mlflow
#: (see needle/tasks/base/training.py), so keeping these lazy avoids paying that cost
#: for callers who only need e.g. MainTask or DownstreamTask.
_SUBMODULE_BY_NAME = {
    "DownstreamTask": "needle.tasks.b2luigi.downstream",
    "EnsembleTask": "needle.tasks.b2luigi.ensemble",
    "EstimatorTask": "needle.tasks.b2luigi.estimator",
    "FoldTask": "needle.tasks.b2luigi.fold",
    "MainTask": "needle.tasks.b2luigi.main",
    "SystematicTask": "needle.tasks.b2luigi.systematic",
    "TrainingTask": "needle.tasks.b2luigi.training",
}


def __getattr__(name: str) -> object:
    module_name = _SUBMODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
