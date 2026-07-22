"""b2luigi_tasks: b2luigi-based workflow backend for NEEDLE.

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
``needle.b2luigi_tasks.workflows.common.configure_b2luigi()``.

Usage (local):
    import b2luigi
    from needle.b2luigi_tasks import MainTask
    b2luigi.process(MainTask(config_file="conf/config.yaml"))

Usage (HTCondor):
    from needle.b2luigi_tasks.workflows.common import configure_b2luigi
    configure_b2luigi(results_path="runs", batch_system="htcondor")
    b2luigi.process(MainTask(config_file="conf/config.yaml"))
"""

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
