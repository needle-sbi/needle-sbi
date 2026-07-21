"""b2luigi_tasks: b2luigi-based workflow backend for NEEDLE.

Mirrors the task DAG of ``needle.law_tasks`` but uses b2luigi for task
scheduling and batch dispatch. Task logic is shared via ``needle.tasks.base``.

Task Hierarchy:
    MainTask (entry point)
    └── EstimatorTask (one per estimator)
         └── SystematicTask (one per systematic variation)
              └── EnsembleTask (one per ensemble group)
                   └── FoldTask (actual training with PyTorch Lightning)

Supporting Components:
    - SnapshotTask: Collects trained models into a snapshot for evaluation
    - DownstreamTask: Runs post-training analysis and evaluation tasks

Batch dispatch is configured via ``settings.json`` at the project root or by
calling ``needle.b2luigi_tasks.workflows.common.configure_b2luigi()``.

Usage (local):
    import b2luigi
    from needle.b2luigi_tasks import MainTask
    b2luigi.process(MainTask(config_file="conf/config.yaml"))

Usage (HTCondor):
    from needle.b2luigi_tasks.workflows.common import configure_b2luigi
    configure_b2luigi(results_path="runs", batch_system="htcondor")
    b2luigi.process(MainTask(config_file="conf/config.yaml"))
"""

from needle.b2luigi_tasks.downstream import DownstreamTask
from needle.b2luigi_tasks.ensemble import EnsembleTask
from needle.b2luigi_tasks.estimator import EstimatorTask
from needle.b2luigi_tasks.fold import FoldTask
from needle.b2luigi_tasks.main import MainTask
from needle.b2luigi_tasks.snapshot import SnapshotTask
from needle.b2luigi_tasks.systematic import SystematicTask

__all__ = [
    "MainTask",
    "EstimatorTask",
    "EnsembleTask",
    "SystematicTask",
    "FoldTask",
    "SnapshotTask",
    "DownstreamTask",
]
