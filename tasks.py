"""b2luigi task definitions, discovered by the ``b2luigi`` CLI.

Scaffolded by ``needle init --backend b2luigi``. Re-exports the needle task DAG so
``b2luigi run <ClassName>`` (and the ``b2luigi batch-runner`` worker command used for
HTCondor/Slurm/LSF submission) can resolve task classes by name. This file must exist
at the project root (or wherever ``--task-file``/``B2LUIGI_TASK_FILE`` points) even
when tasks are only ever submitted through ``needle run --backend b2luigi``: the
b2luigi CLI unconditionally imports it before resolving a class name, batch worker
included.
"""

from needle.tasks.b2luigi import (
    DownstreamTask,
    EnsembleTask,
    EstimatorTask,
    FoldTask,
    MainTask,
    SystematicTask,
    TrainingTask,
)

__all__ = [
    "MainTask",
    "EstimatorTask",
    "EnsembleTask",
    "SystematicTask",
    "FoldTask",
    "TrainingTask",
    "DownstreamTask",
]
