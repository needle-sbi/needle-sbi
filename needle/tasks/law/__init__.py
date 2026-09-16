"""law_tasks: Distributed training orchestration for NEEDLE.

Task Hierarchy:
    MainTask (entry point, law.Task — writes dag_snapshot.json on completion)
    └── EstimatorTask (law.Task, .done marker)
         └── SystematicTask (law.Task, .done marker)
              └── EnsembleTask (law.Task, .done marker)
                   └── FoldTask (law.Task, .done marker)
                        └── TrainingTask (law workflow — the only batch-submitted task)

Supporting Components:
    - DownstreamTask: Runs post-training analysis tasks

Usage:
    law run MainTask --config-file path/to/config.yaml
    law run DownstreamTask --downstream my_analysis_task --config-file path/to/config.yaml
"""
from .downstream import DownstreamTask
from .ensemble import EnsembleTask
from .estimator import EstimatorTask
from .fold import FoldTask
from .main import MainTask
from .systematic import SystematicTask
from .training import TrainingTask

__all__ = [
    "MainTask",
    "EstimatorTask",
    "EnsembleTask",
    "SystematicTask",
    "FoldTask",
    "TrainingTask",
    "DownstreamTask",
]
