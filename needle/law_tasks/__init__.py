"""LAW Tasks - Distributed training orchestration for NEEDLE.

This package defines the complete task DAG for training ensemble models with
systematic variations and cross-validation folds using the LAW (Luigi + Apache Airflow Wrapper)
workflow engine.

Task Hierarchy:
    MainTask (entry point)
    └── EstimatorTask (one per estimator)
         └── SystematicTask (one per systematic variation)
              └── EnsembleTask (one per ensemble group)
                   └── FoldTask (actual training with PyTorch Lightning)

Supporting Components:
    - SnapshotTask: Collects trained models into a snapshot for evaluation
    - DownstreamTask: Runs post-training analysis and evaluation tasks
    - Mixins: HydraMixin for configuration, CollectOutputMixin for debugging
    - Workflows: Support for local, HTCondor, and SLURM execution

Features:
- Multi-level aggregation: folds → ensembles → systematics → estimators
- Remote job dispatch to HPC clusters
- Automatic result aggregation and checkpoint collection
- Workflow visualization and debugging
- Configuration composition with Hydra

Usage:
    law run MainTask --config-file path/to/config.yaml
    law run DownstreamTask --downstream my_analysis_task

See documentation for detailed configuration and usage examples.
"""
from .downstream import DownstreamTask
from .ensemble import EnsembleTask
from .estimator import EstimatorTask
from .fold import FoldTask
from .main import MainTask
from .snapshot import SnapshotTask
from .systematic import SystematicTask

__all__ = [
    "MainTask",
    "EstimatorTask",
    "EnsembleTask",
    "SystematicTask",
    "FoldTask",
    "SnapshotTask",
    "DownstreamTask",
]
