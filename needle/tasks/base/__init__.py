"""Base task classes shared by all workflow backends."""
from needle.tasks.base.downstream import BaseDownstreamMixin
from needle.tasks.base.ensemble import BaseEnsembleTask
from needle.tasks.base.estimator import BaseEstimatorTask
from needle.tasks.base.fold import BaseFoldTask
from needle.tasks.base.main import BaseMainTask
from needle.tasks.base.snapshot import BaseSnapshotTask
from needle.tasks.base.systematic import BaseSystematicTask

__all__ = [
    "BaseFoldTask",
    "BaseEnsembleTask",
    "BaseSystematicTask",
    "BaseEstimatorTask",
    "BaseSnapshotTask",
    "BaseMainTask",
    "BaseDownstreamMixin",
]
