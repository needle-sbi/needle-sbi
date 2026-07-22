"""Base task classes shared by all workflow backends."""
from needle.tasks.base.downstream import BaseDownstreamMixin
from needle.tasks.base.ensemble import BaseEnsembleTask
from needle.tasks.base.estimator import BaseEstimatorTask
from needle.tasks.base.expansion import BaseExpansionTask
from needle.tasks.base.fold import BaseFoldTask
from needle.tasks.base.main import BaseMainTask
from needle.tasks.base.systematic import BaseSystematicTask
from needle.tasks.base.training import BaseTrainingTask

__all__ = [
    "BaseExpansionTask",
    "BaseTrainingTask",
    "BaseFoldTask",
    "BaseEnsembleTask",
    "BaseSystematicTask",
    "BaseEstimatorTask",
    "BaseMainTask",
    "BaseDownstreamMixin",
]
