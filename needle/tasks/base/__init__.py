"""Base task classes shared by all workflow backends."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

#: name -> submodule providing it. Only `BaseTrainingTask` pulls in
#: torch/lightning/mlflow (needle/tasks/base/training.py); keeping these lazy
#: means importing any other sibling submodule (e.g. needle.tasks.base.main,
#: which otherwise forces this __init__ to run first) doesn't pay that cost.
_SUBMODULE_BY_NAME = {
    "BaseDownstreamMixin": "needle.tasks.base.downstream",
    "BaseEnsembleTask": "needle.tasks.base.ensemble",
    "BaseEstimatorTask": "needle.tasks.base.estimator",
    "BaseExpansionTask": "needle.tasks.base.expansion",
    "BaseFoldTask": "needle.tasks.base.fold",
    "BaseMainTask": "needle.tasks.base.main",
    "BaseSystematicTask": "needle.tasks.base.systematic",
    "BaseTrainingTask": "needle.tasks.base.training",
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
    return sorted(__all__)
