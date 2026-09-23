import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.api.config import load_config
    from needle.api.eval import Estimator, aggregate_siblings, load_snapshot
    from needle.api.init import InitResult, init
    from needle.api.law_settings import configure_law
    from needle.api.run import RunResult, UnknownTaskError, run
    from needle.api.train import train_single
    from needle.tasks.b2luigi.workflows.common import configure_b2luigi

__all__ = [
    "load_config",
    "train_single",
    "run",
    "RunResult",
    "UnknownTaskError",
    "init",
    "InitResult",
    "configure_law",
    "configure_b2luigi",
    "Estimator",
    "aggregate_siblings",
    "load_snapshot",
]

_MODULE_BY_NAME = {
    "load_config": "needle.api.config",
    "Model": "needle.api.model",
    "Dataset": "needle.api.dataset",
    "train_single": "needle.api.train",
    "run": "needle.api.run",
    "RunResult": "needle.api.run",
    "UnknownTaskError": "needle.api.run",
    "init": "needle.api.init",
    "InitResult": "needle.api.init",
    "configure_law": "needle.api.law_settings",
    "configure_b2luigi": "needle.tasks.b2luigi.workflows.common",
    "Estimator": "needle.api.eval",
    "load_snapshot": "needle.api.eval",
    "aggregate_siblings": "needle.api.eval",
}


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_NAME.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
