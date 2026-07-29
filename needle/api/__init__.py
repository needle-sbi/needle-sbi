import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.api.config import Config
    from needle.api.dataset import Dataset
    from needle.api.init import InitResult, init
    from needle.api.law_settings import configure_law
    from needle.api.model import Model
    from needle.api.run import RunResult, UnknownTaskError, run
    from needle.api.train import train_single
    from needle.tasks.b2luigi.workflows.common import configure_b2luigi

__all__ = [
    "Config",
    "Model",
    "Dataset",
    "train_single",
    "run",
    "RunResult",
    "UnknownTaskError",
    "init",
    "InitResult",
    "configure_law",
    "configure_b2luigi",
]

#: name -> submodule providing it. Kept lazy so `import needle.api` stays cheap:
#: `Model`/`Dataset` pull in torch/lightning, and `configure_b2luigi` pulls in
#: luigi/b2luigi, only when actually accessed.
_MODULE_BY_NAME = {
    "Config": "needle.api.config",
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
    return sorted(set(__all__) | set(globals()))
