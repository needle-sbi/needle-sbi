import importlib
from typing import TYPE_CHECKING

from needle.utils.dataclass import SerializableDataclass

if TYPE_CHECKING:
    from needle.utils.config_utils import hydra_instantiate, initialize_hydra_config

__all__ = ["SerializableDataclass", "initialize_hydra_config", "hydra_instantiate"]

#: name -> submodule providing it, resolved lazily via module __getattr__ (PEP 562).
_MODULE_BY_NAME = {
    "initialize_hydra_config": "needle.utils.config_utils",
    "hydra_instantiate": "needle.utils.config_utils",
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
