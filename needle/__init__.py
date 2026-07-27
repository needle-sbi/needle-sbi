"""NEEDLE: A Workflow Orchestrator for Neural Simulation Based Inference Methods."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle import etl, ml, utils

__all__ = ["etl", "ml", "utils"]


def __getattr__(name: str) -> object:
    # Lazy: needle.ml (torch/lightning) and needle.etl (dask/uproot) are
    # expensive to import. Keeping `import needle` cheap matters for the
    # `needle` CLI, whose argument parsing and tab-completion must stay fast.
    if name in __all__:
        module = importlib.import_module(f"needle.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
