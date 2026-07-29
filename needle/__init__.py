"""NEEDLE: A Workflow Orchestrator for Neural Simulation Based Inference Methods."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle import etl, ml, utils
    from needle.api import (
        Config,
        Dataset,
        InitResult,
        Model,
        RunResult,
        UnknownTaskError,
        configure_b2luigi,
        configure_law,
        init,
        run,
        train_single,
    )

__all__ = [
    "etl",
    "ml",
    "utils",
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

#: Submodules exposed as-is (`needle.ml`, `needle.etl`, `needle.utils`).
_SUBMODULES = {"etl", "ml", "utils"}

#: Flattened `needle.api` names, so e.g. `needle.Config` works without an
#: explicit `import needle.api`.
_API_NAMES = set(__all__) - _SUBMODULES


def __getattr__(name: str) -> object:
    # Lazy: needle.ml (torch/lightning) and needle.etl (dask/uproot) are
    # expensive to import. Keeping `import needle` cheap matters for the
    # `needle` CLI, whose argument parsing and tab-completion must stay fast.
    if name in _SUBMODULES:
        module = importlib.import_module(f"needle.{name}")
        globals()[name] = module
        return module
    if name in _API_NAMES:
        value = getattr(importlib.import_module("needle.api"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # PEP 562 module __dir__: without this, `dir(needle)` only shows names
    # already cached in globals() by __getattr__, so tab completion in a
    # plain python3 REPL or a notebook wouldn't list `ml`/`Config`/etc. until
    # they'd already been accessed once. Returning only `__all__` (rather than
    # unioning with globals()) keeps implementation details -- `importlib`,
    # `TYPE_CHECKING`, incidental submodule imports like `needle.tasks` -- out
    # of tab completion.
    return sorted(__all__)
