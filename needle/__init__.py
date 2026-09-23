"""NEEDLE: A Workflow Orchestrator for Neural Simulation Based Inference Methods."""

import importlib
from typing import TYPE_CHECKING

import needle.utils.logging  # noqa: F401

if TYPE_CHECKING:
    from needle import etl, ml, utils
    from needle.api import (
        Estimator,
        InitResult,
        RunResult,
        UnknownTaskError,
        configure_b2luigi,
        configure_law,
        init,
        load_config,
        run,
        train_single,
    )

__all__ = [
    "etl",
    "ml",
    "utils",
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
]

_SUBMODULES = {"etl", "ml", "utils"}
_API_NAMES = set(__all__) - _SUBMODULES


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        module = importlib.import_module(f"needle.{name}")
        globals()[name] = module
        return module

    if name in _API_NAMES:
        value = getattr(importlib.import_module("needle.api"), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # PEP 562
    return sorted(__all__)
