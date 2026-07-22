from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import luigi

from needle.tasks.mixins.hydra import HydraParamsMixin


class BaseExpansionTask(HydraParamsMixin, luigi.Task):
    """Common base for all fan-out coordination tasks (Fold, Ensemble, Systematic, Estimator).

    Subclasses define extra parameters, ``abs_results_path``, and ``requires()``.
    The sole output is a ``.done`` marker file.
    """

    results_path: str = luigi.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )
    estimator: str = luigi.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )

    @property
    def abs_results_path(self) -> Path:
        raise NotImplementedError

    def output(self) -> Any:
        return luigi.LocalTarget(os.path.join(str(self.abs_results_path), ".done"))

    def run(self) -> None:
        out = self.output()
        os.makedirs(os.path.dirname(out.path), exist_ok=True)
        Path(out.path).touch()
