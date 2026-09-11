from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type

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

    def _target_class(self) -> Type[Any]:
        """Backend-specific completion-marker target. law overrides this with
        ``law.LocalFileTarget`` since ``law.Task.complete()`` calls ``target.complete()``,
        which plain ``luigi.LocalTarget`` does not implement.
        """
        return luigi.LocalTarget

    def output(self) -> Any:
        return self._target_class()(os.path.join(str(self.abs_results_path), ".done"))

    def run(self) -> None:
        out = self.output()
        os.makedirs(os.path.dirname(out.path), exist_ok=True)
        Path(out.path).touch()
