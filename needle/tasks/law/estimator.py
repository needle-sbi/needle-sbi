"""EstimatorTask (law backend) — thin marker wrapper aggregating SystematicTask instances."""

from __future__ import annotations

from typing import Any, Type

import law
import luigi

from needle.tasks.base.estimator import BaseEstimatorTask


class EstimatorTask(BaseEstimatorTask, law.Task):
    """law EstimatorTask — signals estimator completion via a .done marker."""

    def _systematic_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.law.systematic import SystematicTask

        return SystematicTask

    def _target_class(self) -> Type[Any]:
        return law.LocalFileTarget
