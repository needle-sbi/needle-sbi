from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.estimator import BaseEstimatorTask


class EstimatorTask(BaseEstimatorTask, b2luigi.Task):
    """b2luigi EstimatorTask — marker wrapper aggregating SystematicTask instances."""

    task_namespace = "b2luigi"

    def _systematic_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.systematic import SystematicTask

        return SystematicTask
