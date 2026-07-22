from __future__ import annotations

from typing import Type

import luigi

from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask):
    """b2luigi MainTask — DAG entry point."""

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask
