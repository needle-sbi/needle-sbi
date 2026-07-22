from __future__ import annotations

from typing import Type

import luigi
import b2luigi

from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask, b2luigi.Task):
    """b2luigi MainTask — DAG entry point."""

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask


if __name__ == "__main__":
    b2luigi.process(MainTask())
