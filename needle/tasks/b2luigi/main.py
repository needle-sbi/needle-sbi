"""MainTask - Entry point for the NEEDLE training DAG (b2luigi backend).

Run via ``needle run --backend b2luigi``.
"""

from __future__ import annotations

from typing import Type

import luigi
import b2luigi

from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask, b2luigi.Task):  # type: ignore
    """b2luigi MainTask — DAG entry point."""

    task_namespace = "b2luigi"

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask
