"""MainTask (b2luigi backend) - Entry point for the b2luigi NEEDLE training DAG."""

from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask, b2luigi.Task):
    """b2luigi MainTask — DAG entry point.

    Mirrors the LAW ``MainTask`` but uses b2luigi for task scheduling.
    Requires and abs_results_path logic are inherited from BaseMainTask.
    """

    task_namespace = "b2luigi"

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.b2luigi_tasks.estimator import EstimatorTask

        return EstimatorTask
