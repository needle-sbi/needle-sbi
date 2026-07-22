"""MainTask - Entry point for the NEEDLE training DAG (b2luigi backend).

Run via ``needle run --backend b2luigi`` rather than executing this file
directly — see ``needle/cli.py`` for how the task is constructed and handed
to ``b2luigi.process()``.
"""

from __future__ import annotations

from typing import Type

import luigi
import b2luigi

from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask, b2luigi.Task):  # type: ignore
    """b2luigi MainTask — DAG entry point."""

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask
