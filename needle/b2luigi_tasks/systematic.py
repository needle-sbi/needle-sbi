"""SystematicTask (b2luigi backend)."""

from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.systematic import BaseSystematicTask


class SystematicTask(BaseSystematicTask, b2luigi.Task):
    """b2luigi SystematicTask."""

    task_namespace = "b2luigi"

    def _ensemble_task_class(self) -> Type[luigi.Task]:
        from needle.b2luigi_tasks.ensemble import EnsembleTask

        return EnsembleTask

    # output() and run() inherited from BaseSystematicTask
