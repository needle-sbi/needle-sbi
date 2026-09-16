"""SystematicTask (law backend) — thin marker wrapper aggregating EnsembleTask instances."""

from __future__ import annotations

from typing import Any, Type

import law
import luigi

from needle.tasks.base.systematic import BaseSystematicTask


class SystematicTask(BaseSystematicTask, law.Task):
    """law SystematicTask — signals systematic completion via a .done marker."""

    def _ensemble_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.law.ensemble import EnsembleTask

        return EnsembleTask

    def _target_class(self) -> Type[Any]:
        return law.LocalFileTarget
