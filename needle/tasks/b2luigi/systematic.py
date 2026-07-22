from __future__ import annotations

from typing import Type

import luigi

from needle.tasks.base.systematic import BaseSystematicTask


class SystematicTask(BaseSystematicTask):
    """b2luigi SystematicTask — marker wrapper aggregating EnsembleTask instances."""

    def _ensemble_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.ensemble import EnsembleTask

        return EnsembleTask
