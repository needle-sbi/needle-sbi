from __future__ import annotations

from typing import Type

import luigi

from needle.tasks.base.ensemble import BaseEnsembleTask


class EnsembleTask(BaseEnsembleTask):
    """b2luigi EnsembleTask — marker wrapper aggregating FoldTask instances."""

    def _fold_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.fold import FoldTask

        return FoldTask
