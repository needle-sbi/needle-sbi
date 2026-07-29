"""EnsembleTask (law backend) — thin marker wrapper aggregating FoldTask instances."""

from __future__ import annotations

from typing import Any, Type

import law
import luigi

from needle.tasks.base.ensemble import BaseEnsembleTask


class EnsembleTask(BaseEnsembleTask, law.Task):
    """law EnsembleTask — signals ensemble completion via a .done marker."""

    def _fold_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.law.fold import FoldTask

        return FoldTask

    def _target_class(self) -> Type[Any]:
        return law.LocalFileTarget
