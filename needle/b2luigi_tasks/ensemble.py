"""EnsembleTask (b2luigi backend)."""

from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.ensemble import BaseEnsembleTask


class EnsembleTask(BaseEnsembleTask, b2luigi.Task):
    """b2luigi EnsembleTask.

    No ``input()`` override needed: b2luigi FoldTask output is a plain dict,
    so Luigi's base ``input()`` already returns the correct structure for ``run()``.
    """

    task_namespace = "b2luigi"

    def _fold_task_class(self) -> Type[luigi.Task]:
        from needle.b2luigi_tasks.fold import FoldTask

        return FoldTask

    # output() and run() inherited from BaseEnsembleTask
