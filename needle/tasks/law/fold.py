"""FoldTask (law backend) — thin marker wrapper around TrainingTask."""

from __future__ import annotations

from typing import Any, Type

import law
import luigi

from needle.tasks.base.fold import BaseFoldTask


class FoldTask(BaseFoldTask, law.Task):
    """law FoldTask — signals fold completion via a .done marker.

    The actual training is delegated to ``TrainingTask`` (the law workflow leaf).
    """

    def _training_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.law.training import TrainingTask

        return TrainingTask

    def _target_class(self) -> Type[Any]:
        return law.LocalFileTarget
