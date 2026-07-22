from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.fold import BaseFoldTask


class FoldTask(BaseFoldTask, b2luigi.Task):
    """b2luigi FoldTask — marker wrapper around a single TrainingTask fold."""

    def _training_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.training import TrainingTask

        return TrainingTask
