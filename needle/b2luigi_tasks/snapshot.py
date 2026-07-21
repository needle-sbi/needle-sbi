"""SnapshotTask (b2luigi backend)."""

from __future__ import annotations

from typing import Any, Dict, Type

import b2luigi
import luigi

from needle.tasks.base.snapshot import BaseSnapshotTask


class SnapshotTask(BaseSnapshotTask, b2luigi.Task):
    """b2luigi SnapshotTask."""

    task_namespace = "b2luigi"

    def _main_task_class(self) -> Type[luigi.Task]:
        from needle.b2luigi_tasks.main import MainTask  # avoid circular imports

        return MainTask

    # output() and run() inherited from BaseSnapshotTask
    # BaseSnapshotTask.output() returns luigi.LocalTarget which b2luigi handles natively
