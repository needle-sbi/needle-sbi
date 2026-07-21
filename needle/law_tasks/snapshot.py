"""SnapshotTask - Creates a complete snapshot of the trained ensemble DAG."""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.law_tasks.mixins import HydraMixin
from needle.tasks.base.snapshot import BaseSnapshotTask
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("snapshot")


class SnapshotTask(BaseSnapshotTask, law.Task):
    """Creates a complete snapshot of the trained ensemble DAG.

    This snapshot can be used for evaluation without re-running training.
    """

    def _main_task_class(self) -> Type[luigi.Task]:
        from needle.law_tasks.main import MainTask  # avoid circular imports

        return MainTask

    def output(self) -> Dict[str, Any]:
        """Define output target for the DAG snapshot."""
        return {"dag_snapshot": law.LocalFileTarget(f"{self.abs_results_path}/dag_snapshot.json")}

    # requires() and run() are inherited from BaseSnapshotTask
