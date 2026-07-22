"""MainTask - Entry point for the NEEDLE training DAG."""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.tasks.law.estimator import EstimatorTask
from needle.tasks.base.main import BaseMainTask


class MainTask(BaseMainTask, law.Task):
    """DAG entry point: fans out to EstimatorTasks, then writes dag_snapshot.json.

    Overrides ``output()`` to use ``law.LocalFileTarget`` so law can track
    completion via its target system.
    """

    def _estimator_task_class(self) -> Type[luigi.Task]:
        return EstimatorTask

    def output(self) -> Dict[str, Any]:
        return {"dag_snapshot": law.LocalFileTarget(f"{self.abs_results_path}/dag_snapshot.json")}
