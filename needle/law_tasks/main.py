"""MainTask - Entry point for the NEEDLE training DAG."""

from __future__ import annotations

from enum import Enum
from typing import Type

import law
import luigi

from needle.law_tasks.estimator import EstimatorTask
from needle.law_tasks.mixins import HydraMixin
from needle.tasks.base.main import BaseMainTask
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("dag")


class ConfigStrictness(Enum):
    IGNORE = "IGNORE"
    WARN = "WARN"
    RAISE = "RAISE"


class MainTask(BaseMainTask, law.WrapperTask):
    """This Task serves as the main entry point for all the trainings.

    It is responsible for:

    - Loading and resolving Hydra configuration
    - Creating EstimatorTask instances for all estimators in the config
    - Managing the complete training DAG

    The Task resolves configuration conflicts and manages results paths, then propagates all the
    settings down the Task tree.
    """

    def _estimator_task_class(self) -> Type[luigi.Task]:
        return EstimatorTask

    # requires() and abs_results_path are inherited from BaseMainTask
