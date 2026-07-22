from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.training import BaseTrainingTask


class TrainingTask(BaseTrainingTask, b2luigi.Task):
    """b2luigi TrainingTask — trains one cross-validation fold.

    This is the only task in the b2luigi backend that inherits from ``b2luigi.Task``.
    Batch dispatch is configured globally via ``configure_b2luigi()`` or per-task
    by overriding ``htcondor_settings`` / ``slurm_settings`` class attributes.
    Output paths follow the needle convention (``est__X/syst__Y/ensem__Z/fold__N``).
    """

    task_namespace = "b2luigi"

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask
