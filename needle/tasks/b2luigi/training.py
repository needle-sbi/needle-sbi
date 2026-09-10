from __future__ import annotations

from typing import Any, Dict, Type

import b2luigi
import luigi

from needle.tasks.b2luigi.workflows.common import merged_batch_settings
from needle.tasks.base.training import BaseTrainingTask


class TrainingTask(BaseTrainingTask, b2luigi.Task):
    """b2luigi TrainingTask — trains one cross-validation fold.

    This is the only task in the b2luigi backend that inherits from ``b2luigi.Task``.
    Batch dispatch is configured globally via ``configure_b2luigi()`` or ``settings.json``, with
    per-estimator/systematic ``resources`` (see ``BaseTrainingTask.resources``) merged on top via
    ``htcondor_settings``/``slurm_settings``.
    Output paths follow the needle convention (``est__X/syst__Y/ensem__Z/fold__N``), unless
    ``single=True`` is set, in which case outputs are written flat under ``results_path``.
    """

    task_namespace = "b2luigi"

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.b2luigi.estimator import EstimatorTask

        return EstimatorTask

    @property
    def htcondor_settings(self) -> Dict[str, Any]:
        return merged_batch_settings("htcondor_settings", self.resources)

    @property
    def slurm_settings(self) -> Dict[str, Any]:
        return merged_batch_settings("slurm_settings", self.resources)
