"""FoldTask (b2luigi backend) - Executes the actual training for a single cross-validation fold.

Identical training logic as the LAW backend (inherited from BaseFoldTask).
Batch dispatch is handled by b2luigi via ``batch_system`` and ``htcondor_settings``/
``slurm_settings`` task attributes — no LocalWorkflow inheritance needed.
"""

from __future__ import annotations

from typing import Type

import b2luigi
import luigi

from needle.tasks.base.fold import BaseFoldTask


class FoldTask(BaseFoldTask, b2luigi.Task):
    """b2luigi FoldTask — trains one cross-validation fold.

    Batch dispatch is configured globally via ``configure_b2luigi()`` or per-task
    by overriding ``htcondor_settings`` / ``slurm_settings`` class attributes.
    Output paths follow the needle convention (``est__X/syst__Y/ensem__Z/fold__N``)
    rather than b2luigi's auto-generated param-based paths.
    """

    task_namespace = "b2luigi"

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.b2luigi_tasks.estimator import EstimatorTask  # avoid circular imports

        return EstimatorTask

    # output() and run() are inherited from BaseFoldTask.
    # b2luigi.LocalTarget IS luigi.LocalTarget, so output() works unchanged.
