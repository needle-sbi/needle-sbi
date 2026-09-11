"""TrainingTask - Executes the actual training for a single cross-validation fold.

This is the only law workflow task in the hierarchy. All other tasks above it
(FoldTask, EnsembleTask, etc.) are thin marker wrappers that run locally.
Batch dispatch (HTCondor, SLURM) is handled by the workflow mixin inheritance.
"""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.tasks.base.training import BaseTrainingTask
from needle.tasks.law.workflows import (
    HTCondorWorkflow,
    LocalWorkflow,
    SlurmWorkflow,
    check_batch_system,
)

TrainingTaskOutput = Dict[str, law.LocalFileTarget] | Dict[str, law.TargetCollection]


class TrainingTask(
    BaseTrainingTask,
    LocalWorkflow,
    HTCondorWorkflow,
    SlurmWorkflow,
):
    """Trains a single cross-validation fold; the leaf of the law DAG.

    Supports local, HTCondor, and Slurm execution via workflow mixin inheritance.
    """

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.tasks.law.estimator import EstimatorTask

        return EstimatorTask

    def create_branch_map(self) -> Dict[int, None]:  # type: ignore
        return {0: None}

    def output(self) -> Dict[str, Any]:
        check_batch_system(system=str(self.workflow))  # type: ignore

        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    @staticmethod
    def output_as_dict(task_output: TrainingTaskOutput) -> Dict[str, law.LocalTarget]:
        """Unpack local and remote training outputs.

        Local runs return the plain dict; remote runs wrap outputs in a TargetCollection.
        """
        remote_collection: law.TargetCollection | None = task_output.get("collection")  # type: ignore

        if remote_collection:
            if len(remote_collection) != 1:
                raise NotImplementedError(
                    "Currently the usage of branches in TrainingTask is not supported. "
                    "Folds must each be their own Task instance."
                )
            return remote_collection[0]
        return task_output  # type: ignore

    # run() is inherited from BaseTrainingTask
