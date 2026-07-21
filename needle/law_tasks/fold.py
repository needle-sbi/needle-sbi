"""FoldTask - Executes the actual training for a single cross-validation fold.

This module defines the FoldTask which is responsible for:
- Instantiating the Lightning Trainer, LightningModule, and DataModule
- Running the actual model training using PyTorch Lightning
- Saving checkpoints and training artifacts
- Supporting remote job dispatch (HTCondor, SLURM) or local execution
- Propagating training results to parent EnsembleTask

The task forms the fifth (leaf) level of the task DAG hierarchy:
    MainTask
    └── EstimatorTask (one per estimator)
         └── SystematicTask (one per systematic variation)
              └── EnsembleTask (one per ensemble group)
                   └── FoldTask (one per cross-validation fold) ← actual training happens here

Features:
- Workflow support: Local, HTCondor, SLURM
- Remote job dispatch for distributed training
- Automatic checkpoint management
- GPU support
"""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.law_tasks.mixins import HydraMixin
from needle.law_tasks.workflows import (
    HTCondorWorkflow,
    LocalWorkflow,
    SlurmWorkflow,
    check_batch_system,
)
from needle.tasks.base.fold import BaseFoldTask

#: Return type of :meth:`FoldTask.output` — maps checkpoint keys to LAW file targets.
FoldTaskOutput = Dict[str, law.LocalFileTarget] | Dict[str, law.TargetCollection]


class FoldTask(
    BaseFoldTask,
    LocalWorkflow,
    HTCondorWorkflow,
    SlurmWorkflow,
):
    """Task for training a single cross-validation fold.

    Executes the complete training pipeline for one fold: model instantiation, data loading,
    training via PyTorch Lightning, checkpoint saving, and result serialization. Supports
    optional dependencies on other estimators and generates MLflow experiment tracking.
    """

    def _estimator_task_class(self) -> Type[luigi.Task]:
        from needle.law_tasks.estimator import EstimatorTask  # avoid circular imports

        return EstimatorTask

    def create_branch_map(self) -> Dict[int, None]:  # type: ignore
        """Create branch map for Law task scheduling.

        Returns a single branch (0) since each fold is its own task instance.

        Returns:
            Dict[int, None]: Branch map with single entry {0: None}.
        """
        return {0: None}

    def output(self) -> Dict[str, Any]:
        """Define all output targets for this task.

        Important:
            If using this method in another Task, beware that for remote jobs, the output of this
            method will be wrapped in a Dict of Lists to account for each potential branch of the
            workflow. To avoid encountering this problem, use the `output_as_dict` method instead,
            which flattens the remote output to the same shape as the local version.
        """
        check_batch_system(system=str(self.workflow))  # type: ignore

        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "outputs": base.child("fold_results.json", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    @staticmethod
    def output_as_dict(fold_output: FoldTaskOutput) -> Dict[str, law.LocalTarget]:
        """Unpack local and remote inputs.

        1. Local is simply the Dict defined in the output method of the FoldTask
        2. Remote is instead a DotDict with 'collection' and 'jobs' fields.

        Returns:
            Dict[str, str]: Properly formatted output Dict with key:Target pairs
        """
        remote_collection: law.TargetCollection | None = fold_output.get("collection")  # type: ignore

        if remote_collection:
            if len(remote_collection) != 1:
                raise NotImplementedError(
                    "Currently the usage of branches in FoldTask is not supported. Instead, folds "
                    "have to be their own Task instance required by EnsembleTask."
                )
            return remote_collection[0]
        else:
            return fold_output  # type: ignore

    # run() is inherited from BaseFoldTask — no changes needed
