"""EnsembleTask - Manages training for a single ensemble group.

This module defines the EnsembleTask which is responsible for:
- Creating FoldTask instances for each cross-validation fold
- Managing fold configuration and parameter expansion
- Aggregating results from all folds into an ensemble results object
- Handling ensemble-specific training configurations

The task forms the fourth level of the task DAG hierarchy:
    MainTask
    └── EstimatorTask (one per estimator)
         └── SystematicTask (one per systematic variation)
              └── EnsembleTask (one per ensemble group)
                   └── FoldTask (one per cross-validation fold)
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import law
import luigi

from needle.law_tasks.fold import FoldTask
from needle.law_tasks.mixins import HydraMixin
from needle.tasks.base.ensemble import BaseEnsembleTask
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(BaseEnsembleTask, law.Task):
    """Task representing a single ensemble training run.

    Creates FoldTask instances for each cross-validation fold and aggregates their results
    into an ensemble results object.
    """

    def _fold_task_class(self) -> Type[luigi.Task]:
        return FoldTask

    def input(self) -> List[Dict[str, law.LocalTarget]]:
        """Retrieve and flatten fold task outputs for local use.

        Unwraps remote fold outputs (which are nested in collection/jobs structure) to match
        the local output format, ensuring consistent access to fold results regardless of
        whether tasks ran locally or remotely.

        Returns:
            List[Dict[str, law.LocalTarget]]: Flattened fold outputs, one per fold.
        """
        _remote_fold_outputs = super().input()
        return [FoldTask.output_as_dict(fold_output) for fold_output in _remote_fold_outputs]

    def output(self) -> Dict[str, Any]:
        """Define output target for aggregated ensemble results."""
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("ensemble_results.json", type="f"),
        }

    # run() is inherited from BaseEnsembleTask
