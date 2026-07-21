"""EstimatorTask - Manages training across all systematic variations for one estimator."""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.law_tasks.mixins import HydraMixin
from needle.law_tasks.systematic import SystematicTask
from needle.tasks.base.estimator import BaseEstimatorTask
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("estimator")


class EstimatorTask(BaseEstimatorTask, law.Task):
    """Task representing a single estimator.

    Creates ``SystematicTask`` instances for each systematic shift defined in the config.
    Handles systematic configuration merging and aggregates results from all systematic
    variations.
    """

    def _systematic_task_class(self) -> Type[luigi.Task]:
        return SystematicTask

    def output(self) -> Dict[str, Any]:
        """Define output targets for aggregated estimator results."""
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("estimator_result.json", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    # run() is inherited from BaseEstimatorTask
