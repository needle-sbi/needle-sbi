"""SystematicTask - Manages training for a single systematic variation."""

from __future__ import annotations

from typing import Any, Dict, Type

import law
import luigi

from needle.law_tasks.ensemble import EnsembleTask
from needle.law_tasks.mixins import HydraMixin
from needle.tasks.base.systematic import BaseSystematicTask
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("systematic")


class SystematicTask(BaseSystematicTask, law.Task):
    """Task representing a systematic uncertainty variation.

    Creates ``EnsembleTask`` instances for multiple ensemble runs of the same systematic.
    Aggregates results from all ensembles into a single systematic results object.
    """

    def _ensemble_task_class(self) -> Type[luigi.Task]:
        return EnsembleTask

    def output(self) -> Dict[str, Any]:
        """Define output target for aggregated systematic results."""
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("systematic_results.json", type="f"),
        }

    # run() is inherited from BaseSystematicTask
