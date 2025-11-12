"""
Task for a single ensemble training, includes multiple folds.
"""
import os
from pathlib import Path
from typing import Any, Dict

import law

from law_tasks.fold import FoldTask
from law_tasks.mixins import HydraMixin
from orchestrator.results import EnsembleResults, FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(law.Task, HydraMixin):
    rel_results_path = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="runs/ensembles",
        significant=False,
    )

    def requires(self):
        return [
            FoldTask.req(
                self,
                fold_index=i,
                config_path=self.config_path,
                rel_results_path=self.rel_results_path,
            )
            for i in range(self.config.n_folds)
        ]

    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/ensemble_results.json")}

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    def run(self):
        ensemble_results = EnsembleResults()

        for fold_outputs in self.input():
            fold_result = FoldResults.from_json(fold_outputs["outputs"].path)
            ensemble_results.folds.append(fold_result)

        ensemble_results.to_json(self.output()["outputs"].path)
