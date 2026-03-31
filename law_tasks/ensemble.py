"""
Task for a single ensemble training, includes multiple folds.
"""
import os
from pathlib import Path
from typing import Any, Dict

import law
import luigi

from law_tasks.fold import FoldTask
from law_tasks.mixins import HydraMixin
from orchestrator.config import EstimatorConfig
from orchestrator.results import EnsembleResults, FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(HydraMixin, law.Task):
    results_path: str = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        significant=False,
    )  # type: ignore
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore
    ensemble: int = luigi.IntParameter(
        description="Index of the ensemble (type: int).",
        default=0,
        significant=True,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self):
        return [
            FoldTask(
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=fold_index,
                results_path=os.path.join(self.abs_results_path, f"fold__{fold_index}"),
            )
            for fold_index in range(self.estimator_config.expands.folds)
        ]

    def output(self) -> Dict[str, Any]:
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "outputs": base.child("ensemble_results.json", type="f"),
        }

    def run(self) -> None:
        """Gather results from child FoldTask and merge them into own result container"""
        fold_results = [FoldResults.from_json(fold_output["outputs"].path) for fold_output in self.input()]
        EnsembleResults(folds=fold_results).to_json(self.output()["outputs"].path)
