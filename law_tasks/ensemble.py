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
from orchestrator.results import EnsembleResults, FoldResults
from orchestrator.config import EstimatorConfig
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(law.Task, HydraMixin):
    rel_results_path = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="runs/ensembles",
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
        return os.path.abspath(self.rel_results_path)  # type: ignore
    
    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self):
        return [
            FoldTask.req(
                self,
                config_file=self.config_file,
                estimator=self.estimator,
                systematic=self.systematic,
                fold_index=fold_index,
            )
            for fold_index in range(self.estimator_config.expands.folds)
        ]

    def output(self) -> Dict[str, Any]:
        os.makedirs(self.abs_results_path, exist_ok=True)
        return {"outputs": law.LocalFileTarget(f"{self.abs_results_path}/ensemble_results.json")}

    def run(self):
        EnsembleResults().to_json(self.output()["outputs"].path)
