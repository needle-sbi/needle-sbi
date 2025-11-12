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
    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="config.yaml",
        significant=True,
    )
    results_dir_path = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="runs/ensembles",
        significant=False,
    )
    multiprocessing_type = law.Parameter(
        description="Which multiprocessing library to use, options are `dask` and `torch`",
        significant=False,
        default="torch",
    )

    @property
    def results_dir(self) -> Path:
        return os.path.abspath(self.results_dir_path)  # type: ignore

    def requires(self):
        return [
            FoldTask.req(
                self,
                fold=i,
                config_yaml=self.config_yaml,
                results_dir_path=self.results_dir,
                multiprocessing_type=self.multiprocessing_type,
            )
            for i in range(self.config.n_folds)
        ]

    def output(self) -> Dict[str, Any]:
        return {"outputs": law.LocalFileTarget(f"{self.results_dir}/ensemble_results.json")}

    def run(self):
        ensemble_results = EnsembleResults()

        for fold_outputs in self.input():
            fold_result = FoldResults.from_json(fold_outputs["outputs"].path)
            ensemble_results.folds.append(fold_result)

        ensemble_results.to_json(self.output()["outputs"].path)
