"""
Task for a single ensemble training, includes multiple folds.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import law

from law_tasks.fold import FoldTask
from ml.utils.config import MLConfig
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(law.Task):
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

    @property
    def results_dir(self) -> Path:
        return os.path.abspath(self.results_dir_path)  # type: ignore

    @property
    def config(self):
        return MLConfig.from_yaml(self.config_yaml)  # type: ignore

    def requires(self):
        return [
            FoldTask.req(
                self,
                fold=i,
                config_yaml=self.config_yaml,
                results_dir_path=self.results_dir,
            )
            for i in range(self.config.n_folds)
        ]

    def output(self) -> Dict[str, Any]:
        return {"outputs": law.LocalFileTarget(f"{self.results_dir}/ensemble_done.txt")}

    def run(self):
        with open(self.output()["outputs"].path, "w") as f:
            json.dump("done", f, indent=4)
